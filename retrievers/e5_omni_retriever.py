# retrievers/e5_omni_retriever.py
from __future__ import annotations

import logging
logging.getLogger().setLevel(logging.ERROR)

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

# Required by Qwen2.5-Omni
# pip install qwen-omni-utils  (or whatever package provides qwen_omni_utils in your env)
from qwen_omni_utils import process_mm_info


# Your eval script defines the same ItemKey; we just need compatible fields.
@dataclass(frozen=True)
class ItemKey:
    text: str
    img_path: str
    image: Optional[Image.Image] = None


class Retriever:
    """
    e5-omni-7B embedding retriever (Haon-Chen/e5-omni-7B).

    Embedding rule (as in the HF snippet):
      1) Build chat messages (image + text)
      2) processor.apply_chat_template(..., add_generation_prompt=True) + "<|endoftext|>"
      3) process_mm_info(...) to build multimodal inputs
      4) model.prepare_inputs_for_generation(..., cache_position=arange(seq_len))
      5) forward(output_hidden_states=True)
      6) reps = hidden_states[-1][:, -1] (last token)
      7) L2 normalize
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "Haon-Chen/e5-omni-7B",
        processor_id: str = "Qwen/Qwen2.5-Omni-7B",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = "sdpa",
        normalize: bool = True,
    ):
        self.device = torch.device(device if ("cuda" in device and torch.cuda.is_available()) else "cpu")
        self.model_id = model_id
        self.processor_id = processor_id
        self.dtype = torch_dtype
        self.do_normalize = bool(normalize)

        self.processor = AutoProcessor.from_pretrained(self.processor_id)

        # Match the snippet
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        # Load model
        load_kwargs: Dict[str, Any] = dict(torch_dtype=self.dtype)
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                self.model_id, **load_kwargs
            ).to(self.device).eval()
        except TypeError:
            # Some envs / transformer versions may not accept attn_implementation
            load_kwargs.pop("attn_implementation", None)
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                self.model_id, **load_kwargs
            ).to(self.device).eval()

        # Some Qwen models expose padding_side attribute too
        if hasattr(self.model, "padding_side"):
            self.model.padding_side = "left"

        torch.set_grad_enabled(False)

    # -------------------------
    # Prompting helpers
    # -------------------------
    def _prompt_for_query(self, text: str) -> str:
        # Your eval already prepends dataset instruction when needed.
        return (text or "").strip()

    def _prompt_for_target(self, text: str) -> str:
        # Often empty for gallery; keep it valid.
        t = (text or "").strip()
        return t if t else "Describe the image."

    def _build_conversations(
        self, images: List[Image.Image], prompts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        """
        Returns: list of conversations, each conversation is a list of messages.
        This matches process_mm_info's accepted types:
          Union[List[Dict], List[List[Dict]]]
        """
        conversations: List[List[Dict[str, Any]]] = []
        for img, txt in zip(images, prompts):
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": txt},
                        ],
                    }
                ]
            )
        return conversations

    @staticmethod
    def _ensure_list(x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def _apply_chat_template_batch(self, conversations: List[List[Dict[str, Any]]]) -> List[str]:
        """
        transformers' apply_chat_template can return:
          - a single string
          - a list of strings
        depending on version / input type.
        """
        out = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        if isinstance(out, str):
            texts = [out]
        else:
            texts = list(out)

        # Match snippet: append <|endoftext|>
        return [t + "<|endoftext|>" for t in texts]

    def _process_mm_info_batch(
        self, conversations: List[List[Dict[str, Any]]]
    ) -> Tuple[Any, Any, Any]:
        """
        process_mm_info is expected to accept list-of-conversations.
        If not, we fall back to per-sample processing and concatenate.
        """
        try:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                conversations, use_audio_in_video=True
            )
            return audio_inputs, image_inputs, video_inputs
        except Exception:
            # Fallback: per-sample
            audios, images, videos = [], [], []
            for conv in conversations:
                a, i, v = process_mm_info(conv, use_audio_in_video=True)
                audios.append(a)
                images.append(i)
                videos.append(v)
            return audios, images, videos

    def _to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)
        return inputs

    # -------------------------
    # Core embedding
    # -------------------------
    @torch.inference_mode()
    def _embed_batch(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        assert len(images) == len(prompts)

        conversations = self._build_conversations(images, prompts)
        texts = self._apply_chat_template_batch(conversations)

        audio_inputs, image_inputs, video_inputs = self._process_mm_info_batch(conversations)

        # The processor typically expects:
        #   text=[...], images=[...], videos=[...] (or None), audio=[...] (or None)
        # If process_mm_info returns "None" for a modality, don't pass it.
        kwargs: Dict[str, Any] = dict(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )

        # audio/images/videos can be in different nested forms depending on util version.
        # We pass them through as provided (as in the HF snippet).
        if audio_inputs is not None:
            kwargs["audio"] = audio_inputs
        if image_inputs is not None:
            kwargs["images"] = image_inputs
        if video_inputs is not None:
            kwargs["videos"] = video_inputs

        inputs = self.processor(**kwargs)
        inputs = self._to_device(inputs)

        # cache_position like in snippet
        seq_len = inputs["input_ids"].shape[1]
        cache_position = torch.arange(0, seq_len, device=self.device)

        prepared = self.model.prepare_inputs_for_generation(
            **inputs, use_cache=True, cache_position=cache_position
        )

        outputs = self.model(**prepared, return_dict=True, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)
        reps = last_hidden[:, -1]               # (B, H)

        if self.do_normalize:
            reps = F.normalize(reps, p=2, dim=-1)
        return reps

    # -------------------------
    # Public API
    # -------------------------
    def embed_queries(self, keys: List[ItemKey]) -> torch.Tensor:
        imgs, prompts = [], []
        for k in keys:
            if k.image is None:
                raise ValueError("e5-omni Retriever requires ItemKey.image for queries.")
            imgs.append(k.image)
            prompts.append(self._prompt_for_query(k.text))
        return self._embed_batch(imgs, prompts)

    def embed_targets(self, keys: List[ItemKey]) -> torch.Tensor:
        imgs, prompts = [], []
        for k in keys:
            if k.image is None:
                raise ValueError("e5-omni Retriever requires ItemKey.image for targets.")
            imgs.append(k.image)
            prompts.append(self._prompt_for_target(k.text))
        return self._embed_batch(imgs, prompts)