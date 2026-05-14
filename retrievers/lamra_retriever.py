# retrievers/lamra_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

try:
    from retrievers.models.qwen2_vl import Qwen2VLRetForConditionalGeneration  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import Qwen2VLRetForConditionalGeneration.\n"
        "Copy LamRA's `models/qwen2_vl.py` into `retrievers/models/qwen2_vl.py` "
        "or add the LamRA repo to PYTHONPATH.\n"
        f"Original error: {e}"
    )


def _import_process_vision_info():
    try:
        from qwen_vl_utils.vision_process import process_vision_info  # type: ignore
        return process_vision_info
    except Exception as e:
        raise RuntimeError(
            "Failed to import `process_vision_info` from qwen-vl-utils.\n"
            "Install: `pip install -U qwen-vl-utils torchvision`\n"
            f"Import error: {e}"
        )

process_vision_info = _import_process_vision_info()


@dataclass(frozen=True)
class ItemKey:
    text: str
    img_path: str
    image: Optional[Image.Image] = None


class Retriever:
    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "code-kunkun/LamRA-Ret",
        original_model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 1024,
        normalize: bool = True,
    ):
        self.device = device
        self.model_id = model_id
        self.original_model_id = original_model_id
        self.dtype = torch_dtype
        self.max_length = int(max_length)
        self.do_normalize = bool(normalize)

        self.processor = AutoProcessor.from_pretrained(self.original_model_id)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.model_max_length = self.max_length

        self.model = Qwen2VLRetForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        self.emb_token = "<emb>"
        num_new = self.tokenizer.add_tokens([self.emb_token])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.emb_token_id = self.tokenizer.convert_tokens_to_ids(self.emb_token)
        self.model.config.emb_token_ids = [self.emb_token_id]

        self.debug = bool(int(os.environ.get("LAMRA_DEBUG", "0")))

        if self.debug:
            print("[LAMRA_DEBUG] Retriever initialized")
            print(f"[LAMRA_DEBUG] model_id={self.model_id}")
            print(f"[LAMRA_DEBUG] original_model_id={self.original_model_id}")
            print(f"[LAMRA_DEBUG] emb_token_id={self.emb_token_id}")
            print(f"[LAMRA_DEBUG] tokenize('<emb>')={self.tokenizer.tokenize('<emb>')}")
            print(f"[LAMRA_DEBUG] num_new_tokens_added_for_<emb>={num_new}")

        torch.set_grad_enabled(False)

    def _construct_messages(
        self,
        img: Optional[Image.Image],
        txt: str,
        is_query: bool,
    ) -> List[Dict[str, Any]]:
        txt = (txt or "").strip()
        user_content: List[Dict[str, Any]] = []

        if img is not None:
            user_content.append({"type": "image", "image": img})

        if img is not None and txt:
            user_content.append(
                {"type": "text", "text": f"{txt}\nSummarize above image and sentence in one word: "}
            )
        elif img is None and txt:
            user_content.append(
                {"type": "text", "text": f"{txt}\nSummarize above sentence in one word: "}
            )
        elif img is not None and not txt:
            user_content.append(
                {"type": "text", "text": "\nSummarize above image in one word: "}
            )
        else:
            user_content.append({"type": "text", "text": "Summarize in one word: "})

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]

    def _messages_for_batch(
        self,
        images: List[Optional[Image.Image]],
        texts: List[str],
    ) -> List[List[Dict[str, Any]]]:
        conversations: List[List[Dict[str, Any]]] = []
        for img, txt in zip(images, texts):
            conversations.append(self._construct_messages(img=img, txt=txt, is_query=True))
        return conversations

    def _make_inputs(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        chat_texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in conversations
        ]

        vision_out = process_vision_info(conversations)

        if isinstance(vision_out, tuple) and len(vision_out) == 3:
            image_inputs, video_inputs, vision_infos = vision_out
        elif isinstance(vision_out, tuple) and len(vision_out) == 2:
            image_inputs, video_inputs = vision_out
            vision_infos = None
        else:
            raise RuntimeError(f"Unexpected process_vision_info output: {type(vision_out)}")

        kwargs: Dict[str, Any] = dict(
            text=chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        if image_inputs is not None:
            try:
                if len(image_inputs) > 0:
                    kwargs["images"] = image_inputs
            except TypeError:
                kwargs["images"] = image_inputs

        if video_inputs is not None:
            try:
                if len(video_inputs) > 0:
                    kwargs["videos"] = video_inputs
            except TypeError:
                kwargs["videos"] = video_inputs

        if vision_infos is not None:
            kwargs["vision_infos"] = vision_infos

        inputs = self.processor(**kwargs)

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                if k == "pixel_values":
                    inputs[k] = v.to(self.device, dtype=self.dtype)
                else:
                    inputs[k] = v.to(self.device)

        return inputs

    @torch.inference_mode()
    def _embed_from_text_and_images(
        self,
        images: List[Optional[Image.Image]],
        texts: List[str],
    ) -> torch.Tensor:
        conversations = self._messages_for_batch(images, texts)
        inputs = self._make_inputs(conversations)
        inputs["labels"] = inputs["input_ids"]

        out = self.model(**inputs, inference=True)
        emb = out[0] if isinstance(out, tuple) else out

        if self.do_normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    def embed_queries(self, keys: List[ItemKey]) -> torch.Tensor:
        images: List[Optional[Image.Image]] = []
        texts: List[str] = []
        for k in keys:
            images.append(k.image)
            texts.append(k.text or "")
        return self._embed_from_text_and_images(images, texts)

    def embed_targets(self, keys: List[ItemKey]) -> torch.Tensor:
        images: List[Optional[Image.Image]] = []
        texts: List[str] = []
        for k in keys:
            if k.image is None:
                raise ValueError("LamRA Retriever requires ItemKey.image for targets.")
            images.append(k.image)
            texts.append(k.text or "")
        return self._embed_from_text_and_images(images, texts)