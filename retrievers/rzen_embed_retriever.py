from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


@dataclass(frozen=True)
class ItemKey:
    text: str
    img_path: str
    image: Optional[Image.Image] = None


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO

    return h_bar, w_bar


class Retriever:
    """
    Retriever for qihoo360/RzenEmbed.

    This implementation follows the official rzen_embed_inference.py path
    closely enough for dense retrieval:
      - Qwen2VLForConditionalGeneration base model
      - Qwen-style multimodal prompt formatting
      - last valid token pooling
      - official target-side image instruction
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "qihoo360/RzenEmbed",
        torch_dtype: torch.dtype = torch.bfloat16,
        normalize: bool = True,
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        max_length: int = 2000,
        attn_implementation: str = "sdpa",
        use_fast_processor: bool = False,
    ):
        self.device = device
        self.model_id = model_id
        self.dtype = torch_dtype
        self.do_normalize = bool(normalize)
        self.max_length = int(max_length)
        self.default_instruction = "You are a helpful assistant."
        self.target_instruction = "Represent the given image."

        min_pixels = int(min_image_tokens) * 28 * 28
        max_pixels = int(max_image_tokens) * 28 * 28

        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        config.padding_side = "right"
        config.use_cache = False

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=use_fast_processor,
            trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "right"
        self.image_token = getattr(self.processor, "image_token", None)
        if self.image_token is None and hasattr(self.processor, "tokenizer"):
            self.image_token = getattr(self.processor.tokenizer, "image_token", None)
        if self.image_token is None:
            self.image_token = "<|image_pad|>"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        torch.set_grad_enabled(False)

    def _move(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if "pixel" in k:
                    out[k] = v.to(self.device, dtype=self.dtype)
                else:
                    out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_normalize:
            x = F.normalize(x, dim=-1)
        return x

    def _last_token_pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden[:, -1, :]
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return hidden[:, -1, :]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, sequence_lengths]

    def _process_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        return image.resize((resized_width, resized_height))

    def _build_messages(self, text: Optional[str], image: Optional[Image.Image]) -> tuple[str, List[Image.Image]]:
        input_str = ""
        processed_images: List[Image.Image] = []

        if image is not None:
            processed = self._process_image(image)
            processed_images.append(processed)
            input_str += "<|vision_start|><|image_pad|><|vision_end|>"

        if text:
            input_str += text

        msg = (
            f"<|im_start|>system\n{self.default_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{input_str}<|im_end|>\n"
            "<|im_start|>assistant\n<|endoftext|>"
        )
        return msg, processed_images

    @torch.inference_mode()
    def _embed(self, texts: List[Optional[str]], images: List[Optional[Image.Image]]) -> torch.Tensor:
        if not texts and not images:
            return torch.empty((0, 0))

        prompts: List[str] = []
        flat_images: List[Image.Image] = []
        for text, image in zip(texts, images):
            prompt, processed_images = self._build_messages(text, image)
            prompts.append(prompt)
            flat_images.extend(processed_images)

        image_inputs: Optional[List[Image.Image]] = flat_images if flat_images else None
        batch = self.processor(
            text=prompts,
            images=image_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = self._move(batch)

        outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        reps = self._last_token_pool(hidden, batch.get("attention_mask"))
        return self._normalize(reps)

    def embed_queries(self, keys: List[ItemKey]) -> torch.Tensor:
        if not keys:
            return torch.empty((0, 0))
        texts = [(k.text or "").strip() or None for k in keys]
        images = [k.image for k in keys]
        return self._embed(texts, images)

    def embed_targets(self, keys: List[ItemKey]) -> torch.Tensor:
        if not all(k.image is not None for k in keys):
            raise ValueError("Targets must have images")
        texts = [self.target_instruction for _ in keys]
        images = [k.image for k in keys]
        return self._embed(texts, images)  # type: ignore[arg-type]
