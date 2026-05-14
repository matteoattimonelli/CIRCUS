'''
Docstring per retrievers.mmembed_retriever

env gme
'''


from __future__ import annotations

import os
from typing import List

import torch
from PIL import Image
from transformers import AutoModel


class Retriever:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "nvidia/MM-Embed",
        max_length: int = 4096,
        instruction: str = os.environ.get("Q_INSTRUCTION", "Retrieve a day-to-day image that aligns with the modification instructions of the provided image."),
        batch_size: int = 8,
        image_root: str | None = None,  # <-- add this
    ):
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        self.instruction = instruction
        self.batch_size = batch_size
        self.image_root = image_root or os.environ.get("MMEB_IMAGE_ROOT")


        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def _resolve_img_path(self, img_path: str) -> str:
        img_path = (img_path or "").strip()
        if not img_path:
            return ""
        # If already absolute and exists, keep it
        if os.path.isabs(img_path) and os.path.exists(img_path):
            return img_path
        # Otherwise, join with image_root if provided
        if self.image_root is not None:
            candidate = os.path.join(self.image_root, img_path)
            if os.path.exists(candidate):
                return candidate
        # Fallback: try as-is (will error with a helpful message)
        return img_path


    def _key_to_input(self, key) -> dict:
        d = {}

        txt = (getattr(key, "text", "") or "").strip()
        if txt:
            d["txt"] = txt

        # ✅ prefer in-memory image (attacks)
        img_obj = getattr(key, "image", None)
        if img_obj is not None:
            # ensure RGB + detached copy (avoid shared state surprises)
            if isinstance(img_obj, Image.Image):
                d["img"] = img_obj.convert("RGB")
            else:
                raise TypeError(f"key.image must be a PIL.Image.Image, got: {type(img_obj)}")
            return d

        # 🔁 fallback to img_path only if key.image is None
        img_path = self._resolve_img_path((getattr(key, "img_path", "") or "").strip())
        if img_path:
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Image not found: '{img_path}'. "
                    f"Original relative path was '{getattr(key, 'img_path', '')}'. "
                    f"Set Retriever(image_root=...) or MMEB_IMAGE_ROOT."
                )
            d["img"] = Image.open(img_path).convert("RGB")

        return d

    @torch.no_grad()
    def _encode(self, inputs: List[dict], is_query: bool) -> torch.Tensor:
        outs = []
        for i in range(0, len(inputs), self.batch_size):
            chunk = inputs[i : i + self.batch_size]
            if is_query:
                out = self.model.encode(
                    chunk,
                    is_query=True,
                    instruction=self.instruction,
                    max_length=self.max_length,
                )["hidden_states"]
            else:
                out = self.model.encode(chunk, max_length=self.max_length)["hidden_states"]
            outs.append(out.detach().cpu())
        return torch.cat(outs, dim=0)

    def embed_queries(self, keys: List) -> torch.Tensor:
        inputs = [self._key_to_input(k) for k in keys]
        return self._encode(inputs, is_query=True)

    def embed_targets(self, keys: List) -> torch.Tensor:
        inputs = [self._key_to_input(k) for k in keys]
        return self._encode(inputs, is_query=False)
