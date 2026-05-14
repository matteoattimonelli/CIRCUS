from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


@dataclass(frozen=True)
class ItemKey:
    text: str
    img_path: str
    image: Optional[Image.Image] = None


class Retriever:
    """
    VLM2Vec/VLM2Vec-V2.0 retriever using transformers.

    Important:
      - model weights are loaded from VLM2Vec/VLM2Vec-V2.0
      - processor is loaded from the base Qwen2-VL model
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "VLM2Vec/VLM2Vec-V2.0",
        processor_id: Optional[str] = None,
        base_processor_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype: torch.dtype = torch.bfloat16,
        normalize: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        use_fast: bool = False,
        pooling: str = "last",
    ):
        self.device = device
        self.model_id = model_id
        self.base_processor_id = base_processor_id
        self.dtype = torch_dtype
        self.do_normalize = bool(normalize)
        self.pooling = pooling.lower().strip()
        if self.pooling not in {"last", "mean"}:
            raise ValueError(f"Unsupported pooling={pooling!r}; expected 'last' or 'mean'.")
        self._patched_processor_dir: Optional[tempfile.TemporaryDirectory] = None

        # VLM2Vec reuses the base Qwen2-VL processor.
        if processor_id is None:
            processor_id = self.base_processor_id
        self.processor_id = processor_id

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Processor must come from the base Qwen2-VL model
        self.processor = self._load_processor(self.processor_id, use_fast=use_fast)

        self.model = AutoModel.from_pretrained(
            self.model_id,
            **model_kwargs,
        ).to(self.device).eval()

        self.image_token = getattr(self.processor, "image_token", None)
        if self.image_token is None and hasattr(self.processor, "tokenizer"):
            self.image_token = getattr(self.processor.tokenizer, "image_token", None)
        if self.image_token is None:
            self.image_token = "<|image_pad|>"

        torch.set_grad_enabled(False)

    def _load_processor(self, processor_id: str, use_fast: bool) -> Any:
        try:
            return AutoProcessor.from_pretrained(
                processor_id,
                trust_remote_code=True,
                use_fast=use_fast,
            )
        except ValueError as exc:
            msg = "size must contain 'shortest_edge' and 'longest_edge' keys."
            if msg not in str(exc):
                raise

            patched_dir = self._patch_qwen2vl_preprocessor(processor_id)
            return AutoProcessor.from_pretrained(
                patched_dir,
                trust_remote_code=True,
                use_fast=use_fast,
                local_files_only=True,
            )

    def _patch_qwen2vl_preprocessor(self, processor_id: str) -> str:
        src = self._resolve_local_processor_dir(processor_id)

        preproc_path = src / "preprocessor_config.json"
        if not preproc_path.exists():
            raise FileNotFoundError(f"Missing preprocessor_config.json under {src}")

        data = json.loads(preproc_path.read_text())
        size = data.get("size")
        if isinstance(size, dict) and (
            "shortest_edge" not in size or "longest_edge" not in size
        ):
            data.pop("size", None)

        self._patched_processor_dir = tempfile.TemporaryDirectory(prefix="vlm2vec_proc_")
        dst = Path(self._patched_processor_dir.name)

        for item in src.iterdir():
            target = dst / item.name
            if item.name == "preprocessor_config.json":
                target.write_text(json.dumps(data, indent=2) + "\n")
            else:
                target.symlink_to(item)

        return str(dst)

    def _resolve_local_processor_dir(self, processor_id: str) -> Path:
        src = Path(processor_id)
        if src.exists():
            return src

        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = hub_root / f"models--{processor_id.replace('/', '--')}" / "snapshots"
        snapshots = sorted(repo_dir.glob("*"))
        if snapshots:
            return snapshots[-1]

        raise FileNotFoundError(
            f"Could not locate a local processor snapshot for {processor_id!r}."
        )

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if k in {"pixel_values", "pixel_values_videos"}:
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

    def _last_token_pool(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden[:, -1, :]
        mask = attention_mask.to(dtype=torch.bool)
        token_pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand_as(attention_mask)
        last_idx = token_pos.masked_fill(~mask, -1).max(dim=1).values.clamp_min(0)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, last_idx]

    def _masked_mean_pool(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask[..., None].bool()
        hidden = hidden.masked_fill(~mask, 0.0)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return hidden.sum(dim=1) / denom

    def _pool_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.pooling == "last":
            return self._last_token_pool(hidden, attention_mask)
        return self._masked_mean_pool(hidden, attention_mask)

    def _image_only_prompt(self) -> str:
        return f"{self.image_token} Represent the given image."

    def _multimodal_prompt(self, text: str) -> str:
        text = (text or "").strip()
        if text:
            return f"{self.image_token} {text}"
        return self._image_only_prompt()

    @torch.inference_mode()
    def _embed_text_only(self, texts: List[str]) -> torch.Tensor:
        batch = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = self._move_to_device(batch)

        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        reps = self._pool_hidden(hidden, batch.get("attention_mask"))
        return self._normalize(reps)

    @torch.inference_mode()
    def _embed_image_only(self, images: List[Image.Image]) -> torch.Tensor:
        prompts = [self._image_only_prompt() for _ in images]

        batch = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = self._move_to_device(batch)

        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        reps = self._pool_hidden(hidden, batch.get("attention_mask"))
        return self._normalize(reps)

    @torch.inference_mode()
    def _embed_multimodal(self, images: List[Image.Image], texts: List[str]) -> torch.Tensor:
        prompts = [self._multimodal_prompt(t) for t in texts]

        batch = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = self._move_to_device(batch)

        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        reps = self._pool_hidden(hidden, batch.get("attention_mask"))
        return self._normalize(reps)

    def embed_queries(self, keys: List[ItemKey]) -> torch.Tensor:
        if not keys:
            return torch.empty((0, 0), dtype=torch.float32)

        has_img = [k.image is not None for k in keys]
        has_txt = [bool((k.text or "").strip()) for k in keys]

        if all((not i) and t for i, t in zip(has_img, has_txt)):
            return self._embed_text_only([k.text or "" for k in keys])

        if all(i and (not t) for i, t in zip(has_img, has_txt)):
            return self._embed_image_only([k.image for k in keys])  # type: ignore[arg-type]

        if all(i and t for i, t in zip(has_img, has_txt)):
            return self._embed_multimodal(
                [k.image for k in keys],  # type: ignore[arg-type]
                [k.text or "" for k in keys],
            )

        outs: List[Optional[torch.Tensor]] = [None] * len(keys)

        idx_text = [i for i, k in enumerate(keys) if (k.image is None) and bool((k.text or "").strip())]
        idx_img = [i for i, k in enumerate(keys) if (k.image is not None) and not bool((k.text or "").strip())]
        idx_mm = [i for i, k in enumerate(keys) if (k.image is not None) and bool((k.text or "").strip())]

        if idx_text:
            reps = self._embed_text_only([keys[i].text or "" for i in idx_text])
            for i, rep in zip(idx_text, reps):
                outs[i] = rep

        if idx_img:
            reps = self._embed_image_only([keys[i].image for i in idx_img])  # type: ignore[arg-type]
            for i, rep in zip(idx_img, reps):
                outs[i] = rep

        if idx_mm:
            reps = self._embed_multimodal(
                [keys[i].image for i in idx_mm],  # type: ignore[arg-type]
                [keys[i].text or "" for i in idx_mm],
            )
            for i, rep in zip(idx_mm, reps):
                outs[i] = rep

        if any(x is None for x in outs):
            bad = [i for i, x in enumerate(outs) if x is None]
            raise ValueError(f"Unsupported mixed query batch at positions {bad}")

        return torch.stack([x for x in outs if x is not None], dim=0)

    def embed_targets(self, keys: List[ItemKey]) -> torch.Tensor:
        if not all(k.image is not None for k in keys):
            raise ValueError("All targets must have images.")
        return self._embed_image_only([k.image for k in keys])  # type: ignore[arg-type]
