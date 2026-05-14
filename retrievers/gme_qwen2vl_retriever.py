# retrievers/gme_qwen2vl_retriever.py
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoModel
from transformers.utils.versions import require_version


'''
export GME_USE_FUSED=1
export GME_BATCH_SIZE=32
export GME_T_IS_QUERY=0
export GME_DEVICE_MAP=cuda
export GME_DTYPE=bfloat16   # if it works; else float16

'''


def _chunked(seq: Sequence[Any], bs: int):
    for i in range(0, len(seq), bs):
        yield seq[i : i + bs]


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


class Retriever:
    """
    EmbeddingRetriever for Alibaba-NLP/gme-Qwen2-VL-7B-Instruct.

    Fixes vs naive version:
      - Prefer key.image (PIL) over key.img_path (so evaluator-side attacks work)
      - Resolve relative paths via GME_IMAGE_ROOT
      - If remote code doesn't support PIL, fallback to saving PIL -> temp file paths (cached)
      - IMPORTANT: batch fused queries (so queries aren't 1-by-1 slow)

    Env vars:
      GME_MODEL_NAME         default: "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
      GME_DTYPE              default: "float16"   (or "bfloat16", "float32")
      GME_DEVICE_MAP         default: "auto" (or "cuda" to force a single GPU)
      GME_BATCH_SIZE         default: "16"
      GME_Q_INSTR            default: "Represent the user's input."
      GME_T_IS_QUERY         default: "0"   (0/1)  (targets usually False)
      GME_USE_FUSED          default: "0"   (0/1)  (FashionIQ/CIRR usually needs 1)
      GME_IMAGE_ROOT         default: "" (optional root for relative image paths)

      # Optional: override query-mode behavior when fused disabled:
      # If 1: when both text+image exist but fused is disabled, embed BOTH by using fused anyway.
      # (useful to avoid accidentally running image-only queries)
      GME_FORCE_FUSED_WHEN_BOTH default: "0"
    """

    def __init__(self, device: str = "cuda"):
        require_version(
            "transformers<4.52.0",
            "Remote code has issues with transformers>=4.52.0. "
            "Please downgrade: pip install transformers==4.51.3",
        )

        self.model_name = os.environ.get("GME_MODEL_NAME", "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct")

        dtype_str = os.environ.get("GME_DTYPE", "float16").lower()
        if dtype_str in ("fp16", "float16", "half"):
            self.dtype = torch.float16
        elif dtype_str in ("bf16", "bfloat16"):
            self.dtype = torch.bfloat16
        elif dtype_str in ("fp32", "float32"):
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unknown GME_DTYPE={dtype_str}")

        self.device_map = os.environ.get("GME_DEVICE_MAP", "auto")
        self.batch_size = int(os.environ.get("GME_BATCH_SIZE", "16"))

        self.q_instruction = os.environ.get("Q_INSTRUCTION", "Represent the user's input.")
        self.t_is_query = os.environ.get("GME_T_IS_QUERY", "0").strip() in ("1", "true", "True", "yes", "YES")
        self.use_fused = os.environ.get("GME_USE_FUSED", "0").strip() in ("1", "true", "True", "yes", "YES")

        self.force_fused_when_both = os.environ.get("GME_FORCE_FUSED_WHEN_BOTH", "0").strip() in (
            "1", "true", "True", "yes", "YES"
        )

        self.image_root = os.environ.get("GME_IMAGE_ROOT", "").strip() or None

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()

        # Temp directory + cache for PIL->path fallback
        self._tmpdir = tempfile.TemporaryDirectory(prefix="gme_imgs_")
        self._pil_to_path: Dict[int, str] = {}

    # ---------------- path / image helpers ----------------

    def _resolve_path(self, p: str) -> str:
        p = (p or "").strip()
        if not p:
            return ""
        if os.path.isabs(p) and os.path.exists(p):
            return p
        if self.image_root:
            cand = os.path.join(self.image_root, p)
            if os.path.exists(cand):
                return cand
        return p

    def _pil_to_tmp_path(self, img: Image.Image) -> str:
        """Save PIL to a stable temp file (cached by object id) and return its path."""
        key = id(img)
        prev = self._pil_to_path.get(key)
        if prev and os.path.exists(prev):
            return prev

        out_path = os.path.join(self._tmpdir.name, f"{key}.png")
        img.save(out_path, format="PNG")
        self._pil_to_path[key] = out_path
        return out_path

    def _get_image_input(self, k: Any) -> Union[str, Image.Image, None]:
        """
        Return either:
          - PIL Image (preferred if k.image exists)
          - resolved path (if no PIL)
          - None if no image
        """
        pil = getattr(k, "image", None)
        if isinstance(pil, Image.Image):
            return pil.convert("RGB")

        p = self._resolve_path(_as_str(getattr(k, "img_path", "")))
        if p:
            return p
        return None

    # ---------------- embedding calls ----------------

    @torch.no_grad()
    def _embed_texts(self, texts: List[str], instruction: Optional[str] = None) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for batch in _chunked(texts, self.batch_size):
            emb = self.model.get_text_embeddings(texts=list(batch), instruction=instruction)  # [B,D]
            outs.append(emb.detach().float().cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    @torch.no_grad()
    def _embed_images(self, images: List[Union[str, Image.Image]], is_query: bool) -> torch.Tensor:
        """
        Try PIL first. If remote code rejects PIL, fallback to temp paths.
        """
        outs: List[torch.Tensor] = []
        for batch in _chunked(images, self.batch_size):
            b = list(batch)
            try:
                emb = self.model.get_image_embeddings(images=b, is_query=is_query)  # [B,D]
            except Exception:
                b2: List[Union[str, Image.Image]] = []
                for x in b:
                    if isinstance(x, Image.Image):
                        b2.append(self._pil_to_tmp_path(x))
                    else:
                        b2.append(x)
                emb = self.model.get_image_embeddings(images=b2, is_query=is_query)
            outs.append(emb.detach().float().cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    @torch.no_grad()
    def _embed_fused(self, texts: List[str], images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        Try PIL first. If remote code rejects PIL, fallback to temp paths.
        """
        outs: List[torch.Tensor] = []
        for bt, bi in zip(_chunked(texts, self.batch_size), _chunked(images, self.batch_size)):
            bt_l = list(bt)
            bi_l = list(bi)
            try:
                emb = self.model.get_fused_embeddings(texts=bt_l, images=bi_l)  # [B,D]
            except Exception:
                bi2: List[Union[str, Image.Image]] = []
                for x in bi_l:
                    if isinstance(x, Image.Image):
                        bi2.append(self._pil_to_tmp_path(x))
                    else:
                        bi2.append(x)
                emb = self.model.get_fused_embeddings(texts=bt_l, images=bi2)
            outs.append(emb.detach().float().cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    # ---------------- public API ----------------

    def embed_queries(self, keys: List[Any]) -> torch.Tensor:
        """
        Correct + fast behavior:
          - If fused enabled: batch all keys that have (text AND image) into fused calls.
            Image-only -> image embeddings (is_query=True)
            Text-only  -> text embeddings with instruction
          - If fused disabled:
            By default: image dominates (if image exists -> image embedding), else text.
            Optionally: GME_FORCE_FUSED_WHEN_BOTH=1 makes (text+image) still go fused.
        """
        N = len(keys)
        if N == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        # classify keys
        fused_idx: List[int] = []
        fused_texts: List[str] = []
        fused_imgs: List[Union[str, Image.Image]] = []

        img_idx: List[int] = []
        img_vals: List[Union[str, Image.Image]] = []

        txt_idx: List[int] = []
        txt_vals: List[str] = []

        for i, k in enumerate(keys):
            t = _as_str(getattr(k, "text", "")).strip()
            im = self._get_image_input(k)
            has_t = bool(t)
            has_im = im is not None

            if (self.use_fused or self.force_fused_when_both) and has_t and has_im:
                fused_idx.append(i)
                fused_texts.append(t)
                fused_imgs.append(im)  # type: ignore[arg-type]
            elif has_im:
                img_idx.append(i)
                img_vals.append(im)  # type: ignore[arg-type]
            else:
                txt_idx.append(i)
                txt_vals.append(t)

        out_fused = self._embed_fused(fused_texts, fused_imgs) if fused_texts else None
        out_img = self._embed_images(img_vals, is_query=True) if img_vals else None
        out_txt = self._embed_texts(txt_vals, instruction=self.q_instruction) if txt_vals else None

        # infer D
        D = 0
        for x in (out_fused, out_img, out_txt):
            if x is not None and x.numel():
                D = x.shape[1]
                break

        out = torch.empty((N, D), dtype=torch.float32)
        if out_fused is not None:
            for j, i in enumerate(fused_idx):
                out[i] = out_fused[j]
        if out_img is not None:
            for j, i in enumerate(img_idx):
                out[i] = out_img[j]
        if out_txt is not None:
            for j, i in enumerate(txt_idx):
                out[i] = out_txt[j]

        return out

    def embed_targets(self, keys: List[Any]) -> torch.Tensor:
        """
        Targets:
          - If image exists: image embedding with is_query=GME_T_IS_QUERY (usually False)
          - Else: text embedding (instruction=None)
        """
        N = len(keys)
        if N == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        img_idx: List[int] = []
        img_vals: List[Union[str, Image.Image]] = []

        txt_idx: List[int] = []
        txt_vals: List[str] = []

        for i, k in enumerate(keys):
            t = _as_str(getattr(k, "text", "")).strip()
            im = self._get_image_input(k)
            if im is not None:
                img_idx.append(i)
                img_vals.append(im)  # type: ignore[arg-type]
            else:
                txt_idx.append(i)
                txt_vals.append(t)

        out_img = self._embed_images(img_vals, is_query=self.t_is_query) if img_vals else None
        out_txt = self._embed_texts(txt_vals, instruction=None) if txt_vals else None

        D = 0
        for x in (out_img, out_txt):
            if x is not None and x.numel():
                D = x.shape[1]
                break

        out = torch.empty((N, D), dtype=torch.float32)
        if out_txt is not None:
            for j, i in enumerate(txt_idx):
                out[i] = out_txt[j]
        if out_img is not None:
            for j, i in enumerate(img_idx):
                out[i] = out_img[j]

        return out