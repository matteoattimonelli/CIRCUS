#!/usr/bin/env python3
"""Subset evaluator for CIRCO (multi-positive M-BEIR scoring).

This is a public-release evaluator. Only two query-side ablations from the
paper are runnable:

    * img_black     -- replace the query image with a same-size black image
    * text_drop_all -- drop all text from the query

Both ablations are loaded from the canonical whitelist JSONs in
``ablation_configs/`` and any other JSON shape is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Optional Hugging Face login
# =============================================================================

def maybe_hf_login(log: logging.Logger):
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=True)
        log.info("[hf] Logged in via HF_TOKEN")
    except Exception as e:
        log.warning("[hf] login skipped/failed: %s", str(e))


# =============================================================================
# Contract
# =============================================================================

@dataclass(frozen=True)
class ItemKey:
    text: str
    img_path: str
    image: Optional[Image.Image] = None


class EmbeddingRetriever:
    def embed_queries(self, keys: List[ItemKey]) -> torch.Tensor:
        raise NotImplementedError

    def embed_targets(self, keys: List[ItemKey]) -> torch.Tensor:
        raise NotImplementedError


def load_retriever(module_name: str, class_name: str, device: str) -> EmbeddingRetriever:
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    obj = cls(device=device)
    for m in ("embed_queries", "embed_targets"):
        if not hasattr(obj, m):
            raise TypeError(f"Retriever missing method {m}")
    return obj


# =============================================================================
# Ablations (whitelisted: img_black, text_drop_all)
# =============================================================================

ABLATION_SEED = 123
AblationKind = Literal["img_black", "text_drop_all"]


@dataclass
class Ablation:
    kind: AblationKind


def _sanitize_for_path(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._+\-=]", "-", s)
    if len(s) > max_len:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        s = s[: max_len // 2] + f"__{h}__" + s[-max_len // 2 :]
    return s


def _safe_slug(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._+\-=]", "-", s)
    if len(s) > max_len:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        s = s[: max_len // 2] + f"__{h}__" + s[-max_len // 2 :]
    return s


def ablation_signature(abl: Optional[Ablation]) -> str:
    """Cache-key signature. Format must remain stable for the existing raw cache."""
    if abl is None:
        return "noabl"
    if abl.kind == "img_black":
        base = f"abl=img_black_seed={ABLATION_SEED}"
    elif abl.kind == "text_drop_all":
        base = f"abl=txt_dropall_seed={ABLATION_SEED}"
    else:
        raise ValueError(f"Unsupported ablation kind: {abl.kind}")
    return _sanitize_for_path(base, max_len=160)


_BLACK_CACHE: Dict[Tuple[int, int], Image.Image] = {}


def _black_image_same_size(size: Tuple[int, int]) -> Image.Image:
    if size not in _BLACK_CACHE:
        _BLACK_CACHE[size] = Image.new("RGB", size, (0, 0, 0))
    return _BLACK_CACHE[size].copy()


def apply_ablation(keys: List[ItemKey], abl: Optional[Ablation]) -> List[ItemKey]:
    """Apply the ablation to query keys. Always applies to query side only."""
    if abl is None:
        return keys

    out: List[ItemKey] = []
    for k in keys:
        if abl.kind == "text_drop_all":
            out.append(ItemKey(text="", img_path=k.img_path, image=k.image))
        elif abl.kind == "img_black":
            img = k.image
            if img is not None:
                img = _black_image_same_size(img.size)
            elif k.img_path and os.path.exists(k.img_path):
                with Image.open(k.img_path) as tmp:
                    tmp = tmp.convert("RGB")
                    img = _black_image_same_size(tmp.size)
            out.append(ItemKey(text=k.text, img_path=k.img_path, image=img))
        else:
            raise ValueError(f"Unsupported ablation kind: {abl.kind}")
    return out


def _load_ablation_cfg(path: Optional[str]) -> Tuple[Optional[Ablation], str]:
    """Load a whitelisted ablation JSON. Returns (Ablation|None, json_name).

    The released evaluator ONLY accepts the two shipped configs:
      * ablation_configs/image_only_zero_image.json -> {"name": "img_black"}
      * ablation_configs/text_only_drop_text.json   -> {"name": "text_drop_all"}
    """
    REJECT = ValueError(
        "only img_black and text_drop_all ablations are allowed in the released evaluator"
    )
    if not path:
        return None, "noablation"

    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    if not isinstance(d, dict):
        raise REJECT
    if not set(d.keys()).issubset({"seed", "ablations"}):
        raise REJECT
    ablations_raw = d.get("ablations")
    if not isinstance(ablations_raw, list) or len(ablations_raw) != 1:
        raise REJECT
    op = ablations_raw[0]
    if not isinstance(op, dict) or set(op.keys()) != {"name"}:
        raise REJECT
    name = op.get("name")
    if name not in ("img_black", "text_drop_all"):
        raise REJECT

    return Ablation(kind=name), _sanitize_for_path(p.stem, max_len=80)


# =============================================================================
# Dataset helpers
# =============================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dir(root: Path, name: str) -> Path:
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return root / name


def _normalize_cand_id(cid: str) -> str:
    return cid.split(":", 1)[-1]


def get_instruction(dataset: str) -> str:
    dataset = dataset.lower()
    prompts = {
        "fashioniq":   "Find a fashion image that aligns with the reference image and style note.",
        "fashion200k": "Find a fashion image that aligns with the reference image and style note.",
        "cirr":        "Retrieve a day-to-day image that aligns with the modification instructions of the provided image.",
        "circo":       "Retrieve an image that matches the reference image under the described change.",
        "_default":    "Retrieve the target image that best matches the reference image and the textual modification.",
    }
    return prompts.get(dataset, prompts["_default"])


def infer_split_from_query_name(query_file: str) -> str:
    ql = query_file.lower()
    if "test" in ql:
        return "test"
    if "val" in ql or "dev" in ql:
        return "val"
    return ""


def load_mbeir_queries(query_file: Path) -> List[dict]:
    rows = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_query_path(mbeir_root: Path, query_file: str, query_source: Optional[str]) -> Path:
    if query_source:
        p = Path(query_source).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
    return find_dir(mbeir_root, "query") / "test" / query_file


def load_mbeir_queries_flexible(
    query_path_or_dir: Path,
    query_file_hint: str,
    log: logging.Logger,
) -> List[dict]:
    rows: List[dict] = []
    if query_path_or_dir.is_dir():
        hinted = query_path_or_dir / query_file_hint
        if hinted.exists() and hinted.is_file():
            return load_mbeir_queries(hinted)

        files = sorted(query_path_or_dir.rglob("*.jsonl"))
        if not files:
            log.warning("[queries] no jsonl under directory: %s", query_path_or_dir)
            return []
        for fp in files:
            rows.extend(load_mbeir_queries(fp))
        return rows

    if not query_path_or_dir.exists():
        log.warning("[queries] query path does not exist: %s", query_path_or_dir)
        return []
    return load_mbeir_queries(query_path_or_dir)


def _query_row_id(row: dict) -> str:
    return str(row.get("qid") or row.get("query_id") or row.get("id") or "").strip()


def _resolve_data_path(root: Path, rel_or_abs: Optional[str]) -> Optional[str]:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((root / p).resolve())


def load_query_image_map(
    query_dir: Path,
    dataset: str,
    task_id: Optional[int],
    split_hint: str,
    log: logging.Logger,
) -> Dict[str, str]:
    dataset = dataset.lower()
    split_hint = split_hint.lower() if split_hint else ""

    all_files = sorted(query_dir.rglob("*.jsonl"))
    if not all_files:
        log.warning("[query-map] no jsonl files under %s", query_dir)
        return {}

    def keep(fp: Path) -> bool:
        n = fp.name.lower()
        ok = dataset in n
        if task_id is not None:
            ok = ok and (f"task{task_id}" in n)
        if split_hint:
            ok = ok and (split_hint in n)
        return ok

    files = [p for p in all_files if keep(p)] or all_files

    qid2img: Dict[str, str] = {}
    for fp in files:
        try:
            for row in load_mbeir_queries(fp):
                qid = _query_row_id(row)
                qimg = row.get("query_img_path")
                if not qid or not qimg:
                    continue
                qid2img[qid] = str(qimg)
        except Exception:
            continue

    log.info("[query-map] loaded %d query image paths for dataset=%s split~%s", len(qid2img), dataset, split_hint)
    return qid2img


def load_cand_pool_map(
    cand_pool_dir: Path,
    dataset: str,
    task_id: Optional[int],
    split_hint: str,
    log: logging.Logger,
) -> Dict[str, str]:
    dataset = dataset.lower()
    split_hint = split_hint.lower() if split_hint else ""

    all_files = sorted(cand_pool_dir.rglob("*.jsonl"))
    if not all_files:
        log.warning("[cand_pool] no jsonl files under %s", cand_pool_dir)
        return {}

    def keep(fp: Path) -> bool:
        n = fp.name.lower()
        ok = (dataset in n)
        if task_id is not None:
            ok = ok and (f"task{task_id}" in n)
        if split_hint:
            ok = ok and (split_hint in n)
        return ok

    files = [p for p in all_files if keep(p)] or all_files

    id2path: Dict[str, str] = {}
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        it = json.loads(line)
                    except Exception:
                        continue
                    cid = (
                        it.get("cand_id") or it.get("doc_id") or it.get("id")
                        or it.get("pid") or it.get("docid") or it.get("docId")
                        or it.get("did")
                    )
                    ipath = it.get("img_path") or it.get("image_path") or it.get("path") or it.get("image")
                    if not cid or not ipath:
                        continue
                    cid = str(cid)
                    ipath = str(ipath)
                    id2path[cid] = ipath
                    id2path[_normalize_cand_id(cid)] = ipath
        except Exception:
            continue

    log.info("[cand_pool] loaded %d ids for dataset=%s split~%s", len(id2path), dataset, split_hint)
    return id2path


def load_qrels_multi(qrels_path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not qrels_path.exists():
        return out

    with open(qrels_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.lower().startswith("query_id\tcandidate_id\tscore"):
                    continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            qid, cid, score = parts[0], parts[1], parts[2]
            try:
                score_val = float(score)
            except Exception:
                continue

            if score_val > 0:
                out.setdefault(qid, []).append(cid)

    return out


# =============================================================================
# Image IO
# =============================================================================

def _open_rgb(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_images_threaded(paths: List[str], num_workers: int, desc: str) -> Dict[str, Image.Image]:
    if num_workers <= 1:
        out: Dict[str, Image.Image] = {}
        for p in tqdm(paths, desc=desc):
            img = _open_rgb(p)
            if img is not None:
                out[p] = img
        return out

    out: Dict[str, Image.Image] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_open_rgb, p): p for p in paths}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            p = futs[fut]
            img = fut.result()
            if img is not None:
                out[p] = img
    return out


# =============================================================================
# Cache utils
# =============================================================================

def _retriever_tag(retriever_module: str, retriever_class: str) -> str:
    return f"{retriever_module}.{retriever_class}".replace("/", "__")


def _cache_paths(
    cache_dir: Path,
    dataset: str,
    split_hint: str,
    task_id: Optional[int],
    retriever_tag: str,
    ablation_sig: str,
    ablation_json_name: str,
    query_source_tag: str,
) -> Tuple[Path, Path]:
    # NOTE: cache key format must remain stable for the existing raw cache.
    task = f"task{task_id}" if task_id is not None else "taskNA"
    base = (
        f"{dataset}_{split_hint}_{task}_{retriever_tag}_{ablation_json_name}_"
        f"{ablation_sig}_qsrc={query_source_tag}"
    ).replace("/", "__")
    base = _sanitize_for_path(base, max_len=200)
    return cache_dir / f"gallery_{base}.pt", cache_dir / f"queries_{base}.pt"


# =============================================================================
# Metrics
# =============================================================================

def _dcg_single(rank0: int) -> float:
    return 1.0 / float(np.log2(rank0 + 2))


def _ndcg_at_k_single(rank0: int, k: int) -> float:
    if rank0 < 0 or rank0 >= k:
        return 0.0
    return _dcg_single(rank0)


def _mrr_single(rank0: int) -> float:
    if rank0 < 0:
        return 0.0
    return 1.0 / float(rank0 + 1)


def recall_any_at_k(ranked_ids: List[str], positive_ids: Set[str], k: int) -> float:
    topk = ranked_ids[:k]
    return float(any(cid in positive_ids for cid in topk))


def average_precision_at_k(ranked_ids: List[str], positive_ids: Set[str], k: int) -> float:
    if not positive_ids:
        return 0.0

    ranked_ids = ranked_ids[:k]
    hits = 0
    prec_sum = 0.0

    for i, cid in enumerate(ranked_ids, start=1):
        if cid in positive_ids:
            hits += 1
            prec_sum += hits / i

    denom = min(len(positive_ids), k)
    if denom == 0:
        return 0.0
    return prec_sum / denom


def ndcg_at_k_multi(ranked_ids: List[str], positive_ids: Set[str], k: int) -> float:
    ranked_ids = ranked_ids[:k]
    dcg = 0.0
    for i, cid in enumerate(ranked_ids):
        if cid in positive_ids:
            dcg += 1.0 / float(np.log2(i + 2))

    ideal_hits = min(len(positive_ids), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / float(np.log2(i + 2)) for i in range(ideal_hits))
    return dcg / idcg


# =============================================================================
# Embedding helpers
# =============================================================================

@torch.no_grad()
def _embed_in_batches(fn, keys: List[ItemKey], batch_size: int, desc: str) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for i in tqdm(range(0, len(keys), batch_size), desc=desc):
        chunk = keys[i:i + batch_size]
        out = fn(chunk)
        if out.ndim != 2 or out.shape[0] != len(chunk):
            raise ValueError(f"{desc}: returned {tuple(out.shape)} for batch={len(chunk)}")
        chunks.append(out.detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), dtype=torch.float32)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_mbeir_retriever_fast(
    retriever: EmbeddingRetriever,
    mbeir_root: Path,
    dataset: str,
    query_file: str,
    query_source: Optional[str],
    task_id: Optional[int],
    device: str,
    batch_size: int,
    normalize: bool,
    cache_dir: Optional[Path],
    strict_images: bool,
    num_workers: int,
    ablation: Optional[Ablation],
    ablation_json_name: str,
    log: logging.Logger,
) -> Dict[str, float]:
    dataset_l = dataset.lower()
    query_path = _resolve_query_path(mbeir_root, query_file=query_file, query_source=query_source)
    rows = load_mbeir_queries_flexible(query_path, query_file_hint=query_file, log=log)
    log.info("[queries] loaded %d rows from %s", len(rows), query_path)

    split_hint = infer_split_from_query_name(query_file or query_path.name)
    canonical_query_img_paths: Dict[str, str] = {}
    if query_source:
        query_dir = find_dir(mbeir_root, "query")
        canonical_query_img_paths = load_query_image_map(query_dir, dataset_l, task_id, split_hint, log=log)

    cand_pool_dir = find_dir(mbeir_root, "cand_pool")
    id2relpath = load_cand_pool_map(cand_pool_dir, dataset_l, task_id, split_hint, log=log)

    pool_abs = {_resolve_data_path(mbeir_root, p) for p in id2relpath.values() if p}
    pool_abs.discard(None)
    gallery_paths = [p for p in sorted(pool_abs) if Path(p).suffix.lower() in IMG_EXTS and Path(p).exists()]
    log.info("[gallery] %d images from candidate pool (%s)", len(gallery_paths), split_hint)

    if strict_images and not gallery_paths:
        raise FileNotFoundError("No gallery images found/resolved. Check mbeir_root and cand_pool paths.")

    instruction = get_instruction(dataset_l)

    qrels_multi: Dict[str, List[str]] = {}
    qrels_path = find_dir(mbeir_root, "qrels") / "test.tsv"
    if qrels_path.exists():
        qrels_multi = load_qrels_multi(qrels_path)
        log.info("[qrels] loaded multi-positive qrels for %d queries from %s", len(qrels_multi), qrels_path)

    query_rows: List[Tuple[str, str, str, str, List[str]]] = []
    misses = 0
    ref_paths_needed: List[str] = []
    query_img_fallback_hits = 0

    for it in rows:
        qid = _query_row_id(it)
        cap = it.get("query_txt") or it.get("query_text") or ""
        ref_rel = it.get("query_img_path")
        fallback_ref_rel = canonical_query_img_paths.get(qid)
        pos_list = it.get("pos_cand_list") or []
        pos_id = str(pos_list[0]) if pos_list else None

        if not cap or not ref_rel or not pos_id:
            if not ref_rel and fallback_ref_rel:
                ref_rel = fallback_ref_rel
                query_img_fallback_hits += 1
            else:
                continue

        if not cap or not ref_rel or not pos_id:
            continue

        gt_list = it.get("gt_cand_list") or qrels_multi.get(qid) or pos_list
        gt_list = [str(x) for x in gt_list]

        ref_abs = _resolve_data_path(mbeir_root, ref_rel)
        if (not ref_abs or not os.path.exists(ref_abs)) and fallback_ref_rel:
            fallback_abs = _resolve_data_path(mbeir_root, fallback_ref_rel)
            if fallback_abs and os.path.exists(fallback_abs):
                ref_abs = fallback_abs
                query_img_fallback_hits += 1
        if not os.path.exists(ref_abs):
            if strict_images:
                raise FileNotFoundError(f"Missing query image: {ref_abs}")
            misses += 1
            continue

        t_rel = id2relpath.get(pos_id) or id2relpath.get(_normalize_cand_id(pos_id))
        if not t_rel:
            misses += 1
            continue

        tgt_abs = _resolve_data_path(mbeir_root, t_rel)
        if not os.path.exists(tgt_abs):
            misses += 1
            continue

        query_rows.append((qid, ref_abs, cap, tgt_abs, gt_list))
        ref_paths_needed.append(ref_abs)

    if query_img_fallback_hits:
        log.info("[queries] recovered %d query image paths via canonical mbeir_root/query lookup", query_img_fallback_hits)

    if not query_rows:
        log.warning("[warn] No valid queries built. misses=%d", misses)
        return {
            "R@5": 0.0, "R@10": 0.0, "R@50": 0.0,
            "RecallAny@5": 0.0, "RecallAny@10": 0.0, "RecallAny@50": 0.0,
            "mAP@5": 0.0, "mAP@10": 0.0, "mAP@50": 0.0,
            "nDCG@5": 0.0, "nDCG@10": 0.0, "nDCG@50": 0.0,
            "nDCG_multi@5": 0.0, "nDCG_multi@10": 0.0, "nDCG_multi@50": 0.0,
            "MRR": 0.0,
            "queries": 0, "misses": misses,
            "ablation_sig": "noabl",
        }

    t_load = time.time()
    gal_imgs = load_images_threaded(gallery_paths, num_workers=num_workers, desc="Opening gallery images")
    ref_imgs = load_images_threaded(sorted(set(ref_paths_needed)), num_workers=num_workers, desc="Opening query images")
    log.info(
        "[io] decoded gallery=%d/%d query_imgs=%d unique_refs=%d in %.2fs",
        len(gal_imgs), len(gallery_paths), len(ref_imgs), len(set(ref_paths_needed)), time.time() - t_load
    )

    kept_gallery_paths: List[str] = [p for p in gallery_paths if p in gal_imgs]
    kept_gallery_imgs: List[Image.Image] = [gal_imgs[p] for p in kept_gallery_paths]

    if not kept_gallery_paths:
        log.warning("[warn] No gallery images could be decoded.")
        return {
            "R@5": 0.0, "R@10": 0.0, "R@50": 0.0,
            "RecallAny@5": 0.0, "RecallAny@10": 0.0, "RecallAny@50": 0.0,
            "mAP@5": 0.0, "mAP@10": 0.0, "mAP@50": 0.0,
            "nDCG@5": 0.0, "nDCG@10": 0.0, "nDCG@50": 0.0,
            "nDCG_multi@5": 0.0, "nDCG_multi@10": 0.0, "nDCG_multi@50": 0.0,
            "MRR": 0.0,
            "queries": 0, "misses": misses,
            "ablation_sig": "noabl",
        }

    path2idx = {str(Path(p).resolve()): i for i, p in enumerate(kept_gallery_paths)}
    name2idx = {Path(p).name: i for i, p in enumerate(kept_gallery_paths)}

    idx2cand_id: Dict[int, str] = {}
    for cid, p in id2relpath.items():
        abs_p = str((mbeir_root / p).resolve())
        idx = path2idx.get(abs_p)
        if idx is None:
            idx = name2idx.get(Path(abs_p).name)
        if idx is not None and idx not in idx2cand_id:
            idx2cand_id[idx] = cid

    query_keys: List[ItemKey] = []
    tgt_indices: List[int] = []
    query_positive_sets: List[Set[str]] = []

    for qid, ref_abs, cap, tgt_abs, gt_list in query_rows:
        img = ref_imgs.get(ref_abs)
        if img is None:
            misses += 1
            continue

        tgt_idx = path2idx.get(str(Path(tgt_abs).resolve()))
        if tgt_idx is None:
            tgt_idx = name2idx.get(Path(tgt_abs).name)
        if tgt_idx is None:
            misses += 1
            continue

        q_text = (instruction + "\n" + cap).strip() if instruction else cap
        query_keys.append(ItemKey(text=q_text, img_path=ref_abs, image=img))
        tgt_indices.append(int(tgt_idx))
        query_positive_sets.append(set(str(x) for x in gt_list))

    if not query_keys:
        log.warning("[warn] No valid decoded queries left. misses=%d", misses)
        return {
            "R@5": 0.0, "R@10": 0.0, "R@50": 0.0,
            "RecallAny@5": 0.0, "RecallAny@10": 0.0, "RecallAny@50": 0.0,
            "mAP@5": 0.0, "mAP@10": 0.0, "mAP@50": 0.0,
            "nDCG@5": 0.0, "nDCG@10": 0.0, "nDCG@50": 0.0,
            "nDCG_multi@5": 0.0, "nDCG_multi@10": 0.0, "nDCG_multi@50": 0.0,
            "MRR": 0.0,
            "queries": 0, "misses": misses,
            "ablation_sig": "noabl",
        }

    gallery_keys: List[ItemKey] = [
        ItemKey(text="", img_path=p, image=img)
        for p, img in zip(kept_gallery_paths, kept_gallery_imgs)
    ]

    abl_sig = ablation_signature(ablation)
    if ablation is not None:
        log.info(
            "[ablation] enabled kind=%s sig=%s json=%s",
            ablation.kind, abl_sig, ablation_json_name,
        )
        query_keys = apply_ablation(query_keys, ablation)
    else:
        log.info("[ablation] disabled")
        abl_sig = "noabl"

    retriever_id = _retriever_tag(retriever.__class__.__module__, retriever.__class__.__name__)
    query_source_tag = _safe_slug(str(query_path), max_len=60)
    gallery_query_source_tag = "catalog"
    # gallery is never ablated in this evaluator
    gallery_abl_sig = "noabl"
    gallery_ablation_json_name = "noablation"

    gallery_vecs = None
    qvecs = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        g_path, _ = _cache_paths(
            cache_dir, dataset_l, split_hint, task_id, retriever_id, gallery_abl_sig, gallery_ablation_json_name, gallery_query_source_tag
        )
        if g_path.exists():
            gallery_vecs = torch.load(g_path, map_location="cpu")
            log.info("[cache] loaded gallery embeddings: %s", g_path)
        _, q_path = _cache_paths(
            cache_dir, dataset_l, split_hint, task_id, retriever_id, abl_sig, ablation_json_name, query_source_tag
        )
        if q_path.exists():
            qvecs = torch.load(q_path, map_location="cpu")
            log.info("[cache] loaded query embeddings: %s", q_path)

    if gallery_vecs is None or (isinstance(gallery_vecs, torch.Tensor) and gallery_vecs.shape[0] != len(gallery_keys)):
        gallery_vecs = _embed_in_batches(retriever.embed_targets, gallery_keys, batch_size, "Embedding gallery")
        if normalize:
            gallery_vecs = F.normalize(gallery_vecs, dim=-1)
        if cache_dir is not None:
            g_path, _ = _cache_paths(
                cache_dir, dataset_l, split_hint, task_id, retriever_id, gallery_abl_sig, gallery_ablation_json_name, gallery_query_source_tag
            )
            torch.save(gallery_vecs, g_path)
            log.info("[cache] saved gallery embeddings: %s", g_path)
    else:
        if normalize:
            gallery_vecs = F.normalize(gallery_vecs, dim=-1)

    if qvecs is None or (isinstance(qvecs, torch.Tensor) and qvecs.shape[0] != len(query_keys)):
        qvecs = _embed_in_batches(retriever.embed_queries, query_keys, batch_size, "Embedding queries")
        if normalize:
            qvecs = F.normalize(qvecs, dim=-1)
        if cache_dir is not None:
            _, q_path = _cache_paths(
                cache_dir, dataset_l, split_hint, task_id, retriever_id, abl_sig, ablation_json_name, query_source_tag
            )
            torch.save(qvecs, q_path)
            log.info("[cache] saved query embeddings: %s", q_path)
    else:
        if normalize:
            qvecs = F.normalize(qvecs, dim=-1)

    qvecs = qvecs.to(device)
    gallery_vecs = gallery_vecs.to(device)
    sims = qvecs @ gallery_vecs.T

    recalls5, recalls10, recalls50 = [], [], []
    recall_any5, recall_any10, recall_any50 = [], [], []
    map5, map10, map50 = [], [], []
    ndcg5, ndcg10, ndcg50 = [], [], []
    ndcg_multi5, ndcg_multi10, ndcg_multi50 = [], [], []
    mrrs: List[float] = []

    for s, tgt_idx, pos_set in zip(sims, tgt_indices, query_positive_sets):
        topk = min(50, s.shape[-1])
        _, idx = torch.topk(s, k=topk)

        ranked_idx = idx.tolist()
        ranked_ids = [idx2cand_id[i] for i in ranked_idx if i in idx2cand_id]

        recalls5.append(int((idx[:5] == tgt_idx).any().item()))
        recalls10.append(int((idx[:10] == tgt_idx).any().item()))
        recalls50.append(int((idx[:50] == tgt_idx).any().item()))

        hit = (idx == tgt_idx).nonzero(as_tuple=False)
        rank0 = int(hit[0].item()) if hit.numel() else -1
        ndcg5.append(_ndcg_at_k_single(rank0, 5))
        ndcg10.append(_ndcg_at_k_single(rank0, 10))
        ndcg50.append(_ndcg_at_k_single(rank0, 50))
        mrrs.append(_mrr_single(rank0))

        recall_any5.append(recall_any_at_k(ranked_ids, pos_set, 5))
        recall_any10.append(recall_any_at_k(ranked_ids, pos_set, 10))
        recall_any50.append(recall_any_at_k(ranked_ids, pos_set, 50))

        map5.append(average_precision_at_k(ranked_ids, pos_set, 5))
        map10.append(average_precision_at_k(ranked_ids, pos_set, 10))
        map50.append(average_precision_at_k(ranked_ids, pos_set, 50))

        ndcg_multi5.append(ndcg_at_k_multi(ranked_ids, pos_set, 5))
        ndcg_multi10.append(ndcg_at_k_multi(ranked_ids, pos_set, 10))
        ndcg_multi50.append(ndcg_at_k_multi(ranked_ids, pos_set, 50))

    r5 = 100.0 * float(np.mean(recalls5)) if recalls5 else 0.0
    r10 = 100.0 * float(np.mean(recalls10)) if recalls10 else 0.0
    r50 = 100.0 * float(np.mean(recalls50)) if recalls50 else 0.0

    ra5 = 100.0 * float(np.mean(recall_any5)) if recall_any5 else 0.0
    ra10 = 100.0 * float(np.mean(recall_any10)) if recall_any10 else 0.0
    ra50 = 100.0 * float(np.mean(recall_any50)) if recall_any50 else 0.0

    ap5 = 100.0 * float(np.mean(map5)) if map5 else 0.0
    ap10 = 100.0 * float(np.mean(map10)) if map10 else 0.0
    ap50 = 100.0 * float(np.mean(map50)) if map50 else 0.0

    n5 = 100.0 * float(np.mean(ndcg5)) if ndcg5 else 0.0
    n10 = 100.0 * float(np.mean(ndcg10)) if ndcg10 else 0.0
    n50 = 100.0 * float(np.mean(ndcg50)) if ndcg50 else 0.0

    nm5 = 100.0 * float(np.mean(ndcg_multi5)) if ndcg_multi5 else 0.0
    nm10 = 100.0 * float(np.mean(ndcg_multi10)) if ndcg_multi10 else 0.0
    nm50 = 100.0 * float(np.mean(ndcg_multi50)) if ndcg_multi50 else 0.0

    mrr = 100.0 * float(np.mean(mrrs)) if mrrs else 0.0

    log.info(
        "[done] "
        "R@5=%.2f R@10=%.2f R@50=%.2f | "
        "RecallAny@5=%.2f RecallAny@10=%.2f RecallAny@50=%.2f | "
        "mAP@5=%.2f mAP@10=%.2f mAP@50=%.2f | "
        "nDCG@5=%.2f nDCG@10=%.2f nDCG@50=%.2f | "
        "nDCG_multi@5=%.2f nDCG_multi@10=%.2f nDCG_multi@50=%.2f | MRR=%.2f "
        "(queries=%d misses=%d ablation=%s json=%s)",
        r5, r10, r50,
        ra5, ra10, ra50,
        ap5, ap10, ap50,
        n5, n10, n50,
        nm5, nm10, nm50,
        mrr,
        len(recalls10), misses, abl_sig, ablation_json_name,
    )

    return {
        "R@5": r5,
        "R@10": r10,
        "R@50": r50,
        "RecallAny@5": ra5,
        "RecallAny@10": ra10,
        "RecallAny@50": ra50,
        "mAP@5": ap5,
        "mAP@10": ap10,
        "mAP@50": ap50,
        "nDCG@5": n5,
        "nDCG@10": n10,
        "nDCG@50": n50,
        "nDCG_multi@5": nm5,
        "nDCG_multi@10": nm10,
        "nDCG_multi@50": nm50,
        "MRR": mrr,
        "queries": int(len(recalls10)),
        "misses": int(misses),
        "ablation_sig": abl_sig,
        "ablation_json": ablation_json_name,
        "query_source": str(query_path),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Subset evaluator for CIRCO (multi-positive scoring). "
            "Only the img_black and text_drop_all ablations from the paper are "
            "supported via --ablation_json."
        )
    )
    ap.add_argument("--mbeir_root", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--query_file", type=str, required=True)
    ap.add_argument(
        "--query_source",
        type=str,
        default=None,
        help="Optional path to a query jsonl file or directory of jsonl files. If set, overrides mbeir_root/query/test/<query_file>.",
    )
    ap.add_argument("--task_id", type=int, default=None)

    ap.add_argument("--retriever_module", required=True)
    ap.add_argument("--retriever_class", default="Retriever")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--strict_images", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument(
        "--ablation_json",
        type=str,
        default=None,
        help=(
            "Path to a whitelisted ablation JSON. Only the two shipped configs "
            "(image_only_zero_image.json, text_only_drop_text.json) are accepted; "
            "any other shape is rejected."
        ),
    )
    ap.add_argument("--metrics_out", type=str, default=None)
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    log = logging.getLogger("eval_subset_circo")

    maybe_hf_login(log)

    ablation, ablation_json_name = _load_ablation_cfg(args.ablation_json)

    if ablation is None:
        log.info("[ablation] disabled")
    else:
        log.info(
            "[ablation] enabled json=%s sig=%s kind=%s",
            ablation_json_name, ablation_signature(ablation), ablation.kind,
        )

    if "cuda" in args.device and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    mbeir_root = Path(args.mbeir_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    log.info("Loading retriever: %s.%s device=%s", args.retriever_module, args.retriever_class, args.device)
    retriever = load_retriever(args.retriever_module, args.retriever_class, device=args.device)

    t0 = time.time()
    res = evaluate_mbeir_retriever_fast(
        retriever=retriever,
        mbeir_root=mbeir_root,
        dataset=args.dataset,
        query_file=args.query_file,
        query_source=args.query_source,
        task_id=args.task_id,
        device=args.device,
        batch_size=args.batch_size,
        normalize=args.normalize,
        cache_dir=cache_dir,
        strict_images=args.strict_images,
        num_workers=args.num_workers,
        ablation=ablation,
        ablation_json_name=ablation_json_name,
        log=log,
    )
    res["total_time_s"] = float(time.time() - t0)
    if args.metrics_out:
        metrics_out = Path(args.metrics_out).expanduser().resolve()
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
