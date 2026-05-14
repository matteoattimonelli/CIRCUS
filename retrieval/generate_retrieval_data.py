#!/usr/bin/env python3
"""
================================================================================
generate_retrieval_data.py — Unimodal Retrieval Analysis Data Generator
================================================================================

PURPOSE:
    For every query in an M-BEIR dataset, run retrieval under three conditions:
      1. Multimodal  (query text + reference image)
      2. Text-only   (query text, no image)
      3. Image-only  (reference image, no text)
    and save all results to a single JSON file that can be browsed with the
    companion script `browse_retrievals.py`.

HOW IT WORKS:
    1. Loads query rows from an M-BEIR JSONL file and builds the candidate
       gallery from the corresponding candidate pool.
    2. Decodes all gallery and query reference images (threaded I/O).
    3. Loads the specified embedding retriever and embeds:
       - the gallery once,
       - the queries three times (multimodal, text-only, image-only).
    4. Computes cosine similarities and, for each query × condition, extracts:
       - the rank of the ground-truth target in the full ranking,
       - the top-K retrieved gallery indices.
    5. Categorises each query as:
         "either"             – target in top-K for both text-only AND image-only
         "text_only"          – target in top-K for text-only but not image-only
         "image_only"         – target in top-K for image-only but not text-only
         "multimodal_needed"  – target NOT in top-K for either unimodal condition
    6. Writes everything to a JSON file.

DEPENDENCIES:
    - Python ≥ 3.9
    - torch, torchvision, Pillow, tqdm, numpy
    - A retriever module that implements embed_queries() and embed_targets()
      (see retrievers/ directory in the project root)

OUTPUT FORMAT (JSON):
    {
      "metadata": {
        "dataset": "cirr",
        "task_id": 7,
        "retriever_module": "retrievers.qwen3vl8b_vllm_retriever",
        "retriever_class": "Retriever",
        "k": 10,
        "top_k": 20,
        "normalize": true,
        "mbeir_root": "<path_to_M-BEIR>",
        "total_queries": 4148,
        "generated_at": "2026-03-18T14:22:01"
      },
      "summary": {
        "counts": {
          "either": 312, "text_only": 580,
          "image_only": 423, "multimodal_needed": 2833
        },
        "recall_at_k": {
          "multimodal": 52.3, "text_only": 40.1, "image_only": 28.4
        }
      },
      "gallery_paths": ["/abs/path/img1.jpg", ...],
      "samples": [
        {
          "query_idx": 0,
          "query_text": "make it more red",
          "full_text": "Retrieve the target image ...\nmake it more red",
          "query_image_path": "/abs/path/ref.jpg",
          "target_image_path": "/abs/path/target.jpg",
          "target_gallery_index": 42,
          "category": "text_only",
          "ranks": {"multimodal": 3, "text_only": 7, "image_only": 150},
          "hits_at_k": {"multimodal": true, "text_only": true, "image_only": false},
          "retrievals": {
            "multimodal": [42, 15, 78, ...],
            "text_only":  [15, 42, 200, ...],
            "image_only": [78, 300, 42, ...]
          }
        },
        ...
      ]
    }

    - gallery_paths: ordered list of absolute paths for all gallery images.
    - samples[*].retrievals.*: lists of gallery indices (position in
      gallery_paths), length = top_k.
    - samples[*].ranks.*: 1-based rank of the ground-truth target in the
      full ranking (not clipped to top_k).

RUNNING EXAMPLES:

    # --- Example 1: CIRR dataset with Qwen3-VL-8B (vLLM) ---
    python retrieval/generate_retrieval_data.py \
        --mbeir_root <path_to_M-BEIR> \
        --dataset cirr --task_id 7 \
        --query_file mbeir_cirr_task7_test.jsonl \
        --retriever_module retrievers.qwen3vl8b_vllm_retriever \
        --retriever_class Retriever \
        --normalize --k 10 --top_k 20 \
        --output retrieval/retrieval_data_cirr_qwen3vl8b.json

    # --- Example 2: FashionIQ with CLIP ---
    python retrieval/generate_retrieval_data.py \
        --mbeir_root <path_to_M-BEIR> \
        --dataset fashioniq --task_id 7 \
        --query_file mbeir_fashioniq_task7_test.jsonl \
        --retriever_module retrievers.qwen3vl8b_vllm_retriever \
        --retriever_class Retriever \
        --normalize --k 10 --top_k 50 \
        --output retrieval/retrieval_data_fiq_clip.json

    # --- Example 3: Fashion200K with E5-Omni on specific GPU ---
    python retrieval/generate_retrieval_data.py \
        --mbeir_root <path_to_M-BEIR> \
        --dataset fashion200k --task_id 7 \
        --query_file mbeir_fashion200k_task7_test.jsonl \
        --retriever_module retrievers.e5_omni_retriever \
        --retriever_class Retriever \
        --device cuda:1 --batch_size 16 \
        --normalize --k 10 --top_k 20 \
        --output retrieval/retrieval_data_f200k_e5omni.json

    # --- Example 4: CIRR with ColPali (NOT supported – late-interaction) ---
    # Note: ColPali returns [N, L, D] tensors (multi-vector / late interaction).
    # This script only supports dense [N, D] retrievers. ColPali will NOT work.

NOTES:
    - Run from the project root so that retriever imports resolve correctly.
    - The script automatically adds the project root to sys.path.
    - Gallery embeddings can be large; ensure enough GPU/CPU memory.
    - Output JSON size depends on top_k and number of queries.  Typical sizes:
        ~4000 queries, top_k=20  → ~7–10 MB
        ~4000 queries, top_k=50  → ~15–20 MB
    - The --k flag controls the recall@K threshold for categorisation only.
      The --top_k flag controls how many top retrieved results are saved.
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so we can import helpers.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F
from tqdm import tqdm

from subset_evaluation.eval_subset import (
    IMG_EXTS,
    ItemKey,
    _normalize_cand_id,
    _embed_in_batches as embed_in_batches,
    find_dir,
    get_instruction,
    infer_split_from_query_name,
    load_cand_pool_map,
    load_images_threaded,
    load_mbeir_queries,
    load_retriever,
    _load_ablation_cfg as load_attack_cfg,
    apply_ablation,
    ablation_signature as attack_signature,
)


def apply_attacks(keys, cfg, which: str = "query"):
    """Locked-down shim: only the query-side, two-name ablation set is allowed."""
    if which != "query":
        raise ValueError("released evaluator only ablates the query side")
    return apply_ablation(keys, cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_TEXT_ONLY_ATTACK_JSON = "ablation_configs/image_only_zero_image.json"
DEFAULT_IMAGE_ONLY_ATTACK_JSON = "ablation_configs/text_only_drop_text.json"


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(_PROJECT_ROOT) / path).resolve()


def _safe_slug(text: str, max_len: int = 160) -> str:
    text = (text or "").strip().replace("/", "__")
    text = "".join(ch if ch.isalnum() or ch in "._-+=" else "-" for ch in text)
    if len(text) <= max_len:
        return text
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    head = text[: max_len // 2]
    tail = text[-max_len // 2 :]
    return f"{head}__{digest}__{tail}"


def _retriever_tag(retriever_module: str, retriever_class: str) -> str:
    return f"{retriever_module}.{retriever_class}".replace("/", "__")


def _build_embedding_cache_key(
    dataset: str,
    split_hint: str,
    task_id: int | None,
    retriever_module: str,
    retriever_class: str,
    query_file: str,
    variant: str,
    query_instruction: str,
    normalize: bool,
) -> str:
    base = (
        f"{dataset.lower()}_split-{split_hint or 'na'}_task-{task_id if task_id is not None else 'na'}_"
        f"retriever-{_retriever_tag(retriever_module, retriever_class)}_"
        f"query-{query_file}_variant-{variant}_"
        f"norm-{int(bool(normalize))}_"
        f"instr-{hashlib.md5((query_instruction or '').encode('utf-8')).hexdigest()[:10]}"
    )
    return _safe_slug(base)


def _atomic_torch_save(obj: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def embed_in_batches_resumable(
    fn,
    keys: List[ItemKey],
    batch_size: int,
    desc: str,
    cache_dir: Path | None,
    cache_key: str | None,
    log: logging.Logger,
    normalize: bool,
) -> torch.Tensor:
    if cache_dir is None or not cache_key:
        out = embed_in_batches(fn, keys, batch_size, desc)
        if normalize:
            out = F.normalize(out, dim=-1)
        return out

    run_cache_dir = cache_dir / cache_key
    shard_dir = run_cache_dir / "shards"
    merged_path = run_cache_dir / "embeddings.pt"
    meta_path = run_cache_dir / "meta.json"
    shard_dir.mkdir(parents=True, exist_ok=True)

    expected_batches = (len(keys) + batch_size - 1) // batch_size if keys else 0
    meta = {
        "cache_key": cache_key,
        "desc": desc,
        "num_items": len(keys),
        "batch_size": batch_size,
        "normalize": bool(normalize),
        "expected_batches": expected_batches,
    }
    _atomic_json_dump(meta, meta_path)

    if merged_path.exists():
        try:
            merged = torch.load(merged_path, map_location="cpu")
            if isinstance(merged, torch.Tensor) and merged.shape[0] == len(keys):
                log.info("[cache] loaded merged embeddings for %s from %s", desc, merged_path)
                return merged
            log.warning("[cache] ignoring merged cache with wrong shape for %s: %s", desc, merged_path)
        except Exception as exc:
            log.warning("[cache] failed to load merged cache for %s: %s", desc, exc)

    chunks: List[torch.Tensor] = []
    for batch_idx, start in enumerate(tqdm(range(0, len(keys), batch_size), desc=desc)):
        end = min(start + batch_size, len(keys))
        chunk = keys[start:end]
        shard_path = shard_dir / f"batch_{batch_idx:06d}_{start:08d}_{end:08d}.pt"

        shard_tensor = None
        if shard_path.exists():
            try:
                shard_tensor = torch.load(shard_path, map_location="cpu")
                if not isinstance(shard_tensor, torch.Tensor) or shard_tensor.shape[0] != len(chunk):
                    log.warning("[cache] ignoring bad shard for %s: %s", desc, shard_path)
                    shard_tensor = None
            except Exception as exc:
                log.warning("[cache] failed to load shard for %s (%s): %s", desc, shard_path, exc)
                shard_tensor = None

        if shard_tensor is None:
            shard_tensor = fn(chunk).detach().cpu()
            if normalize:
                shard_tensor = F.normalize(shard_tensor, dim=-1)
            _atomic_torch_save(shard_tensor, shard_path)
            log.info("[cache] saved shard %d/%d for %s -> %s", batch_idx + 1, expected_batches, desc, shard_path)

        chunks.append(shard_tensor)

    merged = torch.cat(chunks, dim=0) if chunks else torch.empty(0)
    _atomic_torch_save(merged, merged_path)
    log.info("[cache] saved merged embeddings for %s -> %s", desc, merged_path)
    return merged

@dataclass
class QueryRecord:
    """Validated, resolved information about a single query."""
    idx: int            # index in the original JSONL file
    ref_abs: str        # absolute path to the reference (query) image
    caption: str        # raw caption / modification instruction
    full_text: str      # instruction prefix + caption
    tgt_abs: str        # absolute path to ground-truth target image


def build_records(
    rows: List[dict],
    mbeir_root: Path,
    id2relpath: Dict[str, str],
    instruction: str,
    log: logging.Logger,
) -> Tuple[List[QueryRecord], List[str], int]:
    """Parse raw JSONL rows into validated QueryRecords.

    Returns (records, ref_paths_needed, miss_count).
    """
    records: List[QueryRecord] = []
    ref_paths_needed: List[str] = []
    misses = 0

    for i, item in enumerate(rows):
        cap = item.get("query_txt") or item.get("query_text") or ""
        ref_rel = item.get("query_img_path")
        pos_list = item.get("pos_cand_list") or []
        pos_id = str(pos_list[0]) if pos_list else None

        if not cap or not ref_rel or not pos_id:
            continue

        ref_abs = str((mbeir_root / ref_rel).resolve())
        if not os.path.exists(ref_abs):
            misses += 1
            continue

        tgt_rel = id2relpath.get(pos_id) or id2relpath.get(_normalize_cand_id(pos_id))
        if not tgt_rel:
            misses += 1
            continue
        tgt_abs = str((mbeir_root / tgt_rel).resolve())
        if not os.path.exists(tgt_abs):
            misses += 1
            continue

        full_text = (instruction + "\n" + cap).strip() if instruction else cap
        records.append(QueryRecord(
            idx=i, ref_abs=ref_abs, caption=cap,
            full_text=full_text, tgt_abs=tgt_abs,
        ))
        ref_paths_needed.append(ref_abs)

    log.info("Valid queries: %d  (misses: %d)", len(records), misses)
    return records, ref_paths_needed, misses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-query retrieval data under three modality conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    ap.add_argument("--mbeir_root", type=str, required=True,
                    help="Path to the M-BEIR data directory.")
    ap.add_argument("--dataset", type=str, required=True,
                    help="Dataset name, e.g. cirr, fashioniq, fashion200k.")
    ap.add_argument("--query_file", type=str, required=True,
                    help="Query JSONL filename inside <mbeir_root>/query/test/.")
    ap.add_argument("--task_id", type=int, default=None,
                    help="Task ID to filter candidate pool files (e.g. 7).")
    ap.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional limit on the number of valid queries to process, for cheap smoke tests.",
    )
    ap.add_argument(
        "--smoke_gallery_size",
        type=int,
        default=None,
        help=(
            "Optional gallery limit for smoke tests. Keeps all targets for the selected "
            "queries, then fills the rest with additional gallery images."
        ),
    )
    ap.add_argument(
        "--query_instruction",
        type=str,
        default=None,
        help=(
            "Optional query instruction override. If omitted, uses the dataset default. "
            "Pass an empty string to disable the instruction prefix."
        ),
    )

    # Retriever arguments
    ap.add_argument("--retriever_module", required=True,
                    help="Dotted module path, e.g. retrievers.qwen3vl8b_vllm_retriever.")
    ap.add_argument("--retriever_class", default="Retriever",
                    help="Class name inside the retriever module (default: Retriever).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Torch device (default: cuda if available, else cpu).")

    # Embedding arguments
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for embedding (default: 32).")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalise embeddings before similarity computation.")
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for resumable embedding caches. "
            "Embeddings are stored per batch and reused on restart."
        ),
    )
    ap.add_argument("--num_workers", type=int, default=8,
                    help="Threads for image decoding (default: 8).")

    # Output control
    ap.add_argument("--k", type=int, default=10,
                    help="Recall@K threshold for categorisation (default: 10).")
    ap.add_argument("--top_k", type=int, default=20,
                    help="Number of top retrieved results to save per condition (default: 20).")
    ap.add_argument("--output", type=str, default="retrieval/retrieval_data.json",
                    help="Output JSON path (default: retrieval/retrieval_data.json).")
    ap.add_argument(
        "--text_only_attack_json",
        type=str,
        default=DEFAULT_TEXT_ONLY_ATTACK_JSON,
        help=(
            "Attack config used to generate the text-only query variant. "
            f"Default: {DEFAULT_TEXT_ONLY_ATTACK_JSON}"
        ),
    )
    ap.add_argument(
        "--image_only_attack_json",
        type=str,
        default=DEFAULT_IMAGE_ONLY_ATTACK_JSON,
        help=(
            "Attack config used to generate the image-only query variant. "
            f"Default: {DEFAULT_IMAGE_ONLY_ATTACK_JSON}"
        ),
    )
    ap.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("generate_retrieval_data")

    if "cuda" in args.device and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    mbeir_root = Path(args.mbeir_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    dataset_l = args.dataset.lower()
    K = args.k
    TOP_K = args.top_k
    t0 = time.time()

    text_only_attack_path = resolve_repo_path(args.text_only_attack_json)
    image_only_attack_path = resolve_repo_path(args.image_only_attack_json)
    text_only_attack_cfg, text_only_attack_name = load_attack_cfg(str(text_only_attack_path))
    image_only_attack_cfg, image_only_attack_name = load_attack_cfg(str(image_only_attack_path))

    # ------------------------------------------------------------------
    # 1. Load queries & candidate pool
    # ------------------------------------------------------------------
    query_path = find_dir(mbeir_root, "query") / "test" / args.query_file
    rows = load_mbeir_queries(query_path)
    log.info("Loaded %d raw query rows from %s", len(rows), query_path.name)

    split_hint = infer_split_from_query_name(args.query_file)
    cand_pool_dir = find_dir(mbeir_root, "cand_pool")
    id2relpath = load_cand_pool_map(
        cand_pool_dir, dataset_l, args.task_id, split_hint, log=log,
    )

    pool_abs = {str((mbeir_root / p).resolve()) for p in id2relpath.values() if p}
    gallery_paths = sorted(
        p for p in pool_abs
        if Path(p).suffix.lower() in IMG_EXTS and Path(p).exists()
    )
    log.info("Gallery: %d images", len(gallery_paths))

    instruction = get_instruction(dataset_l) if args.query_instruction is None else args.query_instruction

    # ------------------------------------------------------------------
    # 2. Build per-query records
    # ------------------------------------------------------------------
    records, ref_paths_needed, misses = build_records(
        rows, mbeir_root, id2relpath, instruction, log,
    )
    if not records:
        log.error("No valid queries — exiting.")
        return
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max_queries must be a positive integer when provided.")
        records = records[: args.max_queries]
        ref_paths_needed = [rec.ref_abs for rec in records]
        log.info("Limiting run to first %d valid queries due to --max_queries", len(records))
    if args.smoke_gallery_size is not None and args.smoke_gallery_size <= 0:
        raise ValueError("--smoke_gallery_size must be a positive integer when provided.")

    if args.smoke_gallery_size is not None:
        target_paths = {str(Path(rec.tgt_abs).resolve()) for rec in records}
        if len(target_paths) > args.smoke_gallery_size:
            raise ValueError(
                f"--smoke_gallery_size={args.smoke_gallery_size} is too small for the selected "
                f"{len(target_paths)} target images. Increase it or reduce --max_queries."
            )

        smoke_gallery_paths: List[str] = []
        added: set[str] = set()

        for tgt_path in sorted(target_paths):
            if tgt_path in gallery_paths and tgt_path not in added:
                smoke_gallery_paths.append(tgt_path)
                added.add(tgt_path)

        for p in gallery_paths:
            if len(smoke_gallery_paths) >= args.smoke_gallery_size:
                break
            if p not in added:
                smoke_gallery_paths.append(p)
                added.add(p)

        gallery_paths = smoke_gallery_paths
        log.info(
            "Limiting gallery to %d images due to --smoke_gallery_size (targets preserved: %d)",
            len(gallery_paths),
            len(target_paths),
        )

    # ------------------------------------------------------------------
    # 3. Load and decode images
    # ------------------------------------------------------------------
    gal_imgs = load_images_threaded(gallery_paths, args.num_workers, "Loading gallery images")
    ref_imgs = load_images_threaded(
        sorted(set(ref_paths_needed)), args.num_workers, "Loading query images",
    )

    kept_gallery_paths = [p for p in gallery_paths if p in gal_imgs]
    kept_gallery_imgs = [gal_imgs[p] for p in kept_gallery_paths]
    log.info("Gallery decoded: %d / %d", len(kept_gallery_paths), len(gallery_paths))

    path2idx = {str(Path(p).resolve()): i for i, p in enumerate(kept_gallery_paths)}
    name2idx = {Path(p).name: i for i, p in enumerate(kept_gallery_paths)}

    valid_records: List[Tuple[QueryRecord, int]] = []
    for rec in records:
        img = ref_imgs.get(rec.ref_abs)
        if img is None:
            continue
        tgt_idx = path2idx.get(str(Path(rec.tgt_abs).resolve()))
        if tgt_idx is None:
            tgt_idx = name2idx.get(Path(rec.tgt_abs).name)
        if tgt_idx is None:
            continue
        valid_records.append((rec, tgt_idx))

    log.info("Queries with decoded images & target in gallery: %d", len(valid_records))
    if not valid_records:
        log.error("No valid decoded queries — exiting.")
        return

    # ------------------------------------------------------------------
    # 4. Load retriever & embed gallery
    # ------------------------------------------------------------------
    log.info("Loading retriever: %s.%s", args.retriever_module, args.retriever_class)
    retriever = load_retriever(args.retriever_module, args.retriever_class, args.device)

    gallery_keys = [
        ItemKey(text="", img_path=p, image=im)
        for p, im in zip(kept_gallery_paths, kept_gallery_imgs)
    ]
    gallery_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant="gallery",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    gallery_vecs = embed_in_batches_resumable(
        retriever.embed_targets,
        gallery_keys,
        args.batch_size,
        "Embedding gallery",
        cache_dir=cache_dir,
        cache_key=gallery_cache_key,
        log=log,
        normalize=args.normalize,
    )
    gallery_vecs_gpu = gallery_vecs.to(args.device)

    # ------------------------------------------------------------------
    # 5. Build three query-variant key lists
    # ------------------------------------------------------------------
    multimodal_keys: List[ItemKey] = []

    for rec, _ in valid_records:
        img = ref_imgs[rec.ref_abs]
        multimodal_keys.append(ItemKey(text=rec.full_text, img_path=rec.ref_abs, image=img))

    text_only_keys = apply_attacks(multimodal_keys, text_only_attack_cfg, which="query")
    image_only_keys = apply_attacks(multimodal_keys, image_only_attack_cfg, which="query")

    log.info(
        "Text-only queries use %s (%s)",
        text_only_attack_path,
        attack_signature(text_only_attack_cfg),
    )
    log.info(
        "Image-only queries use %s (%s)",
        image_only_attack_path,
        attack_signature(image_only_attack_cfg),
    )

    # ------------------------------------------------------------------
    # 6. Embed all three conditions
    # ------------------------------------------------------------------
    log.info("Embedding multimodal queries ...")
    mm_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant="queries_multimodal",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    mm_vecs = embed_in_batches_resumable(
        retriever.embed_queries,
        multimodal_keys,
        args.batch_size,
        "multimodal queries",
        cache_dir=cache_dir,
        cache_key=mm_cache_key,
        log=log,
        normalize=args.normalize,
    )
    mm_vecs = mm_vecs.to(args.device)

    log.info("Embedding text-only queries ...")
    txt_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant=f"queries_textonly_{attack_signature(text_only_attack_cfg)}",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    txt_vecs = embed_in_batches_resumable(
        retriever.embed_queries,
        text_only_keys,
        args.batch_size,
        "text-only queries",
        cache_dir=cache_dir,
        cache_key=txt_cache_key,
        log=log,
        normalize=args.normalize,
    )
    txt_vecs = txt_vecs.to(args.device)

    log.info("Embedding image-only queries ...")
    img_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant=f"queries_imageonly_{attack_signature(image_only_attack_cfg)}",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    img_vecs = embed_in_batches_resumable(
        retriever.embed_queries,
        image_only_keys,
        args.batch_size,
        "image-only queries",
        cache_dir=cache_dir,
        cache_key=img_cache_key,
        log=log,
        normalize=args.normalize,
    )
    img_vecs = img_vecs.to(args.device)

    # ------------------------------------------------------------------
    # 7. Compute similarities
    # ------------------------------------------------------------------
    mm_sims = mm_vecs @ gallery_vecs_gpu.T
    txt_sims = txt_vecs @ gallery_vecs_gpu.T
    img_sims = img_vecs @ gallery_vecs_gpu.T

    topk_cap = min(TOP_K, gallery_vecs_gpu.shape[0])

    # ------------------------------------------------------------------
    # 8. Per-sample analysis
    # ------------------------------------------------------------------
    counts = {"either": 0, "text_only": 0, "image_only": 0, "multimodal_needed": 0}
    samples_out: List[dict] = []

    for (rec, tgt_idx), mm_sim, txt_sim, img_sim in zip(
        valid_records, mm_sims, txt_sims, img_sims,
    ):
        mm_sorted = torch.argsort(mm_sim, descending=True)
        txt_sorted = torch.argsort(txt_sim, descending=True)
        img_sorted = torch.argsort(img_sim, descending=True)

        mm_rank = int((mm_sorted == tgt_idx).nonzero(as_tuple=False)[0].item()) + 1
        txt_rank = int((txt_sorted == tgt_idx).nonzero(as_tuple=False)[0].item()) + 1
        img_rank = int((img_sorted == tgt_idx).nonzero(as_tuple=False)[0].item()) + 1

        mm_hit = mm_rank <= K
        txt_hit = txt_rank <= K
        img_hit = img_rank <= K

        if txt_hit and img_hit:
            category = "either"
        elif txt_hit:
            category = "text_only"
        elif img_hit:
            category = "image_only"
        else:
            category = "multimodal_needed"
        counts[category] += 1

        samples_out.append({
            "query_idx": rec.idx,
            "query_text": rec.caption,
            "full_text": rec.full_text,
            "query_image_path": rec.ref_abs,
            "target_image_path": rec.tgt_abs,
            "target_gallery_index": tgt_idx,
            "category": category,
            "ranks": {
                "multimodal": mm_rank,
                "text_only": txt_rank,
                "image_only": img_rank,
            },
            "hits_at_k": {
                "multimodal": mm_hit,
                "text_only": txt_hit,
                "image_only": img_hit,
            },
            "retrievals": {
                "multimodal": mm_sorted[:topk_cap].cpu().tolist(),
                "text_only": txt_sorted[:topk_cap].cpu().tolist(),
                "image_only": img_sorted[:topk_cap].cpu().tolist(),
            },
        })

    total = len(samples_out)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # 9. Compute summary recall numbers
    # ------------------------------------------------------------------
    mm_recall = 100.0 * sum(1 for s in samples_out if s["hits_at_k"]["multimodal"]) / total if total else 0.0
    txt_recall = 100.0 * sum(1 for s in samples_out if s["hits_at_k"]["text_only"]) / total if total else 0.0
    img_recall = 100.0 * sum(1 for s in samples_out if s["hits_at_k"]["image_only"]) / total if total else 0.0

    # ------------------------------------------------------------------
    # 10. Write JSON
    # ------------------------------------------------------------------
    report = {
        "metadata": {
            "dataset": args.dataset,
            "task_id": args.task_id,
            "retriever_module": args.retriever_module,
            "retriever_class": args.retriever_class,
            "k": K,
            "top_k": TOP_K,
            "max_queries": args.max_queries,
            "smoke_gallery_size": args.smoke_gallery_size,
            "normalize": args.normalize,
            "mbeir_root": str(mbeir_root),
            "query_instruction": instruction,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "text_only_attack_json": str(text_only_attack_path),
            "text_only_attack_name": text_only_attack_name,
            "text_only_attack_sig": attack_signature(text_only_attack_cfg),
            "image_only_attack_json": str(image_only_attack_path),
            "image_only_attack_name": image_only_attack_name,
            "image_only_attack_sig": attack_signature(image_only_attack_cfg),
            "total_queries": total,
            "elapsed_seconds": round(elapsed, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "summary": {
            "counts": counts,
            "recall_at_k": {
                "multimodal": round(mm_recall, 2),
                "text_only": round(txt_recall, 2),
                "image_only": round(img_recall, 2),
            },
        },
        "gallery_paths": kept_gallery_paths,
        "samples": samples_out,
    }

    output_path = Path(args.output)
    _atomic_json_dump(report, output_path)

    log.info("=" * 60)
    log.info("RETRIEVAL DATA GENERATED  (K=%d, top_k=%d, model=%s)", K, TOP_K, args.retriever_module)
    log.info("=" * 60)
    log.info("Total valid queries:  %d", total)
    for cat in ("either", "text_only", "image_only", "multimodal_needed"):
        c = counts[cat]
        pct = 100.0 * c / total if total else 0.0
        log.info("  %-22s %5d  (%5.1f%%)", cat, c, pct)
    log.info("-" * 60)
    log.info("  Recall@%d  multimodal: %.1f%%  text: %.1f%%  image: %.1f%%",
             K, mm_recall, txt_recall, img_recall)
    log.info("  Elapsed: %.1f s", elapsed)
    log.info("  Output:  %s", output_path.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
