#!/usr/bin/env python3
"""
================================================================================
generate_retrieval_data_circo.py — Retrieval Data Generator for CIRCO
================================================================================

PURPOSE:
    Generate the same retrieval-analysis JSON as
    `retrieval/generate_retrieval_data.py`, but with multi-positive ground
    truth handling for CIRCO.

    For every query, the script runs retrieval under three conditions:
      1. Multimodal  (query text + reference image)
      2. Text-only   (query text, no image)
      3. Image-only  (reference image, no text)

    Unlike the generic generator, ranks and hit@K are computed against the full
    positive set:
      - hit@K = whether ANY positive target is retrieved in the top-K
      - rank  = best rank among all positives

    The output remains compatible with `the existing browser tooling` while
    also recording the complete positive-target set for each query.

EXAMPLE:
    python retrieval/generate_retrieval_data_circo.py \
        --mbeir_root /data/M-BEIR \
        --dataset circo \
        --task_id 7 \
        --query_file mbeir_circo_task7_test.jsonl \
        --retriever_module retrievers.qwen3vl8b_vllm_retriever \
        --retriever_class Retriever \
        --normalize \
        --k 10 \
        --top_k 20 \
        --output retrieval/retrieval_data_circo_qwen3vl8b.json
================================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from retrieval.generate_retrieval_data import (
    _atomic_json_dump,
    _build_embedding_cache_key,
    embed_in_batches_resumable,
)
from subset_evaluation.eval_subset_circo import (
    IMG_EXTS,
    ItemKey,
    _normalize_cand_id,
    find_dir,
    get_instruction,
    infer_split_from_query_name,
    load_cand_pool_map,
    load_images_threaded,
    load_mbeir_queries,
    load_retriever,
    load_qrels_multi,
    _load_ablation_cfg as load_attack_cfg,
    apply_ablation,
    ablation_signature as attack_signature,
)


def apply_attacks(keys, cfg, which: str = "query"):
    """Locked-down shim: only the query-side, two-name ablation set is allowed."""
    if which != "query":
        raise ValueError("released evaluator only ablates the query side")
    return apply_ablation(keys, cfg)


DEFAULT_TEXT_ONLY_ATTACK_JSON = "ablation_configs/image_only_zero_image.json"
DEFAULT_IMAGE_ONLY_ATTACK_JSON = "ablation_configs/text_only_drop_text.json"


@dataclass
class QueryRecord:
    idx: int
    query_id: str
    ref_abs: str
    caption: str
    full_text: str
    canonical_tgt_abs: str
    positive_cand_ids: List[str]


def _first_nonempty_lookup(mapping: Dict[str, str], cand_id: str) -> Optional[str]:
    value = mapping.get(cand_id)
    if value is not None:
        return value
    return mapping.get(_normalize_cand_id(cand_id))


def _lookup_gallery_idx(mapping: Dict[str, int], cand_id: str) -> Optional[int]:
    idx = mapping.get(cand_id)
    if idx is not None:
        return idx
    return mapping.get(_normalize_cand_id(cand_id))


def _dedupe_candidate_ids(values: Optional[Sequence[object]]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]

    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        cand_id = str(value)
        if not cand_id or cand_id in seen:
            continue
        seen.add(cand_id)
        out.append(cand_id)
    return out


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(_PROJECT_ROOT) / path).resolve()


def resolve_qrels_path(
    mbeir_root: Path,
    split_hint: str,
    explicit_qrels_path: Optional[str],
) -> Path:
    if explicit_qrels_path:
        return Path(explicit_qrels_path).expanduser().resolve()

    qrels_dir = find_dir(mbeir_root, "qrels")
    candidates: List[Path] = []
    if split_hint:
        candidates.append(qrels_dir / f"{split_hint}.tsv")
    candidates.extend([qrels_dir / "test.tsv", qrels_dir / "val.tsv"])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return qrels_dir / (f"{split_hint}.tsv" if split_hint else "test.tsv")


def build_records(
    rows: List[dict],
    mbeir_root: Path,
    id2relpath: Dict[str, str],
    qrels_multi: Dict[str, List[str]],
    instruction: str,
    log: logging.Logger,
) -> Tuple[List[QueryRecord], List[str], int]:
    records: List[QueryRecord] = []
    ref_paths_needed: List[str] = []
    misses = 0

    for i, item in enumerate(rows):
        qid = str(item.get("query_id", i))
        caption = item.get("query_txt") or item.get("query_text") or ""
        ref_rel = item.get("query_img_path")
        pos_list = _dedupe_candidate_ids(item.get("pos_cand_list"))
        canonical_pos_id = pos_list[0] if pos_list else None

        if not caption or not ref_rel or not canonical_pos_id:
            continue

        ref_abs = str((mbeir_root / ref_rel).resolve())
        if not os.path.exists(ref_abs):
            misses += 1
            continue

        canonical_rel = _first_nonempty_lookup(id2relpath, canonical_pos_id)
        if canonical_rel is None:
            misses += 1
            continue

        canonical_tgt_abs = str((mbeir_root / canonical_rel).resolve())
        if not os.path.exists(canonical_tgt_abs):
            misses += 1
            continue

        positive_cand_ids = _dedupe_candidate_ids(
            item.get("gt_cand_list") or qrels_multi.get(qid) or pos_list
        )
        if not positive_cand_ids:
            misses += 1
            continue

        full_text = (instruction + "\n" + caption).strip() if instruction else caption
        records.append(QueryRecord(
            idx=i,
            query_id=qid,
            ref_abs=ref_abs,
            caption=caption,
            full_text=full_text,
            canonical_tgt_abs=canonical_tgt_abs,
            positive_cand_ids=positive_cand_ids,
        ))
        ref_paths_needed.append(ref_abs)

    log.info("Valid queries: %d  (misses: %d)", len(records), misses)
    return records, ref_paths_needed, misses


def best_rank_and_index_for_positive_set(
    sorted_idx: torch.Tensor,
    positive_idx_set: Set[int],
) -> Tuple[Optional[int], Optional[int]]:
    for rank0, idx in enumerate(sorted_idx.tolist()):
        gallery_idx = int(idx)
        if gallery_idx in positive_idx_set:
            return rank0 + 1, gallery_idx
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate retrieval-analysis JSON for CIRCO with multi-positive GT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--mbeir_root", type=str, required=True,
                    help="Path to the M-BEIR data directory.")
    ap.add_argument("--dataset", type=str, default="circo",
                    help="Dataset name (default: circo).")
    ap.add_argument("--query_file", type=str, required=True,
                    help="Query JSONL filename inside <mbeir_root>/query/test/.")
    ap.add_argument(
        "--query_split",
        type=str,
        default=None,
        help="Optional query split override, e.g. test, val, or train. If omitted, inferred from query_file when possible.",
    )
    ap.add_argument("--task_id", type=int, default=None,
                    help="Task ID to filter candidate pool files (e.g. 7).")
    ap.add_argument("--qrels_path", type=str, default=None,
                    help="Optional path to multi-positive qrels TSV. Defaults to <mbeir_root>/qrels/test.tsv when present.")
    ap.add_argument(
        "--query_instruction",
        type=str,
        default=None,
        help=(
            "Optional query instruction override. If omitted, uses the dataset default. "
            "Pass an empty string to disable the instruction prefix."
        ),
    )

    ap.add_argument("--retriever_module", required=True,
                    help="Dotted module path, e.g. retrievers.mmembed_retriever.")
    ap.add_argument("--retriever_class", default="Retriever",
                    help="Class name inside the retriever module (default: Retriever).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Torch device (default: cuda if available, else cpu).")

    ap.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for embedding (default: 32).")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalise embeddings before similarity computation.")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="Threads for image decoding (default: 8).")

    ap.add_argument("--k", type=int, default=10,
                    help="Recall@K threshold for categorisation (default: 10).")
    ap.add_argument("--top_k", type=int, default=20,
                    help="Number of top retrieved results to save per condition (default: 20).")
    ap.add_argument("--output", type=str, default="retrieval/retrieval_data_circo.json",
                    help="Output JSON path (default: retrieval/retrieval_data_circo.json).")
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for resumable embedding caches. When set, gallery and query "
            "embeddings are stored per batch and reused on reruns."
        ),
    )
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
    log = logging.getLogger("generate_retrieval_data_circo")

    if "cuda" in args.device and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    mbeir_root = Path(args.mbeir_root).expanduser().resolve()
    dataset_l = args.dataset.lower()
    if dataset_l != "circo":
        log.warning("This script is intended for CIRCO-style multi-positive data; got dataset=%s", args.dataset)

    K = args.k
    TOP_K = args.top_k
    t0 = time.time()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    text_only_attack_path = resolve_repo_path(args.text_only_attack_json)
    image_only_attack_path = resolve_repo_path(args.image_only_attack_json)
    text_only_attack_cfg, text_only_attack_name = load_attack_cfg(str(text_only_attack_path))
    image_only_attack_cfg, image_only_attack_name = load_attack_cfg(str(image_only_attack_path))

    query_split = args.query_split or infer_split_from_query_name(args.query_file)
    if not query_split and "train" in args.query_file.lower():
        query_split = "train"
    if not query_split:
        query_split = "test"

    query_path = find_dir(mbeir_root, "query") / query_split / args.query_file
    rows = load_mbeir_queries(query_path)
    log.info("Loaded %d raw query rows from %s", len(rows), query_path.name)

    split_hint = infer_split_from_query_name(args.query_file)
    cand_pool_dir = find_dir(mbeir_root, "cand_pool")
    id2relpath = load_cand_pool_map(
        cand_pool_dir, dataset_l, args.task_id, split_hint, log=log,
    )

    qrels_path = resolve_qrels_path(mbeir_root, split_hint, args.qrels_path)
    qrels_multi = load_qrels_multi(qrels_path)
    if qrels_multi:
        log.info("Loaded multi-positive qrels for %d queries from %s", len(qrels_multi), qrels_path)
    else:
        log.info("No multi-positive qrels loaded from %s; falling back to query-local GT lists", qrels_path)

    pool_abs = {str((mbeir_root / p).resolve()) for p in id2relpath.values() if p}
    gallery_paths = sorted(
        p for p in pool_abs
        if Path(p).suffix.lower() in IMG_EXTS and Path(p).exists()
    )
    log.info("Gallery: %d images", len(gallery_paths))

    instruction = get_instruction(dataset_l) if args.query_instruction is None else args.query_instruction
    records, ref_paths_needed, _ = build_records(
        rows, mbeir_root, id2relpath, qrels_multi, instruction, log,
    )
    if not records:
        log.error("No valid queries — exiting.")
        return

    gal_imgs = load_images_threaded(gallery_paths, args.num_workers, "Loading gallery images")
    ref_imgs = load_images_threaded(
        sorted(set(ref_paths_needed)), args.num_workers, "Loading query images",
    )

    kept_gallery_paths = [p for p in gallery_paths if p in gal_imgs]
    kept_gallery_imgs = [gal_imgs[p] for p in kept_gallery_paths]
    log.info("Gallery decoded: %d / %d", len(kept_gallery_paths), len(gallery_paths))

    path2idx = {str(Path(p).resolve()): i for i, p in enumerate(kept_gallery_paths)}
    name2idx = {Path(p).name: i for i, p in enumerate(kept_gallery_paths)}

    cand_id_to_gallery_idx: Dict[str, int] = {}
    for cand_id, rel_path in id2relpath.items():
        abs_path = str((mbeir_root / rel_path).resolve())
        gallery_idx = path2idx.get(abs_path)
        if gallery_idx is None:
            gallery_idx = name2idx.get(Path(abs_path).name)
        if gallery_idx is None:
            continue
        cand_id_to_gallery_idx[cand_id] = gallery_idx
        cand_id_to_gallery_idx[_normalize_cand_id(cand_id)] = gallery_idx

    valid_records: List[Tuple[QueryRecord, Set[int], Optional[int]]] = []
    for rec in records:
        if ref_imgs.get(rec.ref_abs) is None:
            continue

        positive_idx_set: Set[int] = set()
        for cand_id in rec.positive_cand_ids:
            gallery_idx = _lookup_gallery_idx(cand_id_to_gallery_idx, cand_id)
            if gallery_idx is not None:
                positive_idx_set.add(gallery_idx)

        if not positive_idx_set:
            continue

        canonical_tgt_idx = path2idx.get(str(Path(rec.canonical_tgt_abs).resolve()))
        if canonical_tgt_idx is None:
            canonical_tgt_idx = name2idx.get(Path(rec.canonical_tgt_abs).name)

        valid_records.append((rec, positive_idx_set, canonical_tgt_idx))

    log.info("Queries with decoded images & at least one GT in gallery: %d", len(valid_records))
    if not valid_records:
        log.error("No valid decoded queries — exiting.")
        return

    log.info("Loading retriever: %s.%s", args.retriever_module, args.retriever_class)
    retriever = load_retriever(args.retriever_module, args.retriever_class, args.device)

    gallery_keys = [
        ItemKey(text="", img_path=path, image=image)
        for path, image in zip(kept_gallery_paths, kept_gallery_imgs)
    ]
    gallery_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=query_split or split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant="gallery",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    gallery_vecs = embed_in_batches_resumable(
        retriever.embed_targets, gallery_keys, args.batch_size, "Embedding gallery",
        cache_dir=cache_dir,
        cache_key=gallery_cache_key,
        log=log,
        normalize=args.normalize,
    )
    gallery_vecs_gpu = gallery_vecs.to(args.device)

    multimodal_keys: List[ItemKey] = []

    for rec, _, _ in valid_records:
        image = ref_imgs[rec.ref_abs]
        multimodal_keys.append(ItemKey(text=rec.full_text, img_path=rec.ref_abs, image=image))

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

    log.info("Embedding multimodal queries ...")
    mm_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=query_split or split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant="queries_multimodal",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    mm_vecs = embed_in_batches_resumable(
        retriever.embed_queries, multimodal_keys, args.batch_size, "multimodal queries",
        cache_dir=cache_dir,
        cache_key=mm_cache_key,
        log=log,
        normalize=args.normalize,
    )
    mm_vecs = mm_vecs.to(args.device)

    log.info("Embedding text-only queries ...")
    txt_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=query_split or split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant=f"queries_textonly_{attack_signature(text_only_attack_cfg)}",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    txt_vecs = embed_in_batches_resumable(
        retriever.embed_queries, text_only_keys, args.batch_size, "text-only queries",
        cache_dir=cache_dir,
        cache_key=txt_cache_key,
        log=log,
        normalize=args.normalize,
    )
    txt_vecs = txt_vecs.to(args.device)

    log.info("Embedding image-only queries ...")
    img_cache_key = _build_embedding_cache_key(
        dataset=args.dataset,
        split_hint=query_split or split_hint,
        task_id=args.task_id,
        retriever_module=args.retriever_module,
        retriever_class=args.retriever_class,
        query_file=args.query_file,
        variant=f"queries_imageonly_{attack_signature(image_only_attack_cfg)}",
        query_instruction=instruction,
        normalize=args.normalize,
    )
    img_vecs = embed_in_batches_resumable(
        retriever.embed_queries, image_only_keys, args.batch_size, "image-only queries",
        cache_dir=cache_dir,
        cache_key=img_cache_key,
        log=log,
        normalize=args.normalize,
    )
    img_vecs = img_vecs.to(args.device)

    mm_sims = mm_vecs @ gallery_vecs_gpu.T
    txt_sims = txt_vecs @ gallery_vecs_gpu.T
    img_sims = img_vecs @ gallery_vecs_gpu.T

    topk_cap = min(TOP_K, gallery_vecs_gpu.shape[0])
    counts = {"either": 0, "text_only": 0, "image_only": 0, "multimodal_needed": 0}
    samples_out: List[dict] = []

    for (rec, positive_idx_set, canonical_tgt_idx), mm_sim, txt_sim, img_sim in zip(
        valid_records, mm_sims, txt_sims, img_sims,
    ):
        positive_gallery_indices = sorted(int(idx) for idx in positive_idx_set)
        positive_target_image_paths = [kept_gallery_paths[idx] for idx in positive_gallery_indices]

        display_target_idx = canonical_tgt_idx
        if display_target_idx is None:
            display_target_idx = positive_gallery_indices[0]
        display_target_path = kept_gallery_paths[display_target_idx]

        mm_sorted = torch.argsort(mm_sim, descending=True)
        txt_sorted = torch.argsort(txt_sim, descending=True)
        img_sorted = torch.argsort(img_sim, descending=True)

        mm_rank, mm_best_idx = best_rank_and_index_for_positive_set(mm_sorted, positive_idx_set)
        txt_rank, txt_best_idx = best_rank_and_index_for_positive_set(txt_sorted, positive_idx_set)
        img_rank, img_best_idx = best_rank_and_index_for_positive_set(img_sorted, positive_idx_set)

        mm_hit = mm_rank is not None and mm_rank <= K
        txt_hit = txt_rank is not None and txt_rank <= K
        img_hit = img_rank is not None and img_rank <= K

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
            "query_id": rec.query_id,
            "query_text": rec.caption,
            "full_text": rec.full_text,
            "query_image_path": rec.ref_abs,
            "target_image_path": display_target_path,
            "target_gallery_index": display_target_idx,
            "canonical_target_image_path": rec.canonical_tgt_abs,
            "canonical_target_gallery_index": canonical_tgt_idx,
            "num_gt": len(positive_gallery_indices),
            "positive_candidate_ids": rec.positive_cand_ids,
            "positive_gallery_indices": positive_gallery_indices,
            "positive_target_image_paths": positive_target_image_paths,
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
            "best_positive_gallery_indices": {
                "multimodal": mm_best_idx,
                "text_only": txt_best_idx,
                "image_only": img_best_idx,
            },
            "best_positive_image_paths": {
                "multimodal": kept_gallery_paths[mm_best_idx] if mm_best_idx is not None else None,
                "text_only": kept_gallery_paths[txt_best_idx] if txt_best_idx is not None else None,
                "image_only": kept_gallery_paths[img_best_idx] if img_best_idx is not None else None,
            },
            "retrievals": {
                "multimodal": mm_sorted[:topk_cap].cpu().tolist(),
                "text_only": txt_sorted[:topk_cap].cpu().tolist(),
                "image_only": img_sorted[:topk_cap].cpu().tolist(),
            },
        })

    total = len(samples_out)
    elapsed = time.time() - t0

    mm_recall = 100.0 * sum(1 for s in samples_out if s["hits_at_k"]["multimodal"]) / total if total else 0.0
    txt_recall = 100.0 * sum(1 for s in samples_out if s["hits_at_k"]["text_only"]) / total if total else 0.0
    img_recall = 100.0 * sum(1 for s in samples_out if s["hits_at_k"]["image_only"]) / total if total else 0.0

    report = {
        "metadata": {
            "dataset": args.dataset,
            "task_id": args.task_id,
            "retriever_module": args.retriever_module,
            "retriever_class": args.retriever_class,
            "k": K,
            "top_k": TOP_K,
            "normalize": args.normalize,
            "multi_positive": True,
            "qrels_path": str(qrels_path) if qrels_path.exists() else None,
            "mbeir_root": str(mbeir_root),
            "query_split": query_split,
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
        count = counts[cat]
        pct = 100.0 * count / total if total else 0.0
        log.info("  %-22s %5d  (%5.1f%%)", cat, count, pct)
    log.info("-" * 60)
    log.info("  Recall@%d  multimodal: %.1f%%  text: %.1f%%  image: %.1f%%",
             K, mm_recall, txt_recall, img_recall)
    log.info("  Elapsed: %.1f s", elapsed)
    log.info("  Output:  %s", output_path.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
