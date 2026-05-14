#!/usr/bin/env python3
"""
Aggregate per-retriever retrieval JSONs into a browser-compatible pseudo-retriever.

This script reads multiple `retrieval_data_*.json` files produced by
`generate_retrieval_data.py` or `generate_retrieval_data_circo.py` and combines
them into a single report whose ranks are aggregated across retrievers.

Default definition:
  best_text_rank(i)  = min_p rank_text(p, i)
  best_image_rank(i) = min_p rank_image(p, i)
  best_mm_rank(i)    = min_p rank_mm(p, i)

At a chosen K:
  composition_required if best_text_rank > K and best_image_rank > K
                    and at least M retrievers retrieve multimodally within K
  unresolved          if best_text_rank > K and best_image_rank > K
                    and fewer than M retrievers retrieve multimodally within K
  shortcut_solvable otherwise

The output keeps the original browser-facing shape:
  - `ranks` uses the best rank across retrievers for each modality
  - `retrievals` for each modality come from the retriever that achieved
    the best rank for that modality

Extra aggregation metadata is stored under `metadata["aggregation"]`,
`summary["aggregate_*"]`, and `samples[*]["aggregation"]`.


python retrieval/aggregate_retrieval_data.py \
    --results_dir shortcut_audit \
    --dataset cirr \
    --output shortcut_audit/retrieval_data_cirr_task7_aggregated_best_rank_mm_any.json

"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


LEGACY_CATEGORY_ORDER = ("either", "text_only", "image_only", "multimodal_needed")
DETAIL_LABEL_ORDER = (
    "both_unimodal",
    "text_only",
    "image_only",
    "composition_required",
    "unresolved",
)
HIGH_LEVEL_LABEL_ORDER = ("shortcut_solvable", "composition_required", "unresolved")
MODALITIES = ("multimodal", "text_only", "image_only")


@dataclass
class RetrieverReport:
    retriever_key: str
    path: Path
    data: dict
    gallery_paths: List[str]
    gallery_index_by_path: Dict[str, int]
    samples_by_query_idx: Dict[int, dict]
    query_order: List[int]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate retrieval_data JSONs across retrievers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--json",
        type=str,
        nargs="+",
        default=None,
        help="Explicit input JSON files. If omitted, files are discovered from --results_dir.",
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        nargs="+",
        default=None,
        help="Directories to scan for retrieval_data_*.json files.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to aggregate when using --results_dir.",
    )
    ap.add_argument(
        "--retrievers",
        type=str,
        nargs="+",
        default=None,
        help="Optional retriever suffixes to keep, e.g. e5_omni lamra mmembed.",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="K used for labels and summary. Defaults to the shared source metadata.k.",
    )
    ap.add_argument(
        "--min_mm_success_retrievers",
        type=int,
        default=1,
        help="Minimum number of retrievers that must retrieve multimodally within K to call a sample composition_required.",
    )
    ap.add_argument(
        "--retriever_name",
        type=str,
        default=None,
        help="Retriever suffix for the output filename and browser dropdown entry.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Defaults to a retrieval_data_<dataset>_task<id>_<retriever>.json file beside the first input.",
    )
    ap.add_argument(
        "--mbeir_root",
        type=str,
        default=None,
        help=(
            "Optional local M-BEIR root used to rewrite stored absolute paths "
            "from different machines before gallery-set comparison."
        ),
    )
    ap.add_argument(
        "--rewrite_prefix",
        action="append",
        default=None,
        metavar="OLD=NEW",
        help=(
            "Optional path-prefix rewrite applied before gallery-set comparison "
            "and output emission. May be passed multiple times."
        ),
    )
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args()


def parse_result_filename(path: Path) -> tuple[str, str | None, str | None]:
    m = re.match(r"retrieval_data_(.+?)_task(\d+)_(.+)\.json", path.name)
    if not m:
        return path.stem, None, None
    dataset, task_id, retriever = m.groups()
    return retriever, dataset, task_id


def discover_input_files(
    result_dirs: Sequence[str],
    dataset: str | None,
    retrievers: set[str] | None,
) -> List[Path]:
    found: List[Path] = []
    for result_dir in result_dirs:
        root = Path(result_dir).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.glob("retrieval_data_*.json")):
            retriever_key, ds, _task_id = parse_result_filename(path)
            if retriever_key.startswith("aggregated_"):
                continue
            if dataset is not None and ds != dataset:
                continue
            if retrievers is not None and retriever_key not in retrievers:
                continue
            found.append(path)
    return found


def _apply_rewrite_rules(path: str, rules: list[tuple[str, str]]) -> str:
    if not path:
        return path
    for old, new in rules:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def _build_mbeir_root_rules(data: dict, mbeir_root: str | None) -> list[tuple[str, str]]:
    if not mbeir_root:
        return []

    local_root = str(Path(mbeir_root).expanduser().resolve())
    original_root = data.get("metadata", {}).get("mbeir_root", "")
    if not original_root:
        if data.get("gallery_paths"):
            sample_path = data["gallery_paths"][0]
            for marker in ("mbeir_images", "cand_pool", "M-BEIR"):
                idx = sample_path.find(marker)
                if idx > 0:
                    original_root = sample_path[:idx].rstrip("/")
                    break
    if not original_root or original_root == local_root:
        return []

    rules = [(original_root, local_root)]

    sample_gp = data["gallery_paths"][0] if data.get("gallery_paths") else ""
    if sample_gp and not sample_gp.startswith(original_root):
        original_parent = str(Path(original_root).parent)
        local_parent = str(Path(local_root).parent)
        if original_parent and original_parent != "/" and sample_gp.startswith(original_parent):
            rules.append((original_parent, local_parent))

    rules.sort(key=lambda item: -len(item[0]))
    return rules


def _parse_rewrite_rule(spec: str) -> tuple[str, str]:
    old, sep, new = spec.partition("=")
    if sep != "=" or not old or not new:
        raise ValueError(
            f"Invalid --rewrite_prefix value {spec!r}; expected OLD=NEW"
        )
    return old, new


def _normalize_rewrite_rules(rules: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: dict[str, str] = {}
    for old, new in rules:
        if not old or not new or old == new:
            continue
        deduped[old] = new
    return sorted(deduped.items(), key=lambda item: -len(item[0]))


def _rewrite_report_paths_in_place(data: dict, rules: list[tuple[str, str]]) -> None:
    if not rules:
        return

    rewrite = lambda p: _apply_rewrite_rules(p, rules)
    data["gallery_paths"] = [rewrite(p) for p in data.get("gallery_paths", [])]

    for sample in data.get("samples", []):
        for key in (
            "query_image_path",
            "target_image_path",
            "canonical_target_image_path",
        ):
            if key in sample:
                sample[key] = rewrite(sample.get(key, ""))

        if isinstance(sample.get("positive_target_image_paths"), list):
            sample["positive_target_image_paths"] = [
                rewrite(p) for p in sample["positive_target_image_paths"]
            ]

        if isinstance(sample.get("best_positive_image_paths"), dict):
            sample["best_positive_image_paths"] = {
                k: rewrite(v) if v is not None else None
                for k, v in sample["best_positive_image_paths"].items()
            }

    meta = data.setdefault("metadata", {})
    original_mbeir_root = meta.get("mbeir_root")
    if original_mbeir_root:
        rewritten_mbeir_root = rewrite(original_mbeir_root)
        if rewritten_mbeir_root != original_mbeir_root:
            meta["original_mbeir_root"] = original_mbeir_root
        meta["mbeir_root"] = rewritten_mbeir_root

    if meta.get("qrels_path"):
        meta["qrels_path"] = rewrite(meta["qrels_path"])


def load_retriever_report(
    path: Path,
    mbeir_root: str | None = None,
    extra_rewrite_rules: Sequence[tuple[str, str]] | None = None,
) -> RetrieverReport:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rewrite_rules = _normalize_rewrite_rules(
        _build_mbeir_root_rules(data, mbeir_root) + list(extra_rewrite_rules or [])
    )
    _rewrite_report_paths_in_place(data, rewrite_rules)

    retriever_key, _dataset_from_name, _task_from_name = parse_result_filename(path)
    gallery_paths = list(data.get("gallery_paths", []))
    gallery_index_by_path = {p: idx for idx, p in enumerate(gallery_paths)}
    if len(gallery_index_by_path) != len(gallery_paths):
        raise ValueError(f"{path}: gallery_paths contains duplicates")

    samples_by_query_idx: Dict[int, dict] = {}
    query_order: List[int] = []
    for sample in data.get("samples", []):
        qidx = int(sample["query_idx"])
        if qidx in samples_by_query_idx:
            raise ValueError(f"{path}: duplicate query_idx={qidx}")
        samples_by_query_idx[qidx] = sample
        query_order.append(qidx)

    return RetrieverReport(
        retriever_key=retriever_key,
        path=path,
        data=data,
        gallery_paths=gallery_paths,
        gallery_index_by_path=gallery_index_by_path,
        samples_by_query_idx=samples_by_query_idx,
        query_order=query_order,
    )


def require_shared_metadata(reports: Sequence[RetrieverReport], k_override: int | None) -> dict:
    base = reports[0].data.get("metadata", {})
    dataset = base.get("dataset")
    task_id = base.get("task_id")
    multi_positive = bool(base.get("multi_positive", False))

    query_set = set(reports[0].samples_by_query_idx)
    for report in reports[1:]:
        meta = report.data.get("metadata", {})
        if meta.get("dataset") != dataset:
            raise ValueError(
                f"Dataset mismatch: {report.path} has {meta.get('dataset')} but expected {dataset}"
            )
        if meta.get("task_id") != task_id:
            raise ValueError(
                f"Task mismatch: {report.path} has {meta.get('task_id')} but expected {task_id}"
            )
        if bool(meta.get("multi_positive", False)) != multi_positive:
            raise ValueError(
                f"Multi-positive mismatch: {report.path} is inconsistent with other inputs"
            )
        if set(report.samples_by_query_idx) != query_set:
            raise ValueError(
                f"Query coverage mismatch: {report.path} does not have the same query_idx set"
            )

    k_values = {int(r.data.get("metadata", {}).get("k", 10)) for r in reports}
    if k_override is None and len(k_values) != 1:
        raise ValueError(
            f"Input files disagree on metadata.k: {sorted(k_values)}. Pass --k explicitly."
        )
    out_k = k_override if k_override is not None else next(iter(k_values))

    top_k_values = [int(r.data.get("metadata", {}).get("top_k", 0)) for r in reports]
    out_top_k = min(top_k_values)
    if out_top_k <= 0:
        raise ValueError("Could not determine a positive shared top_k from inputs")

    for key in ("text_only_attack_sig", "image_only_attack_sig", "query_instruction"):
        values = {r.data.get("metadata", {}).get(key) for r in reports}
        if len(values) != 1:
            pretty_values = sorted(repr(v) for v in values)
            raise ValueError(f"Input files disagree on metadata.{key}: {pretty_values}")

    return {
        "dataset": dataset,
        "task_id": task_id,
        "multi_positive": multi_positive,
        "k": out_k,
        "top_k": out_top_k,
        "base_metadata": base,
    }


def build_canonical_gallery(reports: Sequence[RetrieverReport]) -> tuple[List[str], Dict[str, int]]:
    canonical_gallery = list(reports[0].gallery_paths)
    canonical_set = set(canonical_gallery)
    for report in reports[1:]:
        gallery_set = set(report.gallery_paths)
        if gallery_set != canonical_set:
            missing = sorted(canonical_set - gallery_set)[:3]
            extra = sorted(gallery_set - canonical_set)[:3]
            raise ValueError(
                f"{report.path} has a different gallery set. missing={missing} extra={extra}"
            )
    return canonical_gallery, {p: idx for idx, p in enumerate(canonical_gallery)}


def remap_indices(
    retriever: RetrieverReport,
    gallery_indices: Sequence[int] | None,
    canonical_index_by_path: Dict[str, int],
    limit: int | None = None,
) -> List[int]:
    if not gallery_indices:
        return []

    out: List[int] = []
    seen: set[int] = set()
    for raw_idx in gallery_indices:
        idx = int(raw_idx)
        path = retriever.gallery_paths[idx]
        mapped = canonical_index_by_path[path]
        if mapped in seen:
            continue
        seen.add(mapped)
        out.append(mapped)
        if limit is not None and len(out) >= limit:
            break
    return out


def choose_best_retriever(
    reports: Sequence[RetrieverReport],
    query_idx: int,
    modality: str,
) -> tuple[int, RetrieverReport, dict]:
    best: tuple[int, str, RetrieverReport, dict] | None = None
    for report in reports:
        sample = report.samples_by_query_idx[query_idx]
        raw_rank = sample.get("ranks", {}).get(modality)
        if raw_rank is None:
            continue
        rank = int(raw_rank)
        candidate = (rank, report.retriever_key, report, sample)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError(
            f"All retrievers are missing ranks for query_idx={query_idx}, modality={modality}"
        )
    return int(best[0]), best[2], best[3]


def is_hit(rank: int | None, k: int) -> bool:
    return rank is not None and int(rank) <= k


def legacy_category(txt_rank: int | None, img_rank: int | None, k: int) -> str:
    txt_hit = is_hit(txt_rank, k)
    img_hit = is_hit(img_rank, k)
    if txt_hit and img_hit:
        return "either"
    if txt_hit:
        return "text_only"
    if img_hit:
        return "image_only"
    return "multimodal_needed"


def aggregate_label(
    txt_rank: int | None,
    img_rank: int | None,
    mm_support: int,
    k: int,
    min_mm_success_retrievers: int,
) -> tuple[str, str]:
    txt_hit = is_hit(txt_rank, k)
    img_hit = is_hit(img_rank, k)

    if txt_hit and img_hit:
        return "shortcut_solvable", "both_unimodal"
    if txt_hit:
        return "shortcut_solvable", "text_only"
    if img_hit:
        return "shortcut_solvable", "image_only"
    if mm_support >= min_mm_success_retrievers:
        return "composition_required", "composition_required"
    return "unresolved", "unresolved"


def retriever_stats(report: RetrieverReport, k: int) -> dict:
    total = len(report.samples_by_query_idx)
    mm_hits = 0
    txt_hits = 0
    img_hits = 0
    either_hits = 0
    for sample in report.samples_by_query_idx.values():
        mm = is_hit(sample.get("ranks", {}).get("multimodal"), k)
        txt = is_hit(sample.get("ranks", {}).get("text_only"), k)
        img = is_hit(sample.get("ranks", {}).get("image_only"), k)
        mm_hits += int(mm)
        txt_hits += int(txt)
        img_hits += int(img)
        either_hits += int(txt or img)
    pct = lambda n: round(100.0 * n / total, 2) if total else 0.0
    return {
        "recall_at_k": {
            "multimodal": pct(mm_hits),
            "text_only": pct(txt_hits),
            "image_only": pct(img_hits),
            "either_unimodal": pct(either_hits),
        }
    }


def canonical_target_path(sample: dict) -> str:
    return sample.get("canonical_target_image_path") or sample.get("target_image_path")


def build_output(
    reports: Sequence[RetrieverReport],
    out_k: int,
    out_top_k: int,
    min_mm_success_retrievers: int,
    retriever_name: str,
) -> dict:
    base_meta = reports[0].data.get("metadata", {})
    multi_positive = bool(base_meta.get("multi_positive", False))
    canonical_gallery, canonical_index_by_path = build_canonical_gallery(reports)
    query_order = list(reports[0].query_order)

    legacy_counts: Counter[str] = Counter()
    high_level_counts: Counter[str] = Counter()
    detail_counts: Counter[str] = Counter()
    mm_support_hist: Counter[int] = Counter()
    txt_support_hist: Counter[int] = Counter()
    img_support_hist: Counter[int] = Counter()
    either_support_hist: Counter[int] = Counter()

    samples_out: List[dict] = []

    for query_idx in query_order:
        base_sample = reports[0].samples_by_query_idx[query_idx]

        best_ranks: Dict[str, int] = {}
        best_retriever_keys: Dict[str, str] = {}
        best_retriever_samples: Dict[str, dict] = {}
        best_retriever_reports: Dict[str, RetrieverReport] = {}

        for modality in MODALITIES:
            best_rank, best_report, best_sample = choose_best_retriever(
                reports, query_idx, modality
            )
            best_ranks[modality] = best_rank
            best_retriever_keys[modality] = best_report.retriever_key
            best_retriever_samples[modality] = best_sample
            best_retriever_reports[modality] = best_report

        mm_support = 0
        txt_support = 0
        img_support = 0
        either_support = 0
        for report in reports:
            sample = report.samples_by_query_idx[query_idx]
            mm_hit = is_hit(sample.get("ranks", {}).get("multimodal"), out_k)
            txt_hit = is_hit(sample.get("ranks", {}).get("text_only"), out_k)
            img_hit = is_hit(sample.get("ranks", {}).get("image_only"), out_k)
            mm_support += int(mm_hit)
            txt_support += int(txt_hit)
            img_support += int(img_hit)
            either_support += int(txt_hit or img_hit)

        mm_support_hist[mm_support] += 1
        txt_support_hist[txt_support] += 1
        img_support_hist[img_support] += 1
        either_support_hist[either_support] += 1

        legacy = legacy_category(best_ranks["text_only"], best_ranks["image_only"], out_k)
        high_level, detail = aggregate_label(
            txt_rank=best_ranks["text_only"],
            img_rank=best_ranks["image_only"],
            mm_support=mm_support,
            k=out_k,
            min_mm_success_retrievers=min_mm_success_retrievers,
        )
        legacy_counts[legacy] += 1
        high_level_counts[high_level] += 1
        detail_counts[detail] += 1

        target_path = base_sample["target_image_path"]
        target_index = canonical_index_by_path[target_path]

        sample_out = {
            "query_idx": int(base_sample["query_idx"]),
            "query_text": base_sample["query_text"],
            "full_text": base_sample.get("full_text", base_sample["query_text"]),
            "query_image_path": base_sample["query_image_path"],
            "target_image_path": target_path,
            "target_gallery_index": target_index,
            "category": legacy,
            "ranks": best_ranks,
            "hits_at_k": {
                "multimodal": is_hit(best_ranks["multimodal"], out_k),
                "text_only": is_hit(best_ranks["text_only"], out_k),
                "image_only": is_hit(best_ranks["image_only"], out_k),
            },
            "retrievals": {
                modality: remap_indices(
                    best_retriever_reports[modality],
                    best_retriever_samples[modality].get("retrievals", {}).get(modality, []),
                    canonical_index_by_path,
                    limit=out_top_k,
                )
                for modality in MODALITIES
            },
            "aggregation": {
                "label": high_level,
                "detail_label": detail,
                "legacy_category": legacy,
                "best_retriever_by_modality": best_retriever_keys,
                "retriever_support_at_k": {
                    "multimodal": mm_support,
                    "text_only": txt_support,
                    "image_only": img_support,
                    "either_unimodal": either_support,
                },
            },
        }

        if "query_id" in base_sample:
            sample_out["query_id"] = base_sample["query_id"]

        if multi_positive:
            positive_paths = base_sample.get("positive_target_image_paths", [])
            sample_out["canonical_target_image_path"] = canonical_target_path(base_sample)
            sample_out["canonical_target_gallery_index"] = canonical_index_by_path[
                sample_out["canonical_target_image_path"]
            ]
            sample_out["num_gt"] = int(base_sample.get("num_gt", len(positive_paths)))
            sample_out["positive_candidate_ids"] = list(base_sample.get("positive_candidate_ids", []))
            sample_out["positive_target_image_paths"] = list(positive_paths)
            sample_out["positive_gallery_indices"] = sorted(
                canonical_index_by_path[p] for p in positive_paths
            )
            sample_out["best_positive_image_paths"] = {
                modality: best_retriever_samples[modality]
                .get("best_positive_image_paths", {})
                .get(modality)
                for modality in MODALITIES
            }
            sample_out["best_positive_gallery_indices"] = {}
            for modality in MODALITIES:
                best_path = sample_out["best_positive_image_paths"][modality]
                sample_out["best_positive_gallery_indices"][modality] = (
                    canonical_index_by_path[best_path] if best_path is not None else None
                )

        samples_out.append(sample_out)

    total = len(samples_out)
    mm_recall = round(
        100.0 * sum(int(s["hits_at_k"]["multimodal"]) for s in samples_out) / total, 2
    ) if total else 0.0
    txt_recall = round(
        100.0 * sum(int(s["hits_at_k"]["text_only"]) for s in samples_out) / total, 2
    ) if total else 0.0
    img_recall = round(
        100.0 * sum(int(s["hits_at_k"]["image_only"]) for s in samples_out) / total, 2
    ) if total else 0.0

    report = {
        "metadata": {
            "dataset": base_meta.get("dataset"),
            "task_id": base_meta.get("task_id"),
            "retriever_module": f"aggregated.{retriever_name}",
            "retriever_class": "AggregatedRetriever",
            "k": out_k,
            "top_k": out_top_k,
            "normalize": base_meta.get("normalize"),
            "multi_positive": multi_positive,
            "mbeir_root": base_meta.get("mbeir_root"),
            "query_instruction": base_meta.get("query_instruction"),
            "text_only_attack_json": base_meta.get("text_only_attack_json"),
            "text_only_attack_name": base_meta.get("text_only_attack_name"),
            "text_only_attack_sig": base_meta.get("text_only_attack_sig"),
            "image_only_attack_json": base_meta.get("image_only_attack_json"),
            "image_only_attack_name": base_meta.get("image_only_attack_name"),
            "image_only_attack_sig": base_meta.get("image_only_attack_sig"),
            "total_queries": total,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "aggregation": {
                "name": "best_rank_mm_recoverable",
                "retrievers": [r.retriever_key for r in reports],
                "source_files": [str(r.path.resolve()) for r in reports],
                "rank_aggregation": {
                    "multimodal": "min",
                    "text_only": "min",
                    "image_only": "min",
                },
                "query_alignment_key": "query_idx",
                "min_mm_success_retrievers": min_mm_success_retrievers,
                "definition": {
                    "composition_required": (
                        "best_text_rank > K and best_image_rank > K and "
                        "retriever_support_at_k.multimodal >= min_mm_success_retrievers"
                    ),
                    "unresolved": (
                        "best_text_rank > K and best_image_rank > K and "
                        "retriever_support_at_k.multimodal < min_mm_success_retrievers"
                    ),
                    "shortcut_solvable": (
                        "best_text_rank <= K or best_image_rank <= K"
                    ),
                },
            },
        },
        "summary": {
            "counts": {key: legacy_counts.get(key, 0) for key in LEGACY_CATEGORY_ORDER},
            "recall_at_k": {
                "multimodal": mm_recall,
                "text_only": txt_recall,
                "image_only": img_recall,
            },
            "aggregate_counts": {
                key: high_level_counts.get(key, 0) for key in HIGH_LEVEL_LABEL_ORDER
            },
            "aggregate_detail_counts": {
                key: detail_counts.get(key, 0) for key in DETAIL_LABEL_ORDER
            },
            "retriever_support_histograms_at_k": {
                "multimodal": dict(sorted(mm_support_hist.items())),
                "text_only": dict(sorted(txt_support_hist.items())),
                "image_only": dict(sorted(img_support_hist.items())),
                "either_unimodal": dict(sorted(either_support_hist.items())),
            },
            "source_retriever_stats": {
                report.retriever_key: retriever_stats(report, out_k) for report in reports
            },
        },
        "gallery_paths": canonical_gallery,
        "samples": samples_out,
    }

    if multi_positive and "qrels_path" in base_meta:
        report["metadata"]["qrels_path"] = base_meta.get("qrels_path")
    if multi_positive and "query_split" in base_meta:
        report["metadata"]["query_split"] = base_meta.get("query_split")

    return report


def default_retriever_name(report_count: int, min_mm_success_retrievers: int) -> str:
    if min_mm_success_retrievers <= 1:
        return "aggregated_best_rank_mm_any"
    if min_mm_success_retrievers >= report_count:
        return "aggregated_best_rank_mm_all"
    return f"aggregated_best_rank_mm_atleast{min_mm_success_retrievers}of{report_count}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("aggregate_retrieval_data")

    if not args.json and not args.results_dir:
        raise SystemExit("Pass either --json ... or --results_dir ...")
    if not args.json and not args.dataset:
        raise SystemExit("--dataset is required when discovering inputs from --results_dir")

    input_paths: List[Path]
    if args.json:
        input_paths = [Path(p).expanduser().resolve() for p in args.json]
    else:
        input_paths = discover_input_files(
            result_dirs=args.results_dir,
            dataset=args.dataset,
            retrievers=set(args.retrievers) if args.retrievers else None,
        )

    if len(input_paths) < 2:
        raise SystemExit(
            f"Need at least 2 input retrieval JSON files, found {len(input_paths)}"
        )

    t0 = time.time()
    try:
        extra_rewrite_rules = [
            _parse_rewrite_rule(spec) for spec in (args.rewrite_prefix or [])
        ]
    except ValueError as exc:
        raise SystemExit(str(exc))

    reports = [
        load_retriever_report(
            path,
            mbeir_root=args.mbeir_root,
            extra_rewrite_rules=extra_rewrite_rules,
        )
        for path in input_paths
    ]
    shared = require_shared_metadata(reports, args.k)

    report_count = len(reports)
    if args.min_mm_success_retrievers < 1 or args.min_mm_success_retrievers > report_count:
        raise SystemExit(
            f"--min_mm_success_retrievers must be in [1, {report_count}]"
        )

    retriever_name = args.retriever_name or default_retriever_name(
        report_count, args.min_mm_success_retrievers
    )
    output_report = build_output(
        reports=reports,
        out_k=shared["k"],
        out_top_k=shared["top_k"],
        min_mm_success_retrievers=args.min_mm_success_retrievers,
        retriever_name=retriever_name,
    )
    output_report["metadata"]["elapsed_seconds"] = round(time.time() - t0, 1)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        dataset = output_report["metadata"]["dataset"]
        task_id = output_report["metadata"]["task_id"]
        output_name = f"retrieval_data_{dataset}_task{task_id}_{retriever_name}.json"
        output_path = input_paths[0].parent / output_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2, ensure_ascii=False)

    total = output_report["metadata"]["total_queries"]
    detail_counts = output_report["summary"]["aggregate_detail_counts"]
    log.info("=" * 60)
    log.info(
        "AGGREGATED RETRIEVAL DATA GENERATED  (K=%d, retrievers=%d)",
        shared["k"],
        report_count,
    )
    log.info("=" * 60)
    log.info("Dataset: %s", output_report["metadata"]["dataset"])
    log.info("Retrievers: %s", ", ".join(r.retriever_key for r in reports))
    log.info("Total valid queries: %d", total)
    for label in DETAIL_LABEL_ORDER:
        count = detail_counts.get(label, 0)
        pct = 100.0 * count / total if total else 0.0
        log.info("  %-22s %5d  (%5.1f%%)", label, count, pct)
    log.info("-" * 60)
    log.info(
        "  Recall@%d  multimodal: %.1f%%  text: %.1f%%  image: %.1f%%",
        shared["k"],
        output_report["summary"]["recall_at_k"]["multimodal"],
        output_report["summary"]["recall_at_k"]["text_only"],
        output_report["summary"]["recall_at_k"]["image_only"],
    )
    log.info("  Output:  %s", output_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
