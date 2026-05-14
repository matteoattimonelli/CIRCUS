#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASETS = ("cirr", "fashioniq", "lasco", "circo")
DEFAULT_SUBSETS = (
    "composition_required",
    "unresolved",
    "all_validated",
    "composition_required_plus_unsolved",
)
DEFAULT_RECALL_CUTOFFS = (1, 5, 10, 50)
MODALITIES = ("multimodal", "image_only", "text_only")
PAIRWISE_COMPARISONS = (
    ("multimodal", "image_only"),
    ("multimodal", "text_only"),
    ("image_only", "text_only"),
)


DATASET_REPORT_DIRS = {
    "cirr": REPO_ROOT / "retrieval" / "retrieval_results",
    "fashioniq": REPO_ROOT / "retrieval" / "retrieval_results",
    "lasco": REPO_ROOT / "retrieval" / "retrieval_results",
    "circo": REPO_ROOT / "retrieval" / "retrieval_results_circo",
}

DATASET_QUERY_JSONL = {
    "cirr": Path("<path_to_M-BEIR>/query/test/mbeir_cirr_task7_test.jsonl"),
    "fashioniq": Path("<path_to_M-BEIR>/query/test/mbeir_fashioniq_task7_test.jsonl"),
    "lasco": Path("<path_to_lasco_mbeir>/query/test/lasco_test_task7_test.jsonl"),
    "circo": Path("<path_to_circo_mbeir>/query/test/circo_test.jsonl"),
}

CIRCO_CAND_POOL_JSONL = Path("<path_to_circo_mbeir>/cand_pool/circo_test_task7.jsonl")
CACHE_ROOT = REPO_ROOT / "subset_evaluation" / "cache_subset_eval_raw"
STUDY_USERS_DIR = REPO_ROOT / "annotations" / "users"

DATASET_AGGREGATED_JSON = {
    "cirr": REPO_ROOT / "shortcut_audit" / "retrieval_data_cirr_task7_aggregated_best_rank_mm_any.json",
    "fashioniq": REPO_ROOT / "shortcut_audit" / "retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json",
    "lasco": REPO_ROOT / "shortcut_audit" / "retrieval_data_lasco_task7_aggregated_best_rank_mm_any.json",
    "circo": REPO_ROOT / "shortcut_audit" / "retrieval_data_circo_task7_aggregated_best_rank_mm_any.json",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval reports on final subsets without rerunning retrievers. "
            "Uses stored per-query ranks from retrieval_data_*.json files."
        )
    )
    ap.add_argument("--dataset", action="append", choices=sorted(DEFAULT_DATASETS))
    ap.add_argument(
        "--subset",
        action="append",
        choices=sorted(DEFAULT_SUBSETS),
        help=(
            "Repeated. Defaults to composition_required, unresolved, "
            "all_validated, composition_required_plus_unsolved."
        ),
    )
    ap.add_argument(
        "--retriever",
        action="append",
        help=(
            "Repeated retriever short name or retriever module. Examples: qwen3vl8b_vllm "
            "or retrievers.qwen3vl8b_vllm_retriever. Defaults to all open retrievers "
            "discovered from retrieval reports."
        ),
    )
    ap.add_argument("--task_id", type=int, default=7)
    ap.add_argument(
        "--final_dataset_dir",
        default=str(REPO_ROOT / "final_dataset"),
        help="Root containing final_dataset/query_indices/*.json.",
    )
    ap.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "final_evaluation" / "results"),
        help="Output directory for CSV/JSONL artifacts.",
    )
    ap.add_argument(
        "--recall_cutoff",
        action="append",
        type=int,
        help="Repeated recall cutoff. Defaults to 1, 5, 10, 50.",
    )
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_retriever_name(raw: str) -> str:
    value = (raw or "").strip()
    if value.startswith("retrievers."):
        value = value[len("retrievers.") :]
    if value.endswith("_retriever"):
        value = value[: -len("_retriever")]
    return value


def discover_open_retrievers(datasets: Sequence[str], task_id: int) -> List[str]:
    found = set()
    for dataset in datasets:
        report_dir = DATASET_REPORT_DIRS[dataset]
        pattern = f"retrieval_data_{dataset}_task{task_id}_*.json"
        for path in report_dir.glob(pattern):
            retriever = path.stem.split(f"retrieval_data_{dataset}_task{task_id}_", 1)[-1]
            if retriever:
                found.add(retriever)
    return sorted(found)


def report_path_for(dataset: str, task_id: int, retriever: str) -> Path:
    report_dir = DATASET_REPORT_DIRS[dataset]
    return report_dir / f"retrieval_data_{dataset}_task{task_id}_{retriever}.json"


def aggregated_subset_indices(dataset: str, subset: str) -> List[int]:
    payload = load_json(DATASET_AGGREGATED_JSON[dataset])
    out: List[int] = []
    for sample in payload.get("samples", []):
        agg = sample.get("aggregation") or {}
        if agg.get("label") == subset:
            out.append(int(sample["query_idx"]))
    return sorted(out)


def validated_indices(dataset: str, final_dataset_dir: Path) -> List[int]:
    out = set()
    query_indices_dir = final_dataset_dir / "query_indices"
    for path in sorted(query_indices_dir.glob(f"{dataset}_*_validated_query_indices.json")):
        payload = load_json(path)
        out.update(int(x) for x in payload.get("query_indices", []))
    return sorted(out)


def study_union_indices(dataset: str) -> List[int]:
    out = set()
    for user_file in sorted(STUDY_USERS_DIR.glob("*.json")):
        payload = load_json(user_file)
        for assignment in payload.get("assignments", []):
            if str(assignment.get("dataset")) != dataset:
                continue
            if str(assignment.get("hidden_category")) not in {"composition_required", "unresolved"}:
                continue
            try:
                out.add(int(assignment["query_idx"]))
            except Exception:
                continue
    return sorted(out)


def subset_indices(dataset: str, subset: str, final_dataset_dir: Path) -> List[int]:
    if subset == "all_validated":
        return validated_indices(dataset, final_dataset_dir)
    if subset == "composition_required_plus_unsolved":
        return study_union_indices(dataset)
    if subset in {"composition_required", "unresolved"}:
        return aggregated_subset_indices(dataset, subset)
    raise ValueError(f"Unsupported subset: {subset}")


def single_query_ndcg(rank0: int) -> float:
    return 1.0 / math.log2(rank0 + 2.0)


def single_query_mrr(rank0: int) -> float:
    return 1.0 / float(rank0 + 1)


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def l2_normalize_rows(tensor):
    import torch

    norms = torch.linalg.norm(tensor, dim=1, keepdim=True).clamp_min(1e-12)
    return tensor / norms


def circo_multi_positive_ndcg(ranks0: Sequence[int]) -> float:
    if not ranks0:
        return 0.0
    sorted_ranks = sorted(int(r) for r in ranks0)
    dcg = sum(1.0 / math.log2(rank0 + 2.0) for rank0 in sorted_ranks)
    idcg = sum(1.0 / math.log2(i + 2.0) for i in range(1, len(sorted_ranks) + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def circo_multi_positive_mrr(ranks0: Sequence[int]) -> float:
    if not ranks0:
        return 0.0
    return 1.0 / float(min(int(r) for r in ranks0) + 1)


def load_torch_tensor(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "shape"):
        return obj.float()
    raise ValueError(f"Unsupported tensor object in {path}: {type(obj).__name__}")


def find_single_path(base_dir: Path, candidates: Sequence[str]) -> Path | None:
    for pattern in candidates:
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_circo_cache_paths(retriever: str) -> Dict[str, Path] | None:
    retriever_dir = CACHE_ROOT / "circo" / retriever
    if not retriever_dir.is_dir():
        return None

    flat_gallery = find_single_path(retriever_dir, ("**/gallery_*.pt",))
    flat_multi = find_single_path(retriever_dir, ("**/queries_*noattack_noatk*.pt",))
    flat_text = find_single_path(retriever_dir, ("**/queries_*img_black*.pt",))
    flat_image = find_single_path(retriever_dir, ("**/queries_*text_drop_all*.pt",))
    if flat_gallery and flat_multi and flat_text and flat_image:
        return {
            "gallery": flat_gallery,
            "multimodal": flat_multi,
            "text_only": flat_text,
            "image_only": flat_image,
        }

    nested_gallery = find_single_path(retriever_dir, ("**/*variant-gallery*/embeddings.pt",))
    nested_multi = find_single_path(retriever_dir, ("**/*variant-queries_multimodal*/embeddings.pt",))
    nested_text = find_single_path(retriever_dir, ("**/*queries_textonly*/*embeddings.pt", "**/*queries_textonly*/embeddings.pt"))
    nested_image = find_single_path(retriever_dir, ("**/*queries_imageonly*/*embeddings.pt", "**/*queries_imageonly*/embeddings.pt"))
    if nested_gallery and nested_multi and nested_text and nested_image:
        return {
            "gallery": nested_gallery,
            "multimodal": nested_multi,
            "text_only": nested_text,
            "image_only": nested_image,
        }

    return None


def load_circo_positive_gallery_indices(gallery_paths: Sequence[str]) -> Dict[int, List[int]]:
    query_rows = load_jsonl(DATASET_QUERY_JSONL["circo"])
    cand_rows = load_jsonl(CIRCO_CAND_POOL_JSONL)

    filename_to_gallery_index = {
        Path(path).name: idx for idx, path in enumerate(gallery_paths)
    }
    cand_to_gallery_index: Dict[str, int] = {}
    for row in cand_rows:
        cand_id = str(row["cand_id"])
        filename = Path(str(row["img_path"])).name
        gallery_index = filename_to_gallery_index.get(filename)
        if gallery_index is not None:
            cand_to_gallery_index[cand_id] = gallery_index

    out: Dict[int, List[int]] = {}
    for query_idx, row in enumerate(query_rows):
        gt_ids = row.get("gt_cand_list") or row.get("pos_cand_list") or []
        out[query_idx] = [
            cand_to_gallery_index[cand_id]
            for cand_id in gt_ids
            if cand_id in cand_to_gallery_index
        ]
    return out


def compute_circo_multi_positive_rows(
    *,
    retriever: str,
    report_payload: dict,
    selected_query_indices: Sequence[int],
) -> tuple[Dict[int, dict], str | None]:
    try:
        import torch
    except ModuleNotFoundError:
        return {}, "torch unavailable in current env"

    cache_paths = discover_circo_cache_paths(retriever)
    if cache_paths is None:
        return {}, "missing circo full-cache tensors"

    query_rows = load_jsonl(DATASET_QUERY_JSONL["circo"])
    expected_queries = len(query_rows)

    gallery = load_torch_tensor(cache_paths["gallery"])
    queries = {
        modality: load_torch_tensor(cache_paths[modality])
        for modality in MODALITIES
    }
    if gallery.ndim != 2:
        return {}, f"unexpected gallery tensor rank: {tuple(gallery.shape)}"
    for modality, tensor in queries.items():
        if tensor.ndim != 2:
            return {}, f"unexpected {modality} tensor rank: {tuple(tensor.shape)}"
        if tensor.shape[0] != expected_queries:
            return {}, (
                f"{modality} query tensor has {tensor.shape[0]} rows, "
                f"expected full CIRCO set with {expected_queries}"
            )

    gallery = l2_normalize_rows(gallery)
    queries = {k: l2_normalize_rows(v) for k, v in queries.items()}
    positive_indices_by_query = load_circo_positive_gallery_indices(report_payload["gallery_paths"])

    selected = sorted(set(int(x) for x in selected_query_indices))
    rows: Dict[int, dict] = {}
    gallery_t = gallery.t()
    batch_size = 32

    for modality in MODALITIES:
        q_all = queries[modality]
        for start in range(0, len(selected), batch_size):
            batch_qidx = selected[start : start + batch_size]
            batch_tensor = q_all[batch_qidx]
            scores = batch_tensor @ gallery_t
            order = torch.argsort(scores, dim=1, descending=True)
            for row_idx, query_idx in enumerate(batch_qidx):
                pos_indices = positive_indices_by_query.get(query_idx, [])
                if not pos_indices:
                    continue
                ranked = order[row_idx].tolist()
                rank_map = {gallery_idx: rank0 for rank0, gallery_idx in enumerate(ranked)}
                pos_ranks = [rank_map[idx] for idx in pos_indices if idx in rank_map]
                if not pos_ranks:
                    continue
                slot = rows.setdefault(query_idx, {})
                slot[f"{modality}_nDCG_full_multi"] = circo_multi_positive_ndcg(pos_ranks)
                slot[f"{modality}_MRR_full_multi"] = circo_multi_positive_mrr(pos_ranks)

    for query_idx, row in rows.items():
        for left, right in PAIRWISE_COMPARISONS:
            if (
                f"{left}_nDCG_full_multi" in row
                and f"{right}_nDCG_full_multi" in row
                and f"{left}_MRR_full_multi" in row
                and f"{right}_MRR_full_multi" in row
            ):
                row[f"delta_nDCG_full_multi_{left}_minus_{right}"] = (
                    float(row[f"{left}_nDCG_full_multi"]) - float(row[f"{right}_nDCG_full_multi"])
                )
                row[f"delta_MRR_full_multi_{left}_minus_{right}"] = (
                    float(row[f"{left}_MRR_full_multi"]) - float(row[f"{right}_MRR_full_multi"])
                )
    return rows, None


def compute_metric_row(
    *,
    dataset: str,
    subset: str,
    retriever: str,
    per_query_rows: Sequence[dict],
    recall_cutoffs: Sequence[int],
    requested_query_count: int,
    source_datasets: Sequence[str] | None = None,
) -> dict:
    row: Dict[str, object] = {
        "dataset": dataset,
        "subset": subset,
        "retriever": retriever,
        "requested_queries": requested_query_count,
        "evaluated_queries": len(per_query_rows),
        "missing_queries": requested_query_count - len(per_query_rows),
    }
    if source_datasets is not None:
        row["source_datasets"] = list(source_datasets)

    for modality in MODALITIES:
        rank0s = [int(q[f"{modality}_rank0"]) for q in per_query_rows]
        ndcgs = [float(q[f"{modality}_ndcg_full"]) for q in per_query_rows]
        mrrs = [float(q[f"{modality}_mrr_full"]) for q in per_query_rows]
        row[f"{modality}_mean_rank"] = safe_mean([r + 1 for r in rank0s])
        row[f"{modality}_median_rank"] = safe_median([r + 1 for r in rank0s])
        row[f"{modality}_nDCG_full"] = 100.0 * safe_mean(ndcgs)
        row[f"{modality}_MRR_full"] = 100.0 * safe_mean(mrrs)
        for cutoff in recall_cutoffs:
            hit_rate = safe_mean([1.0 if r < cutoff else 0.0 for r in rank0s])
            row[f"{modality}_R@{cutoff}"] = 100.0 * hit_rate

    for left, right in PAIRWISE_COMPARISONS:
        rank_deltas = [float(q[f"rank_delta_{left}_minus_{right}"]) for q in per_query_rows]
        ndcg_deltas = [float(q[f"delta_ndcg_full_{left}_minus_{right}"]) for q in per_query_rows]
        mrr_deltas = [float(q[f"delta_mrr_full_{left}_minus_{right}"]) for q in per_query_rows]
        row[f"delta_mean_rank_{left}_minus_{right}"] = safe_mean(rank_deltas)
        row[f"delta_median_rank_{left}_minus_{right}"] = safe_median(rank_deltas)
        row[f"delta_nDCG_full_{left}_minus_{right}"] = 100.0 * safe_mean(ndcg_deltas)
        row[f"delta_MRR_full_{left}_minus_{right}"] = 100.0 * safe_mean(mrr_deltas)

    multi_positive_available = any("multimodal_nDCG_full_multi" in q for q in per_query_rows)
    row["circo_multi_positive_exact"] = bool(dataset == "circo" and multi_positive_available)
    if dataset == "circo" and multi_positive_available:
        for modality in MODALITIES:
            ndcgs = [float(q[f"{modality}_nDCG_full_multi"]) for q in per_query_rows if f"{modality}_nDCG_full_multi" in q]
            mrrs = [float(q[f"{modality}_MRR_full_multi"]) for q in per_query_rows if f"{modality}_MRR_full_multi" in q]
            row[f"{modality}_nDCG_full_multi"] = 100.0 * safe_mean(ndcgs)
            row[f"{modality}_MRR_full_multi"] = 100.0 * safe_mean(mrrs)
        for left, right in PAIRWISE_COMPARISONS:
            ndcg_deltas = [
                float(q[f"delta_nDCG_full_multi_{left}_minus_{right}"])
                for q in per_query_rows
                if f"delta_nDCG_full_multi_{left}_minus_{right}" in q
            ]
            mrr_deltas = [
                float(q[f"delta_MRR_full_multi_{left}_minus_{right}"])
                for q in per_query_rows
                if f"delta_MRR_full_multi_{left}_minus_{right}" in q
            ]
            row[f"delta_nDCG_full_multi_{left}_minus_{right}"] = 100.0 * safe_mean(ndcg_deltas)
            row[f"delta_MRR_full_multi_{left}_minus_{right}"] = 100.0 * safe_mean(mrr_deltas)
    return row


def load_retriever_report(path: Path) -> Dict[int, dict]:
    payload = load_json(path)
    samples = payload.get("samples", [])
    return {int(sample["query_idx"]): sample for sample in samples}


def build_per_query_rows(
    *,
    dataset: str,
    subset: str,
    retriever: str,
    selected_indices: Sequence[int],
    sample_by_query_idx: Dict[int, dict],
    circo_multi_rows_by_query_idx: Dict[int, dict] | None = None,
) -> tuple[List[dict], List[int]]:
    rows: List[dict] = []
    missing: List[int] = []
    for query_idx in selected_indices:
        sample = sample_by_query_idx.get(int(query_idx))
        if sample is None:
            missing.append(int(query_idx))
            continue

        ranks = sample.get("ranks") or {}
        if any(ranks.get(modality) is None for modality in MODALITIES):
            missing.append(int(query_idx))
            continue

        base = {
            "dataset": dataset,
            "subset": subset,
            "retriever": retriever,
            "query_idx": int(query_idx),
            "query_text": sample.get("query_text"),
            "category": sample.get("category"),
            "target_gallery_index": sample.get("target_gallery_index"),
            "target_image_path": sample.get("target_image_path"),
        }

        for modality in MODALITIES:
            rank0 = int(ranks[modality])
            base[f"{modality}_rank0"] = rank0
            base[f"{modality}_rank1"] = rank0 + 1
            base[f"{modality}_ndcg_full"] = single_query_ndcg(rank0)
            base[f"{modality}_mrr_full"] = single_query_mrr(rank0)

        for left, right in PAIRWISE_COMPARISONS:
            left_rank = int(base[f"{left}_rank1"])
            right_rank = int(base[f"{right}_rank1"])
            base[f"rank_delta_{left}_minus_{right}"] = left_rank - right_rank
            base[f"delta_ndcg_full_{left}_minus_{right}"] = (
                float(base[f"{left}_ndcg_full"]) - float(base[f"{right}_ndcg_full"])
            )
            base[f"delta_mrr_full_{left}_minus_{right}"] = (
                float(base[f"{left}_mrr_full"]) - float(base[f"{right}_mrr_full"])
            )
        if circo_multi_rows_by_query_idx and int(query_idx) in circo_multi_rows_by_query_idx:
            base.update(circo_multi_rows_by_query_idx[int(query_idx)])
        rows.append(base)
    return rows, missing


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_overall_rows(
    *,
    all_per_query_rows: Sequence[dict],
    all_requested_counts: Dict[tuple[str, str], Dict[str, int]],
    recall_cutoffs: Sequence[int],
) -> List[dict]:
    grouped: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    dataset_names: Dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in all_per_query_rows:
        key = (str(row["retriever"]), str(row["subset"]))
        grouped[key].append(row)
        dataset_names[key].add(str(row["dataset"]))

    out: List[dict] = []
    for (retriever, subset), rows in sorted(grouped.items()):
        requested = sum(all_requested_counts.get((retriever, subset), {}).values())
        overall = compute_metric_row(
            dataset="all_selected_datasets",
            subset=subset,
            retriever=retriever,
            per_query_rows=rows,
            recall_cutoffs=recall_cutoffs,
            requested_query_count=requested,
            source_datasets=sorted(dataset_names[(retriever, subset)]),
        )
        out.append(overall)
    return out


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    datasets = args.dataset or list(DEFAULT_DATASETS)
    subsets = args.subset or list(DEFAULT_SUBSETS)
    recall_cutoffs = sorted(set(args.recall_cutoff or DEFAULT_RECALL_CUTOFFS))
    output_dir = Path(args.output_dir).resolve()
    final_dataset_dir = Path(args.final_dataset_dir).resolve()
    ensure_dir(output_dir)

    retrievers = (
        sorted({normalize_retriever_name(p) for p in args.retriever})
        if args.retriever
        else discover_open_retrievers(datasets, args.task_id)
    )
    logging.info("Datasets: %s", ", ".join(datasets))
    logging.info("Subsets: %s", ", ".join(subsets))
    logging.info("Retrievers: %s", ", ".join(retrievers))
    logging.info("Recall cutoffs: %s", ", ".join(str(k) for k in recall_cutoffs))

    summary_rows: List[dict] = []
    all_per_query_rows: List[dict] = []
    all_requested_counts: Dict[tuple[str, str], Dict[str, int]] = defaultdict(dict)
    per_query_dir = output_dir / "per_query"
    ensure_dir(per_query_dir)

    for dataset in datasets:
        for retriever in retrievers:
            report_path = report_path_for(dataset, args.task_id, retriever)
            if not report_path.is_file():
                logging.warning("Skip dataset=%s retriever=%s: missing report %s", dataset, retriever, report_path)
                continue

            report_payload = load_json(report_path)
            sample_by_query_idx = {
                int(sample["query_idx"]): sample for sample in report_payload.get("samples", [])
            }
            logging.info(
                "Loaded report dataset=%s retriever=%s with %d samples",
                dataset,
                retriever,
                len(sample_by_query_idx),
            )

            circo_multi_rows_by_query_idx: Dict[int, dict] = {}
            circo_multi_warning: str | None = None
            if dataset == "circo":
                selected_union = sorted(
                    {
                        idx
                        for subset in subsets
                        for idx in subset_indices(dataset, subset, final_dataset_dir)
                    }
                )
                circo_multi_rows_by_query_idx, circo_multi_warning = compute_circo_multi_positive_rows(
                    retriever=retriever,
                    report_payload=report_payload,
                    selected_query_indices=selected_union,
                )
                if circo_multi_warning:
                    logging.warning(
                        "dataset=circo retriever=%s exact multi-positive unavailable: %s",
                        retriever,
                        circo_multi_warning,
                    )
                else:
                    logging.info(
                        "dataset=circo retriever=%s exact multi-positive rows=%d",
                        retriever,
                        len(circo_multi_rows_by_query_idx),
                    )

            for subset in subsets:
                selected_indices = subset_indices(dataset, subset, final_dataset_dir)
                per_query_rows, missing = build_per_query_rows(
                    dataset=dataset,
                    subset=subset,
                    retriever=retriever,
                    selected_indices=selected_indices,
                    sample_by_query_idx=sample_by_query_idx,
                    circo_multi_rows_by_query_idx=circo_multi_rows_by_query_idx,
                )
                logging.info(
                    "dataset=%s retriever=%s subset=%s selected=%d evaluated=%d missing=%d",
                    dataset,
                    retriever,
                    subset,
                    len(selected_indices),
                    len(per_query_rows),
                    len(missing),
                )

                all_requested_counts[(retriever, subset)][dataset] = len(selected_indices)
                all_per_query_rows.extend(per_query_rows)

                summary_row = compute_metric_row(
                    dataset=dataset,
                    subset=subset,
                    retriever=retriever,
                    per_query_rows=per_query_rows,
                    recall_cutoffs=recall_cutoffs,
                    requested_query_count=len(selected_indices),
                )
                summary_rows.append(summary_row)

                subset_slug = f"{dataset}__{retriever}__{subset}"
                write_jsonl(per_query_dir / f"{subset_slug}.jsonl", per_query_rows)
                write_json(
                    per_query_dir / f"{subset_slug}.manifest.json",
                    {
                        "dataset": dataset,
                        "retriever": retriever,
                        "subset": subset,
                        "report_path": str(report_path),
                        "circo_multi_positive_exact_available": bool(circo_multi_rows_by_query_idx),
                        "circo_multi_positive_warning": circo_multi_warning,
                        "requested_queries": len(selected_indices),
                        "evaluated_queries": len(per_query_rows),
                        "missing_queries": len(missing),
                        "missing_query_indices": missing,
                    },
                )

    overall_rows = group_overall_rows(
        all_per_query_rows=all_per_query_rows,
        all_requested_counts=all_requested_counts,
        recall_cutoffs=recall_cutoffs,
    )

    write_jsonl(output_dir / "summary_rows.jsonl", summary_rows)
    write_csv(output_dir / "summary_rows.csv", summary_rows)
    write_jsonl(output_dir / "overall_rows.jsonl", overall_rows)
    write_csv(output_dir / "overall_rows.csv", overall_rows)
    write_json(
        output_dir / "run_manifest.json",
        {
            "datasets": datasets,
            "subsets": subsets,
            "retrievers": retrievers,
            "task_id": args.task_id,
            "recall_cutoffs": recall_cutoffs,
            "output_dir": str(output_dir),
            "summary_rows": len(summary_rows),
            "overall_rows": len(overall_rows),
            "per_query_rows": len(all_per_query_rows),
        },
    )
    logging.info("Wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()
