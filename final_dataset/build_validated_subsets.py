#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(__file__).resolve().parent
USERS_DIR = REPO_ROOT / "annotations" / "users"
AGG_DIR = REPO_ROOT / "shortcut_audit"


@dataclass(frozen=True)
class DatasetConfig:
    aggregated_json: Path
    source_query_jsonl: Path
    eval_root: Path
    export_name: str


DATASETS: Dict[str, DatasetConfig] = {
    "cirr": DatasetConfig(
        aggregated_json=AGG_DIR / "retrieval_data_cirr_task7_aggregated_best_rank_mm_any.json",
        source_query_jsonl=Path("<path_to_M-BEIR>/query/test/mbeir_cirr_task7_test.jsonl"),
        eval_root=Path("<path_to_M-BEIR>"),
        export_name="mbeir_cirr_task7_test",
    ),
    "circo": DatasetConfig(
        aggregated_json=AGG_DIR / "retrieval_data_circo_task7_aggregated_best_rank_mm_any.json",
        source_query_jsonl=Path("<path_to_circo_mbeir>/query/test/circo_test.jsonl"),
        eval_root=Path("<path_to_circo_mbeir>"),
        export_name="circo_test",
    ),
    "fashioniq": DatasetConfig(
        aggregated_json=AGG_DIR / "retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json",
        source_query_jsonl=Path("<path_to_M-BEIR>/query/test/mbeir_fashioniq_task7_test.jsonl"),
        eval_root=Path("<path_to_M-BEIR>"),
        export_name="mbeir_fashioniq_task7_test",
    ),
    "lasco": DatasetConfig(
        aggregated_json=AGG_DIR / "retrieval_data_lasco_task7_aggregated_best_rank_mm_any.json",
        source_query_jsonl=Path("<path_to_lasco_mbeir>/query/test/lasco_test_task7_test.jsonl"),
        eval_root=Path("<path_to_lasco_mbeir>"),
        export_name="lasco_test_task7_test",
    ),
}

REASON_FIELDS = (
    "INVALID_TEXT_QUERY",
    "INVALID_IMAGE_QUERY",
    "INVALID_TARGET_IMAGE",
    "QUERY_TOO_BROAD",
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_lines(path: Path) -> List[str]:
    rows: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                rows.append(line)
    return rows


def percent(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def grouped_reason_counts(entries: Iterable[dict]) -> Counter:
    counter: Counter = Counter()
    for entry in entries:
        if entry.get("validated") is not False:
            continue
        for reason in REASON_FIELDS:
            if entry.get(reason):
                counter[reason] += 1
    return counter


def collect_unique_annotations() -> Dict[Tuple[str, int, str, str], List[dict]]:
    by_key: Dict[Tuple[str, int, str, str], List[dict]] = defaultdict(list)
    for user_file in sorted(USERS_DIR.glob("*.json")):
        payload = load_json(user_file)
        user = user_file.stem
        for assignment in payload.get("assignments", []):
            annotation = assignment.get("annotation") or {}
            key = (
                str(assignment.get("dataset")),
                int(assignment.get("query_idx")),
                str(assignment.get("source_file_key")),
                str(assignment.get("hidden_category")),
            )
            by_key[key].append(
                {
                    "user": user,
                    "dataset": assignment.get("dataset"),
                    "query_idx": int(assignment.get("query_idx")),
                    "source_file_key": assignment.get("source_file_key"),
                    "source_path": assignment.get("source_path"),
                    "hidden_category": assignment.get("hidden_category"),
                    "selection_mode": assignment.get("selection_mode"),
                    "validated": annotation.get("validated"),
                    "other": annotation.get("other") or "",
                    **{reason: bool(annotation.get(reason)) for reason in REASON_FIELDS},
                }
            )
    return by_key


def dedupe_rows(
    by_key: Dict[Tuple[str, int, str, str], List[dict]]
) -> Dict[Tuple[str, int, str, str], dict]:
    kept: Dict[Tuple[str, int, str, str], dict] = {}
    for key, rows in by_key.items():
        users = {row["user"] for row in rows}
        if users == {"annotator_03", "annotator_08"} and len(rows) == 2:
            picked = next(row for row in rows if row["user"] == "annotator_03")
        else:
            labeled_rows = [row for row in rows if row["validated"] is not None]
            picked = labeled_rows[0] if labeled_rows else rows[0]
        kept[key] = picked
    return kept


def load_aggregated_samples() -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {}
    for dataset, cfg in DATASETS.items():
        data = load_json(cfg.aggregated_json)
        samples = {int(sample["query_idx"]): sample for sample in data.get("samples", [])}
        out[dataset] = samples
    return out


def build_trace_record(annotation: dict, sample: dict) -> dict:
    agg = sample.get("aggregation", {})
    return {
        "dataset": annotation["dataset"],
        "hidden_category": annotation["hidden_category"],
        "query_idx": annotation["query_idx"],
        "validated": annotation["validated"],
        "annotator_kept": annotation["user"],
        "reasons": {
            reason: bool(annotation.get(reason))
            for reason in REASON_FIELDS
        },
        "other": annotation.get("other") or "",
        "query_text": sample.get("query_text"),
        "full_text": sample.get("full_text"),
        "query_image_path": sample.get("query_image_path"),
        "target_image_path": sample.get("target_image_path"),
        "aggregation": {
            "label": agg.get("label"),
            "detail_label": agg.get("detail_label"),
            "legacy_category": agg.get("legacy_category"),
            "retriever_support_at_k": agg.get("retriever_support_at_k", {}),
            "best_retriever_by_modality": agg.get("best_retriever_by_modality", {}),
        },
        "ranks": sample.get("ranks", {}),
        "hits_at_k": sample.get("hits_at_k", {}),
    }


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_query_lines(path: Path, lines: Iterable[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def build_markdown(
    overall: Counter,
    per_split: Dict[Tuple[str, str], Counter],
    reason_counts: Counter,
) -> str:
    lines: List[str] = []
    lines.append("# Final Dataset From Multimodal Validity Study")
    lines.append("")
    lines.append("Deduplication policy:")
    lines.append("- `annotator_03` and `annotator_08` labeled the same 593 examples.")
    lines.append("- `annotator_03` was kept and `annotator_08` was dropped for that duplicate block.")
    lines.append("- Remaining examples are counted at the unique-example level.")
    lines.append("")
    lines.append("Overall:")
    lines.append(f"- Unique examples: `{overall['unique_examples']}`")
    lines.append(f"- Labeled examples: `{overall['labeled_examples']}`")
    lines.append(f"- `validated = true`: `{overall['validated_true_examples']}` ({percent(overall['validated_true_examples'], overall['labeled_examples']):.1f}%)")
    lines.append(f"- `validated = false`: `{overall['validated_false_examples']}` ({percent(overall['validated_false_examples'], overall['labeled_examples']):.1f}%)")
    lines.append(f"- Unlabeled examples: `{overall['unlabeled_examples']}`")
    lines.append("")
    lines.append("## Per Split")
    lines.append("")
    lines.append("| Dataset | Split | Unique | Labeled | Correct | Correct % | Incorrect | Unlabeled |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for (dataset, split), stats in sorted(per_split.items()):
        correct = stats["validated_true_examples"]
        labeled = stats["labeled_examples"]
        lines.append(
            f"| {dataset} | {split} | {stats['unique_examples']} | {labeled} | {correct} | {percent(correct, labeled):.1f}% | {stats['validated_false_examples']} | {stats['unlabeled_examples']} |"
        )
    lines.append("")
    lines.append("## Failure Reasons")
    lines.append("")
    lines.append("Reason counts below are not mutually exclusive. One incorrect example can activate more than one reason.")
    lines.append("")
    for reason in REASON_FIELDS:
        lines.append(f"- `{reason}`: `{reason_counts[reason]}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `CIRR` is the cleanest of the larger benchmarks here.")
    lines.append("- `composition_required` is consistently cleaner than `unresolved`, but still far from noise-free.")
    lines.append("- The dominant failure mode is `QUERY_TOO_BROAD`, not malformed image or text input.")
    lines.append("- For downstream evaluation, the safest subsets are the exported `validated=true` query files.")
    lines.append("")
    lines.append("## Exported Files")
    lines.append("")
    lines.append("- Query-index allowlists: `final_dataset/query_indices/*.json`")
    lines.append("- Repo copies of filtered JSONLs: `final_dataset/query_jsonl/<dataset>/*.jsonl`")
    lines.append("- Direct eval-ready copies: `<dataset_root>/query/test/final_dataset/*.jsonl`")
    lines.append("- Trace JSONL sidecars: `*.trace.jsonl` beside each exported query subset")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    by_key = collect_unique_annotations()
    kept = dedupe_rows(by_key)
    aggregated_samples = load_aggregated_samples()

    overall: Counter = Counter()
    per_split: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    false_entries: List[dict] = []

    correct_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    trace_records_by_split: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    for (dataset, query_idx, _source_file_key, hidden_category), annotation in sorted(kept.items()):
        stats = per_split[(dataset, hidden_category)]
        overall["unique_examples"] += 1
        stats["unique_examples"] += 1

        value = annotation["validated"]
        sample = aggregated_samples[dataset][query_idx]
        trace_records_by_split[(dataset, hidden_category)].append(build_trace_record(annotation, sample))

        if value is None:
            overall["unlabeled_examples"] += 1
            stats["unlabeled_examples"] += 1
            continue

        overall["labeled_examples"] += 1
        stats["labeled_examples"] += 1

        if value is True:
            overall["validated_true_examples"] += 1
            stats["validated_true_examples"] += 1
            correct_indices[(dataset, hidden_category)].append(query_idx)
        else:
            overall["validated_false_examples"] += 1
            stats["validated_false_examples"] += 1
            false_entries.append(annotation)

    reason_counts = grouped_reason_counts(false_entries)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "query_indices").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "query_jsonl").mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(overall=overall, per_split=per_split, reason_counts=reason_counts)
    report_path = OUTPUT_ROOT / "annotation_study.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    summary_manifest = {
        "deduplication_policy": {
            "duplicate_users": ["annotator_03", "annotator_08"],
            "kept_user": "annotator_03",
            "dropped_user": "annotator_08",
            "duplicate_example_count": 593,
        },
        "overall": dict(overall),
        "per_dataset_category": {
            f"{dataset}::{category}": dict(stats)
            for (dataset, category), stats in sorted(per_split.items())
        },
        "reason_counts_false_examples": dict(reason_counts),
        "report_markdown": str(report_path.resolve()),
    }
    write_json(OUTPUT_ROOT / "summary.json", summary_manifest)

    for (dataset, category), query_indices in sorted(correct_indices.items()):
        cfg = DATASETS[dataset]
        source_lines = load_jsonl_lines(cfg.source_query_jsonl)
        query_indices = sorted(query_indices)

        if query_indices and max(query_indices) >= len(source_lines):
            raise SystemExit(
                f"{dataset}/{category}: query_idx out of range for {cfg.source_query_jsonl}"
            )

        allowlist_payload = {
            "dataset": dataset,
            "hidden_category": category,
            "selection_field": "annotation.validated",
            "selection_value": True,
            "deduplication_policy": "keep_first_drop_second_duplicate_block",
            "query_indices": query_indices,
        }
        allowlist_path = OUTPUT_ROOT / "query_indices" / f"{dataset}_{category}_validated_query_indices.json"
        write_json(allowlist_path, allowlist_payload)

        repo_dataset_dir = OUTPUT_ROOT / "query_jsonl" / dataset
        repo_jsonl_path = repo_dataset_dir / f"{cfg.export_name}_validated_{category}.jsonl"
        repo_trace_path = repo_jsonl_path.with_suffix(".trace.jsonl")
        repo_manifest_path = repo_jsonl_path.with_suffix(".manifest.json")

        selected_lines = [source_lines[idx] for idx in query_indices]
        selected_traces = [
            trace
            for trace in trace_records_by_split[(dataset, category)]
            if trace["validated"] is True
        ]
        selected_traces.sort(key=lambda row: int(row["query_idx"]))

        write_query_lines(repo_jsonl_path, selected_lines)
        write_jsonl(repo_trace_path, selected_traces)
        write_json(
            repo_manifest_path,
            {
                "dataset": dataset,
                "hidden_category": category,
                "selected_queries": len(query_indices),
                "source_query_jsonl": str(cfg.source_query_jsonl),
                "repo_query_jsonl": str(repo_jsonl_path),
                "query_index_allowlist": str(allowlist_path),
            },
        )

        eval_dir = cfg.eval_root / "query" / "test" / "final_dataset"
        eval_jsonl_path = eval_dir / repo_jsonl_path.name
        eval_trace_path = eval_jsonl_path.with_suffix(".trace.jsonl")
        eval_manifest_path = eval_jsonl_path.with_suffix(".manifest.json")
        write_query_lines(eval_jsonl_path, selected_lines)
        write_jsonl(eval_trace_path, selected_traces)
        write_json(
            eval_manifest_path,
            {
                "dataset": dataset,
                "hidden_category": category,
                "selected_queries": len(query_indices),
                "source_query_jsonl": str(cfg.source_query_jsonl),
                "eval_query_jsonl": str(eval_jsonl_path),
                "query_file_argument": str(Path("final_dataset") / eval_jsonl_path.name),
                "query_index_allowlist": str(allowlist_path),
            },
        )


if __name__ == "__main__":
    main()
