#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append one structured metrics row to a JSONL file.")
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task_id", type=int, default=None)
    ap.add_argument("--subset_name", required=True)
    ap.add_argument("--retriever_module", required=True)
    ap.add_argument("--retriever_class", default="Retriever")
    ap.add_argument("--ablation_json", required=True)
    ap.add_argument("--ablation_tag", required=True)
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--query_source", required=True)
    ap.add_argument("--eval_script", required=True)
    ap.add_argument("--log_path", required=True)
    ap.add_argument("--raw_cache_dir", required=True)
    ap.add_argument("--gallery_export_dir", default=None)
    ap.add_argument("--query_export_dir", default=None)
    ap.add_argument("--variant_tag", required=True)
    ap.add_argument("--query_mode", default=None)
    ap.add_argument("--query_txt_part", default=None)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--multi_positive", action="store_true")
    ap.add_argument("--export_enabled", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    metrics_path = Path(args.metrics_json).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    row = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "task_id": args.task_id,
        "subset_name": args.subset_name,
        "retriever_module": args.retriever_module,
        "retriever_class": args.retriever_class,
        "ablation_json": args.ablation_json,
        "ablation_tag": args.ablation_tag,
        "query_file": args.query_file,
        "query_source": args.query_source,
        "eval_script": args.eval_script,
        "log_path": args.log_path,
        "raw_cache_dir": args.raw_cache_dir,
        "gallery_export_dir": args.gallery_export_dir,
        "query_export_dir": args.query_export_dir,
        "variant_tag": args.variant_tag,
        "query_mode": args.query_mode,
        "query_txt_part": args.query_txt_part,
        "normalize": args.normalize,
        "multi_positive": args.multi_positive,
        "export_enabled": args.export_enabled,
        "metrics_json": str(metrics_path),
    }
    row.update(metrics)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
