#!/usr/bin/env python3
"""Convert CIRCO into the M-BEIR-style layout expected by this release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CIRCO into the layout expected by the M-BEIR-style evaluator in this release.",
    )
    parser.add_argument("--circo_root", type=Path, required=True, help="Path to the CIRCO root directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output M-BEIR-style directory.")
    parser.add_argument(
        "--query_filename",
        type=str,
        default="mbeir_circo_task7_test.jsonl",
        help="Filename to write under query/test/.",
    )
    parser.add_argument(
        "--cand_filename",
        type=str,
        default="circo_task7.jsonl",
        help="Filename to write under cand_pool/.",
    )
    parser.add_argument(
        "--emit_hidden_test",
        action="store_true",
        help="Also export the CIRCO test split as query/test/circo_hidden_test.jsonl without labels.",
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Store paths relative to output_dir instead of absolute paths.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def validate_expected_files(circo_root: Path) -> Dict[str, Path]:
    circo_root = circo_root.resolve()

    val_candidates = [
        circo_root / "annotations" / "val.json",
        circo_root / "COCO2017_unlabeled" / "annotations" / "val.json",
    ]
    test_candidates = [
        circo_root / "annotations" / "test.json",
        circo_root / "COCO2017_unlabeled" / "annotations" / "test.json",
    ]
    image_info_candidates = [
        circo_root / "COCO2017_unlabeled" / "annotations" / "image_info_unlabeled2017.json",
        circo_root / "annotations" / "image_info_unlabeled2017.json",
    ]
    images_root_candidates = [
        circo_root / "COCO2017_unlabeled" / "unlabeled2017",
        circo_root / "unlabeled2017",
    ]

    val_ann = first_existing(val_candidates)
    test_ann = first_existing(test_candidates)
    image_info = first_existing(image_info_candidates)
    images_root = first_existing(images_root_candidates)

    missing = []
    if val_ann is None:
        missing.append("val.json not found")
    if image_info is None:
        missing.append("image_info_unlabeled2017.json not found")
    if images_root is None:
        missing.append("unlabeled2017/ not found")

    if missing:
        raise FileNotFoundError("Missing required CIRCO files/directories:\n- " + "\n- ".join(missing))

    return {
        "val_ann": val_ann,
        "test_ann": test_ann if test_ann is not None else test_candidates[0],
        "image_info": image_info,
        "images_root": images_root,
    }


def coco_image_filename_from_id(image_id: int) -> str:
    return f"{int(image_id):012d}.jpg"


def normalize_path(path: Path, output_dir: Path, relative_paths: bool) -> str:
    return os.path.relpath(path, output_dir) if relative_paths else str(path.resolve())


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_candidate_rows(
    image_info: dict,
    images_root: Path,
    output_dir: Path,
    relative_paths: bool,
) -> List[dict]:
    rows = []
    for img in image_info.get("images", []):
        image_id = int(img["id"])
        img_path = images_root / coco_image_filename_from_id(image_id)
        rows.append(
            {
                "cand_id": f"circo_img_{image_id}",
                "img_path": normalize_path(img_path, output_dir, relative_paths),
            }
        )
    return rows


def extract_query_id(entry: dict, fallback_idx: int, split_name: str) -> str:
    if "id" in entry:
        return f"circo_q_{split_name}_{entry['id']}"
    return f"circo_q_{split_name}_{fallback_idx}"


def build_query_rows_val(
    annotations: List[dict],
    images_root: Path,
    output_dir: Path,
    relative_paths: bool,
) -> List[dict]:
    rows = []
    for idx, entry in enumerate(annotations):
        qid = extract_query_id(entry, idx, "test")
        ref_img_id = int(entry["reference_img_id"])
        target_img_id = int(entry["target_img_id"])
        ref_path = images_root / coco_image_filename_from_id(ref_img_id)

        rows.append(
            {
                "query_id": qid,
                "query_text": entry["relative_caption"],
                "query_img_path": normalize_path(ref_path, output_dir, relative_paths),
                # the evaluator reads pos_cand_list[0] as the canonical target
                "pos_cand_list": [f"circo_img_{target_img_id}"],
                "gt_cand_list": [f"circo_img_{int(x)}" for x in entry.get("gt_img_ids", [])],
            }
        )
    return rows


def build_query_rows_hidden_test(
    annotations: List[dict],
    images_root: Path,
    output_dir: Path,
    relative_paths: bool,
) -> List[dict]:
    rows = []
    for idx, entry in enumerate(annotations):
        qid = extract_query_id(entry, idx, "hidden_test")
        ref_img_id = int(entry["reference_img_id"])
        ref_path = images_root / coco_image_filename_from_id(ref_img_id)

        rows.append(
            {
                "query_id": qid,
                "query_text": entry["relative_caption"],
                "query_img_path": normalize_path(ref_path, output_dir, relative_paths),
                "pos_cand_list": [],
            }
        )
    return rows


def build_qrels(annotations: List[dict]) -> List[Tuple[str, str, int]]:
    rows = []
    for idx, entry in enumerate(annotations):
        if "gt_img_ids" not in entry:
            continue
        qid = extract_query_id(entry, idx, "test")
        seen = set()
        for gt in entry["gt_img_ids"]:
            gt = int(gt)
            if gt in seen:
                continue
            seen.add(gt)
            rows.append((qid, f"circo_img_{gt}", 1))
    return rows


def write_qrels_tsv(path: Path, rows: List[Tuple[str, str, int]]) -> int:
    with path.open("w", encoding="utf-8") as f:
        f.write("query_id\tcandidate_id\tscore\n")
        for qid, cid, score in rows:
            f.write(f"{qid}\t{cid}\t{score}\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    paths = validate_expected_files(args.circo_root)

    output_dir = args.output_dir.resolve()
    query_test_dir = output_dir / "query" / "test"
    cand_pool_dir = output_dir / "cand_pool"
    qrels_dir = output_dir / "qrels"
    metadata_dir = output_dir / "metadata"

    for d in [query_test_dir, cand_pool_dir, qrels_dir, metadata_dir]:
        ensure_dir(d)

    val_annotations = load_json(paths["val_ann"])
    if not isinstance(val_annotations, list):
        raise ValueError(f"Expected list in {paths['val_ann']}")

    image_info = load_json(paths["image_info"])
    if "images" not in image_info:
        raise ValueError(f"Expected 'images' key in {paths['image_info']}")

    cand_rows = build_candidate_rows(
        image_info=image_info,
        images_root=paths["images_root"],
        output_dir=output_dir,
        relative_paths=args.relative_paths,
    )
    num_cands = write_jsonl(cand_pool_dir / args.cand_filename, cand_rows)

    query_rows = build_query_rows_val(
        annotations=val_annotations,
        images_root=paths["images_root"],
        output_dir=output_dir,
        relative_paths=args.relative_paths,
    )
    num_queries = write_jsonl(query_test_dir / args.query_filename, query_rows)

    qrels_rows = build_qrels(val_annotations)
    num_qrels = write_qrels_tsv(qrels_dir / "test.tsv", qrels_rows)

    hidden_name = "circo_hidden_test.jsonl"
    num_hidden = 0
    if args.emit_hidden_test and paths["test_ann"].exists():
        test_annotations = load_json(paths["test_ann"])
        if not isinstance(test_annotations, list):
            raise ValueError(f"Expected list in {paths['test_ann']}")
        hidden_rows = build_query_rows_hidden_test(
            annotations=test_annotations,
            images_root=paths["images_root"],
            output_dir=output_dir,
            relative_paths=args.relative_paths,
        )
        num_hidden = write_jsonl(query_test_dir / hidden_name, hidden_rows)

    dataset_info = {
        "dataset": "CIRCO",
        "task_id": 7,
        "format": "M-BEIR-style for the evaluator in this release",
        "query_file": str(query_test_dir / args.query_filename),
        "cand_pool_file": str(cand_pool_dir / args.cand_filename),
        "qrels_file": str(qrels_dir / "test.tsv"),
        "counts": {
            "num_candidates": num_cands,
            "num_queries": num_queries,
            "num_qrels": num_qrels,
            "num_hidden_test_queries": num_hidden,
        },
        "paths_are_relative_to_output_dir": bool(args.relative_paths),
    }

    with (metadata_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Wrote candidate pool: {cand_pool_dir / args.cand_filename}")
    print(f"Wrote queries:        {query_test_dir / args.query_filename}")
    print(f"Wrote qrels:          {qrels_dir / 'test.tsv'}")
    if num_hidden > 0:
        print(f"Wrote hidden test:    {query_test_dir / hidden_name}")
    print(f"Wrote metadata:       {metadata_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
