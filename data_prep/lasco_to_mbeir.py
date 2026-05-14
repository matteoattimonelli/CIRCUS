#!/usr/bin/env python3
"""Convert LaSCo into the M-BEIR-style layout expected by this release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LaSCo train+val splits into the M-BEIR-style layout used by the evaluator in this release."
    )
    parser.add_argument(
        "--lasco_root",
        type=Path,
        required=True,
        help="Path to the LaSCo root, containing lasco_{train,val}.json, lasco_{train,val}_corpus.json, train2014/, val2014/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the M-BEIR-style export.",
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Store paths relative to output_dir instead of absolute paths.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_qrels_tsv(path: Path, rows: List[Tuple[str, str, int]]) -> int:
    with path.open("w", encoding="utf-8") as f:
        f.write("query_id\tcandidate_id\tscore\n")
        for qid, cid, score in rows:
            f.write(f"{qid}\t{cid}\t{score}\n")
    return len(rows)


def normalize_path(path: Path, output_dir: Path, relative_paths: bool) -> str:
    if relative_paths:
        return os.path.relpath(path, output_dir)
    return str(path.resolve())


def resolve_required_paths(lasco_root: Path) -> Dict[str, Path]:
    lasco_root = lasco_root.expanduser().resolve()

    paths = {
        "train_queries": lasco_root / "lasco_train.json",
        "train_corpus": lasco_root / "lasco_train_corpus.json",
        "val_queries": lasco_root / "lasco_val.json",
        "val_corpus": lasco_root / "lasco_val_corpus.json",
        "train_images_root": lasco_root / "train2014",
        "val_images_root": lasco_root / "val2014",
    }

    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required LaSCo files/directories:\n" + "\n".join(str(p) for p in missing)
        )

    return paths


def pick_first_present(rec: dict, keys: List[str]):
    for k in keys:
        if k in rec:
            return rec[k]
    return None


def extract_records(obj: Any, kind: str, source_path: Path) -> List[Any]:
    """Tolerate top-level list, dict-with-list, or direct id->image mapping."""
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        candidate_keys = [
            "data",
            "items",
            "images",
            "corpus",
            "queries",
            "annotations",
            "samples",
            "entries",
        ]
        for k in candidate_keys:
            if k in obj and isinstance(obj[k], list):
                return obj[k]

        # Direct mapping case (e.g. lasco_train_corpus.json):
        # {"158307": "train2014/COCO_train2014_000000158307.jpg", ...}
        return [{"id": k, "value": v} for k, v in obj.items()]

    raise ValueError(
        f"Could not extract a list of {kind} records from {source_path}. "
        f"Top-level type={type(obj).__name__}"
    )


def extract_image_path(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and isinstance(value[1], str):
            return value[1]
        for x in value:
            if isinstance(x, str) and ("/" in x or x.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))):
                return x
        for x in value:
            if isinstance(x, str):
                return x
        return None

    if isinstance(value, dict):
        for k in ["path", "img_path", "image", "image_path", "file_name", "filename"]:
            if k in value and isinstance(value[k], str):
                return value[k]
        return None

    return str(value)


def canonical_cand_id(image_name_or_id: Any) -> str:
    if isinstance(image_name_or_id, (list, tuple)):
        if len(image_name_or_id) >= 1:
            first = image_name_or_id[0]
            if isinstance(first, (int, str)) and str(first).isdigit():
                return f"lasco_img_{first}"
        path = extract_image_path(image_name_or_id)
        if path is not None:
            return f"lasco_img_{Path(path).stem}"

    if isinstance(image_name_or_id, dict):
        path = extract_image_path(image_name_or_id)
        if path is not None:
            return f"lasco_img_{Path(path).stem}"

    s = str(image_name_or_id)
    if s.isdigit():
        return f"lasco_img_{s}"
    return f"lasco_img_{Path(s).stem}"


def find_existing_image(image_value: Any, images_root: Path) -> Path:
    image_name = extract_image_path(image_value)
    if image_name is None:
        raise FileNotFoundError(f"Could not extract image path from: {image_value}")

    p = Path(image_name)

    if p.is_absolute() and p.exists():
        return p.resolve()

    candidate = images_root / p
    if candidate.exists():
        return candidate.resolve()

    candidate = images_root / p.name
    if candidate.exists():
        return candidate.resolve()

    candidate = images_root.parent / p
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve image: {image_value} under {images_root}")


def normalize_query_record(rec: dict) -> dict:
    out = {
        "qid": pick_first_present(rec, ["qid", "query_id", "id"]),
        "query_image": pick_first_present(
            rec,
            ["query-image", "query_image", "reference_image", "reference-image"],
        ),
        "query_text": pick_first_present(
            rec,
            ["query-text", "query_text", "caption", "text"],
        ),
        "target_image": pick_first_present(
            rec,
            ["target-image", "target_image", "target", "target-img"],
        ),
    }

    if out["qid"] is None or out["query_image"] is None or out["target_image"] is None:
        raise ValueError(f"Unexpected LaSCo query record format: {rec}")

    if out["query_text"] is None:
        out["query_text"] = ""

    return out


def normalize_corpus_record(rec: Any) -> dict:
    if not isinstance(rec, dict):
        raise ValueError(f"Unexpected corpus record type: {type(rec).__name__} -> {rec}")

    image_value = pick_first_present(
        rec,
        [
            "image",
            "img",
            "image_name",
            "image-name",
            "path",
            "img_path",
            "file_name",
            "filename",
        ],
    )
    cid_value = pick_first_present(rec, ["id", "image_id", "image-id"])

    if image_value is None and "value" in rec:
        value = rec["value"]

        if isinstance(value, str):
            image_value = value

        elif isinstance(value, (list, tuple)):
            image_value = value
            if cid_value is None and len(value) >= 1:
                cid_value = value[0]

        elif isinstance(value, dict):
            image_value = pick_first_present(
                value,
                [
                    "image",
                    "img",
                    "image_name",
                    "image-name",
                    "path",
                    "img_path",
                    "file_name",
                    "filename",
                ],
            )
            if cid_value is None:
                cid_value = pick_first_present(value, ["id", "image_id", "image-id"])

        else:
            image_value = value

    if image_value is None:
        raise ValueError(f"Unexpected LaSCo corpus record format: {rec}")

    if cid_value is None:
        cid_value = image_value

    return {
        "corpus_id": cid_value,
        "image": image_value,
    }


def build_candidate_rows(
    corpus_records: List[Any],
    images_root: Path,
    output_dir: Path,
    relative_paths: bool,
) -> List[dict]:
    rows: List[dict] = []
    seen: set[str] = set()

    for raw in corpus_records:
        rec = normalize_corpus_record(raw)
        img_path = find_existing_image(rec["image"], images_root)
        cand_id = canonical_cand_id(rec["corpus_id"])

        if cand_id in seen:
            continue
        seen.add(cand_id)

        rows.append(
            {
                "cand_id": cand_id,
                "img_path": normalize_path(img_path, output_dir, relative_paths),
            }
        )

    return rows


def build_query_rows(
    query_records: List[Any],
    images_root: Path,
    output_dir: Path,
    relative_paths: bool,
    split_tag: str,
) -> Tuple[List[dict], List[Tuple[str, str, int]]]:
    query_rows: List[dict] = []
    qrels_rows: List[Tuple[str, str, int]] = []

    for raw in query_records:
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected query record type: {type(raw).__name__} -> {raw}")

        rec = normalize_query_record(raw)

        qid = f"lasco_q_{split_tag}_{rec['qid']}"
        qimg_path = find_existing_image(rec["query_image"], images_root)
        target_cand_id = canonical_cand_id(rec["target_image"])

        query_rows.append(
            {
                "query_id": qid,
                "query_text": str(rec["query_text"]),
                "query_img_path": normalize_path(qimg_path, output_dir, relative_paths),
                "pos_cand_list": [target_cand_id],
            }
        )

        qrels_rows.append((qid, target_cand_id, 1))

    return query_rows, qrels_rows


def main() -> None:
    args = parse_args()
    paths = resolve_required_paths(args.lasco_root)

    output_dir = args.output_dir.expanduser().resolve()
    query_train_dir = output_dir / "query" / "train"
    query_test_dir = output_dir / "query" / "test"
    cand_pool_dir = output_dir / "cand_pool"
    qrels_dir = output_dir / "qrels"
    metadata_dir = output_dir / "metadata"

    for d in [query_train_dir, query_test_dir, cand_pool_dir, qrels_dir, metadata_dir]:
        ensure_dir(d)

    train_queries_obj = load_json(paths["train_queries"])
    train_corpus_obj = load_json(paths["train_corpus"])
    val_queries_obj = load_json(paths["val_queries"])
    val_corpus_obj = load_json(paths["val_corpus"])

    train_queries_raw = extract_records(train_queries_obj, "train query", paths["train_queries"])
    train_corpus_raw = extract_records(train_corpus_obj, "train corpus", paths["train_corpus"])
    val_queries_raw = extract_records(val_queries_obj, "val query", paths["val_queries"])
    val_corpus_raw = extract_records(val_corpus_obj, "val corpus", paths["val_corpus"])

    train_cand_rows = build_candidate_rows(
        corpus_records=train_corpus_raw,
        images_root=paths["train_images_root"],
        output_dir=output_dir,
        relative_paths=args.relative_paths,
    )
    test_cand_rows = build_candidate_rows(
        corpus_records=val_corpus_raw,
        images_root=paths["val_images_root"],
        output_dir=output_dir,
        relative_paths=args.relative_paths,
    )

    train_query_rows, train_qrels_rows = build_query_rows(
        query_records=train_queries_raw,
        images_root=paths["train_images_root"],
        output_dir=output_dir,
        relative_paths=args.relative_paths,
        split_tag="train",
    )
    test_query_rows, test_qrels_rows = build_query_rows(
        query_records=val_queries_raw,
        images_root=paths["val_images_root"],
        output_dir=output_dir,
        relative_paths=args.relative_paths,
        split_tag="test",
    )

    train_query_file = query_train_dir / "mbeir_lasco_task7_train.jsonl"
    test_query_file = query_test_dir / "mbeir_lasco_task7_test.jsonl"
    train_cand_file = cand_pool_dir / "lasco_task7_train.jsonl"
    test_cand_file = cand_pool_dir / "lasco_task7.jsonl"
    train_qrels_file = qrels_dir / "train.tsv"
    test_qrels_file = qrels_dir / "test.tsv"

    n_train_queries = write_jsonl(train_query_file, train_query_rows)
    n_test_queries = write_jsonl(test_query_file, test_query_rows)
    n_train_cands = write_jsonl(train_cand_file, train_cand_rows)
    n_test_cands = write_jsonl(test_cand_file, test_cand_rows)
    n_train_qrels = write_qrels_tsv(train_qrels_file, train_qrels_rows)
    n_test_qrels = write_qrels_tsv(test_qrels_file, test_qrels_rows)

    info = {
        "dataset": "LaSCo",
        "splits_exported": {
            "train": "training_as_train",
            "test": "validation_as_test",
        },
        "task_id": 7,
        "task": "(qi, qt) -> ci",
        "format": "M-BEIR-style for the evaluator in this release",
        "files": {
            "train_query": str(train_query_file),
            "test_query": str(test_query_file),
            "train_cand_pool": str(train_cand_file),
            "test_cand_pool": str(test_cand_file),
            "train_qrels": str(train_qrels_file),
            "test_qrels": str(test_qrels_file),
        },
        "counts": {
            "train_queries": n_train_queries,
            "test_queries": n_test_queries,
            "train_candidates": n_train_cands,
            "test_candidates": n_test_cands,
            "train_qrels": n_train_qrels,
            "test_qrels": n_test_qrels,
        },
        "paths_are_relative_to_output_dir": bool(args.relative_paths),
        "image_roots": {
            "train": str(paths["train_images_root"]),
            "test": str(paths["val_images_root"]),
        },
        "source_files": {k: str(v) for k, v in paths.items()},
    }

    with (metadata_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Wrote train queries:   {train_query_file}")
    print(f"Wrote test queries:    {test_query_file}")
    print(f"Wrote train cand pool: {train_cand_file}")
    print(f"Wrote test cand pool:  {test_cand_file}")
    print(f"Wrote train qrels:     {train_qrels_file}")
    print(f"Wrote test qrels:      {test_qrels_file}")
    print(f"Wrote metadata:        {metadata_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
