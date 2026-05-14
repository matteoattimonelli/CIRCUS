#!/usr/bin/env python3
"""
Export gallery/query embedding cache tensors to float16 safetensors plus manifests.

This script is meant to run after the evaluation scripts have populated a
per-run cache directory with `gallery_*.pt` and `queries_*.pt` files.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch

try:
    from safetensors.torch import save_file
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "safetensors is required for export_embedding_cache_safetensors.py"
    ) from exc


DTYPE_CHOICES = ("float16", "bfloat16", "float32")
KIND_CHOICES = ("both", "gallery", "queries")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export cached gallery/query embeddings to safetensors."
    )
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task_id", type=int, default=None)
    ap.add_argument("--subset_name", required=True)
    ap.add_argument("--retriever_module", required=True)
    ap.add_argument("--retriever_class", default="Retriever")
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--query_source", default=None)
    ap.add_argument("--eval_script", required=True)
    ap.add_argument("--ablation_json", default=None)
    ap.add_argument("--query_mode", default="both")
    ap.add_argument("--query_txt_part", default="full")
    ap.add_argument("--export_dtype", choices=DTYPE_CHOICES, default="float16")
    ap.add_argument("--kind", choices=KIND_CHOICES, default="both")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--multi_positive", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    return ap.parse_args()


def export_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported export dtype: {dtype_name}")


def dtype_tag(dtype_name: str) -> str:
    if dtype_name == "float16":
        return "f16"
    if dtype_name == "bfloat16":
        return "bf16"
    return "f32"


def _safe_slug(text: str, max_len: int = 120) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._+\-=]", "-", text)
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _infer_split(query_file: str) -> str:
    ql = (query_file or "").lower()
    if "test" in ql:
        return "test"
    if "val" in ql or "dev" in ql:
        return "val"
    return "na"


def tensor_kind(path: Path) -> str:
    if path.name.startswith("gallery_"):
        return "gallery"
    if path.name.startswith("queries_"):
        return "queries"
    return "unknown"


def output_stem(pt_path: Path, args: argparse.Namespace) -> str:
    kind = tensor_kind(pt_path)
    if kind != "gallery":
        return pt_path.stem

    task = f"task{args.task_id}" if args.task_id is not None else "taskNA"
    split = _infer_split(args.query_file)
    retriever_tag = _safe_slug(f"{args.retriever_module}.{args.retriever_class}", max_len=80)
    ablation_tag = _safe_slug(Path(args.ablation_json).stem if args.ablation_json else "noablation", max_len=40)
    norm_tag = "norm" if args.normalize else "nonorm"
    stem = f"gallery_{args.dataset.lower()}_{split}_{task}_{retriever_tag}_{ablation_tag}_{norm_tag}"
    return _safe_slug(stem, max_len=200)


def export_one(
    pt_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    tensor = torch.load(pt_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{pt_path} does not contain a torch.Tensor")

    out_tensor = tensor.detach().cpu().to(export_dtype(args.export_dtype)).contiguous()
    stem = output_stem(pt_path, args)
    tag = dtype_tag(args.export_dtype)
    out_path = output_dir / f"{stem}.{tag}.safetensors"
    manifest_path = output_dir / f"{stem}.{tag}.manifest.json"

    if not args.force and out_path.exists() and manifest_path.exists():
        return {
            "source_pt": str(pt_path),
            "export_file": str(out_path),
            "manifest_file": str(manifest_path),
            "kind": tensor_kind(pt_path),
            "skipped": True,
        }

    save_file({"embeddings": out_tensor}, str(out_path))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_pt": str(pt_path.resolve()),
        "source_pt_stem": pt_path.stem,
        "export_file": str(out_path.resolve()),
        "kind": tensor_kind(pt_path),
        "canonicalized_name": stem != pt_path.stem,
        "shape": list(out_tensor.shape),
        "source_dtype": str(tensor.dtype),
        "export_dtype": args.export_dtype,
        "dataset": args.dataset,
        "task_id": args.task_id,
        "subset_name": args.subset_name,
        "retriever_module": args.retriever_module,
        "retriever_class": args.retriever_class,
        "query_file": args.query_file,
        "query_source": args.query_source,
        "eval_script": args.eval_script,
        "ablation_json": args.ablation_json,
        "query_mode": args.query_mode,
        "query_txt_part": args.query_txt_part,
        "normalize": args.normalize,
        "multi_positive": args.multi_positive,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "source_pt": str(pt_path),
        "export_file": str(out_path),
        "manifest_file": str(manifest_path),
        "kind": tensor_kind(pt_path),
        "skipped": False,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("export_embedding_cache_safetensors")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files: List[Path] = []
    if args.kind in ("both", "gallery"):
        pt_files.extend(sorted(cache_dir.glob("gallery_*.pt")))
    if args.kind in ("both", "queries"):
        pt_files.extend(sorted(cache_dir.glob("queries_*.pt")))
    if not pt_files:
        raise SystemExit(
            f"No matching pt files found under {cache_dir} for kind={args.kind}"
        )

    exported = []
    for pt_path in pt_files:
        info = export_one(pt_path=pt_path, output_dir=output_dir, args=args)
        exported.append(info)
        action = "SKIP" if info["skipped"] else "SAVE"
        log.info("%s %s -> %s", action, pt_path.name, Path(info["export_file"]).name)

    index_path = output_dir / "export_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": args.dataset,
                "task_id": args.task_id,
                "subset_name": args.subset_name,
                "retriever_module": args.retriever_module,
                "retriever_class": args.retriever_class,
                "query_file": args.query_file,
                "query_source": args.query_source,
                "eval_script": args.eval_script,
                "ablation_json": args.ablation_json,
                "query_mode": args.query_mode,
                "query_txt_part": args.query_txt_part,
                "kind": args.kind,
                "normalize": args.normalize,
                "multi_positive": args.multi_positive,
                "cache_dir": str(cache_dir),
                "export_dir": str(output_dir),
                "files": exported,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log.info("Wrote export index: %s", index_path)


if __name__ == "__main__":
    main()
