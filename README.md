This folder contains the code and data needed to reproduce the experiments in the
paper *"Do Composed Image Retrieval Benchmarks Require Multimodal Composition?"*.

The release is script-first: each stage is a Python entry point with a thin
shell launcher. There is no installable package and no formal test suite.

## Layout

```
cir-audit-release/
├── envs/                 conda env recipes (pinned per retriever stack)
├── data_prep/            convert LaSCo and CIRCO into the M-BEIR layout
├── retrievers/           one Python adapter per retriever (paper Table 1)
├── ablation_configs/     two unimodal ablations: zero-image and drop-text
├── retrieval/            generate per-(dataset, retriever) retrieval JSONs;
│                         aggregate them into the cross-retriever shortcut audit
├── shortcut_audit/       4 aggregated_best_rank_mm_any.json (CIRR/FashionIQ/
│                         LaSCo/CIRCO) — the dataset-level audit artifact
├── shortcut_free_subsets/ ★ DATASET 1 — automatic shortcut-free queries,
│                         per (dataset, label); labels are
│                         {composition_required, unresolved, shortcut_solvable,
│                         shortcut_free}
├── subset_evaluation/    re-evaluate retrievers on the shortcut-free / validated
│                         subsets (with optional caching / safetensors export)
├── annotations/          anonymised per-annotator validation labels and the
│                         inter-annotator agreement report; see
│                         annotations/annotation_instructions.md
├── final_dataset/        ★ DATASET 2 — human-validated subset (well-formed,
│                         shortcut-free queries) + the script that builds
│                         it from the annotations
├── final_evaluation/     consume stored ranks/embeddings to produce the paper
│                         tables and figures (no model is rerun)
└── docs/                 reproduction documentation; see DATA_DOWNLOAD.md for
                          how to obtain CIRR/FashionIQ (M-BEIR) and convert
                          LaSCo/CIRCO into the same layout
```

## Datasets

CIRR and FashionIQ test data come directly from M-BEIR. **LaSCo** and
**CIRCO** are not distributed in M-BEIR layout and must be converted; this
release ships `data_prep/circo_to_mbeir.py` and `data_prep/lasco_to_mbeir.py`
to do so, producing the canonical filenames the launchers expect.

Each dataset can live in its own M-BEIR-shaped root (`<path_to_M-BEIR>` for
CIRR + FashionIQ, `<path_to_lasco_mbeir>`, `<path_to_circo_mbeir>`); every
launcher takes a `--mbeir_root` flag and is invoked once per root. A merged
single-root layout also works. Full step-by-step instructions (download URLs,
conversion commands, sanity checks) are in
[`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md).

## Reproduction pipeline

The pipeline is a four-stage flow. Each stage reads what the previous stage
wrote, so you can also start mid-pipeline using the included artifacts.

### Stage 0 — environments

Each retriever family has incompatible Python deps, so we use one conda env per
stack.

```bash
# create one env at a time
bash envs/create_qwen3emb.sh         # Qwen3-VL-Embedding (2B/8B), Rzen-Embed, VLM2Vec-V2
bash envs/create_gme.sh              # GME-Qwen2VL
bash envs/create_lamra.sh            # LamRA
bash envs/create_lamra_qwen25vl.sh   # LamRA-Qwen2.5VL
bash envs/create_mmemb.sh            # MM-Embed

# or create them all
bash envs/create_all_envs.sh
```

### Stage 1 — per-(dataset, retriever) retrieval

For each open-source retriever and each dataset, generate one
`retrieval_data_<dataset>_task7_<retriever>.json` with the rank of the
ground-truth target under multimodal, text-only and image-only inputs.

```bash
# CIRR + FashionIQ
bash retrieval/run_generate_retrieval_data.sh --mbeir_root <path_to_M-BEIR> \
  --experiment cirr:7:mbeir_cirr_task7_test.jsonl \
  --experiment fashioniq:7:mbeir_fashioniq_task7_test.jsonl

# LaSCo (same launcher, separate root)
bash retrieval/run_generate_retrieval_data.sh --mbeir_root <path_to_lasco_mbeir> \
  --experiment lasco:7:mbeir_lasco_task7_test.jsonl

# CIRCO (dedicated launcher with multi-positive scoring, separate root)
bash retrieval/run_generate_retrieval_data_circo.sh --mbeir_root <path_to_circo_mbeir>

# Commercial APIs (need GEMINI_API_KEY / VOYAGE_API_KEY in the environment;
# pass the appropriate root for each dataset)
bash retrieval/run_generate_retrieval_data_gemini.sh --mbeir_root <path_to_M-BEIR>
bash retrieval/run_generate_retrieval_data_voyage.sh --mbeir_root <path_to_M-BEIR>
```

If you keep a single merged root, pass the same `--mbeir_root` everywhere.

`--mbeir_root` is the only required flag: pass the absolute path to the
M-BEIR-formatted root described in [`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md).
With no other flags the launcher iterates over the full open-source retriever
pool and the three default datasets (CIRR, FashionIQ, LaSCo). Common
overrides (run `bash retrieval/run_generate_retrieval_data.sh --help` for the
full list):

| Flag                              | What it does                                                                 |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `--mbeir_root PATH`               | M-BEIR root; can also be set via `MBEIR_ROOT=...`.                           |
| `--retriever retrievers.<module>` | Restrict to one (or repeated) retriever modules from `retrievers/`.          |
| `--experiment DATASET:TASK:FILE`  | Restrict to one (or repeated) datasets, e.g. `cirr:7:mbeir_cirr_task7_test.jsonl`. |
| `--output_dir PATH`               | Where to write the per-(dataset, retriever) JSONs. Default `retrieval/retrieval_results/`. |
| `--device cuda` / `cuda:0`        | Forwarded to each retriever's constructor.                                   |
| `--batch_size INT`                | Encoder batch size (default 16).                                             |
| `--top_k INT`                     | Top-K to record per query (default 20).                                      |
| `--no-normalize`                  | Disable L2-normalization of embeddings (defaults to normalized).             |
| `--force`                         | Re-run even if the output JSON already exists.                               |

Each retriever runs inside its pinned conda env (`omni`, `gme`, `lamra`,
`lamra2`, `mmemb`, `qwen3emb`); the launcher activates the right one
automatically via `conda run -n <env>`.

The two unimodal ablations are launched implicitly by the evaluator: each query
is encoded three times (multimodal, text-only via `ablation_configs/text_only_drop_text.json`,
image-only via `ablation_configs/image_only_zero_image.json`) and per-modality
ranks are written to the same JSON.

### Stage 2 — aggregate into the cross-retriever shortcut audit

Combine all per-retriever JSONs into a single aggregated artifact per dataset
that defines the dataset-level shortcut labels (composition-required /
unresolved / shortcut-solvable, plus the shortcut-free union).

```bash
python retrieval/aggregate_retrieval_data.py \
    --results_dir retrieval/retrieval_results \
    --dataset cirr \
    --output shortcut_audit/retrieval_data_cirr_task7_aggregated_best_rank_mm_any.json
# repeat for fashioniq, lasco, circo
```

The four aggregated files are already shipped under `shortcut_audit/`, so this
stage is **only required if you re-run Stage 1 with a different retriever pool**.

### Stage 3 — validated subset (annotations → released subsets)

The validated subset is built from the per-annotator label files in
`annotations/users/`:

```bash
python final_dataset/build_validated_subsets.py
```

The output (already shipped under `final_dataset/query_jsonl/<dataset>/{validated_solved,validated_unsolved}/`)
is a per-(dataset, validity) jsonl of queries whose reference image, text, and
target form a coherent CIR instance.

### Stage 4 — re-evaluation on shortcut-free / validated subsets

`subset_evaluation/run_subset_eval.sh` re-encodes a retriever on a subset and
writes per-run metric JSONs to `subset_evaluation/logs/metrics_artifacts/` plus
a single appended summary in `subset_evaluation/logs/results.jsonl`. The
headline metric for this release is **Recall@10**; when the two ablations are
also run, the **Composition Gap** is derived from those Recall@10 numbers.

Two things to set before running:

1. **`PYTHONPATH=$PWD`** when invoked from the repo root, so the launcher's
   `conda run` can import the `retrievers.*` package.
2. **`SUBSETS_ROOT`** controls where `--subset <name>` is looked up:

   | Subset family | Required `SUBSETS_ROOT`                       | Valid `--subset` values                  |
   | ------------- | --------------------------------------------- | ---------------------------------------- |
   | Shortcut-free | `$PWD/shortcut_free_subsets`                  | `shortcut_free`                          |
   | Validated     | `$PWD/final_dataset/query_jsonl` *(default)*  | `validated_solved`, `validated_unsolved` |

The launcher also takes one `--mbeir_root` per invocation, so run it once per
source root: `<path_to_M-BEIR>` for CIRR + FashionIQ, `<path_to_lasco_mbeir>`
for LaSCo, `<path_to_circo_mbeir>` for CIRCO.

#### Recall@10 only (skip the two ablations)

Pass `--no-ablations` to run just the multimodal pass — one row per
`(dataset, subset, retriever)` instead of three:

```bash
# Shortcut-free, CIRR + FashionIQ
PYTHONPATH=$PWD SUBSETS_ROOT=$PWD/shortcut_free_subsets \
    bash subset_evaluation/run_subset_eval.sh \
        --mbeir_root <path_to_M-BEIR> \
        --dataset cirr --dataset fashioniq \
        --subset shortcut_free \
        --retriever retrievers.qwen3vl8b_vllm_retriever \
        --no-ablations

# LaSCo and CIRCO: same call against their own roots
PYTHONPATH=$PWD SUBSETS_ROOT=$PWD/shortcut_free_subsets \
    bash subset_evaluation/run_subset_eval.sh \
        --mbeir_root <path_to_lasco_mbeir> --dataset lasco \
        --subset shortcut_free \
        --retriever retrievers.qwen3vl8b_vllm_retriever --no-ablations

PYTHONPATH=$PWD SUBSETS_ROOT=$PWD/shortcut_free_subsets \
    bash subset_evaluation/run_subset_eval.sh \
        --mbeir_root <path_to_circo_mbeir> --dataset circo \
        --subset shortcut_free \
        --retriever retrievers.qwen3vl8b_vllm_retriever --no-ablations

# Validated (default SUBSETS_ROOT covers both halves of the validated set)
PYTHONPATH=$PWD bash subset_evaluation/run_subset_eval.sh \
    --mbeir_root <path_to_M-BEIR> \
    --dataset cirr --dataset fashioniq \
    --subset validated_solved --subset validated_unsolved \
    --retriever retrievers.qwen3vl8b_vllm_retriever \
    --no-ablations
```

Read the resulting Recall@10 from each `metrics_*.json`, or in bulk from
`results.jsonl`:

```python
import json, pandas as pd
df = pd.DataFrame(json.loads(l) for l in open("subset_evaluation/logs/results.jsonl"))
print(df[df["ablation"] == "noablation"][["dataset", "subset", "retriever", "R@10"]])
```

The validated set ships split into `validated_solved` and `validated_unsolved`.
The single Recall@10 over the full validated split is the count-weighted mean:

```
R@10 = (n_solved · R@10_solved + n_unsolved · R@10_unsolved) / (n_solved + n_unsolved)
```

#### Composition Gap (Recall@10 plus the two ablations)

Drop `--no-ablations` so the launcher runs all three variants (`noablation`,
`image_only_zero_image`, `text_only_drop_text`) per
`(dataset, subset, retriever)`:

```bash
PYTHONPATH=$PWD SUBSETS_ROOT=$PWD/shortcut_free_subsets \
    bash subset_evaluation/run_subset_eval.sh \
        --mbeir_root <path_to_M-BEIR> \
        --dataset cirr --dataset fashioniq \
        --subset shortcut_free \
        --retriever retrievers.qwen3vl8b_vllm_retriever
```

Compute the gap from `results.jsonl`:

```python
import json, pandas as pd
df = pd.DataFrame(json.loads(l) for l in open("subset_evaluation/logs/results.jsonl"))
piv = df.pivot_table(
    index=["dataset", "subset", "retriever"],
    columns="ablation",
    values="R@10",
).reset_index()
piv["compgap"] = piv["noablation"] - piv[
    ["text_only_drop_text", "image_only_zero_image"]
].max(axis=1)
print(piv[["dataset", "subset", "retriever", "noablation", "compgap"]])
```

#### Notes

- Drop `--retriever` to sweep the full default retriever pool.
- Only the two whitelisted ablation configs are accepted via `--ablation_json`:
  `ablation_configs/image_only_zero_image.json` and
  `ablation_configs/text_only_drop_text.json`.
- Raw embedding caches stay under `subset_evaluation/cache_subset_eval_raw/`
  and are reused across subsets and ablations; safetensors exports go to
  `cache_subset_eval_export/`. Pass `--no-export` to skip the export step.

### Stage 5 — final evaluation

The final stage consumes the stored ranks and per-(retriever, subset) caches
without re-encoding any model, producing per-(dataset, retriever, subset)
metric rows (Recall@K, full-catalogue nDCG/MRR, modality deltas) under
`final_evaluation/results/`:

```bash
bash final_evaluation/run_final_evaluation.sh
```

## Retriever pool

The 11 retrievers used in the paper (9 open-source + 2 commercial APIs):

| Module                                            | Conda env  | Type    |
| ------------------------------------------------- | ---------- | ------- |
| `retrievers.e5_omni_retriever`                    | `omni`     | open    |
| `retrievers.gme_qwen2vl_retriever`                | `gme`      | open    |
| `retrievers.lamra_retriever`                      | `lamra`    | open    |
| `retrievers.lamra_qwen25vl_retriever`             | `lamra2`   | open    |
| `retrievers.mmembed_retriever`                    | `mmemb`    | open    |
| `retrievers.qwen3vl2b_vllm_retriever`             | `qwen3emb` | open    |
| `retrievers.qwen3vl8b_vllm_retriever`             | `qwen3emb` | open    |
| `retrievers.rzen_embed_retriever`                 | `qwen3emb` | open    |
| `retrievers.vlm2vec_v2_retriever`                 | `qwen3emb` | open    |
| `retrievers.gemini_embedding_2_retriever`         | (any)      | API     |
| `retrievers.voyage_multimodal_35_retriever`       | (any)      | API     |

Each retriever exposes a `Retriever` class with two methods:

```python
class Retriever:
    def __init__(self, device: str = "cuda"): ...
    def embed_queries(self, keys: List[ItemKey]) -> torch.Tensor: ...
    def embed_targets(self, keys: List[ItemKey]) -> torch.Tensor: ...
```

`ItemKey` is `(text: str, img_path: str, image: Optional[PIL.Image])`.

## Released data summary

The release ships **two** datasets at two different stages of the audit. Pick
the one that matches your experiment:

### 1. Shortcut-free dataset — `shortcut_free_subsets/`

Queries that survived the **automatic shortcut audit**: a query is here only
if no retriever in the pool placed the ground-truth target inside the top-K
under text-only or image-only inputs. Use this if you want the largest
filtered set the audit can produce, before any human validation.

```
shortcut_free_subsets/<dataset>/<label>/
    mbeir_<dataset>_task7_test_multimodal.jsonl        ← queries
    mbeir_<dataset>_task7_test_multimodal.trace.jsonl  ← per-query audit trail
    mbeir_<dataset>_task7_test_multimodal.trace.csv    ← same trail as CSV
    mbeir_<dataset>_task7_test_multimodal.manifest.json
```

`<dataset>` is one of `{cirr, fashioniq, lasco, circo}`. `<label>` is one of:

| Label                  | Meaning                                                                           |
| ---------------------- | --------------------------------------------------------------------------------- |
| `composition_required` | At least one retriever solved the query multimodally; none solved it unimodally.  |
| `unresolved`           | No retriever in the pool solved the query under any modality.                     |
| `shortcut_free`        | Union of the two above (the shortcut-free set used in the paper).                 |
| `shortcut_solvable`    | Complementary set: at least one retriever solved the query unimodally (excluded). |

Per-split sizes — these match **Table 1 of the paper** (shortcut audit
across all 11 retrievers at K = 10):

|             |  total | composition_required | unresolved | shortcut_free | shortcut_solvable |
| ----------- | -----: | -------------------: | ---------: | ------------: | ----------------: |
| CIRR        |  4 170 |                  271 |        414 |           685 |             3 485 |
| FashionIQ   |  6 003 |                1 462 |      2 607 |         4 069 |             1 934 |
| LaSCo       | 30 031 |                2 064 |     16 354 |        18 418 |            11 613 |
| CIRCO       |    220 |                   53 |          3 |            56 |               164 |

### 2. Final validated dataset — `final_dataset/`

The subset of the shortcut-free queries that **also passed the human
validation study** (≥ majority of annotators marked the (reference image,
text, target image) triplet as a coherent CIR instance). Use this if you
want the cleanest evaluation set.

```
final_dataset/
├── query_jsonl/<dataset>/validated_solved/
│       mbeir_<dataset>_task7_test_validated_solved.jsonl       ← queries
│       mbeir_<dataset>_task7_test_validated_solved.trace.jsonl ← per-query trail
├── query_jsonl/<dataset>/validated_unsolved/
│       mbeir_<dataset>_task7_test_validated_unsolved.jsonl
│       mbeir_<dataset>_task7_test_validated_unsolved.trace.jsonl
├── all_validated_queries.jsonl   ← all 1 689 validated queries in one file
├── query_indices/                ← qid → split mapping per (dataset, split)
└── build_validated_subsets.py    ← script that derives this from annotations/
```

`validated_solved` queries were classified as `composition_required` by the
audit; `validated_unsolved` queries were classified as `unresolved`. Per-split
sizes match **Table 2 of the paper**. For CIRR and CIRCO the *full*
shortcut-free residue was audited; for FashionIQ and LaSCo a stratified
sample of 1 000 composition-required + 1 000 unresolved queries was audited
(the rest of those datasets' shortcut-free queries are unlabelled and not
included here):

|              | audited (comp_req) | validated\_solved | audited (unresolved) | validated\_unsolved | total validated |
| ------------ | -----------------: | ----------------: | -------------------: | ------------------: | --------------: |
| CIRR         |                271 |               147 |                  414 |                 156 |             303 |
| FashionIQ    |              1 000 |               368 |                1 000 |                 218 |             586 |
| LaSCo        |              1 000 |               452 |                1 000 |                 306 |             758 |
| CIRCO        |                 53 |                39 |                    3 |                   3 |              42 |
| **Total**    |          **2 324** |       **1 006**   |            **2 417** |           **683**   |       **1 689** |

The `all_validated_queries.jsonl` file is the four-dataset union with extra
fields attached (`dataset`, `split`, `hidden_category`, per-annotator
decisions). For the field schema, see one record in any `validated_*.jsonl`.

### Auxiliary artifacts

* **`shortcut_audit/retrieval_data_*_aggregated_best_rank_mm_any.json`** —
  the dataset-level audit JSONs. Per query: the best rank under each
  modality across the retriever pool, and the audit label that drove the
  shortcut-free split above.
* **`annotations/users/annotator_*.json`** — per-annotator validity labels
  for the main study. **`annotations/agreement_users/annotator_*.json`** —
  per-annotator labels for the IAA double-annotation.
  **`annotations/agreement_report.{json,md}`** — Cohen's κ / Krippendorff α
  write-up. Annotator identities are anonymised to `annotator_NN`.
* **`annotations/annotation_instructions.md`** — the **annotation protocol**
  shown to each annotator during the validation study. It is included **for
  transparency**: the document is the rubric annotators worked against
  (categories, severity ordering, decision flow, the ≥10-alternatives
  heuristic for `QUERY_TOO_BROAD`, and worked examples). All category names
  match the paper (`VALIDATED`, `INVALID_TEXT_QUERY`, `INVALID_IMAGE_QUERY`,
  `INVALID_TARGET_IMAGE`, `QUERY_TOO_BROAD`). Inter-annotator agreement
  results are in
  [`annotations/agreement_report.md`](annotations/agreement_report.md).

Image bytes are **not** shipped — every record's `query_img_path` and every
`cand_pool` entry references a path under your local M-BEIR root.

### Loading the validated subset directly

The validated subset is plain JSONL with the M-BEIR field schema (`qid`,
`query_txt`, `query_img_path`, `pos_cand_list`, …) — no pipeline run
required:

```python
import json
from pathlib import Path

ROOT = Path("final_dataset/query_jsonl")
DATASETS = ["cirr", "fashioniq", "lasco", "circo"]
SPLITS = ["validated_solved", "validated_unsolved"]

queries = []
for dataset in DATASETS:
    for split in SPLITS:
        for jsonl in (ROOT / dataset / split).glob("*.jsonl"):
            if jsonl.name.endswith(".trace.jsonl"):
                continue
            with jsonl.open() as fh:
                for line in fh:
                    record = json.loads(line)
                    record["dataset"], record["split"] = dataset, split
                    queries.append(record)

print(f"loaded {len(queries)} validated queries")
# record["query_img_path"] is relative to <path_to_M-BEIR>/
# record["pos_cand_list"] indexes into <path_to_M-BEIR>/cand_pool/<dataset>_task7.jsonl
```

To load the **shortcut-free** dataset instead, swap the two roots and skip
the `.trace.jsonl` files exactly the same way:

```python
ROOT = Path("shortcut_free_subsets")
SPLITS = ["composition_required", "unresolved", "shortcut_free"]
```

The retrieval pool used to label queries is the 11 retrievers listed in this
release.
