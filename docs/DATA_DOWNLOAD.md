# Data download and conversion

Every retrieval and subset-evaluation script in this release expects an
**M-BEIR-formatted** root, with the same internal layout for any dataset:

```
<root>/
├── query/test/<query_file>.jsonl
├── cand_pool/<dataset>_task7.jsonl
└── mbeir_images/<dataset>_images/    (or absolute paths in the jsonls)
```

CIRR and FashionIQ are distributed in this layout directly. **LaSCo** and
**CIRCO** are not, and are converted by `data_prep/lasco_to_mbeir.py` and
`data_prep/circo_to_mbeir.py`.

You do **not** have to merge all four datasets into a single tree. Every
launcher accepts a `--mbeir_root` flag, so the recommended layout is one root
per source:

```
<path_to_M-BEIR>/        ← CIRR + FashionIQ (official M-BEIR release)
<path_to_lasco_mbeir>/   ← LaSCo, after running data_prep/lasco_to_mbeir.py
<path_to_circo_mbeir>/   ← CIRCO, after running data_prep/circo_to_mbeir.py
```

The launchers invoke each dataset against its own root (see §4 for the
pattern). If you prefer a single merged tree, you can also point all four
datasets at the same `--mbeir_root` — the layout is identical.

## 1. CIRR and FashionIQ — M-BEIR official release

CIRR and FashionIQ test queries, candidate pools, and images come from the
M-BEIR distribution on Hugging Face:

- M-BEIR dataset card: <https://huggingface.co/datasets/TIGER-Lab/M-BEIR>
- Download instructions: <https://huggingface.co/datasets/TIGER-Lab/M-BEIR#downloading-the-m-beir-dataset>

After download, `<path_to_M-BEIR>` should contain:

| File                                          | What it is                                  |
| --------------------------------------------- | ------------------------------------------- |
| `query/test/mbeir_cirr_task7_test.jsonl`      | CIRR test queries (4,170)                   |
| `query/test/mbeir_fashioniq_task7_test.jsonl` | FashionIQ test queries (6,003)              |
| `cand_pool/cirr_task7.jsonl`                  | CIRR candidate pool                         |
| `cand_pool/fashioniq_task7.jsonl`             | FashionIQ candidate pool                    |
| `mbeir_images/cirr_images/`                   | CIRR image bytes                            |
| `mbeir_images/fashioniq_images/`              | FashionIQ image bytes                       |

## 2. CIRCO — convert to M-BEIR format under `circo_mbeir/`

CIRCO is evaluated on its **validation split** (220 queries, multi-positive)
because the official test split has no public ground truth.

### 2.1 Fetch CIRCO and the COCO 2017 unlabeled images

```bash
git clone https://github.com/miccunifi/CIRCO.git
cd CIRCO

# image bytes
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip                 # -> unlabeled2017/*.jpg

# image-info json (extracts annotations/image_info_unlabeled2017.json,
# merging into the existing CIRCO/annotations/ that ships with the repo)
aria2c -x 16 -s 16 http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
unzip image_info_unlabeled2017.zip
```

Then assemble the `COCO2017_unlabeled/` layout the converter expects — this
directory does **not** exist after the unzips, you have to create it and move
the unzipped contents into it:

```bash
mkdir -p COCO2017_unlabeled
mv annotations    COCO2017_unlabeled/
mv unlabeled2017  COCO2017_unlabeled/
```

After this step the layout under `CIRCO/` should be:

```
CIRCO/
└── COCO2017_unlabeled/
    ├── annotations/
    │   ├── val.json                       (from the cloned repo)
    │   └── image_info_unlabeled2017.json  (from the unzip)
    └── unlabeled2017/
        └── *.jpg
```

### 2.2 Convert into the M-BEIR layout

```bash
python data_prep/circo_to_mbeir.py \
  --circo_root /path/to/CIRCO \
  --output_dir <path_to_circo_mbeir>
```

Defaults are pre-aligned with the rest of this release:

- `<path_to_circo_mbeir>/query/test/mbeir_circo_task7_test.jsonl`
- `<path_to_circo_mbeir>/cand_pool/circo_task7.jsonl`
- `<path_to_circo_mbeir>/qrels/test.tsv`

### 2.3 Image bytes

No extra step is needed: the converter writes **absolute** paths to
`/path/to/CIRCO/COCO2017_unlabeled/unlabeled2017/*.jpg` into the jsonls, so
the evaluator reads the bytes directly from there. You don't need to copy
or symlink the images under `<path_to_circo_mbeir>/mbeir_images/`.

## 3. LaSCo — convert to M-BEIR format under `lasco_mbeir/`

LaSCo reuses COCO 2014 images and is evaluated on its **validation split**
(30,031 queries) — the same split the audit operates on.

### 3.1 Fetch LaSCo and the COCO 2014 images

```bash
git clone https://github.com/levymsn/LaSCo.git LASCO
cd LASCO

# the LaSCo repo ships the four JSON annotation files under downloads/;
# the converter expects them at the LASCO root, so move them up
mv downloads/* .

aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2014.zip
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/val2014.zip

unzip train2014.zip
unzip val2014.zip
```

After this step `LASCO/` should contain:

```
LASCO/
├── lasco_train.json
├── lasco_train_corpus.json
├── lasco_val.json
├── lasco_val_corpus.json
├── train2014/
│   └── COCO_train2014_*.jpg
└── val2014/
    └── COCO_val2014_*.jpg
```

### 3.2 Convert into the M-BEIR layout

```bash
python data_prep/lasco_to_mbeir.py \
  --lasco_root /path/to/LASCO \
  --output_dir <path_to_lasco_mbeir>
```

This writes (test split is what the audit and final evaluation use):

- `<path_to_lasco_mbeir>/query/test/mbeir_lasco_task7_test.jsonl` (30,031 queries)
- `<path_to_lasco_mbeir>/cand_pool/lasco_task7.jsonl`
- `<path_to_lasco_mbeir>/qrels/test.tsv`

The training split is also exported for completeness as
`mbeir_lasco_task7_train.jsonl` / `lasco_task7_train.jsonl` but is not used
by the evaluator.

### 3.3 Image bytes

Same as for CIRCO: the converter writes absolute paths into the jsonls
(`/path/to/LASCO/train2014/...` and `/path/to/LASCO/val2014/...`), so the
evaluator finds the bytes directly. No `mbeir_images/` symlink is required.

## 4. Pointing the launchers at the right root

Every launcher takes `--mbeir_root` (or honours `MBEIR_ROOT=...`). Run each
dataset against its own root:

```bash
# CIRR + FashionIQ
bash retrieval/run_generate_retrieval_data.sh \
  --mbeir_root <path_to_M-BEIR> \
  --experiment cirr:7:mbeir_cirr_task7_test.jsonl \
  --experiment fashioniq:7:mbeir_fashioniq_task7_test.jsonl

# LaSCo (same launcher, different root)
bash retrieval/run_generate_retrieval_data.sh \
  --mbeir_root <path_to_lasco_mbeir> \
  --experiment lasco:7:mbeir_lasco_task7_test.jsonl

# CIRCO (dedicated launcher, dedicated root)
bash retrieval/run_generate_retrieval_data_circo.sh \
  --mbeir_root <path_to_circo_mbeir>
```

The same pattern applies to `subset_evaluation/run_subset_eval.sh`: invoke it
once per root, restricting `--dataset` (or `--experiment`) to the datasets
that live in that root.

If you prefer a single merged tree, point all three commands at the same
`--mbeir_root` — the per-dataset filenames don't collide.

## 5. Sanity check

For each root, confirm the files it should contain are in place:

```bash
# CIRR + FashionIQ root
ls <path_to_M-BEIR>/query/test/ | grep -E '^mbeir_(cirr|fashioniq)_task7_test\.jsonl$'

# LaSCo root
wc -l <path_to_lasco_mbeir>/query/test/mbeir_lasco_task7_test.jsonl   # 30031

# CIRCO root
wc -l <path_to_circo_mbeir>/query/test/mbeir_circo_task7_test.jsonl   # 220
```

Cross-check against Table 1 of the paper:

| Dataset    | Test queries |
| ---------- | ------------ |
| CIRR       | 4,170        |
| FashionIQ  | 6,003        |
| LaSCo      | 30,031       |
| CIRCO      | 220          |

## 6. Image storage

The full image set across the four datasets is large (≈100–150 GB). The
launcher scripts only need read access; they do not modify the data roots.
The shipped retrieval, audit, and validated-subset jsonls do **not** embed
image bytes — they reference either paths under `mbeir_images/...` (for
CIRR/FashionIQ from the official M-BEIR release) or the absolute paths
produced by the LaSCo and CIRCO converters, which resolve once the layouts
above exist.
