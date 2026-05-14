# Croissant Files

This directory contains anonymized Croissant metadata for the
`CIR-Benchmarks-Audit` release. The release is metadata-only: it does not
redistribute raw images, and image fields are source-relative references that
users must resolve against the official upstream datasets.

## Files

One Croissant JSON-LD file is generated per source benchmark:

- `cirr.json`
- `fashioniq.json`
- `lasco.json`
- `circo.json`

Each file describes two CSV resources under `exports/`:

- `*_shortcut_free.csv`: the shortcut-free audit trace, including aggregate
  labels, ranks, best retrievers, and retriever support at `K=10`.
- `*_validated.csv`: the retained human-validated query subset, flattened for
  Croissant consumption. Retained annotator identifiers are intentionally
  omitted.

## Row Counts

| Benchmark | Shortcut-free rows | Validated rows |
| --- | ---: | ---: |
| CIRCO | 56 | 42 |
| CIRR | 685 | 303 |
| FashionIQ | 4069 | 586 |
| LaSCo | 18418 | 758 |

The `validated` exports contain retained valid examples. Invalid human
judgments are represented through the validation-reason fields in the source
annotation data, but the public Croissant CSVs are the retained benchmark slice.

## Metadata Scope

The Croissant files declare conformance to:

- `http://mlcommons.org/croissant/1.1`
- `http://mlcommons.org/croissant/RAI/1.0`

The anonymized submission metadata uses:

- `creator`: `Anonymized`
- `publisher`: `Anonymized`
- `version` / `sdVersion`: `0.1.0`
- `dateCreated`, `datePublished`, `dateModified`: `2026-05-04`
- `isLiveDataset`: `false`
- `inLanguage`: `en`
- shortcut-audit retrieval cutoff: `K=10`
- random seed: `123`

The retriever pool is documented in `rai:machineAnnotationTools` with internal
CSV identifiers, release-facing names, and model/API identifiers.

## Upstream Data and Licenses

This release is derived from existing composed image retrieval benchmarks. Users
must separately obtain the official upstream data for CIRR, FashionIQ, LaSCo,
CIRCO, COCO, NLVR2, and M-BEIR as applicable.

The `license` field in each Croissant file follows the source-benchmark terms
currently documented by the paper/release. If the derived metadata is published
under a separate explicit license, update these URLs before public release.

## Validation

Validate with the official Croissant tooling, for example:

```bash
uv run --python 3.11 --with mlcroissant mlcroissant validate --jsonld=cirr.json
```
