# Final Dataset From Multimodal Validity Study

Deduplication policy:
- `annotator_03` and `annotator_08` labeled the same 593 examples.
- `annotator_03` was kept and `annotator_08` was dropped for that duplicate block.
- Remaining examples are counted at the unique-example level.

Overall:
- Unique examples: `4741`
- Labeled examples: `4741`
- `validated = true`: `1689` (35.6%)
- `validated = false`: `3052` (64.4%)
- Unlabeled examples: `0`

## Per Split

| Dataset | Split | Unique | Labeled | Correct | Correct % | Incorrect | Unlabeled |
|---|---:|---:|---:|---:|---:|---:|---:|
| circo | composition_required | 53 | 53 | 39 | 73.6% | 14 | 0 |
| circo | unresolved | 3 | 3 | 3 | 100.0% | 0 | 0 |
| cirr | composition_required | 271 | 271 | 147 | 54.2% | 124 | 0 |
| cirr | unresolved | 414 | 414 | 156 | 37.7% | 258 | 0 |
| fashioniq | composition_required | 1000 | 1000 | 368 | 36.8% | 632 | 0 |
| fashioniq | unresolved | 1000 | 1000 | 218 | 21.8% | 782 | 0 |
| lasco | composition_required | 1000 | 1000 | 452 | 45.2% | 548 | 0 |
| lasco | unresolved | 1000 | 1000 | 306 | 30.6% | 694 | 0 |

## Failure Reasons

Reason counts below are not mutually exclusive. One incorrect example can activate more than one reason.

- `INVALID_TEXT_QUERY`: `318`
- `INVALID_IMAGE_QUERY`: `69`
- `INVALID_TARGET_IMAGE`: `611`
- `QUERY_TOO_BROAD`: `2193`

## Interpretation

- `CIRR` is the cleanest of the larger benchmarks here.
- `composition_required` is consistently cleaner than `unresolved`, but still far from noise-free.
- The dominant failure mode is `QUERY_TOO_BROAD`, not malformed image or text input.
- For downstream evaluation, the safest subsets are the exported `validated=true` query files.

## Exported Files

- Query-index allowlists: `final_dataset/query_indices/*.json`
- Repo copies of filtered JSONLs: `final_dataset/query_jsonl/<dataset>/*.jsonl`
- Direct eval-ready copies: `<dataset_root>/query/test/final_dataset/*.jsonl`
- Trace JSONL sidecars: `*.trace.jsonl` beside each exported query subset
