# Annotation Protocol — Validating Composed Image Retrieval Queries

This document is the protocol shown to each annotator during the validation
study that produced the released **validated** subset of the audit. It is
included **for transparency**: it lets a reader see exactly which judgements
each annotator was asked to make, in what order, and against which rubric.

## What is being annotated

Annotators see a pool of **shortcut-free** Composed Image Retrieval (CIR)
queries — queries the automatic audit identified as those that no retriever
in the pool could solve from a single modality alone (target outside top-K
under text-only and under image-only inputs). At this stage, every shipped
query falls in one of two paper categories, but **the annotator does not
see this hidden category**:

- **composition_required** — at least one retriever in the pool solved the
  query when given both modalities;
- **unresolved** — no retriever in the pool solved the query, even
  multimodally.

The annotator's task is to judge the **data instance itself**, not retriever
performance, by looking at the triplet:

1. **Reference image** — the starting point;
2. **Text modification** — what should change in the target;
3. **Ground-truth target image** — the image the query is supposed to
   retrieve.

For context, annotators are also shown the deduplicated union of the top-10
multimodal retrievals across all retrievers in the pool — the
*aggregate multimodal panel*. This panel is a **diagnostic aid only**; it is
used to judge query specificity and never appears in any label.

## What "validated" means

A query is **VALIDATED** when the triplet (reference image, text
modification, target image) forms a coherent CIR instance: a human reading
the text and looking at the reference image would specifically end up at the
labelled target — not at some other gallery image that satisfies the same
modification just as well.

Validated queries are the union of the two splits the release ships under
`final_dataset/query_jsonl/<dataset>/`:

- `validated_solved` — VALIDATED **and** originally `composition_required`;
- `validated_unsolved` — VALIDATED **and** originally `unresolved`.

All other queries are flagged with one of the four issue categories below
and are excluded from the validated subset.

## Categories

For each query the annotator picks **exactly one** top-level category. When
more than one applies, severity decides; pick the most severe issue:

```
INVALID_TARGET_IMAGE > INVALID_IMAGE_QUERY > INVALID_TEXT_QUERY
> QUERY_TOO_BROAD > VALIDATED
```

### 1. `INVALID_TEXT_QUERY`

The text modification itself is broken — fixing the images would not save
the query.

- **1a. Ungrammatical / Nonsensical** — fragment, garbled, unrelated to the
  reference image, a caption rather than a modification instruction, or
  internally contradictory (e.g. asks for both "lighter" and "black").
- **1b. Underspecified** — grammatically fine but too vague to constrain
  the target on its own (e.g. "make it different", "tall"). Note the
  difference with `QUERY_TOO_BROAD`: here the *text* is vague; there the
  text is specific but the *gallery* contains many equally good matches.
- **1c. Other** — anything else; a free-text note is required.

### 2. `INVALID_IMAGE_QUERY`

The **reference image** is degraded so that a human cannot extract the
visual information the text refers to.

- **2a. Cropped** — relevant subject cut off or incomplete.
- **2b. Low-resolution** — too blurry/small to read the visual context.
- **2c. Other** — corrupted bytes, fully black frame, watermark covering
  the relevant subject; free-text note required.

### 3. `INVALID_TARGET_IMAGE`

The ground-truth target is wrong or degraded.

- **3a. Cropped** — target cropped so the modification cannot be verified.
- **3b. Low-resolution** — target too blurry/small to verify.
- **3c. Other** — common cases: target does not match the text, target is
  identical to the reference (self-retrieval), or target visibly
  contradicts the modification. Free-text note required.

### 4. `QUERY_TOO_BROAD`

The text and images are well-formed and the labelled target is consistent
with the modification — but the combination matches many gallery images
equally well, so the labelled target is **not uniquely correct**. A
retriever returning a semantically valid alternative would still be marked
wrong.

**Working heuristic.** Mark `QUERY_TOO_BROAD` when the query plausibly
admits **≥10 non-ground-truth matches** in the gallery. This is a practical
threshold, not a hard rule — use judgement, and lean on the aggregate
multimodal panel when many of the top retrievals look equally compatible
with the text.

`QUERY_TOO_BROAD` was the dominant failure mode in the audit. Distinguish
it carefully from `INVALID_TEXT_QUERY/Underspecified`:

- *Underspecified text* — the text alone is too vague (e.g. "make it
  different"). The query would be broken even with a small gallery.
- *Query too broad* — the text is specific (e.g. "the sign says 'stop'"),
  but many gallery images independently satisfy it.

### 5. `VALIDATED`

Use this when **all** of the following hold:

- the text modification is clear and describes a specific change;
- the reference image is readable and provides the visual context the
  text relies on;
- the labelled target correctly reflects the requested modification;
- a human would genuinely need **both** modalities to settle on this
  specific target.

A `VALIDATED` query enters the validated subset (under `validated_solved`
or `validated_unsolved` depending on its hidden audit category).

## Decision flow

```
Can you understand the text modification?
  NO  → INVALID_TEXT_QUERY (pick 1a / 1b / 1c)
  YES ↓

Can you read the reference image clearly?
  NO  → INVALID_IMAGE_QUERY (pick 2a / 2b / 2c)
  YES ↓

Does the target image match what the modification asks for?
  NO  → INVALID_TARGET_IMAGE (pick 3a / 3b / 3c)
  YES ↓

Does the (text + reference) admit ≥10 equally good gallery alternatives?
  YES → QUERY_TOO_BROAD
  NO  ↓

VALIDATED
```

## Annotator guidelines

- You are judging the **data instance**, not retriever performance. Do not
  try to predict whether a model will solve the query.
- When two categories apply, pick the more severe one (severity order
  above).
- Any `Other` sub-category requires a free-text note.
- Borderline calls: leave a free-text note explaining your reasoning so
  the case can be re-checked.
- Time budget: aim for 30–60 seconds per query. If you spend more than
  2 minutes, mark the case for adjudication and move on.
- Use the aggregate multimodal panel **only** to judge specificity /
  ambiguity — never to decide whether a model "should have" solved the
  query.

## Worked examples

The following examples were used during onboarding. They illustrate each
non-trivial label; image bytes are not bundled here, but each example
references a query by `<dataset>::<query_idx>` so it can be inspected
against the M-BEIR root.

**`VALIDATED`.** Reference: a long-sleeved patterned dress. Text: *"same
dress, but sleeveless and black"*. Target: the same type of dress,
sleeveless and black. The modification is specific, both images are
readable, and the target uniquely matches.

**`INVALID_TEXT_QUERY` / Ungrammatical-Nonsensical** — `fashioniq::69`.
Text: *"Is lighter and has shorter sleeves and is black and less
revealing."*. The text bundles incompatible constraints (`lighter` and
`black` are mutually exclusive); the failure is in the text itself, not in
how hard the retrieval would be.

**`INVALID_TEXT_QUERY` / Underspecified** — `lasco::34`. Text: *"tall"*.
Reference and target are both ordinary tennis images; the text alone does
not specify any retrieval transformation.

**`INVALID_TARGET_IMAGE` / Other** — `fashioniq::457`. Text: *"White and it
is shorted and with sleeves."*. The query asks for a white, shorter version
with sleeves, but the labelled target is a pink floral short dress: the
target does not reflect the requested edit with respect to the reference.

**`QUERY_TOO_BROAD`** — `lasco::159`. Text: *"The sign says 'stop'"*. The
text is understandable, the labelled target is plausible, but the gallery
contains many other stop-sign images and the top-10 multimodal aggregate
panel surfaces several of them as equally compatible. The issue is gallery
breadth, not a broken instance.

## Inter-annotator agreement

A subset of queries was double-annotated by a separate group of annotators
to estimate inter-annotator agreement. The resulting Cohen's κ /
Krippendorff α values, per dataset and per category, are in
[`agreement_report.md`](agreement_report.md). All annotator identities are
anonymised to `annotator_NN`.
