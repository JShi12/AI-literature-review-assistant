# Evaluation harness

Measures the quality of the pipeline on a fixed set of papers, so prompt, model, or heuristic changes
can be compared with numbers instead of by eyeballing output. Stages, in order: section detection
(no LLM) → claim extraction → retrieval → syntheses → review draft.

It needs no database: each stage calls the same `request_*` function the app uses (same prompt,
instructions, temperature), then mimics persistence (id overriding, location validation, dropping
unknown claim ids, citation cleanup) on in-memory objects.

## Quick start

```bash
python -m evals.snapshot          # once: download the 3 demo arXiv papers, freeze pages + chunks
python -m evals.run               # run all stages on 30 chunks with $OPENAI_CHAT_MODEL, recording responses
python -m evals.run --mode replay # re-score from recordings only: free, offline, no API key
```

Each run writes `evals/reports/<run id>/report.json` (every case and metric) and `summary.md`
(the headline table, worst claim-offset examples, and errors), and prints the summary.

Useful flags:

| Flag | Meaning |
|---|---|
| `--model A --model B` | Evaluate several models side by side |
| `--limit N` / `--seed S` | Number of chunks (0 = all 115), spread evenly across papers |
| `--stop-after claims\|retrieval\|synthesis\|review` | Run only the first stages |
| `--mode record\|replay\|refresh` | Reuse recordings and record misses / recordings only / always call the API |
| `--workers N` | Concurrent claim-extraction calls (default 4) |
| `--judge-model M` / `--no-judge` | Model for the LLM-as-judge checks (default `gpt-4.1`), or skip them |
| `--prune` | Afterwards, delete recordings the run didn't use, e.g. after a prompt change. Use the same `--limit`/`--seed`/models you replay with |

The exit code is 1 if any case couldn't be evaluated (missing recording, API/auth/network error).

## Inputs

- **Snapshot** (`evals/snapshots/demo_papers.json`, gitignored): page text and section-aware chunks
  of the three papers the live demo is seeded with, produced by the production ingestion code.
  Ingestion changes don't affect evals until the snapshot is rebuilt. Each report records the
  snapshot's fingerprint. The fingerprint covers page text and chunk boundaries, not derived
  metadata such as section types, so rebuilding after a section-detection change keeps gold labels
  valid. Build one from your own PDFs with `python -m evals.snapshot --pdf a.pdf --pdf b.pdf --out evals/snapshots/mine.json`.
- **Scenarios** (`evals/datasets/review_scenarios.json`): a review topic, which synthesis types to
  generate, and how many claims to draw on.
- **Gold labels** (`evals/datasets/gold_*.json`, `retrieval_queries.json`, committed): section
  headings for every paper, reference claims for the 30 chunks of the default sample (`--limit 30
  --seed 0`), and 12 retrieval queries with the chunks relevant to each. They were drafted by Claude
  (one labelling agent per paper) and **need human review**. Each file is pinned to a snapshot
  fingerprint and skipped with a warning when run against a different snapshot. Check edits with
  `python -m evals.gold validate`, which confirms every quote and heading exists verbatim.
- **Recordings** (`evals/recordings/<model>/`, committed): one JSON file per LLM response, keyed by a
  hash of model + prompt version + temperature + schema + instructions + input. A prompt edit
  therefore misses the old recording and triggers a fresh call. Recordings store only the input hash,
  not paper text. Embeddings are recorded per input text (`evals/recordings/text-embedding-3-small/`),
  as float16, about 4 KB each.

## Metrics

Most metrics are deterministic and computed in `evals/metrics.py`. The LLM-judge metrics are described
under [LLM judges](#llm-judges).

**Section detection** (runs the *current* `detect_sections` and chunking on the frozen pages)
- *Heading precision / recall*: detected headings that start on the same line as a gold heading.
- *Chunks with correct section type*: this type is sent to the claim extractor in its prompt.

**Claims (per chunk)**
- *Claims dropped: quote not found*: since `claims.v2` the model returns a verbatim `source_quote`
  and production computes offsets by locating it (`pipeline/quotes.py`). Claims whose quote can't
  be found are dropped, so this measures both locating failures and ungrounded quotes.
- *Rejected by location check*: production rejects a chunk if any claim falls outside it. Since
  offsets are computed, this should stay at 0.
- *Span match*: whether the text at `page_text[start_char:end_char]` (the located quote) reads like
  the claim, at similarity ≥ 0.8. Under `claims.v1`, where the model guessed offsets, this was 0.8%.
  Now it mostly measures how far `claim_text` departs from its quote.
- *Verbatim* / *lexical coverage*: how extractive the claim is relative to the chunk. This is a
  rough grounding proxy, not entailment.
- *Gold recall / precision / F1*: matched by **source span, not wording**. Each extracted claim is
  located in the chunk by aligning the source text it was located at, and matches a gold claim when the
  two spans overlap by ≥ 50% of the shorter one. Matching isn't one-to-one, because extractors split
  claims at different granularity. Heavily paraphrased claims can't be located and count as
  unmatched, so the scores lean strict. Precision is "against gold": a valid claim the labeller
  skipped still counts against it.
- *Claim type agrees with gold*: among matched claims. Much of the disagreement is metric vs.
  method, which is partly a taxonomy question.

**Retrieval** (mirrors `find_similar_claims`: embeds `claim_embedding_text`, ranks by cosine)
- Precision@5/10, Recall@20, MRR, nDCG@10 per query, averaged, plus the precision a random ranking
  would get as a baseline. Relevance is labelled per chunk, so a claim counts as relevant when its
  source chunk is relevant to the query. The pool is the claims extracted in this run, so `--limit`
  changes the pool.

**Syntheses**
- *Invented claim ids*: cited ids that weren't in the input. Production silently drops these.
- *No valid support*, *type matches request*, *draws on 2+ papers*.
- *Self-reported counts wrong*: the model's `supporting_claims`/`supporting_papers` disagree with
  its own citation list.

**Review draft**
- *Sentences with linked claims*: structured sentences that cite at least one real claim.
- *Phantom-supported sentences*: production stores `is_supported = true` from any cited id, even
  invented ones, so these sentences show as supported with no claim behind them.
- *Structured sentences found in Markdown*: `sentences[]` and `markdown` are generated separately
  and can drift apart.
- *Substantive Markdown sentences cited*, *invalid citation numbers*, *leaked ids*, *references
  section complete*: hygiene of what the user actually reads.

**Usage**: calls, recorded vs live, tokens, cost (from `evals/pricing.py`, since production
doesn't compute cost), and live latency.

## LLM judges

`evals/judges.py` checks what string matching can't: whether content is actually *supported*. Each judge
sees only the evidence the pipeline gave the model (`gpt-4.1`, temperature 0, versioned rubrics):

- **Claim grounding**: is each claim supported by its chunk? Gives one holistic verdict per claim,
  batched per chunk.
- **Synthesis faithfulness**: the judge splits the title and body into statements and checks each one
  against the cited claims. The verdict is derived in code: all supported is *faithful*, at least half
  unsupported is *unfaithful*, anything else is *partially faithful*. It also checks whether the
  synthesis really is its type, e.g. whether a "contradiction" has two claims that actually disagree.
- **Review citations**: the same statement-level check for each review sentence against its own cited
  claims.

Statement-level checking replaced a single holistic verdict (v1). A single verdict let the judge pass
exactly the subtle overstatements that matter: an extra list item, "widely used", or a finding
attributed to the wrong paper.

### Calibration

`python -m evals.calibrate` scores the judges against `evals/datasets/judge_calibration.json`. The set
has 98 items: real pipeline outputs labelled independently, plus constructed negatives with labels known
by construction. The constructed negatives are claims perturbed to be wrong or overstated, and synthesis
bodies or review sentences paired with someone else's citations. Results with `gpt-4.1`:

| | Claim grounding | Synthesis faithfulness | Review citations |
|---|---|---|---|
| Supported vs. not, binary accuracy | 100% | 89% | 84% |
| Cohen's kappa (3-class) | 0.91 | 0.59 | 0.70 |
| Constructed negatives caught | 100% | 100% | 100% |
| Synthesis-type accuracy | | 63% | |

How to read these:

- The claim judge is reliable.
- The synthesis and review judges reliably catch real failures. Where they disagree with the reference,
  it's mostly at the partial/full boundary, where the reference labels are themselves debatable.
- Type judgement is the weakest check. Treat it as indicative only.
- Caveats:
  - The v2 rubrics were revised after seeing calibration results on the same items, with no separate
    holdout.
  - The reference labels were written by Claude and still need human review.
  - Repeat runs move by a few points, because the judge isn't fully deterministic even at temperature 0.

Calibration recordings live in `evals/recordings/calibration/`, so `evals.run --prune` never deletes
them.

## Not covered yet

- A held-out calibration set, and human review of the judge reference labels.
- Human review of the gold labels.
- Agent tool-trajectory evals.
- A replay-mode CI gate against committed baseline numbers.
