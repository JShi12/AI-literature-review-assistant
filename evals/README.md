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

The exit code is 1 if any case couldn't be evaluated (missing recording, API/auth/network error).

## Inputs

- **Snapshot** (`evals/snapshots/demo_papers.json`, gitignored): page text and section-aware chunks
  of the three papers the live demo is seeded with, produced by the production ingestion code.
  Ingestion changes don't affect evals until the snapshot is rebuilt. Each report records the
  snapshot's fingerprint. Build one from your own PDFs with `python -m evals.snapshot --pdf a.pdf --pdf b.pdf --out evals/snapshots/mine.json`.
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

All metrics are deterministic and computed in `evals/metrics.py`; none use an LLM judge yet.

**Section detection** (runs the *current* `detect_sections` and chunking on the frozen pages)
- *Heading precision / recall*: detected headings that start on the same line as a gold heading.
- *Chunks with correct section type*: this type is sent to the claim extractor in its prompt.

**Claims (per chunk)**
- *Rejected by location check*: production raises on the first claim whose page or offsets fall
  outside the chunk, losing **every** claim from that chunk.
- *Span match*: whether `page_text[start_char:end_char]` actually reads like the claim
  (similarity ≥ 0.8). Being inside the chunk doesn't mean the offsets point at the right text.
- *Verbatim* / *lexical coverage*: how extractive the claim is relative to the chunk. This is a
  rough grounding proxy, not entailment.
- *Ids copied exactly*: whether the model followed the instruction to echo the ids. Production
  overwrites them anyway.
- *Gold recall / precision / F1*: matched by **source span, not wording**. Each extracted claim is
  located in the chunk by aligning its text, never by its offsets, and matches a gold claim when the
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

## Not covered yet

- LLM-judge metrics (claim grounding/entailment, synthesis faithfulness, citation precision).
- Human review of the gold labels.
- Agent tool-trajectory evals.
- A replay-mode CI gate against committed baseline numbers.
