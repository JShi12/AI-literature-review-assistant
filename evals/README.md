# Evaluation harness

Measures the quality of the LLM stages (claims → syntheses → review draft) on a fixed set of papers,
so prompt or model changes can be compared with numbers instead of by eyeballing output.

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
| `--stop-after claims\|synthesis\|review` | Run only the first stages |
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
- **Recordings** (`evals/recordings/<model>/`, committed): one JSON file per LLM response, keyed by a
  hash of model + prompt version + temperature + schema + instructions + input. A prompt edit
  therefore misses the old recording and triggers a fresh call. Recordings store only the input hash,
  not paper text.

## Metrics

All metrics are deterministic and computed in `evals/metrics.py`; none use an LLM judge yet.

**Claims (per chunk)**
- *Rejected by location check*: production raises on the first claim whose page or offsets fall
  outside the chunk, losing **every** claim from that chunk.
- *Span match*: whether `page_text[start_char:end_char]` actually reads like the claim
  (similarity ≥ 0.8). Being inside the chunk doesn't mean the offsets point at the right text.
- *Verbatim* / *lexical coverage*: how extractive the claim is relative to the chunk. This is a
  rough grounding proxy, not entailment.
- *Ids copied exactly*: whether the model followed the instruction to echo the ids. Production
  overwrites them anyway.

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
- Gold labels: section boundaries, gold claims, retrieval relevance (recall@k for topic → claims).
- Agent tool-trajectory evals.
- A replay-mode CI gate against committed baseline numbers.
