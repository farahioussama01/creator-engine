---
name: halve-cost
description: Analyse a batch's cost per keeper and produce a concrete, testable plan to halve it. Use when asked why a batch was expensive or how to make clips cheaper.
---

Work on batch $ARGUMENTS (latest batch if empty).

1. `batch_report` → note cost per keeper, every bucket, stage pass rates, renders per keeper.
2. Rank the buckets. For the top three, say which mechanism produces them:
   - idle GPU → pods held without work (concurrency, teardown, boot time, phase boundaries)
   - model loads & swaps → stage interleaving on one GPU, cold pods
   - rejected clips / dead-end keyframes → which gate (from the funnel) and which format
   - rejected keyframes / best-of-k spares → LoRA quality, prompt issues, k too high
3. For each, propose one change that is a config or prompt edit, with the expected saving computed
   from the report's own numbers (the `levers` list is a starting point, not the answer).
4. Propose a single A/B: the next batch split in two halves, one with the change. Give the exact
   `plan_batch` calls or config diff.
5. Never claim a GPU speed-up you have not measured; GPU switches need `ce bench` numbers first.
