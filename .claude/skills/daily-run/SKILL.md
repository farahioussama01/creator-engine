---
name: daily-run
description: Run today's content batch for a character end to end (allocate formats, render, gate, report, schedule). Spends GPU money, so only the user starts it.
disable-model-invocation: true
---

Run today's batch for $ARGUMENTS (character id, optionally followed by a piece count; default 24).

1. Call `engine_status`. If any endpoint is down or `spend_last_24h_usd` is within 20% of the daily
   budget, stop and tell me.
2. Call `format_experiments` for the character and summarise in two lines which formats are being
   scaled, explored or killed, and why (posts, median lift, P(best)).
3. Call `plan_batch` with the character and count (profile `standard` unless I said otherwise).
4. Call `run_batch`, then poll `batch_status` every ~30 seconds until it is no longer running.
5. Call `batch_report`. Tell me: keepers / planned, cost per keeper, the biggest orange bucket, and the
   top lever from the halving plan.
6. Call `list_pieces` with status `rejected`, `Read` up to 6 of the reject images, and tell me whether
   the gates were right. Flag anything that looks like a false reject.
7. Call `schedule_posts` for the batch and `render_dashboard`. Give me the report path.

Do not publish live. Do not change budgets or gate thresholds; propose changes instead.
