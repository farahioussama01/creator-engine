# creator-engine

Character LoRA stills → 4 checks → Wan 2.2 clips → 2 checks → keepers, with the price of every render
in `runs/ledger.db`. Read `src/creator_engine/pipeline.py` first; everything else hangs off it.

- Tests: `pytest -q` (~15 s, no keys). Lint: `ruff check src tests`.
- Offline try: `ce run --n 12 --provider mock`. Real runs need `FAL_KEY` and a trained LoRA.
- Never quote numbers from a mock batch as real; `report.answer()` refuses them on purpose.
- Don't raise `budget_usd` or switch `provider` to a paid one without the user saying so.
- When reviewing rejects, open the image/clip paths from the `pieces` tool and say what you see.
