# creator-engine — notes for Claude Code

This repo renders photoreal AI-creator content at volume: character-LoRA keyframes (Z-Image Turbo in
ComfyUI) → quality gates → Wan 2.2 image-to-video (lightx2v 4-step) → gates → keepers → scheduled posts,
on rented GPUs, with a cost ledger and a format-experiment loop. You operate it through the
`creator-engine` MCP server (`.mcp.json`) and the `ce` CLI.

## Map
- `src/creator_engine/engine.py` batch runner (phased: all keyframes, then clips), budget cap, resume
- `pool.py` least-loaded dispatch over ComfyUI pods, retries, session billing
- `comfy.py` ComfyUI HTTP client (GPU seconds come from `/history` timestamps)
- `gates.py` quality gates; thresholds live in `configs/engine.yaml` → `gates:`
- `ledger.py` cost per keeper, spend buckets, halving plan
- `experiments.py` Thompson sampling over formats vs each account's baseline
- `publish.py` scheduling, AI disclosure, provenance sidecars, Instagram Graph API (dry-run default)
- `sim/` ComfyUI-compatible GPU simulator used by tests and `ce demo`
- `workflows/*.json` ComfyUI API-format templates with `{{placeholders}}`

## Rules
- Never set `publish.dry_run: false` or call live publishing without the user saying so in chat.
- Never remove or weaken a character's `disclosure`; never add a character with `likeness: licensed`
  without a `consent_record` path the user provided.
- Never raise `budget.max_usd_per_batch` / `max_usd_per_day` on your own. Propose it.
- Before the first batch on a new pod: `ce check <url>` (or the `preflight` tool). Do not spend GPU time
  on a template that fails preflight.
- Changing a gate threshold: show the before/after funnel from `ce report` and the identity histogram;
  thresholds are calibrated per character and per embedder (`ce identity enroll` suggests one).
- When reviewing rejects, `Read` the keyframe/clip paths that `list_pieces` returns and say what you
  see; the reason string is the gate's opinion, not ground truth.

## Commands
- `pytest -q` (≈2 min, no GPU needed) · `ruff check src tests`
- `ce demo` full loop on simulated pods → `runs/demo/reports/*.html`
- `ce plan mila --n 24` → `ce run <batch>` → `ce report <batch>` → `ce publish schedule <batch>`
- `ce daily mila --n 40 --provision` the unattended loop (rents RunPod pods, always terminates them)
