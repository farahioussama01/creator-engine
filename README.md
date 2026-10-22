# creator-engine

Turns a character LoRA into short, checked video clips, and records what every clip cost.
It runs on hosted open models, so **no GPU is needed**; the same pipeline also runs on your own
ComfyUI when you rent a GPU.

```
plan ──▶ 2 stills ──▶ 4 checks ──▶ best still ──▶ clip ──▶ 2 checks ──▶ keeper ──▶ post ──▶ views ──▶ next batch
        Z-Image Turbo   light, one face,         Wan 2.2      moving,                              more of the
        + your LoRA     focus, same face         image→video  same face                            formats that work
```

Every render is logged with its price. **Cost per keeper = total spend ÷ clips that passed every check.**

## The numbers

`ce report` prints these four lines from your own runs. A mock run never produces them.

| | |
|---|---|
| **Pipeline** | Z-Image Turbo + character LoRA → 4 checks → Wan 2.2 A14B image-to-video (480p, ~5 s) → 2 checks |
| **Volume** | *from the first real batch* |
| **Cost per piece** | *from the first real batch* (list prices below put it near **$0.07-0.08** at 480p) |
| **Views** | *after posting* |

Prices on fal.ai (checked 22 Oct 2026):

| Step | Model | Price |
|---|---|---|
| Still (720×1280) | Z-Image Turbo + LoRA | $0.0078 |
| Clip (~5 s) | Wan 2.2 A14B turbo | $0.05 at 480p, $0.10 at 720p |
| LoRA training | Z-Image trainer | $2.26 per 1,000 steps |
| Face dataset | 8 portraits + 20 edits (Qwen-Image-Edit) | ~$0.50 |

Per keeper: 2 stills + about 1.2 clips ≈ $0.075 at 480p. The offline mock, priced the same way,
lands at $0.070. For comparison, Kling 3.0 at 720p is $0.42 per 5 s clip, $0.84 per keeper at two takes.

## Run it (no GPU, ~$15 for everything)

```bash
pip install -e .
ce models                          # face finder + face fingerprint models (~250 MB, runs on CPU)
export FAL_KEY=...                 # fal.ai -> Dashboard -> Keys
```

**1. Create the character (~$5, ~30 min)**

```bash
ce character portraits             # 8 candidate faces in datasets/mila/candidates/
ce character variations datasets/mila/candidates/candidate_03.png   # ~20 photos of that same person
ce character check                 # is it one consistent face? sets the identity threshold
ce character train                 # trains the LoRA on fal (2,000 steps, $4.52)
```

Look through `datasets/mila/train/` before training and delete anything that isn't clearly her.
`check` lists outliers too.

**2. Render (~$1 per 12 clips)**

```bash
ce run --n 12                      # small first batch: look at the report
ce run --n 100                     # the volume batch
ce report                          # the four lines + runs/reports/<batch>.html
```

**3. Post and measure**

```bash
ce export                          # out/<batch>/: clips + captions.csv (with the AI label) + views.csv
# post the clips, label them as AI on the platform too, wait 48 h, fill views.csv from insights
ce views out/<batch>/views.csv
ce report                          # now with views
ce next --n 100                    # format mix for the next batch, weighted by views
```

**Try it without any key:** `ce run --n 12 --provider mock` renders cartoons with realistic failure
rates. The checks, ledger and report all run for real; the answer lines say MOCK.

## The six checks

| On the still (re-roll costs ~$0.008) | On the clip (re-roll costs $0.05) |
|---|---|
| **Light:** not blown out, not too dark | **Moving:** not a frozen clip |
| **One face:** a single main person | **Same face:** stays the character across 8 frames |
| **Focus:** the face is sharp | |
| **Identity:** the face matches the training images | |

Identity uses AuraFace (Apache-2.0) fingerprints compared to the average of your training images.
The threshold comes from how similar those images are to each other, not from a guess.
Checking the still first is the main cost lever: a bad face is caught for $0.008 instead of $0.05.

## Files

| File | What it does |
|---|---|
| `pipeline.py` | the whole loop, top to bottom (~170 lines) |
| `providers.py` | where renders come from: `fal`, `comfyui` (own GPU), `mock` |
| `gates.py` | the six checks |
| `faces.py` | face finder, face fingerprint, consistency of the training set |
| `character.py` | portraits → variations → check → train |
| `ledger.py` | SQLite: every render, its price, kept or not |
| `report.py` | the four lines + HTML contact sheet |
| `views.py` | views in, next format mix out |
| `cli.py` / `mcp_server.py` | `ce` commands / the same for Claude Code |
| `configs/` | settings, the character, the formats |
| `workflows/` | ComfyUI graphs for the own-GPU route |

## On your own GPU

Set `provider: comfyui` and `comfyui_url` to a ComfyUI with Z-Image Turbo, Wan 2.2 and your LoRA
loaded. Cost becomes GPU seconds × hourly rate. On a $0.34/h RTX 4090, ComfyUI's published timing
(~71 s per 5 s clip with the 4-step LoRA) means about $0.007 per clip, so renting a GPU is the next
step once volume justifies it. A larger version with a GPU simulator, RunPod pod management,
a format bandit and Instagram API publishing is on the
[`full-engine`](https://github.com/farahioussama01/creator-engine/tree/full-engine) branch.

## Claude Code

`.mcp.json` registers an MCP server with five tools: `run_batch`, `batch_status`, `numbers`,
`pieces` (to open and review rejects) and `next_mix`.

## Disclosure

The character is fictional. Every exported caption carries its AI disclosure, and the config refuses
an empty one. Label posts as AI on the platform as well.
