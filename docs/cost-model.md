# Cost model: what a keeper costs, and how to halve it

## The unit is cost per keeper

`total spend in a batch ÷ pieces that passed every gate`.

"Total spend" is what the pods billed while the batch held them (per-second billing) plus API calls.
It includes everything that did not become a post: rejected keyframes, rejected clips, best-of-k
spares, model loads, retries after an OOM, and the GPU sitting idle. `ce report` splits it into those
buckets from the execution rows, so every number traces back to a render.

GPU seconds per render come from ComfyUI's own `execution_start` / `execution_success` timestamps,
not from how long the client waited, so a prompt that queued behind another one is not billed twice.

## What one clip costs on a rented GPU

The anchor is the timing note in ComfyUI's official Wan 2.2 14B image-to-video template
(RTX 4090D, 640×640, 81 frames = 5 s at 16 fps, fp8 weights):

| Setting | Warm run | First run |
|---|---|---|
| 20 steps, CFG 3.5 | ~513 s | ~536 s |
| lightx2v 4-step LoRA, CFG 1 | **~71 s** | ~97 s |

480×848 (9:16) has almost the same pixel count as 640×640 (407k vs 410k), so the `standard` profile
lands at ~71 s. On a RunPod community RTX 4090 at $0.34/h that is **$0.0067 per clip render**.
`ce costs` prints the full table; only the 4090 row is tied to a published measurement, other GPUs
use placeholder speeds until `ce bench` replaces them.

| Profile | Clip | Est. s / clip (4090) | Est. $ / clip render |
|---|---|---|---|
| draft | 480×848, 3 s | 41 | $0.0039 |
| standard | 480×848, 5 s | 71 | $0.0067 |
| hero | 720×1280, 5 s | 169 | $0.0160 |
| quality20 (no distillation) | 480×848, 5 s | 510 | $0.0481 |

Keyframes (Z-Image Turbo, 8 steps, ~1 MP) are assumed at ~4 s, about $0.0004 each.

## What the same clip costs hosted

List prices per generated second (sources in `configs/prices.yaml`), 5 s clip, 2 generations per
clip you actually post. Published guides assume 2 to 5 takes per keeper; 2 is the generous end.

| Hosted API | $ / s | Per 5 s generation | Per keeper (×2) |
|---|---|---|---|
| Kling 3.0 720p, no audio | 0.084 | $0.42 | $0.84 |
| Wan 3.0 API 720p | 0.10 | $0.50 | $1.00 |
| Kling 3.0 1080p, no audio | 0.112 | $0.56 | $1.12 |
| Seedance 2.0 Fast 720p | 0.12 | $0.60 | $1.20 |
| Seedance 2.0 720p | 0.15 | $0.75 | $1.50 |
| Kling 3.0 1080p, with audio | 0.168 | $0.84 | $1.68 |
| Seedance 2.5 720p | 0.231 | $1.16 | $2.31 |

## Read the multiple carefully

The simulated demo lands around **a cent per keeper** for the `standard` profile ($0.009-0.011 across
three rounds), ~80× under the cheapest hosted row. That number needs three caveats before anyone
repeats it:

1. **Resolution.** `standard` is 480p; the hosted rows are 720p+. The like-for-like comparison is the
   `hero` profile (720p): ~$0.016 per render, roughly $0.03 per keeper after yield, so ~25-30×.
2. **Quality.** The 4-step distillation trades motion dynamics for speed (ComfyUI's own notes say so).
   Kling and Seedance handle complex motion, longer shots and native audio better. The engine wins on
   formats with simple, repeatable motion, which is most of what volume accounts post; hosted APIs stay
   in the toolbox for hero shots.
3. **Yield.** The simulator's defect rates are invented. Real keyframe and clip pass rates are the
   first thing to measure on a pod, and they move cost per keeper more than anything else.

## The levers, in the order I would pull them

Each one maps to a bucket in `ce report`, and the report computes its counterfactual from the batch's
own rows.

1. **Gate on the keyframe, not the clip.** Rejecting a still costs ~$0.0004; rejecting a clip costs
   ~$0.0067 plus the keyframe it was built on. Identity, head size, exposure, skin and sign text are all
   decided on the still before Wan runs. (Built in.)
2. **Idle GPU → near zero.** Two prompts in flight per GPU so it never waits on a download or a gate;
   phased scheduling so phases don't leave GPUs waiting; pods rented per batch and terminated in a
   `finally`; models on a network volume so a fresh pod boots without pulling ~40 GB. (Built in;
   the ledger shows what is left.)
3. **No model swaps.** On a 24 GB card the keyframe model and Wan's two 13 GB experts cannot all stay
   resident. Render all keyframes, then all clips (phased), or dedicate pods to a stage. (Built in.)
4. **Clip yield.** Most clip rejects are identity morphs and frozen motion. Motion prompts that keep
   the face toward camera and hands away from it, a stricter keyframe identity threshold for formats
   that morph, and at most two clip attempts per keyframe.
5. **Steps and resolution.** 4-step distillation is ~7× cheaper than 20 steps at the same size.
   Render at 480p and upscale only keepers, so rejects never pay for upscaling.
6. **$ per clip, not $ per hour.** Pick hardware by `ce bench` on the actual workflow. A GPU that is
   twice the hourly price and 1.5× faster is the more expensive clip.
7. **Interruptible pods.** Batches are resumable and the pool retries on another pod, so spot capacity
   is usable for everything except the last few renders of a deadline batch.
8. **Next, measured not assumed:** step caching (TeaCache-style), lower-precision Wan weights
   (the LightX2V project publishes FP8/NVFP4 variants), batching latents. Each goes through `ce bench`
   and a split batch before it is trusted.

## What I would measure on day one of the paid test

- `ce bench` warm seconds for `standard` and `hero` on a 4090 and one alternative GPU.
- Keyframe and clip pass rates on 100 real pieces, with 40 of them hand-labelled for `ce gates calibrate`.
- The bucket waterfall from `ce report`, and the halving plan it produces.
