# First real run on a rented GPU

Everything else in the repo runs without a GPU. This is the path to real renders and real numbers.
Budget: roughly **$5-15 of RunPod credit** and an afternoon. You run `ce` on your own machine; it
talks to ComfyUI on the pod over RunPod's HTTPS proxy.

| Phase | Where | Time | Cost |
|---|---|---|---|
| 1. Build the character dataset | ComfyUI pod | 1-2 h | ~$1 |
| 2. Train the LoRA | AI Toolkit pod | ~1-2 h | ~$1-2 |
| 3. Pod setup, preflight, benchmark | ComfyUI pod | ~30 min | ~$0.30 |
| 4. First batches (40, then 200 pieces) | 1-2 ComfyUI pods | 2-4 h | ~$2-5 |

Stop or terminate pods whenever you step away. A network volume (~60 GB, about $4/month) keeps
the models between sessions; delete it when you are done.

## 0. Local setup

```bash
git clone https://github.com/farahioussama01/creator-engine && cd creator-engine
python -m venv .venv && source .venv/bin/activate      # Windows: use WSL
pip install -e ".[dev,vision,ocr,judge]"
ce models download                                     # YuNet + AuraFace for the identity gates
cp configs/engine.gpu.yaml configs/engine.local.yaml
export CE_CONFIG=configs/engine.local.yaml
```

## 1. Build the character dataset (ComfyUI pod)

A LoRA learns whatever face the dataset shows, so the dataset has to be one consistent person
before training.

1. RunPod → Pods → Deploy → RTX 4090 → template **ComfyUI** (On-Demand). Open **Connect → HTTP
   Service [Port 8188]**. First boot can take a while.
2. Make a hero portrait: load the Z-Image Turbo template from ComfyUI's template browser and iterate
   on prompt and seed until you have one face you like. Close-up, neutral light, looking at camera.
3. Make variations of that same face with an image-edit model (the **Qwen-Image-Edit 2509** template,
   Apache-2.0): three-quarter and profile views, different outfits, rooms, light, expressions.
4. Keep 15-25 images where it is unmistakably the same person. Drop anything where the face drifted.
5. Download them into `datasets/mila/` locally, then check consistency before spending on training:

```bash
ce identity enroll mila --refs datasets/mila
```

`outliers` should be empty (remove any image it lists) and the `loo_median` should be well above
the suggested threshold. If the set is not consistent, the LoRA won't be either.

## 2. Train the LoRA (AI Toolkit pod)

1. Deploy a pod from the template **AI Toolkit - ostris - ui - official**, with **>100 GB disk**.
   An RTX 5090 does about 3,000 steps in an hour; a 4090 takes longer.
2. New job, model **Z-Image Turbo** (it uses the Z-Image Turbo training adapter; v1 is the default).
3. Upload the dataset. Trigger word `m1la` (any non-word token; it must match `trigger` in
   `configs/characters/mila.yaml`). Captions are optional.
4. Starting settings: 1024 resolution, rank 16, learning rate 1e-4, ~3,000 steps, save a checkpoint
   every 500 steps, sample prompts that put her in your formats' scenes.
5. Download 2-3 checkpoints (e.g. 2000, 2500, 3000 steps). Terminate the pod.

## 3. Pod setup, preflight, benchmark (ComfyUI pod)

1. Deploy the ComfyUI template on an RTX 4090 again (attach a network volume if you have one).
2. In the pod's web terminal:

```bash
git clone https://github.com/farahioussama01/creator-engine /workspace/creator-engine
bash /workspace/creator-engine/scripts/pod_setup.sh   # downloads ~40 GB of models, skips what exists
```

3. Upload your LoRA checkpoints to `ComfyUI/models/loras/` (the script prints where ComfyUI is).
   Name the one you start with `mila_zimage_v3.safetensors`, or edit `lora:` in the character YAML.
4. Put the pod URL (`https://<pod-id>-8188.proxy.runpod.net`) in `endpoints` in
   `configs/engine.local.yaml`, then from your machine:

```bash
ce check https://<pod-id>-8188.proxy.runpod.net
```

This lists any missing node or model file. Z-Image and Wan 2.2 need a recent ComfyUI; if nodes such
as `WanImageToVideo`, `CreateVideo` or `ModelSamplingAuraFlow` are missing, update ComfyUI from the
Manager and restart. Do not run a batch until `ce check` passes.

```bash
ce bench https://<pod-id>-8188.proxy.runpod.net --stage video --profile standard --n 3 --usd-per-hour 0.34
```

Write the measured seconds down. If the 4090 number differs a lot from ~71 s, update
`sim_gpu_speed` in `configs/prices.yaml` so the cost model uses your measurement.

## 4. Pick the checkpoint, then the first batches

**Checkpoint choice.** For each checkpoint, set `lora:` to it and run a 12-piece draft:

```bash
ce plan mila --n 12 --profile draft && ce run <batch> && ce report <batch>
```

Pick the checkpoint with the best identity pass rate *and* varied keepers (the last checkpoint is
often overfit: stiff poses, same background).

**Calibration batch (40 pieces).**

```bash
ce plan mila --n 40 && ce run <batch> && ce dashboard <batch>
```

Open the report, look at keepers and rejects, and write `labels.csv` (`file,label,issue`) for about
40 renders: `good`/`bad`, and which gate should have caught each bad one. Then:

```bash
ce gates calibrate <batch> --labels labels.csv
```

Update the thresholds in `configs/engine.local.yaml` with what it suggests. The skin texture and
motion thresholds in the GPU config are starting guesses; the identity one should come from
`ce identity enroll`. If you have no Anthropic API key, set `judge.enabled: false`.

**The proof batch (200 pieces).** Add a second pod to `endpoints` to halve wall time, then:

```bash
ce plan mila --n 200 && ce run <batch> && ce report <batch> && ce dashboard <batch>
```

## 5. Put the results in the repo

`runs/` is not committed, so copy what proves the claim into `docs/results/`:

- a screenshot of the report (cost per keeper, buckets, identity histogram)
- a contact sheet of 30-50 keepers (one face across many scenes)
- 3-5 keeper clips (small MP4s) and 2-3 rejects with the gate that stopped them
- the `ce bench` output and the `ce report` text

Then replace the simulated screenshot and the "Results" table in the README with these.

## When something breaks

The real-ComfyUI path is exercised against a faithful simulator but had not been run on a GPU before
this. If `ce check`, `ce bench` or a batch fails, keep the full error message: most likely causes
are a ComfyUI version without one of the nodes, a model filename that differs from the template, or
an output format the client doesn't recognise. All three are small fixes.

## Later: unattended

```bash
ce daily mila --n 60 --provision --webhook "$SLACK_WEBHOOK"
```

`--provision` rents pods through the RunPod API (fill the `runpod:` section, with an image that
starts ComfyUI with your models on a network volume), renders, terminates the pods, schedules posts
and pings Slack/Discord. Publishing stays dry-run until `publish.dry_run: false` and each account's
Graph API token is set.
