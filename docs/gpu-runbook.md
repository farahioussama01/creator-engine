# First real run on a rented GPU

Everything in the repo runs without a GPU against the simulator. This is the path to real renders,
with a budget of roughly $10-20 of RunPod credit for the first character.

## 1. Character LoRA (one-off, ~1 h on a 4090)

- 25-40 reference images of the (fictional) character: varied angles, light, outfits, expressions;
  at least 5 at three-quarter or profile. Same person in every image.
- `ce identity enroll mila --refs path/to/dataset` *before* training: it embeds every image, flags
  outliers (a different-looking face in the set poisons the LoRA) and suggests the identity threshold.
- Train a Z-Image LoRA (e.g. ostris/ai-toolkit, which supports Z-Image) with the trigger from
  `configs/characters/mila.yaml`. Save a checkpoint every 250-500 steps.
- Pick the checkpoint with a draft batch per checkpoint: copy the character YAML with a different
  `lora:` file, plan 12 `draft` pieces each, and compare identity pass rate *and* the variety of the
  keepers. The last checkpoint is often overfit (stiff poses, same background).

## 2. Pod

- RunPod, RTX 4090 (community, ~$0.34/h), any image that runs ComfyUI on port 8188.
- Create a network volume, attach it, then run `COMFY=/workspace/ComfyUI bash scripts/pod_setup.sh`
  once. Later pods mount the volume and boot without downloads.
- Copy the character LoRA into `ComfyUI/models/loras/`.

## 3. Preflight and benchmark

```bash
cp configs/engine.gpu.yaml configs/engine.local.yaml   # set the pod URL, keep secrets in env vars
export CE_CONFIG=configs/engine.local.yaml
ce models download                                     # YuNet + AuraFace for the identity gates
ce check https://<pod-id>-8188.proxy.runpod.net         # nodes, inputs, model files
ce bench https://<pod-id>-8188.proxy.runpod.net --stage video --profile standard --n 3 --usd-per-hour 0.34
```

Put the measured seconds into `configs/prices.yaml` (`sim_gpu_speed`) so the cost model and the
halving plan use real numbers.

## 4. First batch (~$1-3)

```bash
ce plan mila --n 40 --profile standard
ce run <batch-id>
ce report <batch-id>
ce dashboard <batch-id>
```

Label 40 renders and run `ce gates calibrate <batch-id> --labels labels.csv`. Adjust thresholds,
then run a second batch. Commit the report and the contact sheet: that is the proof that one face
holds across hundreds of pieces.

## 5. Unattended

```bash
# rents pods, renders, gates, terminates pods, schedules posts, writes the report, pings a webhook
ce daily mila --n 60 --provision --webhook "$SLACK_WEBHOOK"
```

Cron it (or a scheduled Claude Code run with the `/daily-run` skill for a run that also reviews its
own rejects). Publishing stays dry-run until `publish.dry_run: false` and each account's Graph API
token is set.
