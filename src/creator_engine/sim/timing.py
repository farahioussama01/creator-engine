"""How long a prompt takes on a GPU, for the simulator and for budget estimates.

Video anchor (ComfyUI's official Wan 2.2 14B I2V template notes, RTX 4090D, 640x640, 81 frames,
fp8_scaled): ~513 s at 20 steps / CFG 3.5 and ~71 s with the lightx2v 4-step LoRA at CFG 1, plus
~26 s extra on the first run while models load. Solving the two runs for a per-denoiser-pass cost
and a fixed overhead gives 12.28 s per pass and 21.9 s of VAE/encoder work at that size.

Everything else here (keyframe speed, GPU speed ratios, swap penalties) is an assumption to be
replaced with measurements from ``ce bench`` on the pod you actually rent.
"""

from __future__ import annotations

ANCHOR_MPF = 640 * 640 * 81 / 1e6  # megapixel-frames of the anchor clip
PASS_SECONDS = 12.28
VIDEO_OVERHEAD = 21.9
ATTN_EXPONENT = 1.1  # attention cost grows a bit faster than token count

KF_SECONDS_PER_STEP_MP = 0.45  # assumption: Z-Image Turbo bf16 on a 4090
KF_OVERHEAD = 0.6

COLD_LOAD = {"keyframe": 20.0, "video": 26.0}
SWAP_RELOAD = {"keyframe": 4.0, "video": 10.0}  # assumption: re-load from system RAM after the other stage


def keyframe_seconds(width: int, height: int, steps: int, gpu_speed: float = 1.0) -> float:
    return (steps * (width * height / 1e6) * KF_SECONDS_PER_STEP_MP + KF_OVERHEAD) / gpu_speed


def video_seconds(
    width: int, height: int, frames: int, steps: int, cfg: float, gpu_speed: float = 1.0
) -> float:
    r = (width * height * frames / 1e6) / ANCHOR_MPF
    passes = steps * (2 if cfg > 1.0 else 1)
    return (passes * PASS_SECONDS * r**ATTN_EXPONENT + VIDEO_OVERHEAD * r) / gpu_speed
