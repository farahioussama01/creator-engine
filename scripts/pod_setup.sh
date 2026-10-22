#!/usr/bin/env bash
# Put every model the two workflows need into a ComfyUI install (run on the pod, or once on a
# network volume so later pods boot without downloads). Idempotent: skips files that exist.
#
#   COMFY=/workspace/ComfyUI bash scripts/pod_setup.sh
#   then copy your character LoRAs into $COMFY/models/loras/
#
# Sources are the Comfy-Org repackaged weights linked from ComfyUI's official workflow templates.
# Licences: Wan 2.2 and Z-Image Turbo are Apache-2.0; check each model card before commercial use.
set -euo pipefail
COMFY="${COMFY:-/workspace/ComfyUI}"
M="$COMFY/models"
HF=https://huggingface.co

fetch() {  # folder url
  local dir="$M/$1" url="$2" name
  name="$(basename "$url")"
  mkdir -p "$dir"
  if [[ -s "$dir/$name" ]]; then echo "have   $1/$name"; return; fi
  echo "fetch  $1/$name"
  curl -fL --retry 3 -o "$dir/$name.part" "$url" && mv "$dir/$name.part" "$dir/$name"
}

# Keyframes: Z-Image Turbo (6B, ~8 steps)
fetch diffusion_models "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
fetch text_encoders    "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
fetch vae              "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"

# Clips: Wan 2.2 I2V 14B fp8 (high + low noise experts) + lightx2v 4-step LoRAs
W="$HF/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
fetch diffusion_models "$W/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
fetch diffusion_models "$W/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
fetch loras            "$W/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
fetch loras            "$W/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
fetch vae              "$W/vae/wan_2.1_vae.safetensors"
fetch text_encoders    "$HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

echo "done. character LoRAs go in $M/loras/ (names must match configs/characters/*.yaml)"
