"""ComfyUI API-format workflow templates.

Templates are graphs exported with "Save (API)" in ComfyUI. Values to fill at run time are written
as ``"{{name}}"``. A placeholder that is the whole string is replaced by the typed value (int, float),
so ``"steps": "{{steps}}"`` becomes ``"steps": 4``; a placeholder inside a longer string is
substituted as text.

``preflight`` checks a template against a live server *before* any GPU time is spent: every node
class exists, every required input is wired, and every model file the loaders reference is present.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

# Loader node -> (input name, models folder) used to list the files a graph needs.
MODEL_INPUTS: dict[str, list[tuple[str, str]]] = {
    "UNETLoader": [("unet_name", "diffusion_models")],
    "CLIPLoader": [("clip_name", "text_encoders")],
    "VAELoader": [("vae_name", "vae")],
    "LoraLoaderModelOnly": [("lora_name", "loras")],
    "LoraLoader": [("lora_name", "loras")],
    "CheckpointLoaderSimple": [("ckpt_name", "checkpoints")],
    "UpscaleModelLoader": [("model_name", "upscale_models")],
}

OUTPUT_NODES = {"SaveImage", "SaveVideo", "VHS_VideoCombine", "SaveAnimatedWEBP", "SaveWEBM"}


class TemplateError(ValueError):
    pass


def load_template(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        graph = json.load(fh)
    if not isinstance(graph, dict) or not all(
        isinstance(v, dict) and "class_type" in v for v in graph.values()
    ):
        raise TemplateError(f"{path}: not an API-format ComfyUI graph (use 'Save (API)' in ComfyUI)")
    return graph


def placeholders(graph: dict[str, Any]) -> set[str]:
    found: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, str):
            found.update(PLACEHOLDER.findall(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(graph)
    return found


def fill(graph: dict[str, Any], values: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """Return a copy of ``graph`` with every placeholder replaced."""
    missing = placeholders(graph) - set(values)
    if strict and missing:
        raise TemplateError(f"missing values for placeholders: {sorted(missing)}")

    def sub(v: Any) -> Any:
        if isinstance(v, str):
            whole = PLACEHOLDER.fullmatch(v.strip())
            if whole and whole.group(1) in values:
                return values[whole.group(1)]
            return PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), m.group(0))), v)
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            # keep node links like ["4", 0] untouched
            return [sub(x) for x in v]
        return v

    out = sub(copy.deepcopy(graph))
    # ComfyUI ignores _meta, but keep graphs small on the wire
    return {nid: {k: val for k, val in node.items() if k != "_meta"} for nid, node in out.items()}


def output_nodes(graph: dict[str, Any]) -> list[str]:
    return [nid for nid, node in graph.items() if node["class_type"] in OUTPUT_NODES]


def required_models(graph: dict[str, Any]) -> dict[str, set[str]]:
    """Folder -> filenames referenced by loader nodes (placeholders excluded)."""
    need: dict[str, set[str]] = {}
    for node in graph.values():
        for input_name, folder in MODEL_INPUTS.get(node["class_type"], []):
            val = node.get("inputs", {}).get(input_name)
            if isinstance(val, str) and not PLACEHOLDER.search(val):
                need.setdefault(folder, set()).add(val)
    return need


def validate_against_object_info(graph: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    """Problems that would make ComfyUI reject the prompt. Empty list means OK."""
    problems: list[str] = []
    for nid, node in graph.items():
        cls = node["class_type"]
        info = object_info.get(cls)
        if info is None:
            problems.append(f"node {nid}: class {cls!r} not installed on the server")
            continue
        required = (info.get("input") or {}).get("required") or {}
        inputs = node.get("inputs", {})
        for name in required:
            if name not in inputs:
                problems.append(f"node {nid} ({cls}): missing required input {name!r}")
        for name, val in inputs.items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] not in graph:
                    problems.append(f"node {nid} ({cls}): input {name!r} links to missing node {val[0]}")
    if not output_nodes(graph):
        problems.append("graph has no output node (SaveImage / SaveVideo)")
    return problems
