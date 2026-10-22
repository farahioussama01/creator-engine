"""A ComfyUI-compatible server that simulates a rented GPU.

It speaks the same HTTP API as ComfyUI (``/prompt``, ``/history``, ``/view``, ``/upload/image``,
``/object_info``, ``/models``, ``/queue``, ``/system_stats``, ``/interrupt``, ``/free``), validates
graphs the way ComfyUI does (unknown node classes, missing inputs, missing model files), executes
prompts one at a time like a single GPU, and pays cold-load and model-swap penalties. Instead of
diffusion it renders procedural images and clips (see ``render.py``) with injected defects.

Execution timestamps in ``/history`` are in *simulated* seconds, so the engine's cost ledger works
exactly as it does against a real pod. ``/system_stats`` carries a ``sim`` block with the simulated
uptime, which the engine uses to bill idle time.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import tempfile
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from aiohttp import web
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from . import timing
from .render import KeyframeScene, identity_vector, render_keyframe, render_video, write_mp4

REQUIRED_INPUTS: dict[str, list[str]] = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type"],
    "VAELoader": ["vae_name"],
    "LoraLoaderModelOnly": ["model", "lora_name", "strength_model"],
    "ModelSamplingAuraFlow": ["model", "shift"],
    "ModelSamplingSD3": ["model", "shift"],
    "CLIPTextEncode": ["text", "clip"],
    "ConditioningZeroOut": ["conditioning"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise",
    ],
    "KSamplerAdvanced": [
        "model",
        "add_noise",
        "noise_seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "start_at_step",
        "end_at_step",
        "return_with_leftover_noise",
    ],
    "VAEDecode": ["samples", "vae"],
    "SaveImage": ["images", "filename_prefix"],
    "LoadImage": ["image"],
    "WanImageToVideo": ["positive", "negative", "vae", "width", "height", "length", "batch_size"],
    "CreateVideo": ["images", "fps"],
    "SaveVideo": ["video", "filename_prefix", "format", "codec"],
}
OUTPUT_NODES = {"SaveImage", "SaveVideo"}
MODEL_INPUTS = {
    "UNETLoader": ("unet_name", "diffusion_models"),
    "CLIPLoader": ("clip_name", "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
    "LoraLoaderModelOnly": ("lora_name", "loras"),
}
DEFAULT_MODELS = {
    "diffusion_models": [
        "z_image_turbo_bf16.safetensors",
        "z_image_bf16.safetensors",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ],
    "text_encoders": ["qwen_3_4b.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
    "vae": ["ae.safetensors", "wan_2.1_vae.safetensors"],
    "loras": [
        "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    ],
}
DEFAULT_RATES = {
    "blur": 0.05,
    "overexposed": 0.03,
    "underexposed": 0.03,
    "waxy": 0.07,
    "identity_drift": 0.08,
    "big_head": 0.04,
    "misspelled_sign": 0.30,
    "morph": 0.12,
    "flicker": 0.04,
    "frozen": 0.04,
    "chaos": 0.03,
}


@dataclass
class _Job:
    prompt_id: str
    number: int
    graph: dict[str, Any]


def _link(graph: dict[str, Any], value: Any) -> dict[str, Any] | None:
    if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
        return graph[str(value[0])]
    return None


def _nodes(graph: dict[str, Any], cls: str) -> list[dict[str, Any]]:
    return [n for n in graph.values() if n.get("class_type") == cls]


class SimComfyServer:
    def __init__(
        self,
        name: str = "sim-gpu",
        gpu: str = "rtx4090",
        gpu_speed: float = 1.0,
        time_scale: float = 0.02,
        defect_rates: dict[str, float] | None = None,
        extra_loras: list[str] | None = None,
        seed: int = 0,
        fail_rate: float = 0.0,
        workdir: str | Path | None = None,
    ):
        self.name, self.gpu, self.gpu_speed, self.time_scale = name, gpu, gpu_speed, time_scale
        self.rates = {**DEFAULT_RATES, **(defect_rates or {})}
        self.models = {k: list(v) for k, v in DEFAULT_MODELS.items()}
        self.models["loras"] += list(extra_loras or [])
        self.seed, self.fail_rate = seed, fail_rate
        self._tmp = tempfile.TemporaryDirectory(prefix=f"{name}-") if workdir is None else None
        base = Path(workdir) if workdir else Path(self._tmp.name)
        self.input_dir, self.output_dir = base / "input", base / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: dict[str, dict[str, Any]] = {}
        self.pending: list[_Job] = []
        self.running: _Job | None = None
        self._counter = 0
        self._number = 0
        self._loaded: str | None = None
        self._warm: set[str] = set()
        self._interrupt = False
        self._t0 = time.monotonic()
        self._busy_real = 0.0
        self._busy_sim = 0.0
        self.executed = 0
        self._wake: asyncio.Event | None = None
        self._worker: asyncio.Task | None = None
        self._runner: web.AppRunner | None = None
        self.url = ""

        app = web.Application(client_max_size=64 * 1024**2)
        app.add_routes(
            [
                web.get("/system_stats", self.h_stats),
                web.get("/object_info", self.h_object_info),
                web.get("/models/{folder}", self.h_models),
                web.get("/queue", self.h_queue),
                web.post("/prompt", self.h_prompt),
                web.get("/history/{pid}", self.h_history_one),
                web.get("/history", self.h_history),
                web.get("/view", self.h_view),
                web.post("/upload/image", self.h_upload),
                web.post("/interrupt", self.h_interrupt),
                web.post("/free", self.h_free),
            ]
        )
        self.app = app

    # ------------------------------------------------------------ lifecycle
    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        self._wake = asyncio.Event()
        self._runner = web.AppRunner(self.app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        sock = site._server.sockets[0]  # type: ignore[union-attr]
        self.url = f"http://{host}:{sock.getsockname()[1]}"
        self._worker = asyncio.create_task(self._work())
        return self.url

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        if self._runner:
            await self._runner.cleanup()
        if self._tmp:
            self._tmp.cleanup()

    def sim_uptime(self) -> float:
        real = time.monotonic() - self._t0
        return (real - self._busy_real) / self.time_scale + self._busy_sim

    # ------------------------------------------------------------ handlers
    async def h_stats(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "system": {"os": "posix", "comfyui_version": "sim", "python_version": "sim"},
                "devices": [
                    {
                        "name": f"cuda:0 {self.gpu} (simulated)",
                        "type": "cuda",
                        "vram_total": 24 * 1024**3,
                        "vram_free": 2 * 1024**3,
                    }
                ],
                "sim": {
                    "uptime_seconds": self.sim_uptime(),
                    "busy_seconds": self._busy_sim,
                    "executed": self.executed,
                    "gpu": self.gpu,
                    "gpu_speed": self.gpu_speed,
                },
            }
        )

    async def h_object_info(self, request: web.Request) -> web.Response:
        info = {
            cls: {
                "input": {"required": {name: ["*"] for name in req}, "optional": {}},
                "output_node": cls in OUTPUT_NODES,
                "name": cls,
            }
            for cls, req in REQUIRED_INPUTS.items()
        }
        return web.json_response(info)

    async def h_models(self, request: web.Request) -> web.Response:
        folder = request.match_info["folder"]
        if folder not in self.models:
            return web.json_response([], status=404)
        return web.json_response(sorted(self.models[folder]))

    async def h_queue(self, request: web.Request) -> web.Response:
        run = [[self.running.number, self.running.prompt_id]] if self.running else []
        return web.json_response(
            {"queue_running": run, "queue_pending": [[j.number, j.prompt_id] for j in self.pending]}
        )

    async def h_prompt(self, request: web.Request) -> web.Response:
        body = await request.json()
        graph = body.get("prompt")
        errors = (
            self._validate(graph)
            if isinstance(graph, dict)
            else {"_": {"errors": [{"message": "no prompt"}]}}
        )
        if errors:
            return web.json_response(
                {
                    "error": {
                        "type": "prompt_outputs_failed_validation",
                        "message": "Prompt outputs failed validation",
                        "details": "",
                        "extra_info": {},
                    },
                    "node_errors": errors,
                },
                status=400,
            )
        self._number += 1
        job = _Job(uuid.uuid4().hex, self._number, graph)
        self.pending.append(job)
        assert self._wake is not None
        self._wake.set()
        return web.json_response({"prompt_id": job.prompt_id, "number": job.number, "node_errors": {}})

    async def h_history_one(self, request: web.Request) -> web.Response:
        pid = request.match_info["pid"]
        return web.json_response({pid: self.history[pid]} if pid in self.history else {})

    async def h_history(self, request: web.Request) -> web.Response:
        return web.json_response(self.history)

    async def h_view(self, request: web.Request) -> web.Response:
        q = request.query
        root = self.output_dir if q.get("type", "output") == "output" else self.input_dir
        target = (root / q.get("subfolder", "") / q.get("filename", "")).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            return web.Response(status=404)
        return web.FileResponse(target)

    async def h_upload(self, request: web.Request) -> web.Response:
        reader = await request.multipart()
        subfolder, filename, data = "", None, None
        async for part in reader:
            if part.name == "image":
                filename = Path(part.filename or "upload.png").name
                data = await part.read()
            elif part.name == "subfolder":
                subfolder = (await part.text()).strip().strip("/")
        if data is None or filename is None:
            return web.Response(status=400, text="no image")
        if ".." in subfolder:
            return web.Response(status=400, text="bad subfolder")
        dest = self.input_dir / subfolder
        dest.mkdir(parents=True, exist_ok=True)
        (dest / filename).write_bytes(data)
        return web.json_response({"name": filename, "subfolder": subfolder, "type": "input"})

    async def h_interrupt(self, request: web.Request) -> web.Response:
        self._interrupt = True
        return web.Response(status=200)

    async def h_free(self, request: web.Request) -> web.Response:
        self._loaded = None
        self._warm.clear()
        return web.Response(status=200)

    # ------------------------------------------------------------ validation
    def _validate(self, graph: dict[str, Any]) -> dict[str, Any]:
        errors: dict[str, Any] = {}

        def err(nid: str, cls: str, msg: str) -> None:
            errors.setdefault(nid, {"errors": [], "class_type": cls, "dependent_outputs": []})[
                "errors"
            ].append(
                {
                    "type": "value_not_in_list" if "not in" in msg else "required_input_missing",
                    "message": msg,
                    "details": msg,
                }
            )

        for nid, node in graph.items():
            cls = node.get("class_type")
            if cls not in REQUIRED_INPUTS:
                err(nid, str(cls), f"node class {cls} does not exist")
                continue
            inputs = node.get("inputs", {})
            for name in REQUIRED_INPUTS[cls]:
                if name not in inputs:
                    err(nid, cls, f"Required input is missing: {name}")
            if cls in MODEL_INPUTS:
                key, folder = MODEL_INPUTS[cls]
                val = inputs.get(key)
                if isinstance(val, str) and val not in self.models[folder]:
                    err(nid, cls, f"Value not in list: {key}: '{val}' not in {folder}")
            if cls == "LoadImage":
                val = inputs.get("image", "")
                if not (self.input_dir / str(val)).is_file():
                    err(nid, cls, f"Invalid image file: {val}")
        if not any(n.get("class_type") in OUTPUT_NODES for n in graph.values()):
            errors["_"] = {"errors": [{"type": "prompt_no_outputs", "message": "Prompt has no outputs"}]}
        return errors

    # ------------------------------------------------------------ execution
    async def _work(self) -> None:
        assert self._wake is not None
        while True:
            if not self.pending:
                self._wake.clear()
                await self._wake.wait()
                continue
            job = self.pending.pop(0)
            self.running = job
            t0 = time.monotonic()
            try:
                await self._execute(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad graph must not kill the GPU worker
                now = int(time.time() * 1000)
                self.history[job.prompt_id] = {
                    "prompt": [job.number, job.prompt_id, job.graph, {}, []],
                    "outputs": {},
                    "meta": {},
                    "status": self._status(
                        job, now, 0.0, "execution_error", {"node_type": "?", "exception_message": repr(exc)}
                    ),
                }
            finally:
                self._busy_real += time.monotonic() - t0
                self.running = None

    async def _execute(self, job: _Job) -> None:
        g = job.graph
        kind = "video" if _nodes(g, "SaveVideo") else "keyframe"
        spec = self._parse_video(g) if kind == "video" else self._parse_keyframe(g)
        if kind == "video":
            seconds = timing.video_seconds(
                spec["width"], spec["height"], spec["length"], spec["steps"], spec["cfg"], self.gpu_speed
            )
        else:
            seconds = timing.keyframe_seconds(spec["width"], spec["height"], spec["steps"], self.gpu_speed)
        if kind not in self._warm:
            seconds += timing.COLD_LOAD[kind] / self.gpu_speed
            self._warm.add(kind)
        elif self._loaded not in (None, kind):
            seconds += timing.SWAP_RELOAD[kind] / self.gpu_speed
        self._loaded = kind

        start_ms = int(time.time() * 1000)
        await asyncio.sleep(seconds * self.time_scale)
        rng = random.Random(
            (spec["seed"] * 1_000_003 + zlib.crc32(json.dumps(g, sort_keys=True).encode()) + self.seed)
            % (2**63)
        )
        status: dict[str, Any]
        outputs: dict[str, Any] = {}
        truth: dict[str, Any] = {}
        if self._interrupt:
            self._interrupt = False
            status = self._status(job, start_ms, seconds, "execution_interrupted")
        elif rng.random() < self.fail_rate:
            status = self._status(
                job,
                start_ms,
                seconds,
                "execution_error",
                {
                    "node_type": "KSamplerAdvanced" if kind == "video" else "KSampler",
                    "exception_message": "CUDA out of memory (simulated)",
                },
            )
        else:
            try:
                save_id, out, truth = await asyncio.to_thread(self._render, kind, spec, g, rng)
                outputs[save_id] = out
                status = self._status(job, start_ms, seconds, "execution_success")
            except Exception as exc:  # noqa: BLE001 - surface as a ComfyUI execution error
                status = self._status(
                    job,
                    start_ms,
                    seconds,
                    "execution_error",
                    {
                        "node_type": "SaveVideo" if kind == "video" else "SaveImage",
                        "exception_message": repr(exc),
                    },
                )
        self._busy_sim += seconds
        self.executed += 1
        self.history[job.prompt_id] = {
            "prompt": [job.number, job.prompt_id, g, {"client_id": "sim"}, list(outputs)],
            "outputs": outputs,
            "status": status,
            "meta": {},
            "sim_truth": truth,
        }

    def _status(
        self, job: _Job, start_ms: int, seconds: float, end: str, extra: dict | None = None
    ) -> dict[str, Any]:
        end_ms = start_ms + int(seconds * 1000)
        msgs = [
            ["execution_start", {"prompt_id": job.prompt_id, "timestamp": start_ms}],
            ["execution_cached", {"nodes": [], "prompt_id": job.prompt_id, "timestamp": start_ms}],
            [end, {"prompt_id": job.prompt_id, "timestamp": end_ms, **(extra or {})}],
        ]
        ok = end == "execution_success"
        return {"status_str": "success" if ok else "error", "completed": ok, "messages": msgs}

    # ------------------------------------------------------------ graph parsing
    def _parse_keyframe(self, g: dict[str, Any]) -> dict[str, Any]:
        latent = _nodes(g, "EmptySD3LatentImage")[0]["inputs"]
        ks = _nodes(g, "KSampler")[0]["inputs"]
        pos = _link(g, ks["positive"]) or {}
        loras = [
            n["inputs"]
            for n in _nodes(g, "LoraLoaderModelOnly")
            if "lightx2v" not in n["inputs"]["lora_name"]
        ]
        save_id = next(nid for nid, n in g.items() if n["class_type"] == "SaveImage")
        return {
            "width": int(latent["width"]),
            "height": int(latent["height"]),
            "steps": int(ks["steps"]),
            "cfg": float(ks["cfg"]),
            "seed": int(ks["seed"]),
            "prompt": pos.get("inputs", {}).get("text", ""),
            "lora": loras[0]["lora_name"] if loras else "none",
            "lora_strength": float(loras[0]["strength_model"]) if loras else 0.0,
            "save_id": save_id,
            "prefix": g[save_id]["inputs"]["filename_prefix"],
        }

    def _parse_video(self, g: dict[str, Any]) -> dict[str, Any]:
        wan = _nodes(g, "WanImageToVideo")[0]["inputs"]
        ks = [n["inputs"] for n in _nodes(g, "KSamplerAdvanced") if n["inputs"].get("add_noise") == "enable"]
        ks = ks[0] if ks else _nodes(g, "KSamplerAdvanced")[0]["inputs"]
        img = (_link(g, wan.get("start_image")) or {}).get("inputs", {}).get("image", "")
        fps = _nodes(g, "CreateVideo")[0]["inputs"]["fps"]
        lightning = any(
            "lightx2v" in n["inputs"]["lora_name"] and float(n["inputs"]["strength_model"]) > 0
            for n in _nodes(g, "LoraLoaderModelOnly")
        )
        save_id = next(nid for nid, n in g.items() if n["class_type"] == "SaveVideo")
        return {
            "width": int(wan["width"]),
            "height": int(wan["height"]),
            "length": int(wan["length"]),
            "steps": int(ks["steps"]),
            "cfg": float(ks["cfg"]),
            "seed": int(ks["noise_seed"]),
            "image": img,
            "fps": int(float(fps)),
            "lightning": lightning,
            "save_id": save_id,
            "prefix": g[save_id]["inputs"]["filename_prefix"],
        }

    EXCLUSIVE = (("overexposed", "underexposed"), ("frozen", "chaos"))

    def _defects(self, names: tuple[str, ...], rng: random.Random, mult: dict[str, float]) -> set[str]:
        found = {d for d in names if rng.random() < min(0.95, self.rates.get(d, 0.0) * mult.get(d, 1.0))}
        for a, b in self.EXCLUSIVE:  # opposite failures cancel out in pixels; keep one
            if a in found and b in found:
                found.discard(b)
        return found

    def _next_name(self, prefix: str, ext: str) -> tuple[str, str]:
        self._counter += 1
        sub = str(Path(prefix).parent) if "/" in prefix else ""
        base = Path(prefix).name
        return sub if sub != "." else "", f"{base}_{self._counter:05d}_.{ext}"

    def _render(self, kind: str, spec: dict[str, Any], g: dict[str, Any], rng: random.Random):
        if kind == "keyframe":
            s = spec["lora_strength"]
            mult = {"identity_drift": 1 + 4 * max(0.0, 0.85 - s), "waxy": 1 + 6 * max(0.0, s - 1.0)}
            m = re.search(r"sign[^\"“]*[\"“]([^\"”]{2,30})[\"”]", spec["prompt"], flags=re.I)
            sign = m.group(1) if m else None
            names = ("blur", "overexposed", "underexposed", "waxy", "identity_drift", "big_head")
            defects = self._defects(names + (("misspelled_sign",) if sign else ()), rng, mult)
            img, truth = render_keyframe(
                KeyframeScene(
                    spec["width"],
                    spec["height"],
                    rng.randrange(2**31),
                    identity_vector(spec["lora"]),
                    s,
                    sign,
                    defects,
                )
            )
            sub, fname = self._next_name(spec["prefix"], "png")
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
            meta = PngInfo()
            meta.add_text("prompt", json.dumps(g))
            meta.add_text("ce_sim", json.dumps(truth))
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
                self.output_dir / sub / fname, pnginfo=meta
            )
            return (
                spec["save_id"],
                {"images": [{"filename": fname, "subfolder": sub, "type": "output"}]},
                truth,
            )

        start = cv2.imread(str(self.input_dir / spec["image"]), cv2.IMREAD_COLOR)
        mult = {"frozen": 1.5 if spec["lightning"] else 1.0, "morph": 1.0 if spec["lightning"] else 0.7}
        defects = self._defects(("morph", "flicker", "frozen", "chaos"), rng, mult)
        frames = render_video(
            start, spec["width"], spec["height"], spec["length"], rng.randrange(2**31), defects
        )
        sub, fname = self._next_name(spec["prefix"], "mp4")
        (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        write_mp4(frames, str(self.output_dir / sub / fname), spec["fps"])
        out = {"images": [{"filename": fname, "subfolder": sub, "type": "output"}], "animated": [True]}
        return spec["save_id"], out, {"defects": sorted(defects)}


async def serve_forever(port: int, **kwargs: Any) -> None:
    server = SimComfyServer(**kwargs)
    url = await server.start(port=port)
    print(f"simulated ComfyUI ({server.gpu}) listening on {url}", flush=True)
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await server.stop()
