"""Shared provenance, array validation, and observational resource sampling."""

import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

from z_image_probe import REFERENCES, REVISION


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_environment():
    direct = json.loads(importlib.metadata.distribution("diffusers").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != REFERENCES["diffusers"][1]:
        raise ValueError("Diffusers must be installed from the pinned commit")
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


class Evidence:
    def __init__(self, root, stage, settings):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / (stage + ".json")
        if self.path.exists():
            raise ValueError(f"capture already exists: {self.path}; use a new run directory")
        self.data = {"schema": 1, "stage": stage, "model_revision": REVISION,
                     "references": {k: v[1] for k, v in REFERENCES.items()},
                     "packages": check_environment(), "settings": settings,
                     "arrays": {}, "inputs": {}, "complete": False}
        self.data["tool_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in
                                    ("z_image_capture.py", "z_image_evidence.py", "z_image_probe.py",
                                     "z_image_quantization.py", "z_image_compare.py")}

    def save(self, name, tensor):
        import numpy as np
        source_dtype = str(tensor.dtype)
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
            if str(tensor.dtype) == "torch.bfloat16":
                tensor = tensor.float()
            tensor = tensor.numpy()
        a = np.asarray(tensor)
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite fixture: {name}")
        path = self.root / (name + ".npy")
        np.save(path, a, allow_pickle=False)
        self.data["arrays"][name] = {"file": path.name, "shape": list(a.shape),
                                    "dtype": str(a.dtype), "source_dtype": source_dtype,
                                    "sha256": sha256(path), "bytes": a.nbytes}

    def input(self, manifest):
        path = Path(manifest)
        prior = validate_capture(path)
        if prior["settings"]["prompt"] != self.data["settings"]["prompt"]:
            raise ValueError("conditioning prompt does not match this request")
        if prior["settings"].get("quantized_linears", False) != self.data["settings"].get("quantized_linears", False):
            raise ValueError("conditioning quantization does not match this request")
        if prior["stage"] == "denoise" and prior["settings"] != self.data["settings"]:
            raise ValueError("decode settings do not match denoising request")
        self.data["inputs"][path.name] = sha256(path)

    def finish(self):
        self.data["complete"] = True
        pending = self.path.with_suffix(".pending.json")
        try:
            pending.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
            validate_capture(pending)
            pending.replace(self.path)
        finally:
            pending.unlink(missing_ok=True)


def validate_capture(path):
    import numpy as np
    path = Path(path)
    x = json.loads(path.read_text())
    if not x["complete"] or x["model_revision"] != REVISION:
        raise ValueError("incomplete capture or wrong revision")
    if x["references"] != {k: v[1] for k, v in REFERENCES.items()}:
        raise ValueError("wrong reference revisions")
    expected_inputs = {"encode": set(), "denoise": {"encode.json"},
                       "decode": {"denoise.json"}, "contracts": set()}
    if set(x["inputs"]) != expected_inputs[x["stage"]]:
        raise ValueError("missing or unexpected input provenance")
    for name, digest in x["inputs"].items():
        if sha256(path.parent / name) != digest:
            raise ValueError("input manifest hash mismatch")
        validate_capture(path.parent / name)
    required = {"encode": {"token_ids", "attention_mask", "conditioning"},
                "denoise": {"initial_noise", "final_latents", "timesteps", "sigmas"},
                "decode": {"decoded_pixels"}, "contracts": {"scheduler_updates"}}
    if not required[x["stage"]].issubset(x["arrays"]):
        raise ValueError("missing required fixture")
    if x["stage"] == "denoise":
        count = x["actual_forwards"]
        if count != x["settings"]["steps"] or count != x["scheduler_updates"]:
            raise ValueError("incorrect scheduler/forward count")
        if any(f"latent_{i:02}" not in x["arrays"] for i in range(count)):
            raise ValueError("missing scheduler update")
    for row in x["arrays"].values():
        file = path.parent / row["file"]
        if file.resolve().parent != path.parent.resolve() or sha256(file) != row["sha256"]:
            raise ValueError("fixture path/hash mismatch")
        a = np.load(file, allow_pickle=False)
        if list(a.shape) != row["shape"] or str(a.dtype) != row["dtype"] or a.nbytes != row["bytes"]:
            raise ValueError("fixture shape/dtype/bytes mismatch")
        if not np.isfinite(a).all():
            raise ValueError("non-finite fixture")
    if x["stage"] == "encode":
        ids = np.load(path.parent / x["arrays"]["token_ids"]["file"], allow_pickle=False)
        mask = np.load(path.parent / x["arrays"]["attention_mask"]["file"], allow_pickle=False)
        conditioning = x["arrays"]["conditioning"]
        if ids.shape != (1, 512) or mask.shape != ids.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("invalid conditioning token/mask shape")
        if conditioning["shape"] != [int(mask.sum()), 2560]:
            raise ValueError("invalid conditioning hidden-state shape")
    if x["stage"] == "denoise":
        s = x["settings"]
        shape = [1, 16, s["height"] // 8, s["width"] // 8]
        for key in ["initial_noise", "final_latents"] + [f"latent_{i:02}" for i in range(x["actual_forwards"])]:
            if x["arrays"][key]["shape"] != shape or x["arrays"][key]["dtype"] != "float32":
                raise ValueError("invalid latent shape/dtype")
        if x["arrays"]["timesteps"]["shape"] != [x["actual_forwards"]] or x["arrays"]["sigmas"]["shape"] != [x["actual_forwards"] + 1]:
            raise ValueError("invalid schedule shape")
        trace = x.get("first_step_trace")
        if trace is not None:
            expected_trace = {
                "conditioning": "conditioning",
                "patchification": "patches",
                "noise_refiner": "noise_refiner_output",
                "main_transformer": "main_transformer_output",
                "velocity": "velocity",
                "scheduler_latent": "latent_00",
            }
            if trace != expected_trace:
                raise ValueError("invalid first-step trace declaration")
            patch_count = (s["height"] // 16) * (s["width"] // 16)
            encode = json.loads((path.parent / "encode.json").read_text())
            cap_len = encode["arrays"]["conditioning"]["shape"][0]
            cap_padded_len = (cap_len + 31) // 32 * 32
            trace_shapes = {
                "patches": [patch_count, 64],
                "noise_refiner_output": [1, patch_count, 3840],
                "main_transformer_output": [1, patch_count + cap_padded_len, 3840],
                "velocity": [[1, 16, s["height"] // 8, s["width"] // 8],
                             [16, 1, s["height"] // 8, s["width"] // 8]],
            }
            for name, expected in trace_shapes.items():
                if name not in x["arrays"]:
                    raise ValueError(f"missing first-step trace array: {name}")
                actual = x["arrays"][name]["shape"]
                if name == "velocity":
                    valid = actual in expected
                else:
                    valid = actual == expected
                if not valid or x["arrays"][name]["dtype"] != "float32":
                    raise ValueError(f"invalid first-step trace shape: {name}")
    if x["stage"] == "decode":
        if x["arrays"]["decoded_pixels"]["shape"] != [1, 3, x["settings"]["height"], x["settings"]["width"]]:
            raise ValueError("invalid decoded pixel shape")
        if sha256(path.parent / "image.png") != x["png_sha256"]:
            raise ValueError("PNG hash mismatch")
    return x


def command(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()


def background_snapshot():
    import psutil
    rows = {}
    for proc in psutil.process_iter(["pid", "name", "create_time", "cpu_times"]):
        if proc.pid == os.getpid():
            continue
        info = proc.info
        times = info.get("cpu_times")
        if times is not None:
            rows[(proc.pid, info["create_time"])] = (info["name"], times.user + times.system)
    return time.monotonic(), rows


def background_delta(before, after):
    elapsed = after[0] - before[0]
    rows = [{"pid": key[0], "name": name,
             "cpu_percent": max(0, (value - before[1].get(key, (name, value))[1]) / elapsed * 100)}
            for key, (name, value) in after[1].items()]
    rows.sort(key=lambda row: row["cpu_percent"], reverse=True)
    total = sum(row["cpu_percent"] for row in rows)
    return {"top": rows[:8], "total_cpu_percent": total,
            "quiet": total <= 50 and (not rows or rows[0]["cpu_percent"] <= 20)}


def preflight(require_quiet=True):
    power = command(["pmset", "-g", "batt"])
    if "AC Power" not in power:
        raise RuntimeError("reference measurements require AC power: " + power)
    previous = background_snapshot()
    samples = []
    for _ in range(3):
        time.sleep(2)
        current = background_snapshot()
        samples.append(background_delta(previous, current))
        previous = current
    if require_quiet and not all(row["quiet"] for row in samples):
        raise RuntimeError("reference measurements require quiet background load: " + json.dumps(samples))
    return samples


class Monitor:
    """Instrumented capture timing, not a clean production benchmark."""
    def __init__(self, device):
        self.device = device
        self.rows = []
        self.stop = threading.Event()

    def sample(self):
        import psutil
        import torch
        proc = psutil.Process()
        row = {"elapsed_s": time.monotonic() - self.start, "rss_bytes": proc.memory_info().rss,
               "swap": command(["sysctl", "vm.swapusage"]), "load": list(os.getloadavg()),
               "power": command(["pmset", "-g", "batt"])}
        current = background_snapshot()
        if hasattr(self, "previous"):
            row["background"] = background_delta(self.previous, current)
        self.previous = current
        if platform.system() == "Darwin":
            buf = ctypes.create_string_buffer(1024)
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) == 0:
                row["phys_footprint_bytes"] = ctypes.c_uint64.from_buffer(buf, 72).value
                row["disk_read_bytes"] = ctypes.c_uint64.from_buffer(buf, 144).value
        if self.device == "mps":
            row["mps_live_bytes"] = torch.mps.current_allocated_memory()
            row["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        row["top_cpu"] = command(["ps", "-Ao", "pid,pcpu,comm", "-r"]).splitlines()[:9]
        self.rows.append(row)

    def loop(self):
        while not self.stop.wait(2):
            self.sample()

    def __enter__(self):
        self.start = time.monotonic()
        self.power = command(["pmset", "-g", "batt"])
        if "AC Power" not in self.power:
            raise RuntimeError("reference measurements require AC power")
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()

    def report(self):
        return {"power": self.power, "platform": platform.platform(),
                "elapsed_s": time.monotonic() - self.start, "samples": self.rows,
                "quiet_ac_throughout": all("AC Power" in row["power"] and
                                            row.get("background", {}).get("quiet", True) for row in self.rows),
                "cache_state": "uncontrolled; no cache eviction performed",
                "measurement_kind": "instrumented reference capture"}
