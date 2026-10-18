#!/usr/bin/env python3
"""Write the Multitalker Parakeet stage goldens from a NeMo Frozen E trace.

Reads a trace directory written by nemo_multitalker_trace.py (NeMo Speech
218d0b4, CPU, Frozen E settings) and writes crates/audio/testdata/multitalker_asr/:

  features.npy        [128, steps * 112] f32 NeMo log-mel for the first steps
  steps.json          per step: chunk geometry, active ids, targets, token ids
                      and global frames, cache lengths, decoded lengths
  step_NNNN_*.npy     per step: pre_encoded [rows, 1024], layer0_input
                      [speakers, rows, 1024], encoded [speakers, 1024, rows]
  kernel_case_*.npy   one two-speaker step (pre_encoded, spk, bg, layer0_input)
                      for the speaker kernels in isolation
  manifest.json       sources and sha256 of every file

Usage:
  generate_multitalker_fixture.py --trace ~/models/frozen-e/nemo-trace/IS1004a_00140_00260 \
    --steps 6 --kernel-step 46 --out crates/audio/testdata/multitalker_asr
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHIFT = 112


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--kernel-step", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    trace = a.trace.expanduser()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (trace / "trace.jsonl").open()]
    feats = np.load(trace / "features.npy").astype(np.float32)

    steps, files = [], ["features.npy", "steps.json"]
    for k in range(a.steps):
        r = rows[k]
        z = np.load(trace / "steps" / f"step_{k:04d}.npz")
        # The ASR chunk is [9 cache frames (zero before 0) ; 112 new frames].
        chunk = z["asr_chunk"]
        want = np.zeros_like(chunk)
        lo = k * SHIFT - 9
        src = feats[:, max(lo, 0) : k * SHIFT + SHIFT]
        want[:, want.shape[1] - src.shape[1] :] = src
        np.testing.assert_array_equal(chunk, want)
        for key in ("pre_encoded", "layer0_input", "encoded"):
            name = f"step_{k:04d}_{key}.npy"
            np.save(out / name, np.ascontiguousarray(z[key].astype(np.float32)))
            files.append(name)
        steps.append(
            {
                "step": k,
                "asr_chunk_frames": r["asr_chunk_frames"],
                "asr_chunk_len": r["asr_chunk_len"],
                "drop_extra_pre_encoded": r["drop_extra_pre_encoded"],
                "pre_encoded_len": r["pre_encoded_len"],
                "active": r["active"],
                "spk_targets": r["spk_targets"],
                "bg_targets": r["bg_targets"],
                "tokens": r["tokens"],
                "token_frames": r["token_frames"],
                "texts": r["texts"],
                "decoded_length": r["decoded_length"],
                "cache_last_channel_len": r["cache_last_channel_len"],
            }
        )
    np.save(out / "features.npy", np.ascontiguousarray(feats[:, : a.steps * SHIFT]))
    (out / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")

    kr = rows[a.kernel_step]
    kz = np.load(trace / "steps" / f"step_{a.kernel_step:04d}.npz")
    assert len(kr["active"]) >= 2, "kernel case must have two active speakers"
    kernel = {
        "pre_encoded": kz["pre_encoded"],
        "spk": np.asarray(kr["spk_targets"]),
        "bg": np.asarray(kr["bg_targets"]),
        "layer0_input": kz["layer0_input"],
    }
    for key, arr in kernel.items():
        name = f"kernel_case_{key}.npy"
        np.save(out / name, np.ascontiguousarray(arr.astype(np.float32)))
        files.append(name)
    meta = json.loads((trace / "meta.json").read_text())
    # Clip name only: the trace records an absolute path on the generating machine.
    meta["wav"] = Path(meta["wav"]).name
    manifest = {
        "description": "NeMo Speech 218d0b4 Multitalker Parakeet goldens (Frozen E geometry)",
        "trace_meta": meta,
        "steps": a.steps,
        "kernel_step": a.kernel_step,
        "license": "Audio from the AMI Meeting Corpus (CC BY 4.0), "
        "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
        "generator": "crates/audio/scripts/generate_multitalker_fixture.py",
        "files_sha256": {n: sha256(out / n) for n in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["files_sha256"], indent=1))


if __name__ == "__main__":
    main()
