#!/usr/bin/env python3
"""Write the coupled Nemotron-3 diarizer goldens from a NeMo Frozen E trace.

Reads a trace directory written by nemo_multitalker_trace.py (NeMo Speech
218d0b4, CPU) and writes crates/audio/testdata/nemotron_diarization/coupled/:

  features.npy      [128, steps * 112] f32: the ASR log-mel the diarizer saw.
                    Step k's diarizer input is columns [112k, 112k + 112).
  chunk_preds.npy   [steps, 14, 8] f32: NeMo's 80 ms chunk predictions for each
                    step, before the multitalker driver zeroes speakers >= 4.
  state.json        per-step speaker-cache and FIFO lengths after the step.
  manifest.json     sources and sha256 of every file.

The script asserts that each traced diarizer input equals its feature slice,
so the Rust test can feed contiguous features without reconstructing chunks.

Usage:
  generate_coupled_diar_fixture.py --trace ~/models/frozen-e/nemo-trace/IS1004a_00140_00260 \
    --steps 54 --out crates/audio/testdata/nemotron_diarization/coupled
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHIFT = 112  # mel frames per ASR step for att_context_size [70, 13]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--steps", type=int, default=54)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    trace = a.trace.expanduser()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    feats = np.load(trace / "features.npy").astype(np.float32)
    rows = [json.loads(line) for line in (trace / "trace.jsonl").open()]
    assert len(rows) >= a.steps, f"trace has {len(rows)} steps"
    preds, state = [], []
    for k in range(a.steps):
        z = np.load(trace / "steps" / f"step_{k:04d}.npz")
        diar_in = z["diar_input"]  # [frames, 128]
        assert diar_in.shape == (SHIFT, 128), (k, diar_in.shape)
        np.testing.assert_array_equal(diar_in.T, feats[:, k * SHIFT : (k + 1) * SHIFT])
        assert rows[k]["diar_input_len"] == SHIFT and rows[k]["diar_drop_extra"] == 0
        preds.append(z["diar_total_tail"])
        state.append({"step": k, "spkcache": rows[k]["spkcache_len"], "fifo": rows[k]["fifo_len"]})

    np.save(out / "features.npy", np.ascontiguousarray(feats[:, : a.steps * SHIFT]))
    np.save(out / "chunk_preds.npy", np.stack(preds).astype(np.float32))
    (out / "state.json").write_text(json.dumps(state, indent=1) + "\n")
    meta = json.loads((trace / "meta.json").read_text())
    # Clip name only: the trace records an absolute path on the generating machine.
    meta["wav"] = Path(meta["wav"]).name
    manifest = {
        "description": "NeMo Speech 218d0b4 coupled Nemotron-3 diarizer goldens (Frozen E geometry)",
        "trace_meta": meta,
        "steps": a.steps,
        "license": "Audio from the AMI Meeting Corpus (CC BY 4.0), "
        "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
        "generator": "crates/audio/scripts/generate_coupled_diar_fixture.py",
        "files_sha256": {n: sha256(out / n) for n in ("features.npy", "chunk_preds.npy", "state.json")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["files_sha256"], indent=1))


if __name__ == "__main__":
    main()
