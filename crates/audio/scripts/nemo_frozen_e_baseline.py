#!/usr/bin/env python3
"""Run NVIDIA's NeMo multitalker driver on CPU with the exact Frozen E settings.

This is the NeMo-CPU baseline for the Frozen E port. It writes the NeMo
manifest and reference RTTMs for the frozen clips (each RTTM is checked
against the frozen `rttm_sha256`), then runs the unmodified upstream driver
`examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py`
at NeMo 218d0b4 with the flags of the frozen qualification notebook
(`args_coupled('E')`), changing only paths and `device=cpu`.

Run it with the pinned reference venv:
  crates/audio/scripts/nemo_218d0b4/.venv/bin/python -I \
    crates/audio/scripts/nemo_frozen_e_baseline.py \
    --driver ~/models/frozen-e/nemo-cpu/speech_to_text_multitalker_streaming_infer.py \
    --asr ~/models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --diar ~/models/Nemotron-3-Diarization-f667ed73/Nemotron-3-Diarization.nemo \
    --clips ~/models/frozen-e/clips \
    --references crates/bench/testdata/frozen_e/references.json \
    --out ~/models/frozen-e/nemo-cpu/E
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DRIVER_SHA256 = "7c34077ce6f4286e5ad3f390994683ef957d965bb93e05dea1a51fb2c8b0b56e"
ASR_SHA256 = "afaefe89829e201a0ee22c67a715eb5a467bd06c0704cc1579fbfb370fb5be73"
DIAR_SHA256 = "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d"

# args_coupled('E') from the frozen notebook, minus the path arguments.
FROZEN_E_FLAGS = [
    "max_num_of_spks=4",
    "single_speaker_mode=false",
    "masked_asr=false",
    "parallel_speaker_strategy=true",
    "cache_gating=true",
    "binary_diar_preds=true",
    "att_context_size=[70,13]",
    "fifo_len=264",
    "spkcache_update_period=222",
    "spk_supervision=diar",
    "batch_size=1",
    "precision=32",
    "use_amp=false",
    "generate_realtime_scripts=false",
    "diar_collar=0.0",
    "diar_ignore_overlap=false",
    "print_time=false",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path: Path, want: str) -> None:
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got} != {want}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--asr", type=Path, required=True)
    p.add_argument("--diar", type=Path, required=True)
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ids", default="", help="comma-separated clip ids (default: all)")
    p.add_argument("--extra", action="append", default=[], help="extra key=value override")
    a = p.parse_args()
    driver, asr, diar = (x.expanduser().resolve() for x in (a.driver, a.asr, a.diar))
    clips_dir, out = a.clips.expanduser().resolve(), a.out.expanduser().resolve()
    check(driver, DRIVER_SHA256)
    check(asr, ASR_SHA256)
    check(diar, DIAR_SHA256)

    refs = json.loads(a.references.read_text())
    wanted = [x for x in a.ids.split(",") if x] or [c["id"] for c in refs]
    by_id = {c["id"]: c for c in refs}
    (out / "rttm_ref").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest_E.jsonl"
    with manifest.open("w") as f:
        for cid in wanted:
            c = by_id[cid]
            wav = clips_dir / f"{cid}.wav"
            check(wav, c["audio_sha256"])
            rttm = out / "rttm_ref" / f"{cid}.rttm"
            # Same writer as the frozen notebook cell 4B.
            rttm.write_text(
                "".join(
                    f'SPEAKER {cid} 1 {s["start"]:.3f} {s["end"] - s["start"]:.3f} '
                    f'<NA> <NA> {s["speaker"]} <NA> <NA>\n'
                    for s in c["reference_segments"]
                )
            )
            check(rttm, c["rttm_sha256"])
            text = " ".join(w["text"] for w in c["reference_words"])
            f.write(
                json.dumps(
                    {
                        "audio_filepath": str(wav),
                        "duration": c["duration"],
                        "rttm_filepath": str(rttm),
                        "text": text,
                        "session_id": cid,
                    }
                )
                + "\n"
            )

    overrides = {x.split("=", 1)[0]: x for x in FROZEN_E_FLAGS}
    for x in a.extra:
        overrides[x.split("=", 1)[0]] = x
    argv = [
        sys.executable,
        str(driver),
        f"asr_model={asr}",
        f"diar_model={diar}",
        f"manifest_file={manifest}",
        f"output_path={out / 'seglst.json'}",
        f"diar_output_rttm_dir={out / 'rttm'}",
        "device=cpu",
        *overrides.values(),
    ]
    (out / "command.json").write_text(
        json.dumps(
            {
                "argv": argv,
                "driver_sha256": DRIVER_SHA256,
                "asr_sha256": ASR_SHA256,
                "diar_sha256": DIAR_SHA256,
                "python": sys.version,
            },
            indent=1,
        )
    )
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    start = time.perf_counter()
    with (out / "driver.log").open("w") as log:
        proc = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - start
    (out / "timing.json").write_text(json.dumps({"wall_seconds": elapsed, "clips": len(wanted)}))
    print(f"driver exit {proc.returncode} after {elapsed:.1f}s; log {out / 'driver.log'}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
