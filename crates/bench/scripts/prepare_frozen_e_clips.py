#!/usr/bin/env python3
"""Rebuild the Frozen E AMI IS clips and check them against the frozen hashes.

The frozen qualification notebook (TurboSpark_Nemotron_E_Frozen_Qualification_v3,
cell 4B) cut each clip from the AMI Mix-Headset WAV with soundfile: seek to
round(t * sr), read round(120 * sr) frames as float32, write PCM_16. This
script repeats that cut for every clip in the frozen evaluation manifest and
refuses to continue unless each WAV matches its recorded `audio_sha256`.

Inputs (never downloaded here):
  --ami DIR       holds <meeting>.Mix-Headset.wav (AMI corpus, CC BY 4.0)
  --manifest F    frozen evaluation_manifest.json
  --out DIR       destination for <clip id>.wav

Run with the pinned reference venv, so soundfile/libsndfile match:
  uv run --project crates/audio/scripts/nemo_218d0b4 \
    python crates/bench/scripts/prepare_frozen_e_clips.py \
    --ami ~/models/frozen-e/ami \
    --manifest ~/models/frozen-e/evidence/evaluation_manifest.json \
    --out ~/models/frozen-e/clips
"""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf

MEETING_SHA256 = {
    # AMI_PROVENANCE.json from the frozen run.
    "IS1000a": "25c9a48237c1f23bc419702f89234467a1b9efedbac16d9673f52d6a8c1c6a57",
    "IS1001a": "3630cc7415931e4dc91e3cd022cd7b6123057a055e8ac4114bdf010398ed89f0",
    "IS1002b": "2714833e12db9318fdf2f5165e192a0ee3d4a596e858aac7f85c33c7bf5fac9b",
    "IS1003a": "10d768ad3cf53d02aff96057fb80631c1047e79221ee8f3edb3614dfa23507b5",
    "IS1004a": "c37050e46bf3d339e896cc75baf31b191f4a3c491109d6faa44d9d55cd79666b",
    "IS1005a": "cfb67dda357aa1281f610a98ff99d2247c409a9f0bc5fcef2622175ec5c57e6c",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ami", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    clips = json.loads(a.manifest.read_text())
    checked = set()
    failures = []
    for clip in clips:
        meeting = clip["meeting"]
        src = a.ami.expanduser() / f"{meeting}.Mix-Headset.wav"
        if meeting not in checked:
            got = sha256(src)
            if got != MEETING_SHA256[meeting]:
                raise SystemExit(f"{src}: sha256 {got} != frozen {MEETING_SHA256[meeting]}")
            checked.add(meeting)
        t = float(clip["source_offset"])
        seconds = float(clip["duration"])
        dest = a.out.expanduser() / f"{clip['id']}.wav"
        with sf.SoundFile(str(src)) as f:
            f.seek(round(t * f.samplerate))
            chunk = f.read(round(seconds * f.samplerate), dtype="float32", always_2d=True)
            sr = f.samplerate
        # The frozen cut averaged channels and resampled only when needed.
        chunk = chunk.mean(axis=1) if chunk.shape[1] > 1 else chunk[:, 0]
        if sr != 16000:
            raise SystemExit(f"{src}: sample rate {sr}; the frozen panel is 16 kHz source audio")
        sf.write(str(dest), chunk, 16000, subtype="PCM_16")
        got = sha256(dest)
        ok = got == clip["audio_sha256"]
        print(("OK  " if ok else "BAD ") + clip["id"], got)
        if not ok:
            failures.append(clip["id"])
    if failures:
        raise SystemExit(f"{len(failures)} clip(s) do not match the frozen sha256: {failures}")
    print(f"all {len(clips)} clips match the frozen manifest")


if __name__ == "__main__":
    main()
