#!/usr/bin/env python3
"""Derive crates/bench/testdata/frozen_e/ from the frozen Frozen E evidence.

Reads the evidence files copied from the qualification run
(Drive: TurboSpark_Nemotron_E_Qualification_v3/runs/20261016T120640Z_ce70e6ae)
and writes small, text-only fixtures:

  references.json   per clip: id, meeting, offset, duration, audio/rttm sha256,
                    speakers, reference_words (speaker/start/end/text),
                    reference_segments. Paths and the joined text are dropped.
  E_seglst.json     the frozen E hypothesis, byte for byte.
  frozen_counts.json  per-clip frozen E cpWER counts and the matched-six list.
  manifest.json     sha256 of every source and output file, plus provenance.

No audio is written. Standard library only.

Usage:
  import_frozen_e_evidence.py --evidence ~/models/frozen-e/evidence \
    --out crates/bench/testdata/frozen_e
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# qualification_metrics.json, arm "E".
FROZEN_E = {
    "IS1000a_00510_00630": (25, 239),
    "IS1000a_01241_01361": (54, 347),
    "IS1001a_00390_00510": (28, 286),
    "IS1001a_00634_00754": (38, 364),
    "IS1002b_00880_01000": (47, 466),
    "IS1002b_02053_02173": (15, 415),
    "IS1003a_00320_00440": (31, 370),
    "IS1003a_00446_00566": (30, 288),
    "IS1004a_00140_00260": (2, 219),
    "IS1004a_00608_00728": (33, 326),
    "IS1005a_00440_00560": (27, 160),
    "IS1005a_00692_00812": (25, 334),
}
# dataset_manifest.json of the ONNX parity run: the six clips matched across arms.
MATCHED_SIX = [
    "IS1000a_00510_00630",
    "IS1001a_00390_00510",
    "IS1002b_00880_01000",
    "IS1003a_00320_00440",
    "IS1004a_00140_00260",
    "IS1005a_00440_00560",
]
# Pinned by sha256 in the manifest. Private Drive file ids stay out of the repo.
SOURCES = (
    "evaluation_manifest.json",
    "E_seglst.json",
    "TurboSpark_Nemotron_E_Frozen_Qualification_v3.ipynb",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ev = a.evidence.expanduser()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    clips = json.loads((ev / "evaluation_manifest.json").read_text())
    refs = []
    for c in clips:
        refs.append(
            {
                "id": c["id"],
                "meeting": c["meeting"],
                "source_offset": c["source_offset"],
                "duration": c["duration"],
                "audio_sha256": c["audio_sha256"],
                "rttm_sha256": c["rttm_sha256"],
                "speakers": c["speakers"],
                "reference_words": c["reference_words"],
                "reference_segments": c["reference_segments"],
            }
        )
    assert sorted(r["id"] for r in refs) == sorted(FROZEN_E), "clip set changed"
    (out / "references.json").write_text(json.dumps(refs, indent=1) + "\n")
    shutil.copyfile(ev / "E_seglst.json", out / "E_seglst.json")
    counts = {
        "arm": "E",
        "source": "qualification_metrics.json",
        "matched_six": MATCHED_SIX,
        "clips": {k: {"cpwer_errors": e, "reference_tokens": d} for k, (e, d) in FROZEN_E.items()},
    }
    (out / "frozen_counts.json").write_text(json.dumps(counts, indent=1) + "\n")

    manifest = {
        "description": "Frozen E reference fixtures for the cpWER scorer gate.",
        "provenance": {
            "drive_run": "TurboSpark_Nemotron_E_Qualification_v3/runs/20261016T120640Z_ce70e6ae",
            "scorer": "notebook cell 8A (tokens/edit/cpwer), ported in crates/bench/scripts/frozen_e_score.py",
            "audio_source": "AMI corpus Mix-Headset WAVs; clips rebuilt by crates/bench/scripts/prepare_frozen_e_clips.py",
        },
        "license": "Reference words and segments derive from the AMI Meeting Corpus manual annotations "
        "v1.6.2, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
        "source_sha256": {name: sha256(ev / name) for name in SOURCES},
        "output_sha256": {
            name: sha256(out / name) for name in ("references.json", "E_seglst.json", "frozen_counts.json")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for name, digest in manifest["output_sha256"].items():
        print(digest, name)


if __name__ == "__main__":
    main()
