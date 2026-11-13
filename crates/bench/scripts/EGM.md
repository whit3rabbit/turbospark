# bench/scripts

## Overview
Dev-only Python helpers for the Frozen E meeting-scoring fixtures: derive the committed text fixtures, rebuild the AMI clips, and score a SegLST with the frozen scorer.

## Interface
- `import_frozen_e_evidence.py`: derives `testdata/frozen_e/` (references, E hypothesis, frozen counts, manifest). Only `evaluation_manifest.json` (clip list) and `E_seglst.json` are read from the evidence dir, and `manifest.json` pins the SHA-256 of all three `SOURCES` (those two plus the qualification notebook); the per-clip counts and matched-six list are literal tables (`FROZEN_E`, `MATCHED_SIX`) in the script [import_frozen_e_evidence.py:FROZEN_E, import_frozen_e_evidence.py:main].
- `prepare_frozen_e_clips.py`: cuts 120 s clips from AMI WAVs and checks each against its `audio_sha256` [prepare_frozen_e_clips.py:main].
- `frozen_e_score.py`: reference cpWER scorer (`tokens`, `edit`, `cpwer`) copied from the frozen notebook [frozen_e_score.py:cpwer].

## Invariants
- Source Mix-Headset WAVs must match `MEETING_SHA256` before cutting, or the script stops at once. An output clip that mismatches prints BAD, its WAV is deleted, and the script stops at that clip with a non-zero exit [prepare_frozen_e_clips.py:sha256, prepare_frozen_e_clips.py:main].
- `frozen_e_score.py` refuses (`SystemExit`) any SegLST `session_id` that is not a manifest clip, matching `seglst::group_by_session`; a manifest clip with no segments still scores as all-deleted [frozen_e_score.py:main].
- Scorer logic is unchanged from the frozen notebook; the Rust scorer in [../src/meeting/EGM.md](../src/meeting/EGM.md) must agree with it [frozen_e_score.py:cpwer].
- No audio is written by the import script [import_frozen_e_evidence.py:main].
- Dependencies: `frozen_e_score.py` needs numpy and scipy; `prepare_frozen_e_clips.py` needs soundfile (its header says to use the pinned venv `crates/audio/scripts/nemo_218d0b4`); `import_frozen_e_evidence.py` is stdlib only [frozen_e_score.py:cpwer, prepare_frozen_e_clips.py:main].

## Side Effects
- Writes JSON fixtures to `--out` [import_frozen_e_evidence.py:main].
- Writes PCM_16 WAV clips to `--out` [prepare_frozen_e_clips.py:main].
- `frozen_e_score.py` only reads and prints.

## Verification
- Run: `cargo test -p turbospark-bench` (Rust scorer against committed counts; the scripts have no own tests)
