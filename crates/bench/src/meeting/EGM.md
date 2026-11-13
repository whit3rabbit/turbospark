# bench/src/meeting

## Overview
Speaker-attributed meeting transcription scoring (cpWER) for the Frozen E port. The scorer must reproduce the frozen counts exactly before any native result is compared against them (see module doc in [src/meeting/mod.rs]).

## Interface
- `score_panel`, `PanelScore`, `ClipScore`, `ScoreError`, `read_frozen_counts`, `frozen_e_dir`, `MATCHED_SIX` [mod.rs:score_panel, mod.rs:MATCHED_SIX]
- `cpwer`, `tokens`, `edit_distance`, `CpWer`, `TooManySpeakers`, `MAX_SPEAKERS` [cpwer.rs:cpwer, cpwer.rs:MAX_SPEAKERS]
- `Segment`, `RefWord`, `RefSegment`, `Clip`, `FormatError`, `parse_seglst`, `read_seglst`, `parse_references`, `read_references`, `group_by_session` [seglst.rs:parse_seglst, seglst.rs:group_by_session]

## Invariants
- cpWER refuses more than 8 speakers on either side [cpwer.rs:MAX_SPEAKERS, cpwer.rs:TooManySpeakers].
- A clip with no hypothesis segments scores as all deletions [mod.rs:score_panel].
- An unknown requested clip id is an error, not a skip [mod.rs:ScoreError::UnknownClip].
- A hypothesis segment whose session_id is not a reference clip fails scoring with `FormatError`, it is not skipped [seglst.rs:group_by_session].
- The Python scorer refuses a hypothesis segment whose session_id is not a manifest clip, like `group_by_session` here; a clip omitted via `--clips` is not scored, but its segments are still accepted [../../scripts/frozen_e_score.py:main]. Intended to agree count for count with the Python frozen scorer [../../scripts/frozen_e_score.py:cpwer]; the Rust test checks the committed `frozen_counts.json`, not a live Python run [mod.rs:cpwer_reproduces_frozen_e_per_clip_counts].

## Side Effects
- Reads JSON files from disk only [seglst.rs:read_seglst, mod.rs:read_frozen_counts]. No writes. `meeting_eval` reads `frozen_counts.json` with `unwrap_or_default()`, so a missing file blanks the frozen_errors column instead of failing [../meeting_eval_main.rs:score].

## Verification
- Run: `cargo test -p turbospark-bench`
