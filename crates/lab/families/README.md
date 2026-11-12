# Family profiles

One JSON file per model family the lab can measure: the settings known to
pay, the sweeps worth running first, the findings behind both, the open
questions and the traps. `crates/lab/src/family_profile.rs` reads and
validates them and embeds the set into the lab binary. `tslab family` prints
one, `tslab sweep --sweep ID` runs a sweep from one, and `tslab capabilities`
shows which entries apply to an install.

A profile records. It never decides. A `recommended` entry says what the
ledger and the docs found and names the findings that back it. A `finding`
with `doc` provenance is a `reading` copied from a doc table with its
conditions, not a tslab verdict. The verdict rule, the gates and the bounds
stay in code.

## Files

- Text families: `<ModelFamily::as_str>.json`, one per variant
  (`qwen35.json`, `qwen36.json`, `flint.json`, `gptOss.json`, ...).
  `tests/family_profiles.rs` fails when a variant has no file.
- Audio kinds: `<model_type>.json`, one per `AudioKind`
  (`whisper.json`, `minimax_music3.json`, `kokoro.json`, ...).

## Format (`turbospark.lab.family/1`)

```json
{
  "schema": "turbospark.lab.family/1",
  "family": "qwen35",
  "kind": "text",
  "title": "one line",
  "flow": "qwen",
  "installs": [{"name": "qwen38-27b-mtp", "quant": "MLX affine 4-bit", "machine": "Apple M4 Max 36 GB", "drafters": ["mtp"], "note": ""}],
  "recommended": [
    {"knob": "int4_mma8", "value": "<unset>", "when": [], "why": "...", "findings": ["qwen35.mma8-plain-decode-2026-10-25"]},
    {"speculation": {"drafter": "mtp", "block": 2}, "when": ["mtp"], "why": "...", "findings": ["..."]},
    {"slots": "16", "residency": "streamed", "when": ["moe"], "why": "..."},
    {"chunk": "off", "when": [], "why": "..."}
  ],
  "sweeps": [
    {"id": "spec_block", "description": "...", "when": ["mtp"],
     "metric": "decode_tok_s", "exactness": "bit_exact",
     "workload": {"prompts": [{"protocol": "medium-review"}], "decode_tokens": 128},
     "baseline": "plain", "candidates": ["mtp2~mtp:2", "mtp4~mtp:4"], "note": ""}
  ],
  "findings": [
    {"id": "qwen35.defaults-mtp2-2026-10-25", "date": "2026-10-25",
     "install": "qwen38-27b-mtp", "machine": "Apple M4 Max 36 GB", "power": "ac", "rev": "762e8f65",
     "workload": "what ran, in words",
     "arms": {"off": {}, "mtp2": {"speculation": {"drafter": "mtp", "block": 2}}},
     "metric": "decode_tok_s",
     "values": {"short": {"off": 21.7, "mtp2": 33.1}},
     "verdict": "reading",
     "provenance": {"kind": "doc", "path": "docs/KAPITSA_KERNELS.md", "heading": "Re-measured on the shipped defaults, 2026-10-25"},
     "conditions": "quiet host, load 1.4-2.4",
     "note": ""}
  ],
  "open_questions": ["..."],
  "traps": ["..."],
  "docs": ["docs/KAPITSA_KERNELS.md"]
}
```

## Field rules (enforced by the reader and its tests)

- `schema` is exactly `turbospark.lab.family/1`. `kind` is `text` or
  `audio`. `family` matches the file name; an audio family is an
  `AudioKind` `model_type`, a text family a `ModelFamily::as_str` value.
- `recommended[]` sets exactly one of `knob` plus `value`, `speculation`,
  `slots` and/or `residency`, or `chunk`. A `knob` names a `KNOBS` row in
  `src/knobs.rs` and its `value` is one of that row's `values`. `why` is not
  empty. `findings` name ids in the same file. `when` holds coverage tags
  (`tslab capabilities`): the set in `family_profile::TAGS` plus
  `family:`, `flow:`, `bench:` and `audio:` prefixes. `!t` negates and `a|b`
  is either.
- `sweeps[].id` is `[a-z0-9_]`, unique in the file. `baseline` and every
  `candidates[]` entry parse as an arm
  (`NAME[@SLOTS/RES][~DRAFTER:BLOCK[a]][:KEY=V,...]`) with distinct names.
  `metric` is a lab metric and `exactness` a gate class. An audio family's
  sweep uses `audio_run_s`, `rtf` or `peak_bytes`. `workload` is optional: a
  sweep without one takes the command line's.
- `findings[].id` is `<family>.<slug>-<YYYY-MM-DD>` with a `[a-z0-9_-]`
  slug, unique across every profile. `date`, `install`, `machine`,
  `workload`, `metric` and `conditions` are not empty, and `values` holds at
  least one number. `verdict` is `win`, `loss`, `noise`, `inconclusive` or
  `reading`.
- `provenance.kind` is `ledger` (`experiment` is a lab experiment name and
  `record` a `<unix_ms>-<pid>-<n>` ledger id), `doc` (`path` under `docs/`
  or `crates/`, `heading` the section the table sits under), or
  `upstream_reported` (`source` names the engine's file). A `doc` or
  `upstream_reported` finding is a `reading`. Only a `ledger` finding may
  carry `win`, `loss`, `noise` or `inconclusive`.
- Every string is ASCII (the workspace rule).

## Adding a finding

After a sweep or an A/B, copy the verdict record's id and experiment into a
new `findings[]` entry with the install, machine, power source, revision,
workload and conditions the ledger shows, and the numbers the record holds.
Then, if it changes what an agent should set first, add or edit a
`recommended` entry that names it. A doc table that is not in the ledger
goes in as a `reading` with its heading. Numbers never go in an AGENTS.md.

To try a tweaked profile without editing the embedded one, copy the file into
a directory and pass `--dir DIR` to `tslab family` or `tslab sweep`. A file
with the same `family` replaces the embedded profile for that process.
