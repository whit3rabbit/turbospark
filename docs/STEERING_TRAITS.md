# Trait steering: heretic-style presets, extracted on your machine

The question this page answers: can a user pick a BEHAVIOR ("fewer
refusals", "less slop") from a menu and have the engine extract the
direction for it from their own model at run time, the way
[heretic](https://github.com/p-e-w/heretic) offers decensoring, de-slopping,
de-humoring, de-sycophancy, and emotional steering as options -- without
permanently modifying the model?

Yes. The engine has steered at runtime since `docs/OBLITERATION.md` (four
modes, twelve of fifteen families, -1.7% decode at all layers, speculation
compatible); what this page adds is the trait layer on top: built-in
contrastive prompt corpora, a one-command extraction, and a before/after
check. Everything below is a driver over the measured machinery, so this
page is a USER page, not a measurement page -- the numbers live in
OBLITERATION.md and are not repeated here.

## The pieces

| piece | what it is |
|---|---|
| `turbospark-steer` (`crates/cli/src/bin/steer.rs`) | the CLI: `packs`, `extract`, `check` |
| trait packs (`crates/cli/src/bin/steer_data/*.json`) | contrastive prompt corpora + held-out prompts + keyword markers |
| `turbospark-compute::steering_extract` | difference-of-means reduction + the four diagnostic columns (the Rust twin of `scripts/extract_direction.py`'s reduction half) |
| `TURBOSPARK_RESID_CAPTURE` (`crates/runtime/src/resid_capture.rs`) | the per-layer residual capture, re-armed by `reset()` so a corpus walk costs one open per side |

A pack ships PROMPTS, never a direction. The direction is extracted at run
time from the user's own checkpoint, so the provenance rule the app's
steering surface states -- "nothing ships a direction" (`swift/TurboSparkApp`,
`docs/OBLITERATION.md`) -- holds here too: a pack labels a contrast, not a
behavior, and the extracted `.gguf` lands in a file the user chose.

## Usage

```sh
# What traits exist
turbospark steer packs

# Extract: captures 24 + 24 prompts (two opens, one snapshot per prompt at
# the last prompt token), reduces, prints the four ranking columns, writes
# the .gguf. On the 1.1B tinyllama this is under a minute; on a 27B install
# the two opens dominate.
turbospark steer extract --model tinyllama --trait refusals --out ~/vectors/refusals.gguf

# Check: the pack's held-out prompts, generated twice (unsteered, then
# steered), scored by keyword markers. Greedy decoding makes the arms
# reproducible. Custom prompts ride --prompts-file; markers stay the pack's.
turbospark steer check --model tinyllama --trait refusals \
  --vector ~/vectors/refusals.gguf --scale 0.3
```

`--limit N` truncates each side (paired designs keep their pairing);
`--positive-file`/`--negative-file` replace the pack corpora with your own
(one prompt per line, minimum 4 each) while keeping the pack's markers;
`--system-file` sets a system prompt for the check turns. `check` takes
`--mode`, `--scale`, and `--layers START:END`, which mean what the runtime's
`--steering-mode`, `--steering-scale`, and `--steering-layers` mean
(`docs/OBLITERATION.md` has the table).

## In the app

Model detail pane > Steering card > **Extract from this model** opens a
sheet listing the packs. Extract runs `turbospark-steer extract` as a child
process (the binary ships beside the app executable), writes
`<profile>/steering-vectors/<alias>-<trait>.gguf`, and creates or updates
the preset that points at that file, selected and enabled. The card's
Reload prompt applies it. Rules the sheet enforces:

- The model must not be loaded in the app. The child opens its own copy,
  once per side, so a resident model would be in memory twice. The sheet
  offers Unload.
- Cancel terminates the child. The vector is written to a `.partial` file
  and renamed, so a cancelled re-extraction keeps the previous vector
  intact. A killed run's capture directory in the temp dir is swept by the
  next run after a day.
- A trait id the app has no localized copy for is hidden.

The chat composer's Steering menu (beside Guardrails) is the quick switch:
Off, every registered direction (a direction whose shape the selected
model would refuse is listed but cannot be picked), and an Extract from
this model section for each built-in trait not yet extracted for the
selected install, which opens the same sheet with that trait chosen. When
the loaded model is not running the selection yet, the menu offers Reload
to apply. On a family that does not dispatch steering the menu is disabled
and its tooltip gives the engine's reason.

## Using an extracted vector outside the app

The `.gguf` is an ordinary control vector, so every steering surface takes
it:

```sh
turbospark run tinyllama --chat --steering ~/vectors/refusals.gguf --steering-scale 0.3
turbospark serve --model tinyllama --steering ~/vectors/refusals.gguf --steering-scale 0.3
```

`turbospark run` hands its flags to `turbospark-check`; `--steering-mode`,
`--steering-layers`, `--steering-target`, and `--steering-gate` work the
same way, and the FFI open options (`steering` for the path, `steeringMode`, `steeringScale`,
`steeringLayers`, `steeringTarget`, `steeringGate`) are what the app passes.

## Refusals

`extract` refuses rather than writing a misleading vector when the family
does not dispatch steering, when a capture holds fewer snapshots than its
side has prompts (the capture keys on the last prompt token's non-prefill
pass; a family whose generation loop prefills the whole prompt takes no
such pass), when a rendered prompt reaches the capture window, and when
every layer's direction is zero.

## The five packs

| id | positive side | negative side | intended edit |
|---|---|---|---|
| `refusals` | 24 mild dual-use prompts that reliably elicit refusals (lost-key lockouts, security-class framing, fiction requests) | 24 ordinary helpful requests | `ablate` reduces refusal behavior |
| `slop` | the same 14 story prompts forced into cliches and purple prose | the same 14 forced plain | `ablate` reduces flowery prose |
| `humor` | 16 joke-seeking requests | 16 plain informational requests on the same topics | `ablate` reduces forced jokes and puns |
| `sycophancy` | 16 opinion questions under a flattery-conditioned system prompt | the same 16 under a plain factual system prompt | `ablate` reduces praise openers and reflexive agreement |
| `emotions` | 16 grief/anger/loss writing prompts | 16 matched joy/calm prompts | `ablate` softens negative affect; `add` at a negative scale pushes the other way |

(See `turbospark steer packs` for the summary rendered live. The emotion
pack is the signed one.)

## Verification

**Run, 2026-11-01, real `tinyllama` install (862.5 MiB, dense llama family,
22 layers, hidden 2048):**

- `extract --trait refusals`: 24 + 24 captures in two opens; positions
  sanity line 31..45 and 28..35 (varying with prompt length, which is the
  capture keying on the last prompt token); the four columns rank as the
  OBLITERATION page's own extractions do -- raw norm rising monotonically
  with depth to 10.7 at layer 21, effect size flat-ish (0.46 to 1.10,
  peaking at layer 0, the near-embedding artifact that page documents),
  stream share concentrating late (33.6% at layer 21).
- `check --scale 0.3 --max-new 60`: the steered open applied the edit
  (the engine's own line: `steering: ablate at alpha 0.3 over 21 of
  22 layers`), both arms generated 8 prompts, and TinyLlama 1.1B matched no
  refusal markers on EITHER arm -- a 1.1B Zephyr-tuned model barely refuses,
  so the check's "the unsteered arm already matches no markers" guidance
  fired as designed. Judged by the A/B instead: steered text differs from
  unsteered, coherent both arms.
- Null control through the engine's own path: `--steering-scale 0.0`
  remains the exact identity (OBLITERATION.md's Phase 2 arm, unchanged
  machinery).

The stubbornly honest reading: one real install, one trait, 48 prompts.
TinyLlama is too weak a refuser for the refusal pack to show a marker-rate
drop on it. What this run PROVES is the pipeline (capture, reduce, write,
steered open, score); whether a direction is USEFUL on a stronger
checkpoint is exactly what `check` exists to answer per model, per
extraction, and is the user's call to run. OBLITERATION.md's own caution
holds: a structurally valid direction can still be semantically wrong, and
usable alpha is a property of the direction, not the pack's recommendation.
