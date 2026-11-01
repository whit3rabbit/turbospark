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
`--system-file` sets a system prompt for the check turns. The steering
knobs are the runtime's own (`--steering-mode`, `--steering-scale`,
`--steering-layers` grammar; `docs/OBLITERATION.md` has the table).

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
