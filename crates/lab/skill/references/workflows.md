# tslab workflows

A workflow runs tslab commands in order as one resumable run, holding the
machine lock throughout. Each step calls the same code as the command typed
alone, and lands in the ledger the same way.

```sh
tslab workflow --list
tslab workflow --print new-family                 # the plan as JSON
tslab workflow new-family --model M --experiment E --dry-run
tslab workflow new-family --model M --experiment E --json
tslab workflow new-family --model M --experiment E --step kernels
tslab workflow new-family --model M --experiment E --resume <run>
```

## Built-in plans

| Plan | Runs | Needs |
|---|---|---|
| `new-family` | capabilities, inventory, profile, hashed kernels, null floor, dispatch probe, speculative round costs, verify ladder, tree replay, report | `--model` |
| `kernel-candidate` | hashed kernels, `hash-diff` against a base-build run, null, paired A/B of a switch | `--var against=EXP --var candidate=KNOB=V` |
| `ane-sweep` | hashed `ane_matmul` curves with joules, `ane-plan` placement of the same programs, a report (a `tslab` built with `--features ane-prefill`) | `--var programs=DIR` (from `gen_ane_sweep_programs.py`); `--var tokens=` for the token counts |
| `mode-candidate` | hashed bench curves, a base profile, null floor, `project` from those, paired A/B of the mode's switch, `calibrate` of the projection against that verdict | `--var knob=NAME --var candidate=KEY=V` (a knob with a lane in `tslab knobs`); `--var bench=` and `--var x=` for a lane that is not `int4_matmul` at one row |
| `spec-depth` | chain capture, MTP arms at two blocks with round stats, depth replay priced from the ledger | an install with an MTP head |
| `family-sweep` | capabilities, the family profile, a null floor, every profile sweep whose `when` holds, report | `--model`; the profile's sweeps (`tslab family --model M`) |
| `audio-survey` | capabilities, the family profile, hashed audio kernel benches, profile of a job (music or speak, by the install's task), null floor | an audio install; `--var job=PATH` or `--var speak_job=PATH` to change the job |
| `report` | export (JSON, paths redacted) and HTML beside the ledger | `--experiment` |

A step whose `when` fails for the install is skipped, and the reason is
ledgered. That is how one plan serves every family: run `--dry-run` first to
see which steps will run.

## Plan format (`--plan PLAN.json`)

```json
{
  "name": "my-plan",
  "description": "one line",
  "requires": ["candidate"],
  "vars": {"prompt": "protocol:long-synthesis"},
  "steps": [
    {"id": "caps", "command": "capabilities", "args": {"model": "{model}"}},
    {"id": "probe", "command": "profile", "when": ["probe"], "continue_on_error": true,
     "args": {"model": "{model}", "experiment": "{experiment}", "prompt": "{prompt}",
              "decode": "64", "dispatch-probe": "0,4,8,16"}},
    {"id": "ab", "command": "ab", "needs": ["probe"],
     "args": {"model": "{model}", "experiment": "{experiment}", "baseline": "base",
              "candidate": "cand:{candidate}"}}
  ]
}
```

- `args`: `--key value`. `true` is a bare flag, an array repeats the flag,
  `false` or null omits it. Strings take `{var}` placeholders. An unknown
  placeholder is an error, never passed through.
- Built-in variables: `model`, `experiment` (defaults to the plan name),
  `lab` (the results directory), `lab_src` (the folder holding `workloads/`)
  and `run`. Add your own with `--var KEY=V`.
- `when`: every listed tag must hold. `!tag` negates and `a|b` means either.
  Tags come from `tslab capabilities` (`text`, `audio`, `moe`, `dense`, `gdn`,
  `mla`, `sinks`, `sliding_attention`, `full_attention`, `compressed_attention`,
  `vision`, `drafter`, `mtp`, `dflash`,
  `chunked_prefill`, `steel`, `ane` (dense qwen from a build with the
  `ane-prefill` feature), `probe`, `standin`, `runs`, `profile`,
  `bench:<name>`, `family:<name>`, `flow:<name>`, `audio:music`,
  `audio:transcribe`, `audio:speak`). A plan with no `--model` has only
  `nomodel`.
- `needs`: earlier step ids. A step whose need failed or was skipped is
  skipped.
- A failed step stops the run unless it sets `continue_on_error`.
- Exit code: the worst step's, in the order error, discard, inconclusive.
