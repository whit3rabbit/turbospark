# Kernel modes

A kernel mode is an alternative path for a kernel that ships beside the default
one and is chosen by one switch. The default path stays byte-identical when the
switch is unset, so a mode can be tried, measured, kept, or dropped without a
re-freeze of anything the default produced.

This page is the contract a mode follows and the way `tslab` prices one before
it is built and checks the price after. It condenses what
[INT4_SMALL_M_MMA.md](INT4_SMALL_M_MMA.md) (its checklist),
[SPEEDUP_PLAYBOOK.md](SPEEDUP_PLAYBOOK.md) and `crates/lab/program.md` already
do, so the next mode starts from one page instead of three.

## The contract

1. **One switch.** `TURBOSPARK_<NAME>`, read once through a `OnceLock` at the
   use site. Default off for a new mode. A promoted lane keeps its switch as a
   kill switch (`=0`), so the old path is still one variable away. Setting it
   before the process starts is enough for the lab. An in-process override
   that exists only for tests is invisible to the lab (see "Process-cached
   switches").
2. **The default is untouched.** With the variable unset, every kernel,
   pipeline and output bit is what it was. A mode that needs a new function
   constant puts it in the pipeline cache key
   (`crates/gpu/src/context/device.rs`, `constants_key`) and has a test that
   fails when the constant is dropped from the key. `gdn_tests.rs` is the
   template: removing the constant made the bit test pass on one pipeline
   twice, and only the separate-cache-entry guard failed.
3. **Proof that it ran.** `foundation::engaged!("<name>")` where the lane
   runs, and an `engagement` entry on the knob row. A run whose stderr
   contradicts its arm's switches is ineligible, so an arm that silently ran
   the old path cannot read as a NOISE verdict. A switch read in a crate that
   cannot print the line yet is a gap, named on its row's comment.
4. **An exactness class, stated, and a test that holds it.** One of:
   - **Bit-exact** against the default. A new test file compares `to_bits()`.
   - **Shared arithmetic.** Different bits from the default, but a row's bits
     do not depend on the rows beside it, so decode and verify can both run
     it and stay bit-exact against each other. Frozen digests and quality
     rows for the family re-freeze once. Needs an error bound against an f64
     reference and a row-invariance test.
   - **Drift-bound.** Only some calls take it, so streams can part from the
     default's. Needs a measured KL bound (`int4_mma8_drift_probe.rs` is the
     shape) and an explicit policy decision, never a quiet default.
   Mutation-check the test once: break the kernel on purpose, see it fail,
   restore.
5. **A bench arm at the install's own shapes.** A `KernelBench` in
   `crates/lab/src/kernel/` that takes shapes from the install, not
   constants, and records `out_fnv` and `row_fnv` under `--hash`. The
   exactness hint `tslab kernel --hash` prints (bit-equal or DIFFERS, row
   invariant or VARIES) is what the spec's `exactness` field is copied from.
6. **A lane row.** `knobs::LANES` names the knob, the bench, the two arms (or
   the one arm and the environment assignment that makes the baseline build),
   and the parity test files. `tests/knob_registry.rs` fails when any of them
   stops existing.
7. **A price before a model run.** `tslab project`, below.
8. **A verdict from the fixed evaluator, then a calibration.** `tslab
   run-experiment` or `tslab ab`, then `tslab calibrate`.
9. **Humans decide the rest.** Flipping a default, re-freezing a digest or a
   `BENCHMARKS.md` row, widening a tolerance, or retiring a mode. A retired
   mode keeps its measurement in the topic doc and its name in the do-not-
   revisit list, so it is not re-proposed on someone else's numbers.

## Pricing a mode before building the rest of it

A per-dispatch time is not a round time. The 8-row tile on the small
projections looked faster in a microbench and a dispatch profile. The clean
wall-clock sweep did not show the gain. Its one-row bench also understated the
loss it caused on plain decode. `tslab project` makes that gap visible before
the model run instead of after it:

```sh
tslab kernel  --model M --bench int4_matmul --rows 1,2,4,8 --hash --experiment E
tslab profile --model M --prompt protocol:long-synthesis --decode 64 --arm base --experiment E
tslab project --knob int4_mma8 --curves E --x 1 --base-from E:base --floor-from E
```

For each of the install's shapes it takes `count * (mode_ms - default_ms)`
at the chosen row count. It adds any dispatches the mode removes or adds, at
the measured per-dispatch price. Then it divides the measured base by the base
plus that delta. Three numbers decide how far to trust the result:

- **coverage**: the benched default kernels as a fraction of the base. A
  projection that explains 40% of a token cannot say what the other 60% does.
- **the floor**: the host's own noise floor (or the evaluator's 2% minimum). A
  projection under it means a model A/B cannot see the effect, and the
  hypothesis belongs to a different bucket.
- **the stand-in share** (`--share-from`, `--block`): the same block priced in
  place. If the benched kernels are far from the block's in-place cost, the
  bench does not explain the block.

A projection cannot see everything. Past misses were made of these:

- Extra dispatches and scratch that the mode's wiring adds, such as
  activation prep and the shared prep buffer.
- Fused paths that step aside when the mode serves one of their matrices.
- A coupled side that must move with it. Decode and verify sharing one
  arithmetic means the side the bench did not time pays too.
- Cache state that a pool of weight copies does not reproduce.

Read `coverage` first, run the A/B, then `tslab calibrate`. `tslab calibrate
--all` shows the bias per bench and arm pair.

## What the first calibrations showed

Four projections have been checked against a model A/B so far
([KAPITSA_KERNELS.md](KAPITSA_KERNELS.md), "Measured: the kernel-modes tooling").
The GDN kernel mode projected within the floor at 8 rows and at one row. The
tile projected plain decode's loss close to the documented figure. The
small-projection lever missed by a large margin until the bench itself was
fixed, because a tiny shape's timed pass was dominated by the command buffer's
fixed cost. Read a projection from a tiny shape as suspect until its per-
dispatch time is compared with a many-copy pool. Four points are too few to
subtract a bias; `tslab calibrate --all` keeps counting.

## Process-cached switches

The gpu crate reads some switches once per process, and a bench calls the
kernels directly. Where the unit test has an in-process override (a
thread-local under `cfg(test)`), the lab cannot use it. Such a lane is
compared across two `tslab kernel` processes that differ in one environment
assignment (`Compare::Process` in `knobs.rs`), and priced with `tslab project
--default-from A --mode-from B`. `gdn_dk128` is the example: run the bench
once with `TURBOSPARK_GDN_DK128=0` for the baseline and once without.

## Modes that exist

Defaults and classes are from `crates/lab/src/knobs.rs`. A "bench" of none
means the lab has no arm for it yet, so it cannot be priced by `tslab project`
until one exists.

| Knob | Env | Default | Class | Bench and arms | Parity | Doc |
|------|-----|---------|-------|----------------|--------|-----|
| `int4_mma8` | `TURBOSPARK_INT4_MMA8` | off | shared arithmetic (`1`), drift-bound (`verify`) | `int4_matmul`: `gemv_per_row` to `mma8` | `dequant_int4_mma8_parity.rs`, `qwen38_decode_verify_parity.rs` | [INT4_SMALL_M_MMA.md](INT4_SMALL_M_MMA.md) |
| `int4_mma8_min_rows` | `TURBOSPARK_INT4_MMA8_MIN_ROWS` | 4096 | routing parameter of `int4_mma8`, not a lane | `verify_ladder` | as `int4_mma8` | [INT4_SMALL_M_MMA.md](INT4_SMALL_M_MMA.md) |
| `batched_gemv` | `TURBOSPARK_BATCHED_GEMV` | auto, on for INT4 | KL-bounded | none yet: `int4_matmul` has the arms, the knob-to-arm mapping is unchecked | `dequant_int4_gemm_parity.rs` | [ENV.md](ENV.md) |
| `prefill_steel` | `TURBOSPARK_PREFILL_STEEL` | on | KL-bounded | none yet: which Steel tile it toggles is unchecked | `dequant_int4_steel_parity.rs` | [BATCHED_PREFILL.md](BATCHED_PREFILL.md) |
| `gdn_lazy_tape` | `TURBOSPARK_GDN_LAZY_TAPE` | on | bit-exact | `gdn_delta`: `prefill_eager` to `prefill_lazy` (plus `state_fold`) | `gdn_lazy_fold.rs` | [KAPITSA_KERNELS.md](KAPITSA_KERNELS.md) |
| `gdn_dk128` | `TURBOSPARK_GDN_DK128` | on | bit-exact | `gdn_delta`: `prefill_lazy` across two processes | `gdn_tests.rs` (unit module) | [KAPITSA_KERNELS.md](KAPITSA_KERNELS.md) |
| `attn_decode_kernel` | `TURBOSPARK_ATTN_DECODE_KERNEL` | auto: gqa on the measured dense families (dense Llama, Qwen2, Qwen3, Qwen3-VL, Spark 2.5, plus qwen GDN dense), barrier elsewhere, barrier fallback for unserved shapes | KL-bounded | `attn_decode`: `barrier` to `gqa` (kill-arm form on the flipped families: barrier is the candidate) | `attention_decode_gqa_parity.rs` | [SPEEDUP_PLAYBOOK.md](SPEEDUP_PLAYBOOK.md) |
| `qwen_fused_ffn` | `TURBOSPARK_QWEN_FUSED_FFN` | on | bit-exact | none yet | `dequant_int4_fused_parity.rs` | [KAPITSA_KERNELS.md](KAPITSA_KERNELS.md) |
| `qwen_batched_row_ops` | `TURBOSPARK_QWEN_BATCHED_ROW_OPS` | on | bit-exact | none yet | the batched-vs-sequential suites | [SPEEDUP_PLAYBOOK.md](SPEEDUP_PLAYBOOK.md) |

`gdn_dk128` is read in the gpu crate and prints `[engaged] gdn_dk128` where the
baked pipeline is chosen, so an arm with `=0` that still ran the baked build is
ineligible.

## Where a new mode starts

1. Write the hypothesis as a bucket: which block, which shapes, which row
   count, and the base it sits in (`tslab profile --standin`, `--dispatch-
   probe`, `tslab kernel` curves).
2. Add the bench arm. Run it hashed. Read the exactness hint.
3. `tslab project`. If the effect is under the floor or the coverage is low,
   say so in the hypothesis rather than measure around it.
4. Implement behind the switch with the parity test and the engagement line.
5. `tslab workflow mode-candidate --model M --var knob=NAME --var
   candidate=KEY=V` runs the kernel curves, the base profile, the null floor,
   the projection, the A/B and the calibration as one resumable plan.
6. Record the result, including a negative one, in the topic doc with the
   ledger ids.
