# Non-MoE kernels and the Metal host path (KERN)

*   **Author:** whit3rabbit
*   **Status:** Draft (revised after EGM review, 2026-11-26)
*   **Date:** 2026-11-26

--------------------------------------------------------------------------------

## Overview

This design covers the 30 KERN entries: attention, TurboQuant KV, GDN, dense and
GGUF multiplies, command-buffer handling and dispatch safety.

Ownership: this document owns the idle-stall probe (KERN-05) and spin-poll
(KERN-06). `speedup-moem.md` (MOEM-09) shares the probe, and MOEM-10 there is a
GPU clock keep-alive that overlaps Alternative A. The DFlash2 GPU selector
(KERN-27) belongs to SPEC-09.

State on this branch (turbospark#330, rebased on main after #329 merged):

*   KERN-01 built: every dispatch checks its threadgroup size and panics when
    over the pipeline limit. Step 0 moves the server path to a returned error
    (decision 5).
*   KERN-11 built: the DFlash2 context-KV write batches its norm and RoPE
    dispatches. It was not run on a DFlash2 install and not timed.
*   KERN-04 priced, not built. The price (Q4_K at 0.30 to 0.36 of the INT4
    rate, Q8_0 at 0.28 to 0.36) was measured on zero weights. It is not valid
    under this document's bench rule until re-measured on real weights.
*   KERN-05: one reading from the probe in turbospark#329, on one install
    (Inkling 2 Mini). The first step after 2 s or more of idle is about 200 ms
    slower than the 42 ms step after a 0.5 s gap. The cause is not identified.

## The Problem

*   **GGUF decode kernels are untimed on real weights in the engine.** The
    zero-weight price suggests a gap but does not establish it. The bench rule
    (section 3) must confirm it. The entry's bar to proceed is two parts: the
    current kernel under 75% of INT4 on the same real bytes, and a new arm at
    1.25x or better on the same bytes. Neither is measured yet.
*   **TurboQuant attention is still on the old kernel,** one threadgroup per
    query head with barriers per position. The FP16 path gained 1.24x to 1.76x
    from its rewrite. The TurboQuant ratio on the same layer is not yet measured.
*   **Gemma 4 has no shared-KV decode attention,** because that path covers
    head_dim 128 and 256 and Gemma uses 512. Its share of a token is not
    measured.
*   **Reordered GDN kernels cannot be judged by the current bench.** Its only
    pass rule is bit equality, so every reordering idea is rejected without a
    timing result.
*   **Several host-path ideas may have low ceilings.** One figure: host encode
    is 0.65 ms of about 50 ms per token on the dense 27B. Neither figure is
    linked to a run yet.

## Goals

*   Find the cause of the idle stall (KERN-05) before building a fix. Record the
    result either way (section 1).
*   Build wider-unit GGUF multiply kernels only if a bench arm reaches 1.25x on
    real bytes, with a token's bits not depending on the number of rows per
    dispatch.
*   Price TurboQuant attention (KERN-02) and Gemma head_dim 512 (KERN-03) with
    bench arms before touching the engine. KERN-02 builds if the TurboQuant
    kernel takes over 5% of decode at 16k, and passes at 1.3x. KERN-03 builds if
    attention takes over 5% of a token at 16k, and passes by gaining at least 3%
    decode tok/s.
*   Add the FP64 reference helper and oracle (KERN-13). Prefix-difference
    profiling (KERN-12) is deferred (decision 4).
*   Run KERN-11 on a DFlash2 install: equal accepted stream and equal written K
    rows.
*   No decode-numeric change lands before the decode gate (LAB-01, section
    "Gates").

## Non-Goals

*   Metal 4 tensor API for prefill or K-quant GEMM (KERN-28, declined).
*   GDN fused prework, ReplaySSM and snapshot elision (KERN-20, declined on an
    estimate of under 2% per dispatch that is not yet measured).
*   Chunked WY GDN prefill, closed in `ROADMAP.md` item 15.
*   Kernels for families that are not registered (KERN-29 waits).
*   Taking a measured default from one family to another without its own
    quality, memory and paired speed gates.

## Open Questions

### Owner decisions (answered 2026-11-26)

1.  **Bench weight source (answered: find a bench).** Q8_0 bytes come from the
    local file `~/models/gguf-inkling2-dspark/dspark-Inkling-2-Mini-Q8_0.gguf`
    (504 MB, 27 Q8_0 tensors with Inkling 2 Mini shapes), available now. Q4_K bytes
    come from `~/models/gguf-bench/Qwen_Qwen3-8B-Q4_K_M.gguf` (5,027,784,224 bytes,
    downloaded from `bartowski/Qwen_Qwen3-8B-GGUF`). Its tensors are 217 Q4_K, 37
    Q6_K (the output head, reported separately), and 145 F32 norms. The Unsloth
    Inkling-Small UD-Q4_K_M file (163 GB, a different model) cannot be the bench.
2.  **TurboQuant default for `--kv-bits` (answered: amend CLAUDE.md).** AGENTS.md
    (the file CLAUDE.md links to) now carves out one default that applies to every
    family accepting the flag. That default is decided once, by the decode gate and
    the TurboQuant layer-kernel gate, run on each family that accepts the flag.
    Family-specific paths keep their own verdicts. The default is still gated: it
    does not flip until those gates pass.
3.  **Partial outcome (answered: lab only).** A result that cuts the stall by at
    least half but misses the gate stays a lab-only arm, with no production knob.
4.  **Prefix-difference tooling (answered: defer).** Section 2.
5.  **KERN-01 panic (answered: address the underlying issue).** Step 0 moves the
    server path to a returned error, so a thread-limit violation reaches the caller
    as `Err` instead of a panic. The unchecked commit still panics, and Step 0
    lists the callers that must move to the checked form.
6.  **Unsafe boundary (answered: yes).** The existing `#[allow(unsafe_code)]`
    precedent for raw Objective-C in `crates/gpu/src/context/pass.rs` covers the
    new module in section 1.
7.  **Gemma trigger context (answered: 16k).** Same length as KERN-02.

### Other questions

*   **Is a lossy opt-in kernel wanted at all?** KERN-24 spends accuracy for
    speed. Its threshold is in section 5.

## Critical User Journeys (CUJs)

### CUJ 1: user runs a GGUF Q4_K_M install

*   **Given**: a dense GGUF install with real weights.
*   **When**: a wider-unit kernel ships after its bench rule and decode gate pass.
*   **Then**: decode tok/s rises by the gain measured on that install, the decode
    gate passes, and the current kernel can be selected with
    `TURBOSPARK_GGUF_KERNEL=current`.

### CUJ 2: app runs on a Mac with a lower thread limit

*   **Given**: a chip whose compiled kernel allows fewer threads than requested.
*   **When**: the dispatch is encoded.
*   **Then** (target): the run stops with an error that names the kernel and both
    numbers, and produces no output. Step 0 makes this the server path's behavior.
    Until it lands, the guard panics.

## Technical Plan (Architecture)

Order: the cause test for the idle stall first, then bench-only pricing, then
kernels behind the decode gate.

```text
0  thread-limit guard on the server path (KERN-01)       build here
1a idle-stall cause test (touch and power-state arms)   needs only #329
1b residency set                                         only if 1a supports it
2  FP64 oracle helper                                     build here
3  bench arms (GGUF first, then attention)                bench files present
4  kernels                                                each needs its bench win AND the decode gate
```

Preconditions: turbospark#329 is on main (merged 2026-11-26). This branch
(turbospark#330) overlaps `crates/gpu/src/context/pass.rs` and `dispatch.rs`, so
rebase on main before Step 1a.

### Gates

*   **Decode gate (LAB-01).** The teacher-forced decode gate defined with the lab
    tooling in turbospark#329 (`docs/SPEEDUP_PLAYBOOK.md` line 1796). Any change to
    decode numbers must pass it before merge. Its KL bound is pooled and
    floor-relative. The absolute 1e-3 bound in the playbook was replaced
    (playbook line 282), and this document does not restate a KL number.
*   **Idle-stall outcome table.** Defined in section 1.
*   **GGUF bench rule.** On the same real bytes: current kernel under 75% of INT4
    in GB/s, and an arm at 1.25x or more in GB/s. GB/s is the metric because
    weights per second mixes format size into the ratio.
*   **Paired speed gate.** Any measured default needs its own family gate
    (CLAUDE.md). The pair count and protocol come from `docs/BENCHMARKS.md`.

### 0. Thread-limit guard on the server path (KERN-01)

Status: every dispatch is checked against the pipeline limit (`pass.rs`), and a
violation panics. `commit_and_wait_checked` (`crates/gpu/src/context/pass.rs:209`)
returns `Err(GpuError::InvalidInput)` when a command buffer did not complete.
`commit_and_wait` (`pass.rs:204`) returns nothing.

Design (decision 5):

*   Record a thread-limit violation in `PassEncoder` instead of panicking, and skip
    the dispatch.
*   `commit_and_wait_checked` returns `Err` naming the kernel and both numbers when
    a violation was recorded.
*   The unchecked commits abort on a recorded violation. This still panics, so the
    panic is narrowed to unchecked callers, not removed.
*   Server-reachable callers move to the checked form. The runtime callers in
    `crates/runtime/src/qwen3_asr_metal.rs`, `nemotron_diar_metal.rs` and
    `multitalker_metal.rs` already use it. The rest are listed by a search of
    `crates/runtime`, `crates/ffi` and `crates/server` for `commit_and_wait()` and
    `commit()` before merge.

Test: a fixture with a lowered limit returns `Err` through the checked commit and
does not panic. The unchecked path aborts on the recorded violation.

Built (guard half): `PassEncoder` records the first violation and skips that
dispatch. `commit_and_wait_checked` and a new non-waiting `commit_checked`
return `Err` without committing, so outputs keep their old contents. `commit`,
`commit_and_wait`, `commit_and_wait_with_gpu_time` and `commit_signaling` panic
on a recorded violation. The standalone helpers in `context/dispatch.rs` still
panic before encoding. Tests are in `pass.rs`; the checked-commit test was
mutation-checked.

Survey (2026-11-26, `rg '\.commit_and_wait\(\)|\.commit\(\)'`): about 126
unchecked sites in about 70 files under `crates/runtime/src` (84 waiting, 42
non-waiting) and 2 in `crates/image`; none in `crates/ffi` or `crates/server`,
which reach the GPU only through the runtime.

Migrated (caller half): every non-test runtime site moved to `commit_checked()`,
either `pass.commit_checked()?` or `pass.commit_checked()?.wait()`, with the
module's existing error mapping (`RealForwardError::Gpu`, `to_string`, or
`gpu_err`). This keeps device-fault handling as before: the wait still warns
and does not fail, so only the thread-limit violation changes from a panic to
`Err`. `commit_and_wait_checked` would also have turned faults into errors,
which is a separate decision. Left unchecked: test modules (`ane_split.rs`,
`moonshine.rs`, `llama/attn_tests.rs`) and `crates/image`, whose
`commit_deferred` returns a bare handle to many callers and is outside this
step's scope. Those still panic.

### 1. Idle stall (KERN-05)

Evidence (turbospark#329, one reading, host timed, Inkling 2 Mini UD-Q4_K_XL):
after 0.5 or 1 s idle the next step takes 42 ms; after 2, 4 or 8 s the first
command buffer waits 180 to 225 ms and the step takes 220 to 260 ms. The cause is
not identified.

Terms in this section, fixed for the rest of the document:

*   **Baseline:** the step time after a 0.5 s gap, from the same run (42 ms above).
*   **Stall:** the step time after a gap of 2 s or more, minus the baseline
    (about 180 to 220 ms above).
*   **Cut by half:** the stall falls to half its value or less on the same run.

#### Step 1a: cause test first (lab-only arms)

Run with `tslab idle-gap` at 2, 4 and 8 s idle, same install, two fresh processes.

*   **Arm T (touch on wake):** one small GPU write to each resident buffer before
    the first step after a gap. Lab only.
*   **Arm P (power state):** record GPU power and clock state at the first step.
    The sampling tool is not named yet and must be chosen in this step.

Built (2026-11-26, not run): `gpu::touch::IdleProbe` (`crates/gpu/src/touch.rs`)
and `crates/lab/src/gpu_pstate.rs`, wired into the idle-gap probe in
`crates/lab/src/worker.rs`. Arm selection is `TURBOSPARK_LAB_IDLE_PRESTEP=none|wake|touch`.
Three choices differ from the text above:

*   Arm T reads one word per 16 KB page of the resident weights buffer. It does
    not write, so it cannot change decode output. It covers the trunk only:
    mapped experts, KV and scratch are not touched.
*   An added arm W (`wake`) commits one empty dispatch. If W alone removes the
    stall, the cause is GPU power or scheduling state, not paging.
*   Arm P's tool is IOReport `GPU Stats / GPU Performance States` (channel
    `GPUPH`, states `OFF`, `P1` to `P15` on the M4 Max). It needs no sudo and,
    like the energy meter, reads the whole machine. It is recorded on every
    run, not as a separate arm.

The original reading's install (`inkling2-mini-ud-q4kxl`) is no longer on this
Mac. The run uses `inkling2-mini-mlx4.gturbo`, so the baseline is re-measured
there first, with no other Inkling 2 process running.

Reading the result:

*   If arm T brings the after-gap step to 50 ms or less with an equal decode
    digest, paging of resident buffers is the likely cause, and Step 1b is the
    next step.
*   If arm T does not, the residency set is not built on this evidence. Arm P is
    measured next. A failed touch shows only that touching does not help. It does
    not show that a residency set would fail, because a touch may not keep pages
    wired the way a set does.

Result (2026-11-26, `inkling2-mini-mlx4`, one run per arm, AC, load under 2;
ledger `inkling2_loop.idle-gap-cause-2026-11-26`). After a gap of 2 s or more,
the first command buffer waited:

| Arm | Wait after 2, 4, 8 s gap (ms) | Prestep after the gap (ms) |
|---|---|---|
| none | 178, 199, 206 | |
| wake | 194, 191, 194 | 6, 10, 13 |
| touch | 147, 171, 162 | 34, 49, 52 (59,378 pages) |

After 0.5 or 1 s the wait is under 3 ms in every arm. Decode digests are
equal across arms at every gap.

*   Arm T does not bring the after-gap step to 50 ms or less, so **Step 1b is
    not built on this evidence**.
*   Arm W shows the stall is not the GPU waking for any command buffer: an
    empty dispatch runs in 6 to 13 ms after the gap, and the step after it
    still waits about 190 ms. The stall follows the step's own work or the
    resources it binds.
*   The weights touch costs 34 to 52 ms after a gap (1 to 2 ms after a short
    one) and lowers the wait by about the same amount: the trunk's pages are a
    part of the stall, not most of it.
*   Arm P (`GPUPH`) shows the GPU `OFF` for the whole of a 0.5 s gap as well,
    so this counter does not tell the stalled gaps from the fast ones.

Per the order below, this is inconclusive and the step stops here. The next
cause test would touch every buffer the step binds (KV, scratch, expert slots),
not only the weights. That is a design change, so it waits for the owner.

#### Step 1b: residency set (only if Step 1a supports it)

Author decision, 2026-11-26: a residency set through `metal::objc` messages. Its
justification is the Step 1a result, not the earlier reading.

*   New module `crates/gpu/src/residency.rs`, macOS only. A `ResidencySet` owns
    the residency-set object, removes it from the queue, then releases it on drop.
*   Bindings: `metal::objc`, the crate's existing re-export (already used by
    `crates/gpu/src/context/pass.rs`). No new Cargo dependency.
*   Support check at run time: the device answers
    `newResidencySetWithDescriptor:error:` (macOS 15.0 and later). On macOS 14 the
    constructor returns `None`, and the default path is unchanged. The app's floor
    stays `.macOS(.v14)`.
*   Calls: `addAllocation:`, `commit`, `requestResidency`, then `addResidencySet:`
    on the context's command queue.
*   Members: the resident weight buffer (`ResidentGpuWeights::buffer()`), KV
    buffers and long-lived scratch. **Never** a mapped or streamed expert buffer.
    The exclusion is enforced by a test (Step 1b file list).
*   Size guard: compare against `gpu::recommended_max_working_set()`
    (`crates/gpu/src/device_memory.rs:45`). That number is advisory. If it returns
    `None`, the set is not built and the reason is printed.
*   Failure handling: a failed commit or attach is logged, the set is dropped, and
    decoding continues without it.
*   Knob `TURBOSPARK_RESIDENCY_SET=off|on`, default `off`. Lab arms may also use
    `once` (residency at open) and `request` (residency at generation start).
    Neither is shipped until the outcome table below passes.

Invariants (copied into `crates/gpu/AGENTS.md` as part of Step 1b):

1.  Remove the set from the queue before releasing it.
2.  The set never holds a mapped or streamed expert buffer.
3.  Every Objective-C message runs inside `gpu::autorelease_pool`.
4.  A failed commit or attach drops the set and never fails a decode.

The unsafe boundary is settled by decision 6: the existing raw Objective-C
precedent in `pass.rs` covers this module.

Outcome table, using `tslab idle-gap` on the same install and gap schedule as
Step 1a:

| Outcome | Condition | Action |
|---|---|---|
| Pass | after-gap step 50 ms or less, decode digest equal, `phys_footprint` and swap unchanged, steady decode within the null-run spread | keep the set. The default stays off until decided under the family rule. |
| Partial | not a pass, stall cut by at least half, digest equal | keep as a lab-only arm with no production knob, and reopen decision 3 |
| Negative | stall cut by less than half, or decode digest changes | delete `residency.rs`, its test and the knob; keep the recorded finding. A digest change is a correctness failure. |

Null-run spread: the spread of steady decode tok/s over two identical runs with
the knob off. The protocol for that spread is to be confirmed in Step 1a.

Limit of this test: Inkling 2's mapped experts are not in the set, so on that
install the set holds the trunk only. A negative result there is recorded as
inconclusive for the expert path, not as proof that no fix exists.

### 2. FP64 reference and oracle (KERN-13)

Status: neither the prefix-difference tool nor the oracle mode exists on this
branch. Checked: no `--prefix-diff` in `crates/lab/src/cli.rs`, and no oracle mode
in `crates/lab/src/kernel/gdn_delta.rs`. Both are planned work.

*   **FP64 reference helper:** one shared module, used first by the existing
    parity tests and then by the GGUF and attention arms in Step 3. Location is
    chosen in Step 2.
*   **Oracle tolerance:** `max(floor, 3x the incumbent's own error)`. The floor is
    a fixed value set in Step 2 from the incumbent's measured error on the pooled
    fixture set, then recorded.
*   **Negative test:** a deliberately perturbed arm must fail the oracle.

Built (2026-11-26): the helper is `crates/compute/src/oracle.rs` (FP64
gated-delta step, error stats, `oracle_tolerance`), chosen over
`crates/gpu/tests/` so the lab can use it too (`crates/lab` gained a path
dependency on `turbospark-compute`, owner-approved). `tslab kernel --bench
gdn_delta --oracle on` checks decode and eager prefill (8 rows) over four
seeds. Floors, relative to the largest reference value, from the incumbent's
measured error at the Qwen3.6 geometry and a small one: y measured 2.3e-4 to
3.8e-4, floor 2^-11 (one `half` rounding step); f32 state measured 1.3e-7 to
3.6e-7, floor 8 x 2^-23. The negative test is a `dt_bias` shift of 0.01, which
moves the state by 6.5e-4 to 3.2e-1. Both mutation checks (no shift, a reference
without beta) fail the test. Not yet used by the existing parity tests.

**Deferred: prefix-difference profiling (KERN-12).** No gate in this document reads
its output. Build it when a kernel is blocked on attribution. Decision 4.

### 3. GGUF multiplies (KERN-04)

Status: Q4_K and Q8_0 are priced, not built. The price came from zero weights and
is not valid under the bench rule. The Q8_0 bench runs now on the local Inkling 2
Mini file, and the Q4_K bench runs on the downloaded Qwen3 8B Q4_K_M file (decision 1).

Stage A result (two runs: battery at 1-minute load near 16 from the build, then
AC power at 1-minute load 2.9 and 5-minute load 6.5). Test: `crates/gpu/tests/gguf_real_bytes_price.rs`. Ratio is GB/s
against the INT4 GEMV at the same shapes, same process.

*   Q4_K, Qwen3 8B, 36 FFN gate tensors, 1.02 GB footprint: real bytes 0.359x,
    zero-weight control 0.360x, real/zero 0.999x.
*   Q8_0, dspark Inkling 2 Mini, 27 tensors, 278 MB footprint (3 layers, so partly
    cache-warm): real bytes 0.323x, zero-weight control 0.322x.
*   AC re-run: Q4_K real 0.364x, zero control 0.364x; Q8_0 real 0.329x, zero
    control 0.330x. Battery and AC agree within about 2%.

Read: real bytes give the same rate as zeros, so the zero-weight price is not an
artifact. Both formats are below the 75% bar, so the first half of the proceed
rule is met on real bytes. The second half needs an arm at 1.25x, which is not
written. Q8_0 has 1.0625 bytes per weight against 0.5625 for INT4, so its
weights-per-second ratio is about 0.17, not 0.32. The proceed rule uses GB/s,
which measures bandwidth. Quote the AC run. It still started with a 5-minute load
near 6.5, so a quieter machine could move these by a few percent.

Stage A wide-loads arm (2026-11-26). Built as `dequant_q4_k_gemv_wide` in
`dequant_q4_k.metal`, a bench arm the engine does not call. Each lane makes one
`uint` load of 4 nibble bytes from one group, two `half4` loads of x and one
`uint` for d and dmin, and unpacks only its group's two scale pairs instead of
all eight. It sits beside the current kernel in
`crates/gpu/tests/gguf_real_bytes_price.rs` rather than in a new lab file,
because that probe already times the same real bytes and the INT4 baseline. It
is held to the CPU decoder by `the_wide_load_arm_matches_cpu_reference`, which
fails when the high-nibble x load is pointed at the wrong element. The same 36
Q4_K FFN gate tensors of Qwen3 8B were used, three runs, AC power. The first run
had a browser at about 100% CPU; the other two did not:

| Run | current GB/s | wide GB/s | wide / current |
|---|---|---|---|
| 1 | 63.7 | 99.7 | 1.566x |
| 2 | 38.8 | 61.9 | 1.594x |
| 3 | 59.4 | 89.5 | 1.506x |

Absolute GB/s moved by nearly 2x between runs (clocks, and another session's
idle server on the machine). The arms interleave each round, so the ratio is the
reading. The wide arm clears the 1.25x bar on real bytes, so for Q4_K **both
halves of the proceed rule are met**. Stage B (the other arms, rows 2 to 8,
other formats) is now open for Q4_K. Q8_0 has no wide arm yet: its 34-byte
block is not 4-byte aligned, so it needs a different load plan.

Step 4a (2026-11-26). `TURBOSPARK_GGUF_KERNEL=current|wide` (default
`current`) selects the wide GEMV and a wide twin of the batched verify GEMM
(`dequant_q4_k_gemm_wide`) together. `dequant_q4_k_wide_parity.rs` asserts the
twin is bit-identical to B GEMV calls at every width 1 to 16; pointing the batched
path back at the current kernel fails it. The wide lane prints
`[engaged] gguf_q4k_wide`. On Qwen3 8B Q4_K_M (an install built from the local
GGUF with `gguf_local_install`), `tslab gate` with the trusted evaluator
(`~/.turbospark/lab/bin/22d70656`) gives:

*   Quality: PASS. Mean KL 1.1e-5; teacher-forced decode over 1,280 positions has
    mean KL 1.3e-5 (p99 1.1e-4) and 1 flip, not a sure one; PPL 1.390 in both arms.
*   Memory: peak 820 MB in both arms.
*   Paired decode tok/s: **inconclusive**. 2 clean pairs of 3 needed (median
    +20%, 95% CI -10% to +50%, 1 better and 1 worse). 5 of 10 runs were dirty
    from other sessions' compiles. Ledger `1795740462768-70075-33`.

The default stays `current` until a re-run on a quiet host gives a verdict.

Stage B for Q4_K (2026-11-26, two runs on a quiet host, same 36 real tensors;
`crates/gpu/tests/gguf_stage_b_price.rs`). Ratios against the current GEMV:

| Arm | Run 1 | Run 2 |
|---|---|---|
| wide | 1.915x | 1.855x |
| wide, unroll 2 | 1.614x | 1.949x |
| wide, unroll 4 | 1.637x | 1.966x |
| verify GEMM B=2, wide / current | 1.725x | 1.607x |
| verify GEMM B=4, wide / current | 1.442x | 1.693x |
| verify GEMM B=8, wide / current | 2.311x | 2.441x |

On a quiet host the wide arm is about 1.9x. Unrolling is not a consistent gain
over it (below wide in run 1, at or just above it in run 2), so it is not
adopted. The current verify GEMM falls to 0.24x to 0.26x of the GEMV rate at
B=8, and the wide twin is faster at every width, so speculation on a Q4_K trunk
has the most to gain. Two rows per simdgroup (NR0=2) was not built.

Bench (new file `crates/lab/src/kernel/gguf_gemv.rs`, to be created):

*   **Stage A (first):** the current kernel and one unit-wide loads arm, at row 1,
    with an INT4 baseline on the same bytes, for each format's real bytes: Q8_0
    from the local Inkling 2 Mini file, and Q4_K from Qwen3 8B Q4_K_M. Time only
    the 217 Q4_K tensors for the Q4_K arm, and report the Q6_K output head apart from
    them. Report GB/s. Confirm each file's tensor types with a header scan before the
    run.
*   **Stage B (only if Stage A reaches 1.25x):** the other arms (unroll 2 and 4,
    two simdgroups per row with mask fixup), rows 1, 2, 4 and 8, and the other
    formats (Q8_0, IQ3_XXS, IQ4_XS, IQ4_NL, IQ2_XXS).
*   The bench reuses the existing bandwidth bench pattern in
    `crates/gpu/tests/gemv_bandwidth_bench.rs`, in which the ratio is the deliverable.
*   Table placement for IQ codebooks (KERN-23) is not here. It is a separate entry
    with its own gate (section 5).

Proceed only if the bench rule in "Gates" passes.

Build, only if it passes: new entry points in `dequant_q4_k.metal` and
`dequant_q8_0.metal`, with every function constant in the pipeline cache key, and
the current kernel selectable by `TURBOSPARK_GGUF_KERNEL=current|wide` (name
proposed, default `current`).

Gate: the decode gate (LAB-01); a token's bits do not depend on rows per dispatch;
parity with the `crates/compute` CPU decoders; and the real-bytes bench above.

### 4. Attention (KERN-02, KERN-03)

*   Add `tq_barrier` and head_dim 512 arms to `crates/lab/src/kernel/attn_decode.rs`.
*   **KERN-02 build trigger:** the TurboQuant kernel takes over more than 5% of
    decode at 16k. Report the TurboQuant-to-FP16 layer time ratio at 16k as a
    measured fact before the build decision. That ratio alone is not a trigger.
*   **KERN-02 pass:** 1.3x faster than the current TurboQuant kernel at 16k on the
    layer kernel, with KL inside the decode gate.
*   **TurboQuant rewrite** in `attention_tq.metal`: the pipeline cache key lists
    every kernel constant (CLAUDE.md invariant; list in Step 4b).
*   **Default:** decided once for all families under the AGENTS.md carve-out, after
    the decode gate and the layer gate pass on each family that accepts
    `--kv-bits`. Until then TurboQuant stays opt-in.
*   **Kill switch:** `TURBOSPARK_ATTN_DECODE_KERNEL` is read only at
    `crates/gpu/src/attention_decode.rs:794`, and `attention_tq.rs` does not read
    it. A TurboQuant selector is added instead (name proposed:
    `TURBOSPARK_ATTN_TQ_KERNEL=barrier|rewrite`). `TURBOSPARK_ATTN_DECODE_KERNEL`
    also changes verify scratch (`docs/SPEEDUP_PLAYBOOK.md` lines 226-229), so it
    is not a clean A/B baseline.
*   **KERN-03 build trigger:** attention takes over more than 5% of a token at 16k.
*   **KERN-03 pass:** gains at least 3% decode tok/s, with the digest and KL gates
    green and peak footprint unchanged.
*   **Gemma default:** only after its own quality, memory and paired speed gates
    (CLAUDE.md).

### 5. Everything else

Then: `record` writes the result to the ledger. `close` marks the entry done.
`separate design doc` means the entry gets its own design. `per entry` means the
threshold is set when the entry is picked up, and must be written down before its
first measurement.

| Entry | Needs | Threshold | Then |
|---|---|---|---|
| KERN-11 | a DFlash2 install | equal accepted stream and written K rows | close, with the decode gate applied |
| KERN-06 | GPU-clock-under-CPU-load bench (needs an install not on this Mac) | GPU time up under 5% with one spin-poll thread | hybrid wait A/B |
| KERN-07, 16 | existing knob and bench (needs an install not on this Mac) | under 1% cost; 3% gain | record |
| KERN-08 | real K and V rows | error down at 2 and 3 bit | GPU fold (move the norm correction into the GPU kernel) |
| KERN-09, 22 | KERN-02 | per entry | separate design doc |
| KERN-10, 14, 15, 18, 19, 21, 25, 26, 30 | one price each | per entry | separate design doc if cleared |
| KERN-23 | a settled audit (ledger: contested) | table placement gains at least 3% on the same bytes | resolve the audit, then measure |
| KERN-24 | ALU probe: fp16 against fp32 FMA | threshold set before the probe | opt-in arm, KL inside the decode gate |
| KERN-17 | verify tile, 9 to 24 rows | width 9-16 gains at least 1.3x over width 8 | two-tile arm |
| KERN-27 | owned by SPEC-09 | | |
| KERN-29 | a registered family | | |

## Alternatives Considered

"Rejected by author" marks a choice made while this was drafted. "Not chosen" is a
drafting choice covered by approval of the technical plan. "Closed" cites an
earlier record. D numbers are rows of the declined register in
`SPEEDUP-IMPROVE.md`, a working file outside this repository, so those citations
cannot be checked from here.

### Alternative A: Heartbeat kernel for the idle stall

*   **Pros:** works on macOS 14. Acts on GPU power state as well as memory.
*   **Cons:** costs power and heat while idle, unless bounded (see below).
*   **Reason for Rejection:** Rejected by author. A residency set was chosen as
    the mechanism test. Counter-evidence stays on record: llama.cpp ships its set
    with a 5 ms heartbeat, and PolyStrata reports a set alone did not help
    (upstream, not reproduced here). This is why Step 1a tests the cause before
    any set is built. A bounded keep-warm (stay warm for N seconds after the last
    token, sized from Step 1a) is not yet priced, and MOEM-10 covers it.

### Alternative B: Residency set through objc2-metal

*   **Pros:** typed bindings and less hand-written unsafe code.
*   **Cons:** adds a dependency (`objc2-metal`) that is not in `Cargo.lock`, and a
    second Objective-C binding next to `metal::objc`.
*   **Reason for Rejection:** Rejected by author for the second binding. The crate
    already uses `metal::objc`. Typed bindings can be revisited if the raw calls
    grow.

### Alternative C: Find the cause of the stall before designing a fix

*   **Pros:** avoids building a fix for the wrong mechanism.
*   **Cons:** slower to a user-visible result.
*   **Reason for Rejection:** Partly adopted. Step 1a now tests the cause before the
    set is built. The full version (no set until arm P is also measured) is
    deferred. The outcome table in Step 1b covers the case where the set does not
    address the cause.

### Alternative D: Typed error from the thread-limit guard (KERN-01)

*   **Pros:** no panic in a server process.
*   **Cons:** if every encode method returns `Result`, every caller changes.
*   **Reason for Rejection:** Not chosen as the design. The narrower option is
    adopted as Step 0 (decision 5).

### Alternative E: Metal 4 tensor API for prefill

*   **Pros:** Apple-maintained matrix path.
*   **Cons:** no gain measured in two other engines (sources in `SPEEDUP-IMPROVE.md`
    D44, not in this repository).
*   **Reason for Rejection:** Closed. The 9 to 24 row verify tile stays open under
    KERN-17.

### Alternative F: Fuse the GDN decode steps (KERN-20)

*   **Pros:** fewer dispatches.
*   **Cons:** estimated under 2% at our per-dispatch cost. Not measured.
*   **Reason for Rejection:** Closed in Non-Goals. Reopen if a dispatch probe (not
    yet defined) shows more than 2%.

### Alternative G: Probe a resident dense and a mapped-expert install before building the residency set

*   **Pros:** confirms the stall beyond one family.
*   **Cons:** neither install is on this Mac, so this cannot run now.
*   **Reason for Rejection:** Not possible yet. Step 1a runs on the one install
    available. A trunk-only result is recorded as inconclusive for the expert path
    (section 1). Revisit when a second install is available.

## Risks & Limits

| Risk | Mitigation | Gate that catches it |
|---|---|---|
| The residency set does not address the stall's cause | Step 1a cause test before any set is built | Outcome table (Step 1b) |
| Residency pins weights, KV and scratch, so memory cost is hidden | Measure wired memory as well as `phys_footprint` (proposed addition) | Outcome table, memory checks |
| The set holds an expert buffer and faults in a mapping | Exclusion test; invariant 2 | `crates/gpu/tests/residency.rs` |
| Raw Objective-C misuse causes a crash or leak | Autorelease pool, release order, invariants 1 to 4 | Tests and decision 6 |
| macOS 14 silently has no residency | Default off; the gap is recorded as a con of this option | Two-OS check (proposed) |
| The TurboQuant default changes behavior for every family | Default follows the AGENTS.md carve-out (decision 2); opt-in until its gates pass | Decode gate and layer gate on each family that accepts `--kv-bits` |
| A zero-weight price is taken as a real gain | Bench rule requires real bytes | GGUF bench rule |
| The server process panics on a thread-limit violation | Decision 5; Alternative D | Decision 5 |
| Gemma head_dim 512 overflows threadgroup memory | Budget check against the device limit before the build | Step 4b check |
| A GQA path change reaches families without their gates | List every caller; no family gets the default from this change | Step 4b caller list |

## Additional Resources

*   [SPEEDUP_PLAYBOOK.md](../SPEEDUP_PLAYBOOK.md): the first pass over all entries
    (2026-11-26) and the prices measured in it.
*   `SPEEDUP-IMPROVE.md`: the source plan with the full text of each entry. A
    working file kept outside the repository.
*   turbospark#329 (lab tooling), merged into `main` on 2026-11-26. turbospark#330
    (first pass) is this branch.
*   [speedup-spec.md](speedup-spec.md): speculative decoding
*   [speedup-moek.md](speedup-moek.md): MoE compute
*   [speedup-moem.md](speedup-moem.md): MoE streaming and memory
*   [speedup-serv.md](speedup-serv.md): serving, caching and prefill
*   [speedup-qfam.md](speedup-qfam.md): quantization, families and media

## Detailed Implementation Plan

Scope: technical plan sections 1 to 4 and the gates. Section 5 rows are
measurements, closures or later designs.

Precondition: turbospark#329 is on main (merged 2026-11-26). Rebase this branch on
main before Step 1a.

### Order

0.  Step 0: thread-limit guard on the server path (KERN-01).
1.  Step 1a: idle-stall cause test. Stop here if the result is inconclusive.
2.  Step 1b: residency set, only if Step 1a supports it. Apply the outcome table.
3.  Step 2: FP64 reference helper and oracle.
4.  Step 3: GGUF bench, Stage A first, on the Q8_0 and Q4_K files named in section 3.
5.  Step 4a: GGUF kernels, only if the bench rule passes.
6.  Step 4b: attention kernels, each only if its trigger in section 4 is met. The
    default follows the AGENTS.md carve-out (decision 2).

### Step 0: thread-limit guard (KERN-01)

*   `crates/gpu/src/context/pass.rs` (Modify): violation record, dispatch skip,
    checked-commit `Err`, unchecked-commit abort on a recorded violation.
*   Server-reachable callers (Modify): move to the checked commit. Survey list
    produced from `crates/runtime`, `crates/ffi` and `crates/server` before merge.
*   `crates/gpu/tests/` (Create): lowered-limit fixture; checked commit returns
    `Err`; unchecked path aborts.

### Step 1a: idle-stall cause test

*   `crates/gpu/src/touch.rs` (Create): lab-only touch dispatch over resident
    buffers, behind a lab knob.
*   `crates/lab/src/checks.rs` (Modify): add arms T and P to the idle-gap check.
*   `crates/lab/families/inkling2_loop.json` (Modify): record the result, using
    the existing idle-gap entry as the template.

### Step 1b: residency set (conditional)

Precondition: any uncommitted edits to `crates/runtime/src/real_forward_open.rs`
and `crates/runtime/src/families/inkling2/state.rs` in the main checkout are
committed or discarded by their owner before this step. Check that `git status`
in the main checkout shows neither file.

*   `crates/gpu/src/residency.rs` (Create): `ResidencySet` with construction
    returning `None` where unsupported; add, commit, request, attach to queue;
    remove and release on drop; every message inside `gpu::autorelease_pool`.
*   `crates/gpu/src/lib.rs`, `crates/gpu/src/context/device.rs` (Modify): export
    the module; the context owns the set so it outlives its command buffers.
*   `crates/runtime/src/real_forward_open.rs` (Modify): build the set at open when
    the knob is on, for resident weights; never add mapped or streamed expert
    buffers; apply the size guard and print the reason on skip.
*   `crates/runtime/src/families/inkling2/state.rs`, `crates/runtime/src/families/qwen/state.rs`
    (Modify): register KV and long-lived scratch for these two families only.
    Other families are out of scope until measured.
*   `crates/runtime/src/real_forward_traits.rs` (Modify): request mode only, and
    only if the Step 1b outcome keeps it lab-only (decision 3). The request hook is `try_reuse_prefix`, called
    from `crates/runtime/src/raw_completion.rs:244` and
    `raw_completion_chunked.rs:104`. The speculative loop does not call it, so
    speculative generations are excluded from request mode, and that exclusion is
    stated in the code comment.
*   `crates/lab/src/knobs.rs` (Modify): register the knob name. The runtime reads
    the variable directly, as `attention_decode.rs:794` already does.
*   `docs/ENV.md`, `docs/LOAD_GUARD.md`, `CHANGELOG.md`, `crates/gpu/AGENTS.md`
    (Modify): the knob, the wired-memory note, and the invariants above.
*   `crates/gpu/tests/residency.rs` (Create): construct, add, commit and drop on a
    real device; an expert buffer is refused; the unsupported branch is exercised
    through an injected support flag, so it runs on a macOS 15 machine.

`Cargo.toml` and `Cargo.lock` are not in this list. Confirm with `git diff --stat`
that neither changed.

If the Step 1b outcome is Negative, revert every file above. Keep the Step 1a files
and the recorded result.

### Step 2: FP64 reference and oracle

*   The shared FP64 reference helper (Create). Location chosen in this step:
    candidates are `crates/compute` and `crates/gpu/tests/`.
*   `crates/lab/src/kernel/gdn_delta.rs` (Modify): oracle mode using the helper,
    and a deliberately perturbed arm that must fail.
*   `crates/lab/skill/references/commands.md` (Modify): document the oracle
    command.

Prefix-difference tooling (`crates/gpu/src/dispatch_profile.rs`, `crates/lab/src/cli.rs`,
`worker.rs`, `trace.rs`) is deferred (section 2).

### Step 3: GGUF bench (Q8_0 and Q4_K files both on disk)

*   `crates/lab/src/kernel/gguf_gemv.rs` (Create), `crates/lab/src/kernel/mod.rs`
    (Modify): Stage A (Q4_K, current, unit-wide loads, row 1, INT4 baseline), then
    Stage B only if Stage A reaches 1.25x.
*   `crates/lab/src/kernel/attn_decode.rs` (Modify): `tq_barrier` arm and head_dim
    512 shapes at 4k, 16k, 32k and 64k positions.
*   Bytes: Q8_0 from the local Inkling 2 Mini file. Q4_K from
    `~/models/gguf-bench/Qwen_Qwen3-8B-Q4_K_M.gguf`. Shapes are recorded per file.

### Step 4a: GGUF kernels (only if the bench rule passes)

Selection rule: the arm with the highest GB/s among arms that reach 1.25x on the
same bytes. Ties go to the arm earlier in the Stage A or Stage B list. If no arm
passes, record the result and create no kernel file.

*   `crates/gpu/src/shaders/dequant_q4_k.metal`, `crates/gpu/src/shaders/dequant_q8_0.metal`
    (Modify): the winning arm as a new entry point.
*   `crates/gpu/src/dequant_q4_k_gemv.rs`, `crates/gpu/src/dequant_q8_0_gemv.rs`
    (Modify): select it with `TURBOSPARK_GGUF_KERNEL`, default `current`; constants
    in the pipeline cache key.
*   `crates/lab/src/knobs.rs`, `docs/ENV.md`, `CHANGELOG.md` (Modify): the switch.
*   `crates/gpu/tests/dequant_q4_k_gemv_parity.rs`,
    `crates/gpu/tests/dequant_q8_0_gemv_parity.rs` (Modify): CPU decoder parity,
    and bit equality between one row and many rows per dispatch.

### Step 4b: attention (each only if its section 4 trigger is met)

*   `crates/gpu/src/shaders/attention_tq.metal`, `crates/gpu/src/attention_tq.rs`
    (Modify): TurboQuant rewrite. Pipeline cache key: group, simdgroups, head_dim,
    unroll, k bits, v bits. Add the TurboQuant selector
    `TURBOSPARK_ATTN_TQ_KERNEL`. The default stays opt-in until the AGENTS.md
    carve-out gates pass.
*   `crates/gpu/src/shaders/attention.metal` (Modify): `kGqaMaxPerLane` from 8 to 16
    (line 297). `tg_o` (line 322) grows with it, so the new threadgroup memory size
    must fit the device limit. Measure the limit with the device query in this step,
    not from memory.
*   `crates/gpu/src/attention_decode.rs` (Modify): extend
    `attention_decode_gqa_supported` (lines 1134-1139) to head_dim 512. Every
    caller changes behavior: `crates/lab/src/kernel/rows_attn.rs:93`,
    `crates/lab/src/kernel/attn_decode.rs:166`,
    `crates/runtime/src/families/lfm3/state.rs:77`,
    `crates/runtime/src/families/llama/prefill_batched.rs:149`,
    `crates/runtime/src/families/qwen/state.rs:454`,
    `crates/runtime/src/families/qwen/dflash_tree.rs:58`, and
    `attention_decode.rs` lines 1018 and 1234. Each caller keeps its own test. No
    family gets the GQA default from this change.
*   New entry points stay outside default JIT warmup (CLAUDE.md: warmup is opt-in).
*   `crates/gpu/tests/attention_tq_parity.rs`,
    `crates/gpu/tests/attention_decode_gqa_parity.rs` (Modify): new shapes, and the
    KV ring buffer wrap-around case.
*   `docs/SPEEDUP_PLAYBOOK.md`, `CHANGELOG.md` (Modify). The TurboQuant doc path
    (`docs/TURBOQUANT.md`, previously written `TRUBOQUANT.md`) still needs
    confirming.

The GGUF kernels ship behind `TURBOSPARK_GGUF_KERNEL`, default `current`. Making a
new kernel the default re-freezes rows, so it needs owner approval and that family's
gates. The TurboQuant default follows the AGENTS.md carve-out (decision 2).

### Verification Plan

*   **Build and tests:** `cargo build --workspace`, `cargo test --workspace`,
    `cargo fmt --check`, and `cargo clippy --workspace --tests` (CLAUDE.md minimal
    verification). For Step 1b, also `make swift-lib` and `make swift-test`, because
    the app links the gpu crate.
*   **Cross-target:** Step 1b adds cfg-gated code. Run the portable cross-target
    check from `.claude/docs/verification.md` (command to confirm there).
*   **macOS 14 path:** tested through the injected support flag in
    `residency.rs`, not by running macOS 14.
*   **Idle-stall outcome:** the table in Step 1b, on the same install and gap
    schedule as Step 1a.
*   **Kernel:** the decode gate (LAB-01), paired timing on a real install with the
    pair count from `docs/BENCHMARKS.md`, and the bench rule for GGUF.

## Entry ledger

Every entry in this area. "Plan" is the disposition recorded in
`SPEEDUP-IMPROVE.md` (a working file outside the repository, which holds the full
entry text). "State" is from the first pass recorded in `docs/SPEEDUP_PLAYBOOK.md`
on this branch, 2026-11-26, corrected where turbospark#329 showed otherwise.

Plan labels, as used in `SPEEDUP-IMPROVE.md`: DO-NOW (scheduled), QUEUE (planned,
not started), NEEDS-DATA (waits for a measurement), CONTESTED (disputed, not
decided), DECLINE (not pursued).

| Id | Entry | Plan | State |
|---|---|---|---|
| KERN-01 | Central pipeline thread-limit guard on every dispatch | DO-NOW | built on this branch |
| KERN-02 | TurboQuant decode attention: GQA-shared, barrier-free, simdgroup-stripe rewrite | DO-NOW | waits on the decode gate |
| KERN-03 | head_dim 512 GQA-shared decode attention (Gemma 4) | DO-NOW | waits on the decode gate |
| KERN-04 | GGUF K-quant / Q8_0 / IQ decode GEMV: first measure, then wide-unit kernels | DO-NOW | Q4_K wide-loads arm 1.51x to 1.59x the current kernel on real bytes (bar 1.25x); bench arm only, not in the engine; Q8_0 arm not written |
| KERN-05 | Idle stall: cause test, then residency set (Step 1a, 1b) | DO-NOW | Step 1a run: stall reproduced on mlx4; wake and weights touch do not remove it; Step 1b not built |
| KERN-06 | Spin-poll on command-buffer completion versus sleep (settle with a paired A/B including a hybrid) | CONTESTED | needs an install not on this Mac |
| KERN-07 | Price precise (safe) math for text decode and the GDN gate transcendentals | DO-NOW | needs an install not on this Mac |
| KERN-08 | TurboQuant norm-correction fold, K/V split and boundary-layer sweep | DO-NOW | needs an install not on this Mac |
| KERN-09 | `--kv-bits` blocks batched prefill and speculative verify: two-source prefill and multi-row quantized attention | QUEUE | queued, not started |
| KERN-10 | Windowed, sink-aware multi-row flash attention for Flint/LFM3 prompt batches | NEEDS-DATA | needs data, not started |
| KERN-11 | DFlash2 context-KV write: batch the per-row norm and rope dispatches | DO-NOW | built, not run on a DFlash2 install |
| KERN-12 | Prefix-difference profiling and function-constant probes for `tslab` | DEFERRED (decision 4) | not present on this branch after the #329 merge |
| KERN-13 | Numerical-oracle harness for kernel changes (FP64 reference, nonzero inputs, canary) | DO-NOW | not present on this branch after the #329 merge |
| KERN-14 | Command-buffer submission path: unretained references, chunked commit, replayable decode segments | NEEDS-DATA | needs data, not started |
| KERN-15 | Automatic memory-range barrier tracking and graph reorder for the concurrent encoder | NEEDS-DATA | needs data, not started |
| KERN-16 | Attention decode launch geometry sweep and tuner anti-drift procedure | DO-NOW | needs an install not on this Mac |
| KERN-17 | Few-row MMA: 9-24 row verify tile, in-place B loads, per-shape crossover table | QUEUE | waits on the decode gate |
| KERN-18 | GDN prefill: re-price share for MoE installs; column-owner staged scan variants | NEEDS-DATA | needs data, not started |
| KERN-19 | Sixteen-bit GDN recurrent state (fp16/bf16, position-keyed stochastic rounding) | NEEDS-DATA | needs data, not started |
| KERN-20 | GDN decode and verify restructurings: fused prework/gnorm, ReplaySSM, snapshot copy elision (priced, decline) | DECLINE | declined |
| KERN-21 | Prefill attention share versus context length, work cap, and mask block map | NEEDS-DATA | needs data, not started |
| KERN-22 | Cheaper-dequant KV formats (affine8, K-only int8, UltraQuant-style FP4) alongside TurboQuant | QUEUE | queued, not started |
| KERN-23 | IQ codebook tables: constant versus threadgroup memory (audits disagree) | CONTESTED | contested, not settled |
| KERN-24 | fp16 accumulation window for K-quant/IQ GEMV (opt-in, quality-costing) | NEEDS-DATA | waits on the decode gate |
| KERN-25 | qwen4_exp QSA kernels: O(N^2) top-k, per-head-barrier scoring, indexed attention without GQA sharing | NEEDS-DATA | needs data, not started |
| KERN-26 | Dispatch-count fusion bundle (qk-norm+rope+KV write, sandwich norm, add+RMSNorm, sigmoid-gate, MTP pre-FC norms, HC inject+norm, Inkling conv_mix) | NEEDS-DATA | needs data, not started |
| KERN-27 | DFlash2 candidate selector on the GPU | QUEUE | needs an install not on this Mac |
| KERN-28 | Metal 4 tensor API (`matmul2d`, MPP) on M4 Max | DECLINE | declined |
| KERN-29 | Mamba-2 SSD chunked scan and KDA recurrence (family-conditional) | QUEUE | queued, not started |
| KERN-30 | Chunked no-copy weight wrap beyond `maxBufferLength`; MTP prompt rows on a second queue; CPU matrix-unit co-execution | NEEDS-DATA | needs data, not started |

--------------------------------------------------------------------------------
