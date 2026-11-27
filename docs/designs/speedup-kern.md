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

#### Re-read of Step 1a (2026-11-26, recorded data and code, no new runs)

The three arms fit one mechanism. **Likely cause:** after about 1 to 2 s with
no command buffer, macOS drops the GPU-side mapping of the process's buffers,
and the next command buffer pays, in the kernel and before the GPU starts, to
map again the buffers it names.

*   The engine already measured this cost at open: `inkling2_prewarm`
    (`crates/runtime/src/families/inkling2/prefill.rs`, 2026-11-23) exists
    because the first buffer naming the resident experts and trunk waits 1.2 to
    2.8 s while the driver maps them. The idle stall is the same wait, smaller,
    coming back after idle.
*   The wait is `GPUStartTime - kernelStartTime`, and `GPUPH` reads `OFF` for
    82 to 87% of a stalled first step (8 to 21% after a short gap). The earlier
    "counter cannot tell them apart" holds for the gap window only.
*   `wake` names an 8-byte buffer, maps nothing, and completes in 6 to 13 ms:
    GPU power-up is cheap.
*   The cost moves with the buffer named. `touch` names the trunk (928 MiB),
    and prestep plus remaining wait equals the baseline wait (34+147=181 vs
    178, 49+171=220 vs 199, 52+162=214 vs 206 ms). The GPU is `OFF` for 99.8%
    of the 4 s and 8 s touch presteps, so that time is kernel-side too.

**Correction to Step 1a and Step 1b:** the mlx4 run did not use mapped
experts. `auto` resolved to resident (`resident_experts_decision`, peak
footprint 17.74 GiB in the ledger). One step names the trunk plus 24 copied
expert buffers (about 14.6 GiB) plus KV, so the touch arm covered about 6% of
the named bytes, and invariant 2 does not exclude these buffers.

**Open point:** about 36 to 56 ms per GiB for the trunk (every page read)
against about 10 to 12 ms per GiB averaged over the experts (only routed
strides read). The cost is either per page the GPU reads or per buffer kind
(file-backed no-copy trunk vs copied expert buffers). The `prewarm` and
`touch-all` arms below separate these. The Step 1a host had 8 GB of swap in
use and about 1M compressions per run, a confound.

**Upstream:** the PolyStrata audit (section 9.5) reports the same symptom and
threshold and ships a one-thread kernel that calls `useResource` on every
buffer every 250 ms, for at most 120 s after the last command buffer, paused
under memory pressure; a residency set alone and a keep-alive naming no buffer
did not help there. llama.cpp (PR 11427) keeps per-buffer residency sets and
calls `requestResidency` every 5 ms for 180 s, citing memory "collected by the
OS after 1 second of inactivity". Apple documents no residency duration.

**Consequence:** Step 1b as written (a set requested once or per generation)
is the variant upstream reports as not working, so it is not built first. The
fix primitive already exists (the prewarm); the question is when to run it
again. Alternative A's bounded form is what PolyStrata ships, so it is
reopened as a lab arm only. A production default stays a separate decision.

The first step after any gap of 0.5 s or more also runs 12 to 16 ms slow while
the GPU ramps from P1 to P15. That is separate and out of scope (the
2026-11-21 ALU keep-alive was tried and not kept).

#### Steps 0 to 2 of the re-plan (lab only)

Built (2026-11-26): `runtime::IdleProbeAction` (`Wake`, `TouchWeights`,
`Prewarm`, `TouchAll`) replaces the `touch: bool` of `run_idle_probe`.
`Prewarm` shares its encode-and-commit body with the open-time prewarm
(`inkling2_prewarm_commit`), waits on it, and is refused by name without
resident experts. tslab arms: `TURBOSPARK_LAB_IDLE_PRESTEP=prewarm|touch-all`,
and keep-warm `TURBOSPARK_LAB_IDLE_KEEPWARM_MS=<n>` with
`TURBOSPARK_LAB_IDLE_KEEPWARM=<action>`, which runs the action every `n` ms
strictly inside each gap on the probe thread (no engine change, no
background thread). Rows add the summed first-step waits, the first five busy
steps, gap energy, footprint, and swap and compressor movement.

Reading, Step 1:

| Result | Meaning |
|---|---|
| `prewarm` prestep about 200 ms, wait under 3 ms, digest equal | naming is enough. The existing prewarm is the fix primitive |
| `prewarm` cheap and wait unchanged, `touch-all` absorbs it | the cost is per page read. A keep-alive must touch what a step reads |
| neither moves the wait | the diagnosis is wrong. Stop and re-plan |

Outcome, Step 2 (against `none`, arms every 250 ms, every 750 ms, and `wake`
every 250 ms as the negative control):

| Outcome | Condition | Action |
|---|---|---|
| Pass | first-buffer wait under 3 ms at 2, 4 and 8 s, digest equal, footprint and swap unchanged | price the production forms (prewarm on intent, bounded keep-warm), separate approval |
| Partial | stall cut by half or more | record. Try a shorter interval or the touch form |
| Negative | cut by less than half | record, drop keep-warm, re-plan |

Result (2026-11-26 and 27, M4 Max, AC, load under 2, swap 0 after a reboot,
tslab built from `3c8343a0`; two fresh processes per arm, interleaved;
ledgers `kern05_idle` and `kern05_idle_dense`; reading
`inkling2_loop.idle-gap-fix-2026-11-27`). First-buffer wait after a 2, 4 and
8 s gap, ms, one triple per run:

| Install, arm | Wait (ms) | Prestep or keep-warm cost |
|---|---|---|
| mlx4 `none` (4 runs) | 197/239/262, 157/205/214, 172/222/198, 151/188/194 | |
| mlx4 `wake` | 152/191/183, 194/220/206 | prestep 4 to 9 ms |
| mlx4 `touch` (trunk) | 132/160/167, 198/192/183 | prestep 19 to 42 ms |
| mlx4 `prewarm` | 1.8/1.7/1.4, 4.3/2.0/2.3 | prestep 142 to 196 ms, 0 pages |
| mlx4 `touch-all` | 1.5/11.1/2.4, 1.8/1.3/1.4 | prestep 152 to 179 ms, 1,017,842 pages |
| mlx4 keep-warm `prewarm` / 250 ms | 4.5/5.5/2.3, 4.8/4.1/5.9 | about 1 ms per action |
| mlx4 keep-warm `prewarm` / 750 ms | 5.1/5.8/4.1, 4.8/4.5/4.3 | about 1 ms per action |
| mlx4 keep-warm `wake` / 250 ms | 142/191/196, 137/185/198 | about 1 ms per action |
| qwen3-8b q4km `none` | 47/48/49, 48/34/48 | |
| qwen3-8b q4km `touch` | 1.1/1.4/1.7, 1.5/1.1/1.3 | prestep 36 to 51 ms, 306,471 pages |

After 0.5 and 1 s every arm waits under 3 ms. Decode digests are equal across
every arm at every gap on each install (29 mlx4 and 4 dense runs, and equal to
the Step 1a digests). Footprint stays 17.72 to 17.74 GiB on mlx4 and swap does
not grow. Four runs carry tslab's DIRTY flag (Spotlight re-indexing after the
reboot, or foreign CPU over 150%); their waits sit inside the clean runs'
spread.

*   **Step 1: row 1, the diagnosis holds.** Naming every resident buffer moves
    the whole stall into the prestep, and the first step returns to the
    short-gap baseline (31 to 41 ms). `touch-all` reads every page of about 15.5
    GiB, `prewarm` one element of each of 25 buffers, and they cost about the
    same, so the cost is per buffer
    named after idle, not per page read. That settles the open point. The
    existing prewarm is the fix primitive.
*   **The prestep form moves the cost, it does not remove it.** Prestep plus
    remaining wait is about the `none` wait. Re-issuing the prewarm when a
    request arrives hides the stall only behind CPU work it overlaps.
*   **Dense (Alternative G):** the stall is present on a dense install with one
    file-backed buffer, at the same 1 to 2 s threshold, and `touch` moves at
    least 95% of it into the prestep. The size was not as predicted: about 47
    ms for 4.7 GB (about 10 ms per GiB), not 170 to 260 ms. The cost does not
    scale with trunk bytes at the Inkling trunk's rate.
*   **Step 2: Partial by the table.** A keep-warm `prewarm` cuts the stall by
    97 to 98%, and each action costs about 1 ms instead of about 160 ms: named
    every 250 or 750 ms, the buffers are never unmapped. The wait left is 2.3
    to 5.9 ms, over the 3 ms Pass bar at 2 and 4 s. The `wake` control, at the
    same rate and cost, leaves the whole stall, which matches the PolyStrata
    report that a keep-alive naming no buffer does not help.
*   **The table's Partial action does not apply as written.** 250 and 750 ms
    leave the same residual, so a shorter interval is not expected to help, and
    the touch form names the same buffers. The residual is probably GPU
    power-up (an empty dispatch after a gap takes 4 to 9 ms, and the prestep
    form, where the step follows the prewarm at once, leaves 1.4 to 2.3 ms).
    That is an inference, not measured.
*   **Energy is not priced.** Gap energy is machine-wide IOReport, and on this
    host it overlaps across arms (`none` reads 3.4 to 27 J for the same gaps).
    A keep-warm's idle power needs a quieter host or a per-process meter before
    any default.

Owner decision (2026-11-27): the 2 to 6 ms residual is acceptable, and the
production form is the bounded keep-warm prewarm. Step 1b's residency set is
not needed for this mechanism.

#### Step 3: bounded keep-warm (built, default off)

Built (2026-11-27): `gpu::KeepWarm` (`crates/gpu/src/keep_warm.rs`), started at
open by `inkling2_start_keep_warm` when `TURBOSPARK_INKLING2_KEEPWARM_MS=<50..=5000>`
is set; `TURBOSPARK_INKLING2_KEEPWARM_WINDOW_S=<1..=3600>`, default 120 (the
PolyStrata bound).

*   Its own thread and its own command queue; one single-thread dispatch per
    buffer reads one word and writes only its own scratch, so decode output
    cannot change. Names what the prewarm names: every resident expert buffer
    and the trunk. A streamed install is refused by name (its experts are
    mapped and must stay droppable), and so is a malformed value.
*   Fires once a full interval has passed since the buffers were last named,
    by a pass on the runner's context or by its own tick. `MetalContext`
    stamps the time at every pass begin (one atomic store), so a busy decode
    costs nothing. It polls at a quarter interval, so the longest unnamed
    stretch is about 1.25 intervals.
*   Stops once the window has passed since the runner's last pass. Skips any
    tick where `kern.memorystatus_vm_pressure_level` is not normal (or
    unread).
*   Tests: parse, decision and refusal unit tests; `crates/gpu/tests/keep_warm.rs`
    on the device (fires while idle, silent while the context is busy, stops
    after the window, joins on drop). Mutants killed: passes no longer stamp
    activity, the window never closes, only critical pressure pauses. tslab
    rows count the runner's ticks over each gap (`runner_keep_warm`).

Result (2026-11-27, M4 Max, AC, mlx4, tslab built from the uncommitted tree
on `d3d7ec72`; two fresh processes per arm, interleaved; ledger `kern05_idle`
records `1795791967676-81247-2` to `1795793676996-23134-2`). First-buffer
wait after a 2, 4 and 8 s gap, ms:

| Arm | Wait (ms) | Keep-warm ticks over the gap |
|---|---|---|
| `none` | 228/263/1440, 1633/2224/146 | |
| `KEEPWARM_MS=750` | 5.7/5.4/4.0, 3.6/4.2/4.7 | 2/4-5/10, 0 paused, 0 failed |
| `KEEPWARM_MS=750`, window 3 s | 6.5/1719/178, 5.1/176/159 | 2/3/2-3, then none |

*   **Production knob reproduces the Step 2 lab arm.** 3.6 to 5.7 ms at every
    gap of 2 s or more in both runs, against 146 to 263 ms for `none` rows
    without paging. Decode digests are equal across all six runs at every gap,
    and equal to the Step 1 digests.
*   **The window bounds it.** With a 3 s window the 2 s gap stays warm, and
    the 4 and 8 s gaps stall again (159 to 178 ms on rows without paging): ticks stop
    at the window, and the mappings drop within about 1 s of the last one.
*   **The pressure pause engaged live.** The host swapped up to about 7 GB
    from foreign work during these runs. One window-arm row logged 3 paused
    ticks, and `none` and window rows that hit paging show a different, larger
    stall (1.4 to 2.2 s, swap growth, footprint dropping to 16.06 GiB) that a
    keep-warm does not claim to fix. Those rows are excluded from the
    comparison above. Two runs carry tslab's DIRTY flag (foreign CPU, one a
    compile); their waits sit inside the clean runs' spread.
*   **Not priced:** idle power over the window (gap energy is machine-wide and
    overlaps across arms here too), a default-on decision, and the app's
    settings surface. The knob is off unless set; nothing in Swift sets it.

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

Owner decision, 2026-11-26: the wide Q4_K pair is the default (`cb8be041`)
without waiting for a clean paired A/B, on the quality pass and the Stage A and
B bench readings. `TURBOSPARK_GGUF_KERNEL=current` restores the old pair. The
knob A/B queued at base `5fe079f5` (default still `current` there) is now a
check of that decision. From `cb8be041` on, a Q4_K A/B must set
`TURBOSPARK_GGUF_KERNEL=current` on its baseline explicitly. Q8_0 stays
`current` until its own A/B.

Knob A/B and confirm (2026-11-26, AC power, quiet host, trusted evaluator
`~/.turbospark/lab/bin/5fe079f5`, `current` vs `wide`, 3 clean pairs each):

| Format | Quality (mean KL) | run-experiment median (95% CI) | confirm, held-out (95% CI) | Ledger (confirm) |
|---|---|---|---|---|
| Q4_K (Qwen3 8B Q4_K_M) | PASS, 1.11e-5, top-1 1.00 | +50.8% (+50.2, +50.8) | +45.5% (+41.7, +47.4) | `1795755163935-70733-21` |
| Q8_0 (Qwen3 8B Q8_0) | PASS, 1.50e-5, tokens equal | +43.3% (+42.6, +47.5) | +49.2% (+49.2, +50.0) | `1795756390103-6512-25` |

Peak memory is unchanged in both. Both verdicts are `keep`, so the Q4_K
decision holds and Q8_0 is now wide by default too. `TURBOSPARK_GGUF_KERNEL=current`
restores both previous pairs, and a GGUF A/B from here on must set it on the
baseline.

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

Q8_0 Stage A (2026-11-26, `crates/gpu/tests/gguf_q8_wide_price.rs`, the 27
Q8_0 tensors of the dspark Inkling 2 Mini file). `dequant_q8_0_gemv_wide`
uses `ushort` quant loads because 34-byte blocks keep block starts only 2-byte
aligned, plus `half4` x loads and the scale factored out of each block's sum.
`_r2` adds two rows per simdgroup sharing x (llama.cpp's NR0 = 2):

| Arm | Run 1 | Run 2 |
|---|---|---|
| wide / current | 1.595x | 1.594x |
| wide, 2 rows / current | 1.628x | 1.620x |

Both clear 1.25x. These 278 MB are partly cache-warm (the INT4 arm read about
635 GB/s, above DRAM bandwidth), so the ratio needs the full-model A/B. Step 4a
for Q8_0 uses the one-row arm: `TURBOSPARK_GGUF_KERNEL` now also selects it and
its batched twin `dequant_q8_0_gemm_wide`, each format with its own default
(both `current`). The pair parity test first asserts that the fixture separates
the two reductions: at 19 to 67 rows the half rounding of y hid the difference
and a verify path left on the current kernel passed; 2048 rows of 133 blocks
does separate them, and the mispaired mutation then fails at batch 1. The
knob A/B for both formats (Qwen3 8B Q4_K_M and Q8_0, base `5fe079f5`, from a
clean worktree) waits for AC power and a quiet host.

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

Step (a) price, 2026-11-26. `attn_decode` gains `tq4_barrier` and
`tq3.5_barrier` arms (today's TurboQuant kernel over the same rows, quantized
once with `kv_quantize_tq`), and `attn_decode_gemma4` times Gemma 4's shapes on
any install. Both runs were on battery with foreign CPU, so the times are not
quotable; the arms interleave in one process.

Qwen3 8B geometry (32q over 8kv, head_dim 128, 36 layers), ms per token over
all layers:

| Context | FP16 barrier | FP16 GQA (default) | TQ 4 | TQ 3.5 |
|---|---|---|---|---|
| 4k | 23.5 | 2.2 | 26.4 | 26.5 |
| 16k | 114.9 | 6.8 | 111.0 | 110.1 |
| 32k | 232.5 | 13.1 | 248.0 | 236.2 |

`--kv-bits` attention runs at the barrier kernel's cost, about 16x the FP16
default at 16k, so the KERN-02 trigger (slower than FP16 GQA) is met. The
share-of-decode reading from `tslab profile` on `qwen38-27b` is still owed.

Gemma 4 shapes (head_dim 512): full layers (5) take 7.4 ms barrier and 16.3 ms
TQ 4 at 16k; sliding layers (25, window 1024) take about 2.4 ms. Against the
25 to 29 ms Gemma token in the frozen rows that is roughly a third of a token
at 16k, so the KERN-03 trigger looks met. That share is an estimate: no Gemma
install is on this Mac for `tslab profile`.

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
*   **Status (2026-11-27):** reopened as a lab arm only. A keep-warm that names
    every resident buffer every 250 or 750 ms cuts the stall by 97 to 98%; one
    that names none does not (section 1, Steps 0 to 2 result). Idle power is
    not priced, and no production form is built.

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
*   **Status (2026-11-27):** the dense half ran on
    `qwen3-8b-q4km.gturbo`: the stall is present (about 47 ms) and a trunk
    touch moves at least 95% of it. A mapped-expert install is still not
    probed.

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
| KERN-05 | Idle stall: cause test, then residency set (Step 1a, 1b) | DO-NOW | Cause confirmed (2026-11-27): after 1 to 2 s idle the next buffer pays to map again every buffer it names; re-running the prewarm moves the stall out of the step, a keep-warm prewarm every 250 or 750 ms cuts it 97 to 98% (Partial: 2 to 6 ms left), a dense install shows it too. Step 1b not needed. Step 3 built (2026-11-27): opt-in bounded keep-warm `TURBOSPARK_INKLING2_KEEPWARM_MS`, 3.6 to 5.7 ms after 2 to 8 s idle, digests equal, window and pressure pause shown live; default stays off |
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
