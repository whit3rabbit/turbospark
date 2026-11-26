# tslab improvements: status of the LAB plan

The plan is `TSLAB-IMPROVE.md` (36 entries, LAB-01 to LAB-36), kept outside
the repository with its source audits. This page records, per entry, what the
lab now does and what it does not, with the reason. Commands and flags are in
`crates/lab/skill/references/commands.md`.

Status words:

- `done`: the lab capability exists and has its own test.
- `partial`: part exists; the missing part and its reason are named.
- `not built`: no code; the reason is named.

"Not measured" is a separate thing: building a capability is not a finding.
No benchmark number in this page is a frozen row. The host was on battery and
shared while this was built, so no speed claim was taken from it.

## Wave 0

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-07 one-knob lint | done | A spec's arms may differ only in `knob_env`; anything else needs `also_changes` with a reason, recorded in the decision. `ab` warns. The five specs frozen under the lab dir on this host pass the lint. | `expect_engaged` is exempt: it asserts a lane and changes no behavior. |
| LAB-05 type-I audit, non-inferiority | done | `tslab type1` replays stored null calibrations: on this host (163 pair improvements from 56 calibrations) the false WIN rate at 3 pairs was 0.6%, under the 5% line, so `min_pairs` stays. `noninferiority_limit` and `ab --noninferior` (10 pairs, one-sided bounds, limit at most 3%). The sweep prints its false-WIN bound. A null run with fewer than 3 clean pairs records no floor, and older such records are ignored. | The replay resamples pairs as independent. The autoresearch loop's multiplicity over many specs is not corrected beyond the single held-out `confirm`. |
| LAB-06 route attestation | partial | Print-only engagement lines for the lane knobs, now including the single call site switches (23 diagnostic knobs are unquotable and need none). `ab` and `sweep` warn when a baseline lost or gained a lane (`baseline_route`). | Four quotable knobs have no line: they size or locate something rather than pick a lane (`int4_mma8_min_rows`, `ane_prefill_dir`, `inkling2_panel_rows`, `inkling2_grouped_min_rows`). The route comparison sees engagement lines only. The frozen A/Bs were not re-run. |
| LAB-14 page-cache eviction | done | `streaming::evict_file_pages` (`msync(MS_INVALIDATE)`) and the `~cold` arm. Measured by test on this macOS: a hot 8 MiB file reads 0 physical bytes, and 8 MiB after eviction. | The install-walk use (`--evict` for quantize and stream installs) is an installer change, not a lab change. `docs/BENCHMARKING.md` still asks for `sudo purge` for frozen cold rows: one file's test is not evidence for a whole install. |
| LAB-26 launch counts | done | `TURBOSPARK_LAUNCH_COUNT=1` counter in the gpu crate and `tslab launch-count`; run on an inkling2-mini install. | Verify rounds are not split from plain decode. |
| LAB-16 slot curve | done | `slot-curve` workflow; `sweep` records and prints each candidate's decode expert hit rate. | The curve itself was not measured (needs a quiet AC host). Encode CPU ms and eviction-scan time are not columns: the runtime exposes no counter for either. |

## Wave 1

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-01 teacher-forced decode gate | done | Worker mode `forced_decode`; every KL or bit-exact gate of a decoding workload replays the baseline's greedy ids through the candidate (up to 256 steps on 64, 500 and 2000 token prompts) and judges per position. A decode-only change can no longer pass on prefill logits. `gate --decode-trace` forces it. | Speculative and chat workloads stay token-identity only: the runtime loop does not hand out logits. The qwen38 and qwen36 pilots the plan names were not run (neither install is in the catalog store on this host). |
| LAB-02 per-token KL statistics | partial | KL mean, median, p99 and SE, flip rate and flips by reference margin, PPL per arm, paired NLL delta and SE, over the forced decode trace and the prefill logits at every chunk end. The trace has two text classes, documents and chat, each held to the bound on its own. | Prefill positions are chunk ends, not every token: the chunked kernels return one row per chunk, and a family whose chunk driver skips the head records none. Prefill positions are not split by class (that corpus is all documents). No compact u16 reference. |
| LAB-03 A/A quality noise | partial | `gate --noise-arms`, `tslab noise`, and the `token_divergence` class for a change of verify width: both arms run under the speculative probe, and outputs may part only on a near tie (baseline margin under 0.5 logits) and no more often than the model's own `noise` record (zero without one). | Covers speculative verify width only, not batched decode or prefill batch width. Tree rounds record no margin, so a divergence there fails. No first-target runs (qwen36, MTP 2 against 3). |
| LAB-04 floor table | partial | `tslab floor` records a family's arm against itself; `gate` prints limit, floor and margin. No path widens a bound. | The CPU or llama.cpp reference half stays `xengine-kl`, run separately. Floors for the listed families were not collected. |
| LAB-08 golden answers | partial | `tslab sanity`: 8 exact cases, `--tools`, `--needle` (4K, 16K, 64K at depths 10, 50, 90) and `--candidate` as the functional gate; 15 of 15 on inkling2-mini-ud-q4kxl with the tool and 4K needle cases. A `sanity` step in `new-family`. | The 16K and 64K needles and the per-family mutation pilot were not run. |

## Wave 2

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-11 route trace v2 and replays | partial | The router trace adds `gate_weights` and `token_index`, with positions for every family's chunked prefill, per-token decode and verify; `expert_policy_replay.py` adds `admit:`, `--keep`, `--felt-fraction`, `--layer-cost` and `predicted_gain`; `calibrate --predicted` joins it to a verdict. | No qwen36 or Gemma 4 capture was taken, so no policy is ranked. |
| LAB-12 layer-major prefill replay | done | `prefill_bank_replay.py --layer-major` over chunk sizes and slot budgets. | With private per-layer slot caches, layer-major equals batch-major LFU by construction; it only beats a bank other layers overwrite. A build case needs shared slot memory, which is not modeled. No long-prompt capture replayed. |
| LAB-13 union pricing | done | `moe_route_replay.py --streams` and `--phase verify` (distinct experts per verify), `depth-replay --cost-model union`, and per verify expert reads and bytes in `tslab spec-rounds`. | Reads are requested, not physically read. The streamed 35B pilot was not run. |
| LAB-10 verify width beyond 16 | done | `verify_ladder` covers widths 1 to 64 and 96 through the sliced GEMM route, with step edges and a logits-memory budget. | No speculative verify accepts more than 16 rows today, so wider points are a what-if cost. The qwen38 and qwen36 sweeps were not run (not in the catalog store here). |
| LAB-17 fill ladder | done | `fill-ladder` workflow (0.3K, 8K, 32K); `ab` and `sweep` print and record a per-context-bucket decode ratio and footprint, never pooled into a verdict. | The 64K and larger rungs are not in the built-in plan: the corpus ends near 100K tokens and a long rung on a streamed MoE competes with its slot pool. Not run on installs here. |
| LAB-24 GPU-idle gap | partial | `trace_tables.py` per-token idle, busy and command buffers; `tslab idle-gap` reports the first step after each gap and how long its first command buffer waited between kernel scheduling and GPU start. | The token count for the trace is given by the caller. The one reading below was host timed on a shared machine; the GPU-side delay has not been read on a model. |

## Wave 3

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-18 server bench | partial | `tslab serve-bench` over loopback HTTP/SSE. The server now sends llama.cpp-style `timings` and `usage.prompt_tokens_details.cached_tokens`, so server tok/s and cached tokens are readable. | No queue or coalescing arms. The server-tax comparison was not run. |
| LAB-19 open loop and goodput | partial | Closed, periodic and Poisson arrivals with SLO and goodput. `GET /v1/metrics` now carries `turbospark_text_*` series on every server. | `serve-bench` records each series' growth over the run (`server_metrics`); they are counters, not histograms. No open-loop pilot. |
| LAB-35 soak | done | `tslab soak`; a 60 s run against the real server completed with no failure. | The hour-long soak was not run. |
| LAB-20 agent workloads | partial | `chat:` `edit_variants` and `chats/edit-last-turn.chat.json`. | `warm_start` is NOT built: it needs KV-cache persistence to disk across every family's state layout, a product feature of its own (the server accepts `--paged-ssd-cache-dir` and does not honour it). A message-span checkpoint arm likewise needs runtime work. |
| LAB-15 cold-state hygiene | partial | Opt-in compressor and swap dirty rules; `file_age_s` per run; `open_pressure` (compressor pages, swap growth, and peak swap sampled once a second across the open). | Thresholds are not calibrated. No file-age refusal: no tslab command times two installs against each other. The sampling is in the lab worker, not in the bench memory oracle. |
| LAB-21 memory ballast | done | `~ballast:GIB` (at most 12), refused when the host already swaps, dirty on swap growth. | Never run with a real model: this 36 GB host has been rebooted by memory pressure before, and the plan asks for an AC, quiet host. |
| LAB-22 quant_ab | done | `tslab quant-ab`; the pass rule lives in `gates.rs`. | PPL is on the reference's own greedy ids, not a held-out multilingual and code slice. Not run on a pair of installs. |
| LAB-29 release gate | partial | `tslab release-gate` with `--base-worker`, declared changes, SKIP rows. | A command, not a workflow file: it needs a second built worker, which no workflow step can produce. The HEAD against HEAD~20 pilot was not run. |
| LAB-30 toolchain provenance | done | `toolchain_id` in every provenance; `tslab anchor` keyed by family and toolchain. | No anchor file is committed: anchors belong to a release host's installs. |

## Unscheduled

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-09 methodology id | done | Every run and verdict records `methodology`; the logit trace has a row-count footer; `export` and the HTML report split a group that mixes ids; `report --md` says so. | Old records are not backfilled (they read `unrecorded`): their prompt bytes cannot be recovered reliably. `pinned` holds upstream settings, not ledger series, so it has nothing to group. Quality logit files keep their size check. |
| LAB-23 synthetic acceptance | partial | `tslab spec-rounds --synth-len` forces acceptance to a mean committed length through a thread-local runtime probe no server or app path installs; such runs are marked unquotable and never feed round costs. | Greedy chain and copy rounds only (not tree or sampled). One smoke run only: no break-even per family yet. |
| LAB-25 busy-age round cost | partial | Every probed round records its time into the run; `spec-rounds` prints verify median by width and run age (0-6, 6-15, 15-60, 60+ s). | Age is time since the probe started, not since the last GPU idle gap. The depth rule is unchanged until a measurement moves the ratio. No 120 s run. |
| LAB-27 tile sweeps and no-harm rule | partial | The no-harm-per-bucket rule, attached to the `rows_attn`, `moe_routed` and INT4 sweeps as a chip-keyed record. | It proposes a pin and changes no default. No decode-GEMV threadgroup arms where the gpu crate takes no tile parameter. No sweep was run for a pin. |
| LAB-28 real routes in `moe_routed` | partial | `moe_routed --routes trace:\|uniform\|zipf:`, rows-per-expert histogram, pair against expert-sorted grid with hash equality (synthetic test). | Not run on a real capture or install: the bench synthesizes affine INT4 experts and no such MoE install is here. |
| LAB-31 host-quiet extensions | done | Low Power Mode and memory-pressure level in every host sample; `--require-full-power` and `--require-normal-pressure`. | Opt-in: old ledgers never recorded either, so the number of runs a default rule would dirty cannot be counted yet. |
| LAB-32 markdown report | done | `report --md` with the derived "claims not made" list. | |
| LAB-33 secondary guard | done | Spec `secondary_guard` and `sweep --guard`, including `expert_hit_rate` as the guard metric. | The allowance is a relative change (at most 2%), not percentage points. |
| LAB-34 GDN trajectory | partial | `gdn_trajectory` bench with a one-ULP fault self-test (synthetic GDN test). | Not run on a real GDN install (none here). |
| LAB-36 image denoise stage gate | partial | `tslab image-stage` runs each Z-Image denoise step from the reference latent through `MetalImageBackend::denoise_one_step` and reports update error, cosine and output hash; with `--against` it fails on any changed step. | Never run on a model: no Z-Image install or capture on this host, so neither the new backend entry nor its parity test (`one_step_from_a_rollout_latent_reproduces_the_next_latent`, ignored) has executed. No quality bound: it needs a measured reference floor. Z-Image only. |

## Found by these checks and fixed

Tool calling on Inkling 2 mini was broken twice over. The chat template calls
`tojson(separators=...)`, which the template engine refused, so no tool
request rendered; and the model's call frame had no parser, so a rendered
request would have returned the call as reply text. `crates/tokenizer` now
accepts the `json.dumps` arguments (byte-identical to Python on the real
template, other calls unchanged) and parses
`<|content_invoke_tool_json|>` into a tool call.

## What was run on hardware

All on this Mac with inkling2-mini installs; none is a speed claim.

- `tslab floor`: prefill and 768 forced decode positions bit-identical run to
  run (mean KL 0, no flips).
- `tslab sanity`: 8 of 8. The first version passed on this family's
  reasoning text, which uses its own segment markers rather than `<think>`;
  `golden.rs` now reads the answer segment and has a test for it.
- `tslab serve-bench` and a 60 s `tslab soak` against `turbospark-server`.
- `tslab launch-count`.
- `tslab kernel --bench moe_routed,verify_ladder,gdn_trajectory`: the ladder
  ran, the other two refused this install with their reason.
- `tslab type1` over the stored null calibrations (LAB-05) and the msync
  eviction test (LAB-14).
- `cargo test --workspace` passed locally. The Swift gates were not run: no
  Swift or FFI source changed.

## Later runs, 2026-11-26

All on inkling2-mini, on a Mac another session was also using at times.

- Tool calling end to end against `turbospark-server` (UD-IQ1_S): the call
  returns as `tool_calls`, the tool result is answered, streaming matches,
  and `timings` and `/v1/metrics` are populated.
- `sanity --tools --needle 4096` (UD-Q4_K_XL): 15 of 15.
- `spec-rounds` with DFlash at depth 3: 2.94 committed per round under the
  model, 2.18 with acceptance forced to a mean of 2. No expert reads were
  reported: Inkling 2's flow does not fill that counter.
- Prefill-position gate: 49 chunk-end positions per run, bit-identical
  against itself.
- `idle-gap`: first decode step about 41 ms after a 0.5 or 1 s gap and 240 to
  290 ms after 2, 4 and 8 s (busy median 26 to 33 ms). One reading, host
  timed: it opens the question, it does not name the cause.
