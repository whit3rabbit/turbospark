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
| LAB-05 type-I audit, non-inferiority | done | `tslab type1` replays stored null calibrations: on this host (163 pair improvements from 56 calibrations) the false WIN rate at 3 pairs was 0.6%, under the 5% line, so `min_pairs` stays. `noninferiority_limit` and `ab --noninferior` (10 pairs, one-sided bounds, limit at most 3%). The sweep prints its false-WIN bound. The stale `decide` comment is fixed. | The replay resamples pairs as independent, so within-session drift is not in it. The autoresearch loop's own multiplicity over many specs is not corrected: it is held by the single held-out `confirm`. |
| LAB-06 route attestation | partial | Print-only engagement lines and knob rows added for part of the unattested knobs; `launch-count` lists the pipelines each arm used. | The agent stopped before listing which knobs stay unattested. No automatic default-route warning in `ab` (a counted run is not a timing run). The frozen A/Bs were not re-run. |
| LAB-14 page-cache eviction | done | `streaming::evict_file_pages` (`msync(MS_INVALIDATE)`) and the `~cold` arm. Measured by test on this macOS: a hot 8 MiB file reads 0 physical bytes, and 8 MiB after eviction. | The install-walk use (`--evict` for quantize and stream installs) is an installer change, not a lab change. `docs/BENCHMARKING.md` still asks for `sudo purge` for frozen cold rows: one file's test is not evidence for a whole install. |
| LAB-26 launch counts | done | `TURBOSPARK_LAUNCH_COUNT=1` counter in the gpu crate and `tslab launch-count`. | Not run on a real install; per-verify-round counts are not split out. |
| LAB-16 slot curve | done | `slot-curve` workflow; `sweep` records and prints each candidate's decode expert hit rate. | The curve itself was not measured (needs a quiet AC host). Encode CPU ms and eviction-scan time are not columns: the runtime exposes no counter for either. |

## Wave 1

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-01 teacher-forced decode gate | done | Worker mode `forced_decode`; every KL or bit-exact gate of a decoding workload replays the baseline's greedy ids through the candidate (up to 256 steps on 64, 500 and 2000 token prompts) and judges per position. A decode-only change can no longer pass on prefill logits. `gate --decode-trace` forces it. | Speculative and chat workloads stay token-identity only: the runtime loop does not hand out logits. The qwen38 and qwen36 pilots the plan names were not run (neither install is in the catalog store on this host). |
| LAB-02 per-token KL statistics | partial | KL mean, median, p99 and SE, flip rate, share of flips under 0.5 logits of reference margin, flips at 2 logits or more (fails a KL class), PPL per arm, paired NLL delta and SE, per prompt. | Prefill-side stays final-position. No split by text class beyond per prompt (the corpus is one class). No compact u16 reference: it is the same size as the f16 trace, which is the saved reference `quant-ab` reuses. The pilots were not run. |
| LAB-03 A/A quality noise | partial | `gate --noise-arms` compares each arm with itself; `tslab noise` records distinct outputs and first divergence over K fresh processes, at any arm including a verify width. | The `TokenDivergence` gate class is not built: it needs the reference margin at a divergence inside the speculative loop, which the runtime does not expose, and a looser class must not exist without that evidence (root rule on tolerances). |
| LAB-04 floor table | partial | `tslab floor` records a family's arm against itself; `gate` prints limit, floor and margin. No path widens a bound. | The CPU or llama.cpp reference half stays `xengine-kl`, run separately. Floors for the listed families were not collected. |
| LAB-08 golden answers | partial | `tslab sanity` (8 exact cases through the chat template, think-leak and advisory garbage flag), a `sanity` step in `new-family`. | The needle grid (4K to 64K) and strict tool-call class (`gate --class functional`) are queued in the plan; not built. The per-family mutation pilot was not run. |

## Wave 2

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-11 route trace v2 and replays | partial | The router trace adds `gate_weights` and `token_index`; `expert_policy_replay.py` adds `admit:`, `--keep`, `--felt-fraction`, `--layer-cost` and `predicted_gain`; `calibrate --predicted` joins it to a verdict. | `token_index` is -1 for families whose batched prefill opens no scope. No qwen36 or Gemma 4 capture was taken, so no policy is ranked. |
| LAB-12 layer-major prefill replay | done | `prefill_bank_replay.py --layer-major` over chunk sizes and slot budgets. | With private per-layer slot caches, layer-major equals batch-major LFU by construction; it only beats a bank other layers overwrite. A build case needs shared slot memory, which is not modeled. No long-prompt capture replayed. |
| LAB-13 union pricing | partial | `moe_route_replay.py --streams` (union growth) and `--phase verify`; `depth-replay --cost-model union`. | No per-verify `[spec-stats]` line: it needs a new producer method and per-family plumbing. Bytes are requested bytes, not physical. The streamed 35B pilot was not run. |
| LAB-10 verify width beyond 16 | partial | `verify_ladder` extended past 16 rows. Code from a subagent that was stopped before it reported: present in the tree, only what the test run covers is verified. | Not reviewed or measured. |
| LAB-17 fill ladder | done | `fill-ladder` workflow (0.3K, 8K, 32K); `ab` and `sweep` print and record a per-context-bucket decode ratio and footprint, never pooled into a verdict. | The 64K and larger rungs are not in the built-in plan: the corpus ends near 100K tokens and a long rung on a streamed MoE competes with its slot pool. Not run on installs here. |
| LAB-24 GPU-idle gap | partial | `trace_tables.py` reports idle and busy ms per token and command buffers per token (`trace --window-tokens N`). | The token count is given by the caller: the trace window is not aligned with decode. The `idle-gap` sleep probe is not built: its kernel-start half needs `kernelStartTime` exposed by the gpu crate. |

## Wave 3

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-18 server bench | partial | `tslab serve-bench` over a loopback HTTP/SSE client. Code from a subagent that was stopped before it reported: present in the tree, only what the test run covers is verified. | No run against a real server; no in-process server-tax comparison confirmed. |
| LAB-19 open loop and goodput | partial | Closed, periodic and Poisson arrivals with SLO and goodput. Code from a subagent that was stopped before it reported: present in the tree, only what the test run covers is verified. | No `/metrics` scrape: the server has no text metrics surface. No pilot. |
| LAB-35 soak | partial | `tslab soak`. Code from a subagent that was stopped before it reported: present in the tree, only what the test run covers is verified. | Never run against a real server. |
| LAB-20 agent workloads | partial | `chat:` files take `edit_variants` (A B C C B A from one history); `chats/edit-last-turn.chat.json`; per-turn reuse and TTFT already print. | `warm_start` and a message-span checkpoint arm need runtime features that do not exist. |
| LAB-15 cold-state hygiene | partial | Opt-in `--max-compressions` and `--max-swap-growth-mib` dirty rules; every run records `file_age_s`. | Off by default: no threshold was calibrated from this host's nulls. No file-age refusal in `sweep` (a sweep compares knobs on one install, never two layouts). No 1 Hz compressor sampling in the memory oracle (`crates/bench`, outside the lab). |
| LAB-21 memory ballast | done | `~ballast:GIB` (at most 12), refused when the host already swaps, dirty on swap growth. | Never run with a real model: this 36 GB host has been rebooted by memory pressure before, and the plan asks for an AC, quiet host. |
| LAB-22 quant_ab | done | `tslab quant-ab`; the pass rule lives in `gates.rs`. | PPL is on the reference's own greedy ids, not a held-out multilingual and code slice. Not run on a pair of installs. |
| LAB-29 release gate | partial | `tslab release-gate` with `--base-worker`, declared changes, SKIP rows. | A command, not a workflow file: it needs a second built worker, which no workflow step can produce. The HEAD against HEAD~20 pilot was not run. |
| LAB-30 toolchain provenance | done | `toolchain_id` in every provenance; `tslab anchor` keyed by family and toolchain. | No anchor file is committed: anchors belong to a release host's installs. |

## Unscheduled

| Id | Status | What the lab does now | Not done, and why |
|---|---|---|---|
| LAB-09 methodology id | partial | Every run and verdict records `methodology` (kind, version, hash of workload and prompt bytes); the logit trace has a row-count footer; `report --md` flags mixed ids. | `export`, `pinned` and the HTML report do not group by it. Old records are not backfilled (they read `unrecorded`). Quality logit files keep their size check, no footer. |
| LAB-23 synthetic acceptance | not built | none | It needs an override of verify decisions inside the production speculative loop, including KV and GDN rollback of intentionally wrong tokens. That is a runtime change with correctness risk in the shipped path and needs a decision of its own. |
| LAB-25 busy-age round cost | not built | none | The runtime reports round costs per call, not per round with timestamps, and the plan marks it NEEDS-DATA behind LAB-10's measured ladder. |
| LAB-27 tile sweeps and no-harm rule | partial | A no-harm-per-bucket rule (`kernel/tile_pin.rs`). Code from a subagent that was stopped before it reported: present in the tree, only what the test run covers is verified. | Not reviewed; no sweep run. |
| LAB-28 real routes in `moe_routed` | partial | `moe_routed --routes` (`kernel/routes.rs`). Code from a subagent that was stopped before it reported: present in the tree, only what the test run covers is verified. | Not reviewed; no real capture replayed. |
| LAB-31 host-quiet extensions | done | Low Power Mode and memory-pressure level in every host sample; `--require-full-power` and `--require-normal-pressure`. | Opt-in: old ledgers never recorded either, so the number of runs a default rule would dirty cannot be counted yet. |
| LAB-32 markdown report | done | `report --md` with the derived "claims not made" list. | |
| LAB-33 secondary guard | done | Spec `secondary_guard` and `sweep --guard` (at most 2%). | No expert hit-rate guard: the hit rate is printed beside the verdict instead. |
| LAB-34 GDN trajectory | partial | `gdn_trajectory` bench with a one-ULP fault self-test. Code from a subagent that was stopped before it reported: present in the tree, only what the test run covers is verified. | Not run on a real GDN install. |
| LAB-36 image denoise stage gate | not built | none | Needs per-step reference latents captured from the BF16 or FP32 reference and an image bench kind; no such fixtures exist, and it depends on LAB-04 floors for its bounds. |

## What was run on hardware

- `tslab floor` on `inkling2-mini-gguf-ud-iq1s`: prefill and 768 forced
  decode positions bit-identical run to run (mean KL 0, no flips).
- `tslab type1` over this host's stored null calibrations (see LAB-05).
- `msync_invalidate_evicts_hot_pages` (see LAB-14).
- Nothing else was run against a model. No speed claim was taken.

## Open at the time of the first commit

Work stopped on a usage limit. Known and not fixed:

- `knobs::tests::a_knob_set_on_a_path_that_prints_nothing_is_flagged` fails
  (`mapped_demand_prep=touch`): the engagement rows are unfinished (LAB-06).
- `kernel_trajectory_and_routes::route_sources_run_and_the_sorted_grid_changes_no_bit`
  fails: a route source repeats an expert in one pass (LAB-28).
- `cargo clippy`, `cargo test --workspace`, the Python script tests after the
  last edits, and the Swift gates were not run.
