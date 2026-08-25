# turbospark-selection

Candidate token selection (`select`, `select_from_logits`) from per-candidate logit vectors under a validated shaping configuration (temperature, top-k, top-p, repetition penalty, seed, step position, distribution guards).

## Directory & File Structure

```
crates/selection/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library root
|   +-- shaping.rs          # Main select and select_from_logits entry points
|   +-- choose.rs           # Categorical distribution sampling and argmax selection
|   +-- penalty.rs          # Frequency and repetition penalty application logic
|   +-- truncation.rs       # Top-k and top-p (nucleus) candidate truncation
|   \-- derive.rs           # Derived selection parameters and helper functions
\-- tests/
    +-- determinism.rs      # Verifies seed determinism across selection runs
    +-- distribution.rs     # Verifies temperature and logit distribution behavior
    +-- domain_guards.rs    # Edge case guards (NaN/Inf logits, empty candidates)
    +-- host_sampler_cost.rs# Micro-benchmark profiling host sampler component costs
    +-- penalty.rs          # Repetition penalty calculation unit tests
    +-- rank_top_k.rs       # Top-k partial ranking vs full sort equivalence tests
    +-- shaping_validation.rs# Parameter range validation unit tests
    \-- truncation_order.rs # Top-k and top-p truncation order unit tests
```

## Key Modules

- `shaping.rs`: Main entry points `select` and `select_from_logits`.
- `choose.rs`: Categorical sampling from probability distributions.
- `penalty.rs`: Repetition penalty application based on token history.
- `truncation.rs`: Top-k and top-p (nucleus) filtering logic.
- `derive.rs`: Derived selection parameters and helper functions.

## Development & Test Commands

```sh
# Run tests for turbospark-selection
cargo test -p turbospark-selection
```

## Crate Gotchas

1. **Logits in, Probs Out Internally**: `selection::select` accepts raw unnormalized LOGITS and applies softmax internally. Never pass pre-softmaxed probabilities into `select`. Passing probabilities causes double-softmaxing (`softmax(softmax(z))`), collapsing top-k/top-p distributions toward uniform and ruining sampling.
2. **Deterministic Parity**: Selection uses a deterministic pseudorandom generator seeded per run to ensure exact reproducibility across runs when a seed is provided.
3. **This crate runs per decoded token at the FULL vocabulary, and no profiler in this repo sees it.** `select` is called from `run_raw_completion` after `LogitProducer::produce` returns, so it is outside every `MFERENCE_PHASES=1` bucket and every dispatch ranking (AGENTS.md Gotcha 23). At Gemma 4's V=262144 an O(n log n) step here costs more than the entire GPU forward pass: the full sort `rank_top_k` replaced was 18.9 ms/token against a ~25 ms pass, and was the whole of this port's one-time 1.5x decode gap against Swift (`docs/BENCHMARKS.md`). Treat any new whole-domain pass as a throughput change and measure it with `cargo test -p turbospark-selection --release --test rank_top_k -- --ignored --nocapture`.

   **WHAT IS LEFT IS NOT WHERE IT WAS ASSUMED TO BE**, split 2026-08-15 by `tests/host_sampler_cost.rs` at V=262144, T=0.2, top-k 64 (release, one discarded warmup, parts summing to 97.7% of the whole so nothing is hiding):

   | part | ms/call | share |
   |---|---|---|
   | `rank_top_k_u32_into` | 2.061 | 70% |
   | exp pass, f64 as shipped | 0.669 | 23% |
   | widen f16 -> f32 + finite check | 0.142 | 5% |
   | `select` whole | 2.940 | |

   The standing note that the residual is "softmax over 262144 f64 plus the index Vec" had the ORDER of those two backwards. The exp pass is 23%, and narrowing it to f32 saves 0.181 ms -- 0.72% of a ~25 ms token, which does not pay for touching the sampler at all (and see Gotcha 4 for why it would not even be the change it looks like). **The partial ranking was 70% of the sampler even after the 2026-08-07 fix**, and its cost was INDIRECTION rather than complexity: `select_nth_unstable_by` partitions 262144 `u32` indices under a comparator that random-accesses a 2 MiB `f64` key array, so almost every comparison is a cache miss, on top of a 1 MiB identity permutation written first.

   **FIXED the same day by the sequential cut in `rank_top_k_u32_into`**, which finds the k-th largest value in one streaming pass and ranks only the indices that reach it: `select` as a whole went 2.940 -> 1.181 ms/call, and the ranking 2.061 -> 0.326. Because absolute timings on this machine drift ~25% with desktop load (AGENTS.md Gotcha 43), the number to quote is the PAIRED one from `the_cut_path_against_the_partition_it_replaced`, which times both arms interleaved in one process and reads **6.5-7.0x** across rounds. Output is bit-identical: greedy `67a23bb5...` and sampled `ebfba17a...` unchanged on the real Gemma 4 install, and `quality_gate` reproduced perplexity 37.4176 with both frozen digests.

   **THE WIN IS A FUNCTION OF `top_k`, AND THE REPO'S GREEDY SMOKE CANNOT SEE IT.** At `--top-k 1` the old path called `select_nth_unstable_by(0, ..)`, which is already `O(n)` with no sort, so the greedy smoke moves barely at all -- the first end-to-end check of this change was run greedy and read a null result, which looked like the micro-benchmark being wrong. At the CLI's SAMPLED defaults (T=0.2, top-k 64, top-p 0.95), which is what a user actually runs, it is the full effect. Same species as Gotcha 16: greedy is not a weaker test here, it is a test of a different code path.

   Confirmed end to end by the Gotcha 23 subtraction rather than by tok/s, because a residual is a WITHIN-RUN quantity and survives the contention a throughput row does not. `wall - phases_total` per forward pass, three interleaved pairs at the sampled defaults: **2.903 / 2.903 / 2.982 ms/call before against 1.210 / 1.236 / 1.325 after, i.e. 1.657-1.693 ms/call saved**, against the 1.759 the micro-benchmark predicts. Decode read +10.0% on two of three pairs. That subtraction is also the standing check that nothing has crept back outside `produce`: it now accounts for ~98% of a decode run, where before the 2026-08-07 sampler fix it accounted for 63%.

   The exp pass is now the sampler's largest single part. It is still not worth narrowing, for the reason above.

4. **The ranking is INVARIANT under exp precision far below f32, and the reason is upstream of this crate.** `LogitValue` is f16, so two logits that differ at all differ by at least one f16 ULP (~0.008 at magnitude 8); `exp` is monotone and that gap survives at any float width worth discussing, while equal f16 logits give exactly equal exps at every width and fall to the same ascending-index tie-break. Measured rather than argued: the top-64 order is unchanged by an f32 exp pass and first moves at **7 mantissa bits**, against f32's 24 (`the_rankings_sensitivity_to_exp_precision_is_measured_not_assumed`). That test reports the breaking point instead of asserting either answer on purpose -- a fixture whose top-k were too separated to reorder at any precision would report 1 bit and be visibly useless, which is the failure mode a plain `assert_eq` would have hidden. **This does NOT make an f32 exp pass output-identical**: `exps[i] / sum` is read again by the top-p cumulative walk and by `reweight`'s `powf(1/T)`, so the surviving set would be the same while the draw's weights moved in their low bits. The ranking is safe; the draw is not, and no measurement here bounds that.
5. **Only a prefix of the ranked order is observable.** Probability-mass truncation keeps a prefix, and rank truncation caps that prefix, so with `top_k > 0` nothing past rank `top_k` can change the result -- which is what licenses the partial ranking. The full sort remains the fallback for `top_k == 0`. `tests/rank_top_k.rs` pins the usize reference and the `_u32_into` scratch variants to agree, ties and all; a tie-break difference would change sampled output while every throughput number improved, and greedy would not notice (Gotcha 16's failure mode).
6. **The hot path never materializes the probability vector.** `select` keeps thread-local scratch (`working`, `exps`, `ranked`) and ranks unnormalized `exp(s - max)` keys via `rank_top_k_u32_into`; division by the positive normalizer is monotone, so the order matches ranking probabilities, and `exps[i] / sum` reproduces the retired full vector's entries bit for bit where they are read (top-p cumsum, reweight). Any change here must keep the real-model sampled smoke md5-identical, not just the greedy one.
7. **`--temperature 0.0001` is not the fast path.** `is_deterministic` is `temperature == 0.0` exactly, so the repo's "greedy" smoke runs the full sampled pipeline. Only an exact `0` takes the `argmax` early return.
