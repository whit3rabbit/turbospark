# MiniMax-M2: Phase 0

Status, 2026-09-10: GGUF intake and text execution implemented, and the pinned
Q4_K_M install completed. Synthetic Metal inference, the real memory oracle,
and greedy/sampled short-answer EOS checks pass. The standard low-temperature
coastal-wetlands smokes repeat reasoning and exhaust 400 tokens. These failures
block release; there is no catalog row or accepted MiniMax performance baseline.

## Witness and scope

The header-only probe read `general.architecture = minimax-m2` from
[Unsloth's first Q4_K_M shard](https://huggingface.co/unsloth/MiniMax-M2-GGUF/blob/06e952eab9e8e136847e9df067a0032582696778/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf).
`crates/repack/tests/minimax_network.rs` validates all three headers at
checkpoint revision `06e952eab9e8e136847e9df067a0032582696778`.
Tokenizer/config sidecars are pinned to MiniMaxAI/MiniMax-M2 revision
`757303d492a50514c312788b5247a4f696a4c6a3`.

```sh
cargo run -p turbospark-cli --bin turbospark-model -- probe \
  unsloth/MiniMax-M2-GGUF \
  --file Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf \
  --sidecar-repo MiniMaxAI/MiniMax-M2
```

Start with GGUF intake. The original checkpoint uses FP8 block quantization;
that is a separate source-format project. Do not infer M2.1 or later model
support from this witness. MTP and vision are outside this initial scope.

## Checkpoint facts

The [publisher's config](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/757303d492a50514c312788b5247a4f696a4c6a3/config.json)
was read on 2026-09-10. HF spells the type `minimax_m2`.

| Property | Value |
| --- | --- |
| Trunk layers / hidden width | 62 / 3072 |
| Attention | Full attention throughout; no sliding window |
| Q heads / KV heads / head width | 48 / 8 / 128 |
| Rotary width / theta | 64 / 5000000 |
| Experts / selected / expert FFN width | 256 / 8 / 1536 |
| Shared expert width | 0 |
| RMS epsilon | 1e-6 |
| Vocabulary / trained context | 200064 / 196608 |
| Embedding head | Untied |

The explicit head width matters: hidden width divided by Q heads is 64,
which is not this checkpoint's 128. `mlp_intermediate_size` is not the routed
expert width. The eight selected experts determine the minimum slot count.

## Layer contract

Two independent implementations agree:
[MLX MiniMax](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py)
and [llama.cpp MiniMax-M2](https://github.com/ggml-org/llama.cpp/blob/master/src/models/minimax-m2.cpp).
Implementation references, resolved on 2026-09-10:

- [MLX source at 7453524](https://github.com/ml-explore/mlx-lm/blob/745352405f0909540760fd9b9ff16d933fd9c82b/mlx_lm/models/minimax.py).
- [llama.cpp source at e5a8d43](https://github.com/ggml-org/llama.cpp/blob/e5a8d439cef31f27fad6938233da10dae1ba5631/src/models/minimax-m2.cpp).

Attention uses separate bias-free Q/K/V projections. Learned Q/K RMS norms
span the entire projected vectors, before the heads are reshaped: widths
6144 and 1024. They are not per-head norms. V is not normalized. Rotate the
leading 64 coordinates of each 128-wide head using split-half pairing;
attention scales by `1/sqrt(128)`.

Each block is pre-norm attention plus residual, then pre-norm routed SwiGLU
plus residual. There are no sandwich norms, shared expert, or output gate.
The final learned RMS norm feeds an untied linear head returning raw logits.

MLX makes the routing distinction explicit:

```text
scores = sigmoid(router(x))
selected = top8(scores + correction_bias)
weights = scores[selected] / sum(scores[selected])
output = sum(weights * selected_expert_outputs)
```

The correction bias changes selection only. Putting it into the weights, or
substituting softmax, changes the model. Preserve router ranking when reducing
expert outputs, following this repo's existing deterministic slot contract.

## Implementation and intake evidence

`ModelFamily::MiniMaxM2` is appended with manifest spelling `minimax_m2`;
GGUF uses `minimax-m2`. HF recognition does not enable safetensors intake.
The shared Llama flow now selects no, per-head, or whole-projection Q/K
normalization by family. MiniMax uses FP32 router weights, FP32 accumulation,
and the original FP32 correction bias. Other F32 norms narrow to BF16 under
the existing resident contract. Experts retain their original GGUF blocks.
Mapped expert residency remains refused for MiniMax.

`GgufSet` merges the inventory into logical offsets and dispatches each
range to its source shard. Probe and install share discovery and validation:
complete numbered shards, consistent split metadata, unique tensor names,
checked file bounds, and declared total tensor count. A layer may reference
multiple shards. Installation streams to `.gturbo` without staging source
weight files; progress and cancellation reach every shard's HTTP source.

The pinned artifact contains 809 tensors: 288 / 299 / 222 across its three
shards. There are 375 Q4_K, 61 Q6_K, and 373 F32 tensors. Every layer has 13
roles: four attention projections, four norms, router, correction bias,
and three expert projections. Q/K norms have shapes [6144] / [1024]; the
router is [256, 3072] and `exp_probs_b.bias` is [256]. Gate/up experts use
Q4_K; down projections mix Q4_K and Q6_K by layer.

| Header-derived quantity | Bytes |
| --- | ---: |
| Source download, all three shards | 138342385120 |
| Resident file, including index | 2606941184 |
| Padded expert files | 135819952128 |
| Install allowance, including 64 MiB sidecars | 138494002176 |
| Largest padded expert stride | 9191424 |
| Eight expert slots across all layers | 4244373504 |
| FP16 KV at 8192 context | 2080374784 |

These are storage/allocation calculations, not measured process footprint.
Per-layer strides matter: padding every layer to the largest stride would
overstate expert storage. The required free-space preflight is the complete
install allowance plus at least 20 GiB. No existing models are deleted.

The completed install at `~/models/minimax-m2-q4km.gturbo` contains all 62
expert files: 135819952128 bytes, plus the 2606941184-byte resident file.
Both exactly match the header calculation. Final free space was 27.4 GiB
after reclaiming disposable compiler incremental caches, without deleting
models. Manifest SHA-256:
`4e72c852af141a05be96af8fdefd21fa1a38e66ba32c50f18809265067a0a453`.

## Validation record

- Three pinned headers: all 809 tensors mapped and sized; PASS.
- The catalog network guard now uses shared split discovery for aggregate
  GGUF download sizes. Both network checks pass for the existing catalog;
  the MiniMax row remains gated on real-checkpoint execution and measurements.
- Split fixtures: missing shards/count metadata, duplicate tensors, inconsistent numbering,
  architecture metadata (including keys first introduced by later shards),
  out-of-file ranges (including a truncated empty shard header), total count,
  filename, and a layer spanning two shards;
  PASS. The direct repack path also refuses non-F32 router/bias tensors and
  zero experts before sizing; all 14 intake fixtures pass, including the
  untied-head guard described below.
- Independent Metal FP32 router dot products with buffer offsets and a
  non-SIMD-multiple width; PASS.
- Whole-projection RMS normalization at widths 6144 and 1024 against scalar
  arithmetic with nonuniform learned weights, in place; PASS.
- Attention at 48 Q heads / 8 KV heads / width 128 agrees with the scalar
  reference. Distinct KV-head means make incorrect 6-to-1 grouping visible.
- Selection-only correction bias, sigmoid weights versus softmax, and
  deterministic expert-index ties; PASS.
- Synthetic two-layer MiniMax: Q4_K gate/up, down switching Q6_K to Q4_K between layers, frozen logit digest
  `9422bdb22e2f00f2`, sequential/chunked agreement, reset repeatability, and
  8/16-slot agreement; PASS.
- Router and correction bytes preserved exactly through repacking; PASS.
- Manifest round-trip initially lost `routerScoringFunc`, causing the
  synthetic model to reopen as softmax. The writer now persists it.
- Mutation checks caught all eleven split-fixture failures, router selection
  bias, sigmoid weighting, tie order, FP32 router row/accumulation,
  whole-projection normalization, RoPE tail, FP32 resident tag, runtime norm
  dispatch, raw correction-bias low-bit loss, direct FP32 intake guards,
  the zero-expert guard, family registration, the pinned architecture
  baseline, 6-to-1 GQA grouping, and duplicate BOS. A first asymmetric comparator mutation
  survived; adding the reversed-order bias case caught it.
- MiniMax tokenizer loading initially failed on Gemma's required `<pad>`.
  A MiniMax token resolver now uses EOS as unused padding, takes the
  checkpoint stop token, and adds no BOS prefix. The real Jinja template
  renders with its original leading whitespace and forced-open `<think>`.
  A small fixture tests loading without pad, EOS, no duplicate BOS, missing
  template refusal, and thought splitting without native tool parsing.
  Because this template opens thought unconditionally, the turn splitter
  builds its decoder even at effort `off` and emits reasoning separately.
  Both activation and event classification are mutation-checked.
- MiniMax's untied-head contract is enforced at both boundaries: GGUF
  intake refuses a missing `output.weight` instead of inferring tied
  embeddings, and runtime refuses a tied-head MiniMax manifest. Each new
  case failed before its guard, passed afterward, and failed alone when
  its guard was removed. Both guards are restored; the valid fixture's
  frozen logits still match.
- The family-enumeration regression now includes the pre-existing
  `Qwen3Dense` variant and appended `MiniMaxM2`; that test passes.
  The explicit Q/K mode also fixes dense Qwen3's omitted per-head norms
  (the old boolean selected only Qwen3 MoE). Its pinned 0.6B Q8_0 real
  quality gate and old-mode mutation check pass, as recorded below.
- Workspace build, tests, formatting, and Clippy with Metal access: PASS
  after the final untied-head guards (full rerun completed 14:29 CDT).
  The intake target passes all 14 cases, and the MiniMax runtime target
  passes all three cases. Release inference and dedicated gate targets
  build. The portable nine-crate check also passes. Mistral and dense Qwen3
  real-model regressions pass, as do Qwen3 MoE's existing quality and memory
  gates (completed 15:06 CDT). The
  catalog's Mistral 7B Q4_K_M checkpoint has been restored at
  `~/models/mistral7b-dense.gturbo` (4371570688 resident bytes) after a
  storage preflight. Its manifest SHA-256 is
  `7f284bb560f487735cb885556d60d65c5243ed6c006193c717813739678907d0`.
  Qwen3-0.6B Q8_0 has also been installed at
  `~/models/qwen3-06b-regression.gturbo`, then leaving 21.07 GiB free. Source:
  `Qwen/Qwen3-0.6B-GGUF@23749fefcc72300e3a2ad315e1317431b06b590a`,
  sidecars `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
  Resident bytes: 633413632; manifest SHA-256:
  `d2867ccc5cab168b596d13e7c090f13bff2d95bfeca24102cf147423eefbeabf`.
  Dedicated dense Qwen3 memory and quality targets pin this artifact;
  they cover per-head normalization and the tied output head.
  Mixtral remains unavailable locally. Qwen3 MoE installation completed
  after reclaiming disposable debug build artifacts. Its pinned source is
  `Qwen/Qwen3-30B-A3B-GGUF@e4d4bafdfb96a411a163846265362aceb0b9c63a`,
  with sidecars `Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
  The 18556685824-byte source plus conservative padding and sidecar bounds
  fits a 19 GiB install allowance. Preflight free space was 42255335424
  bytes, exceeding that allowance plus 20 GiB headroom (41875931136 bytes).
  No model was deleted; release binaries and completed check logs were kept.
  The completed install contains 960235520 resident bytes and 17565745152
  expert-bin bytes (plus 4738426 bytes of expert layout metadata), leaving
  23687290880 bytes free. Manifest SHA-256:
  `2369b7b42885d2498ddd87774d14411420156ba0cf5d319d5029f9fa1f9ca2ed`.

## Real-checkpoint diagnosis

Machine: Apple M4 Max, 38654705664 bytes RAM, macOS 26.6.2 (25G83), AC power.
Runs use 8192 context, eight slots, FP16 KV, and the installed chat template.

- Standard coastal-wetlands greedy smoke (seed 1, temperature 0.0001,
  top-k 1): repeats the planning paragraph, then `MaxTokens` at 400.
- Same prompt sampled at CLI defaults (seed 20260721, temperature 0.2,
  top-k 64, top-p 0.95): repeats planning, then `MaxTokens` at 400.
- Short question, capital of France in one sentence: correct answer and
  `EndOfTurn` in both greedy and sampled runs, 109 / 57 generated tokens
  respectively (56 prompt tokens each). The sampled case uses temperature
  0.2, top-k 64, top-p 0.95, and seed 20260721.
- These truncated, repetitive smokes are not accepted benchmark cases.
  No kernel change has been justified by the symptom. In-place normalization
  and 6-to-1 GQA pass independent checks; the pinned llama.cpp model selects
  NeoX and its [MiniMax converter](https://github.com/ggml-org/llama.cpp/blob/e5a8d439cef31f27fad6938233da10dae1ba5631/conversion/minimax.py)
  does not permute Q/K weights.
- The pinned `generation_config.json` and
  [publisher guidance](https://github.com/MiniMax-AI/MiniMax-M2) recommend
  temperature 1.0, top-p 0.95, top-k 40. A temperature-only control
  (retaining top-k 64 and the same seed) avoids repetitive planning, but
  reaches `MaxTokens` at both 400 and 1024 while still writing the answer.
  With a 4096-token budget it completes coherently at `EndOfTurn`, 1240
  generated tokens (51 prompt tokens, 28.63 seconds prefill, 694.66 seconds
  decode). Its output matches the earlier 1024-token run through that cutoff.
  The Qwen3 MoE download overlapped this diagnostic, so its 1.785 tok/s is
  not a performance baseline. This does not prove that low-temperature
  repetition is intrinsic to the checkpoint.
  An [independent local-model puzzle experiment](https://big-stupid-jellyfish.github.io/GFMath/pages/parlor-reasoning)
  also reports reasoning loops in original M2 across several quantizations.
  Its workload and Q4_K_XL artifact differ from this gate, so that observation
  supports investigating model behavior but does not establish engine parity
  or waive this checkpoint's smoke failure.
- The exact frozen `short-explanation` protocol prompt at default sampling
  completes a coherent answer with `EndOfTurn`: 89 prompt tokens and 885
  generated tokens. Its explicit scope and word limit differ from the
  ad-hoc coastal-wetlands smoke. This supports testing the provisional
  4096-token budget; it does not retroactively pass either 400-token smoke.

The first oracle run passed (2026-09-10, 10:03-12:32 CDT). Its measured cases are:

| Case | Stop | Prompt tokens / prefill s | Generated tokens / decode s | Decode tok/s | Peak so far, MiB |
| --- | --- | --- | --- | --- | --- |
| short-explanation | EndOfTurn | 89 / 59.47 | 885 / 521.23 | 1.698 | 6278.4 |
| medium-review | EndOfTurn | 451 / 292.12 | 1393 / 862.51 | 1.615 | 6285.1 |
| long-synthesis | EndOfTurn | 2785 / 1820.25 | 975 / 638.64 | 1.527 | 6286.2 |

The oracle uses sequential prefill; the CLI diagnostic used chunks of 128.
All three cases fit the 4096-token generation budget. The first steady-state
replay ended at 6286.2 MiB with +0.00 MiB growth. The session passed its
explicit provisional 16384 MiB ceiling. No throughput floor was asserted;
these values are not a frozen baseline.
CPU-only verification and the Mistral regression-checkpoint download
overlapped subsequent cases of this exploratory run; its throughput must
not be frozen as an idle-machine baseline.

Two fresh-process quality runs passed determinism checks (12:32-12:56 and
12:56-13:19 CDT, 1392.69 / 1395.41 seconds). Both reported reference-answer
perplexity 6.1254 and identical digests:

- Greedy: `864f777bbeaa8cc0d830b2a5fa08e717508e5cb10ad5e2b3957c2872595fcb77`.
- Sampled: `d9fbb32a5f812ae92f4cfbf2f88dc03a3de48e59992b4bc1aa627a5aead008d4`.

This is the repository's fixed assistant-answer corpus, not a published
benchmark or a comparison against another engine. The first measurements
have no prior numerical baseline to assert against. The table remains
unfrozen while the low-temperature smoke diagnosis is open; determinism
alone does not establish acceptable generation quality.

A larger-budget greedy diagnostic repeated the same paragraph several
times and was interrupted after 349.57 seconds. Its requested 4096-token
budget was not exhausted, and the interruption is not an EOS or a passing
completion. No production arithmetic change is justified by this run alone.

## Shared-flow regression checks

Mistral 7B Q4_K_M greedy and sampled 400-token smokes are coherent (both
reach `MaxTokens`, which is permitted for a coherence smoke). Dense Qwen3
0.6B Q8_0 produces coherent answers and reaches `EndOfTurn` in 251 greedy
and 252 sampled tokens. These use the standard coastal-wetlands prompt,
8192 context, checkpoint chat templates, and chunked prefill. Their memory
and quality results are listed below; smoke throughput is not a new frozen row.

- Mistral's existing oracle passed its frozen 1300 MiB ceiling and 12 tok/s
  floor: 1197.4 MiB peak, 30.025 / 23.234 / 19.576 tok/s, all three cases
  `EndOfTurn` (447 / 600 / 370 generated tokens), +0.00 MiB replay growth.
  Its target has no quality baseline; none was borrowed from another model.
- Dense Qwen3's two fresh quality processes reproduced perplexity 33.5298,
  greedy digest `d108df5120d92b47419e9d8096a6bc54120e0f1eb18422295b08ecf13e6a1637`,
  and sampled digest `73e57a2bacd00b20be5febce9289d8f781806faaf5bdacf128d9bf63752d36d0`.
  The dedicated quality target now carries this checkpoint-specific row.
  Mutation-checking the old dense no-norm mode changes perplexity to
  379949.9645 and fails this target. Restoring per-head normalization
  reproduces 33.5298 and both frozen digests. The mutation is restored.
- Dense Qwen3's memory oracle passed its provisional 2048 MiB ceiling at
  8192 context: 1121.7 MiB peak, all cases `EndOfTurn` (262 / 495 / 281
  generated tokens), +0.00 MiB replay growth. No throughput floor is frozen
  for this new small-checkpoint target.

Qwen3 MoE (30B-A3B Q4_K_M) uses its existing 4096-context, 16-slot
protocol. Both coastal-wetlands smokes complete coherently with `EndOfTurn`,
385 greedy and 380 sampled tokens. Its frozen quality gate passes unchanged:
perplexity 14.5988, greedy digest
`b9211e34142a2e9c6fd60feca13600d1a2e668885d3dadedeeaa25debd5cc848`,
and sampled digest
`92c294a3bb2ec623583e76f412c61d705381d9aa122ebace8d8d97be14a1c874`.
The constrained eight-slot arm reproduces the greedy digest exactly.
Its existing memory oracle passes the frozen 2900 MiB ceiling and 12 tok/s
floor: 2735.4 MiB peak, 27.657 / 24.561 / 16.786 tok/s, all cases
`EndOfTurn` (370 / 482 / 333 generated tokens), and +0.00 MiB replay
growth. Background `systemstats` CPU activity was recorded during these
regressions; no new throughput baseline is frozen from them.

## Release gates still pending

1. Resolve the low-temperature repetition without weakening the release gate.
2. Greedy and sampled 400-token smokes. Short-answer EOS checks now pass
   in both modes using the checkpoint-provided chat template and EOS tokens.
3. An idle-machine performance baseline. The memory oracle passes at 8192
   context and eight slots; its 4096-token budget covers the measured
   885 / 1393 / 975-token completions without truncation.
4. Quality baseline review: both fresh processes reproduce perplexity and
   digests; freeze only explained results.
5. Catalog row and support promotion only after that exact artifact passes.

MLX/FP8 intake, native tool-call parsing, MTP, vision, and later MiniMax
variants remain deferred.

## Reproduction commands

The pinned three-shard contract test reads headers only:

```sh
cargo test -p turbospark-repack --test minimax_network -- --ignored --nocapture
```

After confirming free space for the install allowance plus 20 GiB:

```sh
cargo run --release -p turbospark-cli --bin turbospark-model -- pull \
  --repo unsloth/MiniMax-M2-GGUF@06e952eab9e8e136847e9df067a0032582696778 \
  --file Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf \
  --sidecar-repo MiniMaxAI/MiniMax-M2@757303d492a50514c312788b5247a4f696a4c6a3 \
  --alias minimax-m2-q4km --out ~/models/minimax-m2-q4km.gturbo
```

The memory target requires an explicit provisional ceiling until measurements
establish a baseline. A supplied ceiling is a run limit, not a measured peak.
Both targets use eight slots and 8192 context; the memory target additionally
requires complete answers with its provisional 4096-token generation budget.
The quality target currently measures perplexity and asserts determinism;
its empty baseline table must be populated only after two validated runs.

```sh
TURBOSPARK_MINIMAX_INSTALL_DIR=~/models/minimax-m2-q4km.gturbo \
TURBOSPARK_MINIMAX_CEILING_MIB=16384 \
  cargo test -p turbospark-bench --test minimax_memory_oracle --release -- --ignored --nocapture

TURBOSPARK_MINIMAX_INSTALL_DIR=~/models/minimax-m2-q4km.gturbo \
  cargo test -p turbospark-bench --test minimax_quality_gate --release -- --ignored --nocapture
```
