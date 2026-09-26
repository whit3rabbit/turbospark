# Header probes, convention checks and decode-path probes

Split out of the repository's `AGENTS.md`, which is loaded into every
session; this page is loaded when you follow the link. Nothing here is a
new fact, and `AGENTS.md` remains the map.

The cheap instruments. Most read KB off a published file or run against an
install already on disk, and none downloads a checkpoint. Reach for these
BEFORE a re-stream: a convention question settled by patching an install in
place costs seconds where a repack costs half an hour (`AGENTS.md` Gotcha 33).

All commands run from the REPOSITORY ROOT, not from this directory.

## Cross-engine mismatch: compare the first divergent boundary

Use [SlotStream](https://github.com/carloslfu/slotstream) as a debugging
reference when an independently run model disagrees with TurboSpark. Its
[`current_backend_reference.py`](https://github.com/carloslfu/slotstream/blob/main/Tools/current_backend_reference.py)
loads selected tensors and emits early Qwen4 layer outputs, while
[`qwen4_exp.py`](https://github.com/carloslfu/slotstream/blob/main/Tools/reference/qwen4_exp.py#L2513-L2524)
documents the GDN Q/K L2 normalization math. Follow the same sequence:

1. Freeze the source artifact/revision, rendered prompt, token IDs, and dtype.
2. Capture a narrow activation before and after the operation under suspicion.
3. Recompute the expected output with a separate implementation and compare
   at the same dtype boundary.
4. Move to the next layer only after the current boundary agrees.

Do not compare quantized GGUF activations directly to SlotStream's MLX
safetensors outputs and call that parity. Use SlotStream to locate useful
boundaries and verify equations. For the Qwen4 Q/K epsilon check, TurboSpark
can emit a one-token capture and verify it with:

```sh
TURBOSPARK_QWEN4_GDN_NORM_CAPTURE=/tmp/qwen4-gdn.json \
TURBOSPARK_QWEN4_GDN_NORM_CAPTURE_POSITION=61 \
TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-iq2-xs.gturbo \
  cargo test -p turbospark-bench --test qwen4exp_swift_first_token_probe \
  --release -- --ignored --nocapture
python3 scripts/check_qwen4_gdn_norm_capture.py /tmp/qwen4-gdn.json
```

The capture is FP16 and covers the first GDN layer's `conv_out` immediately
before and after Q/K normalization. It is an operator-level check, not a
full-layer or model-quality comparison.

Use the Qwen4 residual trace to compare layer entry after PLE, after the
attention join, and after the MoE join. `after_moe_join` is the same boundary
SlotStream writes as `layer_i.bin`. Set the requested layer count to 48 for a
full Qwen4 trace. To compare a CLI chat run's rendered input with a direct
probe, set `TURBOSPARK_DEBUG_PROMPT_IDS=1`; it prints the exact token IDs after
chat-template rendering. This example also records TurboSpark's selected
expert IDs:

The public `turbospark run` command delegates generation to the sibling
`turbospark-check` binary. When checking a release build after source edits,
rebuild that peer (`cargo build -p turbospark-cli --release --bin
turbospark-check`) or invoke it directly. Rebuilding only the wrapper can leave
the generation path stale.

```sh
TURBOSPARK_QWEN4_LAYER_CAPTURE=/tmp/qwen4-layers.json \
TURBOSPARK_QWEN4_LAYER_CAPTURE_LAYERS=48 \
TURBOSPARK_QWEN4_LAYER_CAPTURE_POSITION=61 \
TURBOSPARK_ROUTER_HIST=/tmp/qwen4-router.json \
TURBOSPARK_ROUTER_TRACE=1 \
TURBOSPARK_ROUTER_LOGITS_LAYER=23 \
TURBOSPARK_ROUTER_LOGITS_PASS=61 \
TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-iq2-xs.gturbo \
  cargo test -p turbospark-bench --test qwen4exp_swift_first_token_probe \
  --release -- --ignored --nocapture
```

For an independent CPU trace of the same GGUF, build
`scripts/capture_qwen4_llama_callback.cpp` against the llama.cpp revision used
for the reference. It freezes the 62 prompt-token IDs from the Rust probe and
captures the last prompt token's residual boundaries, router logits, routing
weights, and selected experts. Use an empty output directory:

```sh
c++ -std=c++17 \
  -I"$(brew --prefix llama.cpp)/include" \
  -I"$(brew --prefix ggml)/include" \
  scripts/capture_qwen4_llama_callback.cpp \
  -L"$(brew --prefix llama.cpp)/lib" -L"$(brew --prefix ggml)/lib" \
  -Wl,-rpath,"$(brew --prefix llama.cpp)/lib" \
  -Wl,-rpath,"$(brew --prefix ggml)/lib" \
  -lllama -lggml -lggml-base -o /tmp/qwen4-llama-capture
/tmp/qwen4-llama-capture /path/to/model-00001-of-00002.gguf /tmp/qwen4-callback
python3 scripts/check_qwen4_layer_boundary_capture.py /tmp/qwen4-layers.json \
  --llama-callback-dir /tmp/qwen4-callback \
  --router-trace-json /tmp/qwen4-router.json --router-trace-position 61
```

The TurboSpark trace is FP16 at one decode position. The llama.cpp callback
files are float32 vectors for the final prompt token. The checker can also
compare SlotStream outputs with `--reference-dir` and
`--reference-position`. Compare as parity only when source weights and input
token IDs match. Set `TURBOSPARK_ROUTER_LOGITS_LAYER` and
`TURBOSPARK_ROUTER_LOGITS_PASS` together to save one raw f32 router row in the
router JSON. Pass is the zero-based routed-forward count for that layer; it
matches token position 61 for this one-token-at-a-time prompt. When present,
the checker compares the row against llama.cpp's `ffn_moe_logits` callback
tensor and prints both top-k orders. A matching layer boundary localizes a
divergence; it does not establish model quality.
```sh
# What is in a Hugging Face repo, before downloading any of it. Gotcha 47 says
# to parse the config first and these are the mechanism: KB each, no clone and
# no `hf` login. `?blobs=true` sums to the install size, which is the first
# refusal on a large checkpoint -- the 2026-09-01 qwen4_exp survey rejected a
# 104 GB variant against 60 GiB free without transferring a byte, and found an
# expert-pruned 68 GiB sibling in the same pass. `turbospark-model probe` is
# the richer answer and is GGUF-only; these work on a safetensors repo, which
# is where a new ARCHITECTURE usually lands first.
curl -s "https://huggingface.co/api/models?search=<name>&limit=30"
curl -s "https://huggingface.co/api/models/<owner>/<repo>?blobs=true"
curl -s "https://huggingface.co/<owner>/<repo>/raw/main/config.json"

# GGUF intake (ROADMAP Phase G). Reads only the HEADER of the real
# published GGUFs -- a few MB off a 20-27 GB file, ~4 s each -- and checks it
# three ways: the parser agrees with what llama.cpp's converter writes, every
# tensor name maps, and the ArchConfig derived from GGUF metadata equals the
# one the corresponding .gturbo install declares. Set the install vars to get
# the last two cross-checks; without them it still parses and reports.
# The three `scopes_phase_s_*` cases in the same file are the ROADMAP Phase S
# ingest survey: same cost, and their output is a ggml TYPE HISTOGRAM in
# bytes. Read that histogram's UNSIZED rows before its percentages -- a type
# with no `ggml_type_block` row is one this port cannot ingest, which on a
# mixed file is usually the routed experts, and printing it as 0 bytes once
# made an imatrix file look like it was 76% Q8_0.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# ROADMAP Phase M1: the admission gate for `arch_registry.rs`'s planned
# architecture rows. Re-reads the header of the real published file each row
# was taken from and asserts the `general.architecture` string still matches,
# so the table cannot rot into folklore. A header per row, seconds each, no
# install and no checkpoint. Run it after adding a row.
cargo test -p turbospark-repack --test arch_registry_network --release -- --ignored --nocapture

# Settles which half of Gemma's fused ffn_gate_up_exps is the gate (Stage 2's
# first use of the Q8_0 reference). Correlates a dequantized layer 0 expert 0
# against the same expert in the MLX install. Also a real-data check on the
# Q8_0 dequant itself: a sign, scale or block-layout error cannot correlate
# at +0.9957 with an independently-produced INT4 install. Few KB, ~5 s.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# The evidence behind the transcode decision (Gotcha 29): GGUF's F32 norms
# are upcast BF16 and narrow back bit-exactly, and INT8-transcoding its F32
# router does not move the routing decision. No install needed, few KB, ~6 s.
cargo test -p turbospark-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# The same real-data check for Q4_K, against the real Qwen 3.6 Q4_K_M: a
# dequantized layer 0 expert 0 gate row correlates +0.9930 with the same row
# in the MLX install, while the unrelated `up` matrix reads +0.12. It is the
# one check that can catch a decoder and its fixture quantizer being wrong
# TOGETHER, which nothing in crates/compute can. Few KB, ~5 s. Read Gotcha 30
# before believing a low number out of it.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_q4_k_network --release -- --ignored --nocapture

# ROADMAP Phase G Stage 2 item 10: Qwen's V-head convention (Gotcha 33), on
# EVERY layer rather than the layer 0 the transforms were characterized on.
# Reports per tensor how well the bytes on disk and the de-interleaved
# candidate agree with the MLX install. Read-only; TURBOSPARK_QWEN_PATCH=1
# rewrites the install in place, which is how a coherence test costs seconds
# instead of a 23-minute repack. Idempotent: it patches only what improves.
# ~40 s, no network, needs both Qwen installs.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_convention_patch --release -- --ignored --nocapture

# The diagnostic that found the five QUANTIZED tensors on that axis, which a
# BF16 probe structurally cannot see. Recovers the permutation outright where
# a tensor has one row per head. ~1 s, needs both Qwen installs.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_quant_probe --release -- --ignored --nocapture

# The GGUF install's resident BF16 core must be BIT-IDENTICAL to the MLX
# install's: norms, router.scale, per_expert_scale, layer_scalar. Settles
# the Gemma norm "+1" convention and the shape-matched name mappings.
# ~1 s, no network, needs both installs.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_norm_convention_probe --release -- --ignored --nocapture

# ROADMAP M4 Phase 0: which dense `llama` checkpoint the dense half should
# target. Three real headers, no download, ~36 s. The two gates it reads are
# whether the file carries `rope_freqs.weight` (Llama 3.1's RoPE scaling,
# which ships as a TENSOR and has no kernel input here) and whether every
# block type in it already has a kernel. Also prints each candidate's
# RESIDENT FLOOR, which for a dense model is the whole weight file: nothing
# streams, so Gotcha 36's slot-cache multiplication does not apply and the
# 1.6-2.2 GiB band cannot.
cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- \
  --ignored --nocapture scopes_the_dense_llama_candidates

# The determinism check nothing else makes: one runner, the same greedy
# generation six times, asserting ONE distinct output. This is what caught
# Gotcha 27's cache-state-dependent reduce order; run it at 8/16/32 slots
# after any change to the routed-expert path. ~15 s per slot count.
TURBOSPARK_PROBE_SLOTS=32 TURBOSPARK_PROBE_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test gguf_nondeterminism_probe --release -- --ignored --nocapture

# Phase D1's gate: does rollback return the engine to the state a fresh run
# would have reached? Asserts BIT-IDENTICAL logits against a reset-and-replay
# ground truth, because a stale recurrent state or a clobbered KV row produces
# fluent wrong text that no coherence smoke can see. Run it on BOTH families:
# Qwen exercises the recurrent half (30 linear layers of 40, unbounded
# rollback), Gemma the sliding-window ring (25 of 30, rollback capped at the
# ring's 128-token slack). ~4 s each.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test rollback_probe --release -- --ignored --nocapture

# Phase D2's last unknown: how many proposed tokens does the target accept?
# Needs NO batched kernels -- only the ratio matters, so the verify pass runs
# sequentially. The drafter is n-gram/prompt-lookup (zero weights), which is a
# LOWER bound on a trained one. Also the end-to-end losslessness check: every
# block size must produce a token stream byte-identical to speculation off.
# ~40 s.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test accept_length_probe --release -- --ignored --nocapture

# The DFlash2 block drafter's two probes (`docs/DFLASH2.md`). The accept
# probe is the gate: it asserts every block byte-identical to speculation
# off AND carries the functional-drafter guard. The bisect probe localizes a
# broken drafter, and its `the_context_write_is_the_only_variable` case is
# the shape to copy -- one runner, one base, drafted twice, one variable.
TURBOSPARK_DFLASH2_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
  cargo test -p turbospark-bench --test dflash2_accept_length_probe --release -- --ignored --nocapture
TURBOSPARK_DFLASH2_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
  cargo test -p turbospark-bench --test dflash2_bisect_probe --release -- --ignored --nocapture

# The two measurement surfaces behind ROADMAP's speculative-decoding item
# (its Phase D0 gate: does a batched verify pay on this engine, and at what
# block size?). Neither needs a drafter or a new kernel.
#
# 1. Expert union. `TURBOSPARK_ROUTER_TRACE=1` adds the top-k ids IN PASS
#    ORDER to the histogram file, which the counts throw away; a batched
#    verify of M tokens reads their UNION, so its expert cost is the
#    distinct-expert count over a window of M consecutive passes. Pass the
#    run's PROMPT TOKEN COUNT as the second argument -- the trace covers
#    prefill too, and prefill routes differently. Read `breakeven_accept`:
#    a block pays, on this axis, only if it accepts more than that.
TURBOSPARK_ROUTER_HIST=/tmp/rq.json TURBOSPARK_ROUTER_TRACE=1 \
  ./target/release/turbospark-check --model ~/models/qwen36.gturbo \
  --messages-file /tmp/p.json --max-new 300 --temperature 0.0001 --top-k 1
python3 scripts/router_window.py /tmp/rq.json 22 2,4,8,16

# 2. Compute headroom. Is `dequant_int4_gemv_simd` already bandwidth-bound
#    at the shapes decode dispatches? If it is, batching cannot help. It is
#    at the big projections and is NOT at the routed-expert shape. Reports
#    ratios against the same kernel at a large row count, never against a
#    spec number; read its module doc before quoting an absolute GiB/s.
cargo test -p turbospark-gpu --test gemv_bandwidth_bench --release -- --ignored --nocapture

# MAPPED EXPERT RESIDENCY (`docs/EXPERT_RESIDENCY.md`): the routed experts
# read IN PLACE out of one `mmap` per layer instead of `pread`-copied into a
# pinned slot. Gemma 4 only so far, OFF by default, an A/B seam in the shape
# `TURBOSPARK_ROUTED_BATCH` has -- both arms MUST produce identical tokens.
# Every other family REFUSES it by name rather than ignoring it, so a caller
# cannot measure the streamed engine and report it under the mapped label.
# NOTE the first mapped run on a cold page cache pays its faults up front
# (prefill 74.8s against 2.5s); warm it is slower on prefill and faster on
# decode, so it is a one-time cost for a faster steady state.
TURBOSPARK_EXPERT_RESIDENCY=mapped ./target/release/turbospark-check \
  --model ~/models/gemma4.gturbo --messages-file /tmp/p.json --max-new 400

# The measurement that licensed it, and the one that refuted `crates/bench`
# Gotcha 1 as it stood: map the whole 12.3 GB expert table, wrap all 30 layer
# files in Metal buffers, read an expert on each through the GPU, and sample
# phys_footprint at every stage. ~6 s, no decode.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test mapped_expert_probe --release -- --ignored --nocapture
```
