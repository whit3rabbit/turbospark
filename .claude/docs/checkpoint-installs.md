# Installing the real checkpoints

Split out of the repository's `AGENTS.md`, which is loaded into every
session; this page is loaded when you follow the link. Nothing here is a
new fact, and `AGENTS.md` remains the map.

Every install streams its checkpoint a layer or a tensor at a time and NEVER
writes it to disk whole, and none of them CAN RESUME: a failure restarts the
walk. Minutes to half an hour each, gigabytes over the network. The sidecars
are fetched and verified FIRST, so a wrong sidecar list costs seconds rather
than a re-stream (`AGENTS.md` Gotcha 47).

All commands run from the REPOSITORY ROOT, not from this directory.

```sh
# Install it (19.4 GB in, ~15 GB out, ~21 min, never written to disk whole).
TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
  cargo test -p turbospark-repack --test museglimmer_checkpoint_network --release -- --ignored --nocapture

# ROADMAP Phase G Stage 2 item 8: install the REAL published Gemma 4 Q8_0
# GGUF. Streams the 26.9 GB file from HF a layer at a time (it is never
# written to disk) and produces a ~25 GB install. ~24 min. Must NOT point at
# the MLX-derived ~/models/gemma4.gturbo; the test asserts it does not.
TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_install_network --release -- --ignored --nocapture

# ROADMAP Phase M2: install the real published Mixtral 8x7B Q4_K_M. Streams
# the 26 GB file from HF a layer at a time (never written to disk) into a
# ~29 GB install. The THIRD family, and the first whose architecture string
# (`llama`) covers two different models: a dense Llama 3.1 reports the same
# string and is refused at open, because only `expert_count` says which half
# a file is. Also the first real file to put Q6_K in a ROUTED expert (16 of
# 32 layers, against Q4_K on the other 16) and Q5_K on `attn_output`.
TURBOSPARK_MIXTRAL_INSTALL_DIR=~/models/mixtral-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_mixtral_install_network --release -- --ignored --nocapture

# The SIXTH family (ROADMAP M5), and the one whose expert block type has NO
# resident GEMV: MXFP4 lives in `ffn_*_exps` and nowhere else, with attention,
# `token_embd` and `output` all Q8_0. Streams the 12.1 GB file a layer at a
# time (never written to disk) into a ~12 GB install, ~25 min. It is also the
# first file with a BIAS beside every projection, a sink vector per block, and
# a RANK-2 per-expert bias inside the routed blob -- all three exercised
# against `SyntheticGptOssShape` first, which is what kept the two walk holes
# it exposed to milliseconds each instead of a re-stream apiece (Gotcha 42).
TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
  cargo test -p turbospark-repack --test gguf_gptoss_install_network --release -- --ignored --nocapture

# The FOURTH family, and the one Mixtral's granularity finding asked for
# (Gotcha 36): install the real published Qwen3-30B-A3B Q4_K_M. Streams the
# 17.3 GB file a layer at a time (never written to disk) into a ~18 GB
# install. Same LAYER GRAPH as Mixtral and the same decode flow
# (`crates/runtime/src/families/llama/`), differing only in per-head q/k
# norms and an RMS epsilon of 1e-6 -- but 128 experts of 2.5 MiB against
# Mixtral's 8 of 108.9, so its slot cache is 1.90 GiB at 16 slots rather
# than 54.5 and the memory oracle and quality gate are worth running on it.
# NO new kernels: Q4_K and Q6_K are both already executable, asserted off
# the header before the download.
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen3moe_install_network --release -- --ignored --nocapture

# ROADMAP M4: the FIFTH family install and the first DENSE one. Streams the
# real Mistral-7B-Instruct-v0.3 Q4_K_M (4.1 GB, never written to disk) into a
# ~4.1 GB install with ZERO packed-expert files -- a dense model has no routed
# experts, so nothing streams at decode time. ~5 min. Its `[INST]` chat
# dialect landed back in M2, so `--messages-file` renders correctly for it.
# Gotcha 36's slot-cache multiplication does not apply; nor does the
# resident-floor-is-the-working-set reading of Gotcha 19 (see Gotcha 40:
# measured peak is 684 MiB against 4.07 GiB of weights).
TURBOSPARK_MISTRAL_INSTALL_DIR=~/models/mistral7b-dense.gturbo \
  cargo test -p turbospark-repack --test gguf_mixtral_install_network --release -- --ignored --nocapture repacks_the_real_mistral

# The same family at a twentieth the size: TinyLlama-1.1B-Chat Q6_K, 0.84 GB
# in and ~0.9 GB out, ~3 min. The cheapest real dense checkpoint on the shelf
# and the one that caught Gotcha 39 (its head_dim is 64, where every other
# real `llama` file agrees with the Mixtral baseline's 128). Without the env
# var it installs to a temp dir and deletes it, which is the walk-only case.
# Its chat framing is Zephyr, not `[INST]`, and `--messages-file` now renders
# that correctly: the checkpoint's own template wins over the dialect (Gotcha
# 41). This used to be M4's one open item and needed a raw `--prompt`.
TURBOSPARK_DENSE_LLAMA_INSTALL_DIR=~/models/tinyllama-dense.gturbo \
  cargo test -p turbospark-repack --test gguf_mixtral_install_network --release -- --ignored --nocapture repacks_a_real_dense_llama_gguf

# ROADMAP Phase G Stage 2 item 9: the same, for the K-quants. Streams the real
# published Qwen 3.6 Q4_K_M (~20 GB, never written to disk) into a ~20 GB
# install. This is the MIXED case: Q4_K experts and embedding, Q8_0 attention,
# one Q6_K tensor. Must NOT point at ~/models/qwen36.gturbo; asserted.
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_install_network --release -- --ignored --nocapture

# ROADMAP Phase S: install the real sub-4-bit candidate. Streams the 12 GB
# `unsloth/gemma-4-26B-A4B-it-UD-Q3_K_M` from HF a layer at a time (never
# written to disk) into a ~12 GB install whose experts are 9.6 GiB against the
# MLX install's 12. ~18 min. MIXED ALONG TWO AXES, which is what it is for:
# IQ3_XXS gate/up over an IQ4_NL down in one expert, and a layer 29 that is
# IQ4_XS over Q8_0. Asserts the per-layer stride saved over 30% on disk.
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-repack --test gguf_iq_install_network --release -- --ignored --nocapture

# The SECOND checkpoint of the `qwen3_5` family (2026-08-14), and the one
# that makes the pair a CONTROLLED comparison: `Qwen/Qwen3.8-27B` shares
# Bonsai-27B's architecture EXACTLY (33 of 35 text_config keys equal; the
# two that differ reach no ArchConfig field), so the only thing that varies
# between the two installs is the quantization -- 1-bit group 128 against
# INT4 group 64. Streams the 16.08 GB mlx-community artifact a tensor at a
# time (never written to disk) into a ~16 GB install. Nothing in `src/`
# changed for it; the offline `qwen35_config` target is what says so, and it
# is the gate to run BEFORE this one.
TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
  cargo test -p turbospark-repack --test qwen38_checkpoint_network --release -- --ignored --nocapture

# THE VISION TOWER'S INTAKE (ROADMAP M-V3), and the ORDER of these two is
# load-bearing rather than a preference (`crates/repack` Gotcha 8). The
# FIXTURE runs in 0.1 s and makes every assertion the download would
# otherwise make minutes at a time -- including the one whose absence shipped
# a headless install for a release, that BOTH writers carry the component.
cargo test -p turbospark-repack --test synthetic_qwen35_vision

# Only then the real 16 GB stream, into its OWN directory. Separate from
# `~/models/qwen38-27b.gturbo` on purpose: that install is what
# `qwen38_memory_oracle` and `qwen38_quality_gate` assert their frozen rows
# against, and adding ~0.9 GiB of tower would move its footprint and force a
# re-freeze for a component neither gate exercises. ~25 min, CANNOT RESUME.
# It is also the only place the BF16 arm of `convert_raw_to_fp16` runs: this
# checkpoint's tower is BF16 where Bonsai's (and the fixture's) is F16.
TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
  cargo test -p turbospark-repack --test qwen38_checkpoint_network --release -- \
  --ignored --nocapture repacks_the_real_qwen38_27b_checkpoint_with_its_vision_tower

# The THIRD checkpoint of the `qwen3_5` family (ROADMAP's ternary entry) and
# the one that made it a WIDTH rather than a family: MLX affine at TWO bits,
# group 128, FP16 companions. Its `text_config` is Bonsai-27B's to the key --
# the two files differ in their `quantization` object ALONE -- so nothing in
# `crates/model-io` or `crates/runtime`'s flow moved for it. Streams the
# 8.49 GB artifact a tensor at a time (never written to disk) into a ~7.6 GB
# install, ~14 min. The offline `qwen35_config` target asserts the parse and
# is the gate to run BEFORE this one.
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
  cargo test -p turbospark-repack --test ternary_checkpoint_network --release -- --ignored --nocapture

# ORNITH-1.5, two checkpoints of the SAME shipped architectures and three
# installs. Neither is a new family: the 35B derives `qwen_gdn_moe_35b_a3b()`
# field for field from its HF config AND from llama.cpp's GGUF metadata
# independently, and the 9B is the dense half at a new shape. Install them:
TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
  cargo test -p turbospark-repack --test ornith_install_network --release -- --ignored --nocapture installs_the_real_ornith_9b
TURBOSPARK_ORNITH35B_GGUF_INSTALL_DIR=~/models/ornith35b-gguf.gturbo \
  cargo test -p turbospark-repack --test ornith_install_network --release -- --ignored --nocapture installs_the_real_ornith_35b

# The INT4-affine 35B, which is the one to prefer: 1.63-1.67x the Q8_0
# install's decode at half the expert stride and half the disk, and the only
# one that clears `speculation_blocker`'s dtype arm (it still cannot draft --
# the publisher's MLX conversion drops `mtp.*`, like mlx-community's Qwen3.8
# one). Q8_0 rather than Q4_K_M on the two GGUF rows is FORCED, not preferred:
# this family's V-head de-interleave runs on the COLUMNS of
# `linear_attn.out_proj` and a 128-column head is half a 256-element K-quant
# superblock (`crates/repack` Gotcha 7).
TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
  cargo test -p turbospark-repack --test ornith_mlx_install_network --release -- --ignored --nocapture

# The other #[ignore]d tests: real checkpoint downloads (many GB).
cargo test -p turbospark-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
cargo test -p turbospark-repack --test hf_checkpoint_network --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture
TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
  cargo test -p turbospark-repack --test qwen35_checkpoint_network --release -- --ignored --nocapture
```
