# Per-family memory oracles and quality gates

Split out of the repository's `AGENTS.md`, which is loaded into every
session; this page is loaded when you follow the link. Nothing here is a
new fact, and `AGENTS.md` remains the map.

The per-family gate matrix. Each family needs its own target and its own
install variable: an oracle asserts a whole-session peak and the families
have different ceilings, so they cannot share a process. Read a peak or a
tok/s row WITH the context window and slot count it was taken at
(`AGENTS.md` Gotchas 36, 40 and 58); `docs/BENCHMARKING.md` explains the
modes and `docs/BENCHMARKS.md` holds the frozen rows.

All commands run from the REPOSITORY ROOT, not from this directory.

```sh
# The cheapest gate in the repo and the one to run BEFORE the quality gates
# whenever prompt rendering moves. Its second case is the same guard for
# `--reasoning` (Gotcha 55): that `off` renders the frozen bytes on every
# install, and that a level either MOVES the render or is refused -- there is
# no third outcome that is not a silent no-op. It asserts that each family's own chat
# template differs from the per-dialect renderer by nothing but `trim`, and
# pins per family WHETHER that template trims -- which is the axis that
# decides whether the frozen digests move, and the one that already moved
# `qwen3moe`'s. It compares the real protocol prompt, trailing newline and
# all, because on a tidy one-line string `trim` is a no-op and the guard
# sees nothing. Seconds, no GPU, no model load beyond the tokenizer.
# See Gotcha 41.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-tokenizer --test installed_template -- --ignored --nocapture

# The memory oracle: asserts endOfTurn on every protocol case, peak
# footprint under the ceiling, no growth on a replayed warm case, and
# (where a row exists) a decode tok/s floor. Each row records whether it
# came from Swift's docs/BENCHMARKS.md or from this port measuring itself
# -- printed every run. Skips with a note if the env var is unset. Takes
# ~10 minutes. See docs/BENCHMARKING.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture

# The quality gate (ROADMAP Phase Q): teacher-forced perplexity of a fixed
# reference answer in the ASSISTANT slot (an instruction-tuned checkpoint
# is never trained to predict prompt tokens, so scoring those measures
# nothing), plus frozen greedy and sampled output digests, plus a
# constrained-working-set repeat at 8 expert-cache slots (digest frozen,
# throughput floored). Split per family for the same one-model-per-process
# reason as the oracle. About 80 seconds each. Numbers and caveats:
# docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-bench --test qwen3moe_quality_gate --release -- --ignored --nocapture

# Proof that the gate above can SEE quantization damage, rather than just
# asserting it could. Clones the install (APFS clonefile, so the original
# is untouched and only written pages cost disk), shifts one quantization
# level in a strided subset of the routed experts, and re-measures. About
# 30 seconds. Curve and floor: docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture

# Same oracle for Qwen 3.6. A SEPARATE target, not a second #[test]: the
# footprint assertion is a whole-session peak and the two families have
# different ceilings (~2,200 vs ~1,600 MiB), so they need one process each.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# Same oracle for Qwen3-30B-A3B (`qwen3moe`). Its ceiling is 2,900 MiB, ABOVE
# the other two families' 1,600-2,300: the slot cache is
# `slots x layers x expert_stride` and this model is 48 layers deep at a
# ~2.9 MiB expert, i.e. 2,094 MiB of slot capacity at 16 slots plus a 916 MiB
# pinned resident core. It streams (the whole expert table is 16.36 GiB); it
# just does not land inside the band the README quotes.
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-bench --test qwen3moe_memory_oracle --release -- --ignored --nocapture

# The dense family's oracle (ROADMAP M4). RUNS AT 8,192 CONTEXT where the
# other three run at 4,096, and that is not a knob: the protocol freezes the
# PROSE, and its token count belongs to the checkpoint's tokenizer -- the same
# text is 3,444 tokens under Mistral's 32k vocab against qwen3moe's 2,842, and
# 3444 + 1024 does not fit 4,096, so `long-synthesis` does not run at all.
# Raising the shared PROTOCOL_MAX_CONTEXT would resize KV for every family and
# move every frozen peak, so the window is a per-target parameter and is
# printed on every run beside the ceiling. Its 1,300 MiB ceiling is NOT
# comparable to the MoE rows: 1,024 of the measured 1,201 MiB is KV, and the
# 4.07 GiB of weights count for nothing (Gotcha 40). ~8 min.
TURBOSPARK_MISTRAL_INSTALL_DIR=~/models/mistral7b-dense.gturbo \
  cargo test -p turbospark-bench --test mistral_memory_oracle --release -- --ignored --nocapture

# The SIXTH family's two gates (ROADMAP M5). BOTH differ from their four
# siblings in a way that has to be read with the numbers. The oracle runs at
# 8,192 context AND a 3,072-token budget, because Harmony puts the model's
# reasoning in an `analysis` channel BEFORE its answer -- the three cases need
# 818 / 2,153 / 1,108 tokens to reach `<|return|>`, so at the shared 1,024 two
# of three stop on maxTokens and the validity gate refuses them. Its ceiling
# is 5,700 MiB, far above the README's 1.6-2.2 GiB band, and that is
# `slots x layers x expert_stride` = 4,854 MiB of slot cache, not a
# regression (Gotcha 36). The quality gate PINS A DATE, because Harmony's
# template writes `Current date: ` into its system preamble and a digest over
# a clock-reading prompt expires at midnight.
TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
  cargo test -p turbospark-bench --test gptoss_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
  cargo test -p turbospark-bench --test gptoss_quality_gate --release -- --ignored --nocapture

# The SEVENTH family's two gates (dense, MLX INT4). Its protocol runs at
# 8,192 context AND a 2,048 budget, because the model REASONS to a `to=self`
# message before its `to=user` answer -- the three cases need 1,132 / 1,498 /
# 1,552 sampled tokens, so the SHORT one already exceeds the shared 1,024.
# The quality gate needs an ASSISTANT PREFIX (` to=user<|message|>`): the
# generation prompt ends at `<|start|>assistant` and the model's next
# emission is a RECIPIENT, so a reference answer spliced in raw lands in no
# message at all -- gpt-oss's 148,421.76 failure, one family over.
TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
  cargo test -p turbospark-bench --test museglimmer_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
  cargo test -p turbospark-bench --test museglimmer_quality_gate --release -- --ignored --nocapture

# Its two gates, the FIRST this family has ever had (Bonsai got neither, so
# the DENSE half of `families/qwen/` had no sentinel at all until now).
TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
  cargo test -p turbospark-bench --test qwen38_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
  cargo test -p turbospark-bench --test qwen38_quality_gate --release -- --ignored --nocapture

# The FIFTEENTH family's two gates (dense, MLX INT4, the llama flow's third
# family). The oracle drives TWO of three protocol cases: sampled at the
# protocol's temperature, this checkpoint's medium-review answer had not
# terminated by 3,072 new tokens (greedy ends it at 683) -- a checkpoint
# verbosity property recorded in the oracle file, tinyllama precedent.
TURBOSPARK_QWEN3VL_INSTALL_DIR=~/.turbospark/models/text/qwen3vl-4b.gturbo \
  cargo test -p turbospark-bench --test qwen3vl_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_QWEN3VL_INSTALL_DIR=~/.turbospark/models/text/qwen3vl-4b.gturbo \
  cargo test -p turbospark-bench --test qwen3vl_quality_gate --release -- --ignored --nocapture

# The qwen3_vl DEEPSTACK gates (vision half, 2026-09-19): the four in-repo
# deepstack gates plus the seven-stage mlx-vlm parity, all on the combined
# install (2.9 GiB, re-pull per ROADMAP's artifact table). The page is
# `scripts/make_vision_test_page.py --size 1024 1280`; the dump comes from
# `scripts/vision_tower_probe.py --mode dump` (it captures the three
# deepstack mergers too).
TURBOSPARK_QWEN3VL_VISION_INSTALL_DIR=~/.turbospark/models/text/qwen3vl-4b-vision.gturbo \
TURBOSPARK_QWEN3VL_VISION_PAGE=~/.turbospark/vision-pages/page.png \
  cargo test -p turbospark-runtime --test qwen3vl_vision --release -- --ignored --nocapture
TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/.turbospark/models/text/qwen3vl-4b-vision.gturbo \
TURBOSPARK_VISION_DUMP_DIR=/tmp/qwen3vl-vision-dump \
  cargo test -p turbospark-runtime --test vision_tower_parity --release -- --ignored --nocapture the_tower_agrees

# Its two gates. NOTE the perplexity is NOT comparable to `qwen38_quality_gate`'s
# as a quantization result even though the two share an architecture: one is
# prism-ml's QAT checkpoint and the other Qwen's release, so a TRAINING
# separates them as well as a width.
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
  cargo test -p turbospark-bench --test ternary_quality_gate --release -- --ignored --nocapture
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
  cargo test -p turbospark-bench --test ternary_memory_oracle --release -- --ignored --nocapture

# The Bonsai-2 line's two gates (2026-09-19, the HADAMARD-FOLDED checkpoint;
# `docs/BONSAI2.md` is the contract's page). The install carries
# `hadamard.bin` plus the manifest section; the frozen rows are the proof the
# activation transforms are right -- a missing transform reads as fluent
# output with a catastrophic perplexity, not as an error.
TURBOSPARK_BONSAI2_INSTALL_DIR=~/.turbospark/models/text/bonsai2.gturbo \
  cargo test -p turbospark-bench --test bonsai2_quality_gate --release -- --ignored --nocapture
TURBOSPARK_BONSAI2_INSTALL_DIR=~/.turbospark/models/text/bonsai2.gturbo \
  cargo test -p turbospark-bench --test bonsai2_memory_oracle --release -- --ignored --nocapture

# Their four gates. The 9B's oracle is what VERIFIED `protocol_parameters`'
# `qwen3_5` row, which had been placed on the tokenizer's evidence alone and
# said so: all three cases reach endOfTurn at the shared 4,096/1,024.
TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
  cargo test -p turbospark-bench --test ornith9b_quality_gate --release -- --ignored --nocapture
TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
  cargo test -p turbospark-bench --test ornith9b_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
  cargo test -p turbospark-bench --test ornith35b_quality_gate --release -- --ignored --nocapture
TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
  cargo test -p turbospark-bench --test ornith35b_memory_oracle --release -- --ignored --nocapture

# Its quality gate. A SEPARATE target with its own chip row, not the Gemma
# gate with an env var moved: the existing rows are keyed on the chip and
# freeze the MLX INT4 goldens, so pointing an install var at a different
# artifact asserts the wrong digests. ~2 min.
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-bench --test iq3_quality_gate --release -- --ignored --nocapture

# The `deepseek2` family's two gates (DeepSeek-V2-Lite-Chat Q8_0, the MLA
# bring-up's witness). The oracle runs the protocol at the family's OWN
# 8,192/1,024 row -- the compressed 576-half KV cache is why it can take the
# dense window for a quarter of the KV cost (`real_model_params`) -- and the
# quality gate needs NO assistant prefix: V2's answer slot is plain prose
# after `Assistant:`, with no think frame to close and no channel marker.
TURBOSPARK_DSV2_INSTALL_DIR=~/.turbospark/models/dsv2lite-16b.gturbo \
  cargo test -p turbospark-bench --test dsv2_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_DSV2_INSTALL_DIR=~/.turbospark/models/dsv2lite-16b.gturbo \
  cargo test -p turbospark-bench --test dsv2_quality_gate --release -- --ignored --nocapture
```
