# MiniCPM-V 4.7 35B-A3B (MLX mxfp4): implementation and evidence

Status: installable and verified, text only. The catalog alias is
`minicpm-v47-35b-mxfp4`. On a 36 GB M4 Max it decodes at 24-26 tok/s with a
1,491 MiB process peak at 4,096 context and 16 expert-cache slots. Prefill
is the weak spot. A 2,940-token prompt took 141-145 s sequentially and
107-120 s on the chunked driver that landed on 14 October 2026.

The checkpoint is a MiniCPM-V wrapper around an exact Qwen 3.6 35B-A3B
trunk. So the work was not a new model family. It was a new quantization
format (MLX `mode: "mxfp4"`) and a way to drop a vision tower this port does
not implement. This page records what was built, how it was checked, what
was measured, and what is still open.

Deliberate divergences and their ceilings also live in
[`DEVIATIONS.md`](../DEVIATIONS.md) (the "MiniCPM-V 4.7 35B-A3B" section).
The frozen rows live in [`BENCHMARKS.md`](BENCHMARKS.md).

## Pinned inputs

- Repository: `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4`, revision
  `810419dd09762d63aad6da5b6a418c4af8946e60`.
- Root `model_type` is `minicpmv4_7`. The trunk resolves through the
  `text_config` fallthrough to `qwen3_5_moe`, so the install's family is
  `qwen36` (`QwenGdnMoe`). Nothing in `ArchConfig` or the decode flow changed.
- Trunk shape: 40 layers (30 gated-DeltaNet, 10 full attention), hidden 2048,
  256 routed experts of width 512, one shared expert, vocab 248,144, untied
  head.
- Quantization: global `mxfp4` at group 32. The two router gates per layer
  (`mlp.gate`, `mlp.shared_expert_gate`) override to affine INT8 at group 64.
  The embedding, the head, and the GDN projections ship as BF16.
- Download is 22,509,241,916 bytes. The install is about 20.5 GiB.

```sh
turbospark-model pull minicpm-v47-35b-mxfp4
turbospark-check --model minicpm-v47-35b-mxfp4 --chat
```

## What was built

### The MLX mxfp4 codec

`crates/repack/src/gemma4_checkpoint/mlx_mxfp4.rs` reads MLX's two planes:
`.weight` (u32 words holding eight E2M1 codes) and `.scales` (one E8M0
exponent per 32 elements). There is no `.biases` plane. Neither consumer in
this port reads that layout, so the codec converts it, and both conversions
are exact.

- **Routed experts are re-blocked into GGUF MXFP4** (`reblock_to_gguf_mxfp4`).
  The existing MXFP4 MoE kernels in `moe_gguf.metal` read GGUF's 17-byte
  block. The 4-bit codes and the exponent byte carry over unchanged. Only
  the nibble order moves: MLX packs adjacent elements per byte, and GGUF
  pairs element `j` with `j + 16`.
- **Resident matrices are dequantized to BF16** (`dequant_to_bf16`). That
  covers full-attention q/k/v/o and the shared expert. E2M1 has at most two
  significant bits, so every value lands on BF16's grid and nothing is
  discarded. The cost is bytes, not numerics.
- **Exponent byte 255 is refused.** MLX reads it as NaN and ggml as a finite
  2^127, so copying it through would turn a poisoned weight into a large one.
- **Elements that overflow f32 are refused.** At exponent 253 a magnitude of
  4 or 6, and at 254 anything from 2 up, reaches 2^128, which every decoder
  (MLX's included) turns into infinity.

The nibble order is the trap. Reading one layout as the other gives a
correctly scaled permutation of the weights, which correlates well with the
right answer and is wrong. The rule is recorded in
[`crates/repack/AGENTS.md`](../crates/repack/AGENTS.md).

### Supporting changes

- **MoE activation chosen by family.** The MXFP4 phase-1 kernel used to
  hardcode gpt-oss's `swiglu_oai` (alpha 1.702, clamp 7). It now takes
  `Mxfp4Swiglu::for_family` (`crates/runtime/src/real_forward_layout.rs`):
  `GptOss` for gpt-oss and `Plain` for everything else. A clamped
  `swiglu_oai` on a Qwen layer decodes finite, fluent, wrong text, which is
  why this is selected by `ArchConfig.family` and never by tensor names.
- **Per-tensor group sizes** (`group_overrides` in
  `gemma4_checkpoint/config.rs`) for an mxfp4 checkpoint's affine INT8 gates.
  An affine checkpoint never fills it, so existing installs read exactly as
  before.
- **A raw-BF16 manifest slot** for the embedding and head.
- **A BF16 embedding lookup kernel** (`embed_lookup_bf16` in
  `crates/gpu/src/shaders/gemv_bf16.metal`), reached through
  `encode_embed_any`.
- **`has_foreign_vision_tower`** (`crates/repack/src/arch_registry.rs`). For
  this family the `vision_tower.` prefix normally means the Qwen tower. Under
  the `minicpmv4_7` wrapper it is MiniCPM's SigLIP, so the tower and both
  mergers (`vision_tower.`, `vit_merger.`, `merger.`) are dropped and the
  install is text only. The check matches the root `model_type` exactly, so
  a real Qwen VL checkpoint cannot match it, and an unknown wrapper fails
  closed.
- **The catalog probe** recognizes MLX mxfp4 and the foreign tower, and
  reports the checkpoint `RUNNABLE`. One routed expert is 1,671,168 bytes
  (1.6 MiB), so 16 slots over 40 layers come to 1,020 MiB.

## How it was checked

- **Codec.** `crates/repack/tests/mlx_mxfp4.rs` checks both nibble orders
  against an independent OCP decoder, ggml's own decoder, and MLX's
  `quantized_matmul`. The tolerance is tight enough to reject the wrong order.
- **Config intake.** `crates/repack/tests/minicpm_v47_config.rs` reads the
  pinned `config.json` fixture, including the per-tensor INT8 gates.
- **Synthetic install.** `crates/repack/tests/synthetic_qwen_mxfp4.rs` and
  `crates/runtime/tests/real_forward_qwen_mxfp4.rs` push a small mxfp4
  checkpoint through the real repack walk and the Metal forward pass.
- **Embedding kernel.** `crates/gpu/tests/embed_lookup_bf16_parity.rs`, with
  parity and a mutation check.
- **Real install.** Greedy and sampled 400-token smokes were coherent, and a
  short question stopped at end of turn.
- **Review.** An independent review of `b6128dd2` and `e628ae6d` found no
  high or medium severity issues. Its three low-severity notes (the
  overflow refusal above, a test assertion that passed for the wrong
  reason, and three stale comments) were fixed in `48d881cd`, with
  mutation checks.

## Measured results

All on one Apple M4 Max with 36 GB, on AC power. The frozen rows were taken
on 13 October 2026. The profiles were taken on 14 October 2026 with other
sessions on the host and about 3.9 GB of swap in use, so treat them as
rankings, not frozen numbers.

### Frozen gates

| Gate | Result |
| --- | --- |
| Memory oracle (`minicpm_v47_memory_oracle`) | peaks 1,491 and 1,481 MiB, ceiling 1,650 MiB, every case reaches end of turn |
| Decode at 16 slots | 24.0-25.6 tok/s on the quieter reading, floor 17.5 tok/s |
| Quality gate (`minicpm_v47_quality_gate`) | perplexity 7.5693, greedy digest `cfa82442`, sampled `68850b09`, three processes agree |
| 8-slot digest | equal to the 16-slot one |

The quality gate also checks that halving the expert cache to 8 slots keeps
at least half the 16-slot decode rate. That ratio is noisy on a shared host:
0.49x (failed), 1.20x, 0.68x, and 0.76x across four runs. The 0.76x rerun on
14 October 2026 passed with the perplexity and both digests unchanged.

The peak sits below Qwen 3.6 INT4's even though the resident file is about
three times larger. Mapped resident weights are not charged to
`phys_footprint`, so the extra BF16 bytes show up as decode bandwidth instead.
The perplexity is a different fine-tune at a different quantization, so it
does not rank against Qwen 3.6's 6.2536.

### Prefill and decode by prompt length

`turbospark-bench`, 16 slots, 4,096 context:

| Case | Prompt | Prefill | Decode | Peak |
| --- | ---: | ---: | ---: | ---: |
| short-explanation | 62 tokens | 2.3 s | 24.3 tok/s | 1,481 MiB |
| long-synthesis | 2,940 tokens | 137.8 s | 19.4 tok/s | 1,491 MiB |

The memory oracle's own run of the long case prefilled in about 111 s.
Either way, sequential prefill runs at roughly decode speed: every prompt
token is a full decode pass with a blocking router readback on each layer.

### Chunked prefill

The qwen MoE flow gained a chunked prefill driver on 14 October 2026
(`crates/runtime/src/families/qwen/moe_prefill.rs`). It runs the attention
and router half of a whole micro-batch (up to 16 tokens) before ONE wait
per layer, then the routed experts token by token in router ranking order.
Every dispatch is the one the sequential flow makes, so the output is
byte-identical, not just close. `turbospark-check` and `turbospark-server`
use it by default. `TURBOSPARK_QWEN_MOE_CHUNKED=0` turns it off.

Byte identity on the real install, same binary, the switch's two arms, a
161-token prompt at 16 slots:

| Run | Sequential stdout md5 | Chunked stdout md5 | Prefill |
| --- | --- | --- | --- |
| greedy | `a4fc6840` | `a4fc6840` | 9.23 s to 5.68 s |
| sampled, seed 7 | `ab398d73` | `ab398d73` | 8.05 s to 5.69 s |

Prefill speed, `turbospark-bench --case long-synthesis --expert-cache-slots 16
--prefill-chunk off|auto`, two interleaved pairs on AC with no other model
running:

| Pair | Sequential | Chunked | Speedup | Peak, sequential / chunked |
| --- | ---: | ---: | ---: | ---: |
| 1 | 141.17 s | 107.39 s | 1.31x | 1,420 / 1,522 MiB |
| 2 | 145.15 s | 119.60 s | 1.21x | 1,423 / 1,445 MiB |

- 1.21-1.31x is the same order as Gemma 4's 1.22x from the same kind of
  driver. It removes per-token waits, not weight reads, so prefill still
  reads every resident matrix once per token.
- Both arms generated the same 609 tokens.
- The chunked peak was 22 to 101 MiB higher. The two widened buffers are
  kilobytes, so they are not the cause, and the band is too wide to call a
  fixed cost. It stays under the 1,650 MiB ceiling. Not attributed.
- Decode after a chunked prefill read 19.1-19.9 tok/s against 20.5-20.6 in
  both pairs. One possible cause is that the expert cache ends prefill
  holding different experts (the pipelined loop protects the previous
  token's slots, which changes eviction). Not verified.

### Where a decoded token's time goes

`TURBOSPARK_PHASES=1 TURBOSPARK_DISPATCH_PROFILE=1 turbospark-check`, 16
slots, 4,096 context, 400 tokens. The dispatch profiler inflates absolute
times (17.6 tok/s under it), so read the shares.

| Bucket | Share | ms/token |
| --- | ---: | ---: |
| GPU work, all layers | 52.7% | 32.7 |
| Expert reads (`pread`) | 28.6% | 17.7 |
| Output head | 7.5% | 4.7 |
| Encode and logit readback | 6.9% | 4.3 |

- The expert cache hit 52.7% at 16 slots, so each token requested about
  242 MiB of expert weights. Most of that comes out of the page cache (see
  the A/B below).
- Inside the GPU work, `bf16_gemv_rows` takes about 64% of the routed
  command buffer (305 dispatches per token). The two MXFP4 expert kernels
  take about 23%.
- Most of those BF16 bytes are the GDN projections and the head, which are
  BF16 in the source checkpoint. The mxfp4-at-source matrices that repack
  widens to BF16 add roughly 18% to the bytes read per token.

### Expert-cache size: 96 slots beat 16, and disk reads do not change

With no flags, `turbospark-check` picks 96 slots on this machine. An A/B on
14 October 2026 alternated 16 and 96 slots with the same binary and prompt,
4,096 context, and `TURBOSPARK_EXPERT_DISK_IO=1`:

| Run | Slots | Decode | Hit rate | Requested | Physical disk | pread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 17.83 tok/s | 52.7% | 241 MiB/token | 35.6 MiB/token | 16.1 ms/token |
| 1 | 96 | 18.20 tok/s | 87.3% | 65 MiB/token | 26.6 MiB/token | 11.1 ms/token |
| 2 | 16 | 16.83 tok/s | 52.7% | 241 MiB/token | 26.7 MiB/token | 15.1 ms/token |
| 2 | 96 | 18.05 tok/s | 87.3% | 65 MiB/token | 26.8 MiB/token | 11.5 ms/token |

- 96 slots decode 2-7% faster and spend about 4.5 ms less per token in
  expert reads.
- Physical disk reads are about 27 MiB per token in both arms. At 16 slots,
  85-89% of the requested expert bytes come out of the page cache, so the
  larger cache saves memory copies rather than SSD traffic.
- A single earlier run with no flags at all (96 slots, 262,144 context,
  5,120 MiB of KV) read 19.75 tok/s, so the large context reservation did
  not slow decode either.

An earlier reading suggested the defaults were slower. That compared
`turbospark-bench` (24.3 tok/s) against `turbospark-check` with
`TURBOSPARK_PHASES=1` (19.75 tok/s), two different binaries and prompts.
Every `turbospark-check` run with phase timing read 17-20 tok/s at either
slot count. Why it sits below `turbospark-bench` is not established.

## Where it ranks

`turbospark-model recommend` and the app's onboarding share one ranking:
fits, then evidence, then measured speed. On the 36 GB M4 Max this install
comes 5th of the 12 rows that fit, below `qwen36` (33-38 tok/s, 16.8 GiB) and
`ornith35b` (32-42 tok/s, 18.2 GiB). Those run the same trunk faster and
smaller. Text only, MiniCPM has no advantage over them. Its value is the
MiniCPM vision tower, which this port does not run yet, so it is not
promoted.

## Open work

In priority order.

1. **Batched prefill for the qwen MoE flow.** The per-token chunked driver
   landed (above) and bought 1.21-1.31x. The larger step batches the weight
   reads themselves. For `qwen36` and `ornith35b` (INT4 affine) the M-row
   kernels already exist in the dense driver and the MTP verify pass. For
   MiniCPM it needs a new `bf16_gemm_rows` kernel for the BF16 resident
   matrices and a qwen copy of gpt-oss's batched MXFP4 routed pair. Neither
   `qwen36` nor `ornith35b` is installed on the development machine, so
   their real-install checks of this driver are still owed too.
2. **Explain the `turbospark-check` gap.** It reads 17-20 tok/s against
   `turbospark-bench`'s 24.3 at the same slot count. Phase timing is ruled
   out: the runtime never reads `TURBOSPARK_PHASES` (its counters always
   run, and the variable only gates the CLI's printout). The power profile
   is ruled out too: both default to `performance`, and Low Power Mode was
   off. What is left is the workload (a 13-token greedy prompt against the
   bench's 62-token sampled protocol prompt, so routing differs), other
   sessions' load during the `turbospark-check` runs, and the CLI's
   per-token streaming output. Test on a quiet host: interleave the two
   binaries, then `turbospark-check --quiet`, before quoting a
   `turbospark-check` rate.
3. **A resident MXFP4 GEMV,** wrapping `dequant_mxfp4_row_simd`. It is
   lossless and specific to this checkpoint, but it recovers only the
   widened attention and shared-expert bytes, about 10% of decode at best.
4. **Cross-engine check against mlx-vlm's `minicpmv4_7` class.** Nothing
   outside this port has checked its numerics. The gates are sentinels
   against their own past. It needs the raw 22.5 GB MLX checkpoint (the
   install was pulled with the download cache off, so it is not on disk)
   and `mlx-vlm`. See
   [`.claude/docs/cross-engine-kl.md`](../.claude/docs/cross-engine-kl.md).
5. **Vision.** The MiniCPM SigLIP tower (27 layers, patch 14,
   `insert_layer_id` 6, canvas mRoPE) is a bring-up of its own.

The playbook for speed work on this family is
[`SPEEDUP_PLAYBOOK.md`](SPEEDUP_PLAYBOOK.md). Profile before touching a
kernel, and keep the old path one env var away.

## Reproduce

```sh
# Memory oracle (one real model per process)
TURBOSPARK_MINICPM_V47_INSTALL_DIR=~/.turbospark/models/text/minicpm-v47-35b-mxfp4.gturbo \
  cargo test -p turbospark-bench --test minicpm_v47_memory_oracle --release -- --ignored --nocapture

# Quality gate
TURBOSPARK_MINICPM_V47_INSTALL_DIR=~/.turbospark/models/text/minicpm-v47-35b-mxfp4.gturbo \
  cargo test -p turbospark-bench --test minicpm_v47_quality_gate --release -- --ignored --nocapture

# Prefill and decode by case
cargo run --release -p turbospark-bench --bin turbospark-bench -- \
  --model ~/.turbospark/models/text/minicpm-v47-35b-mxfp4.gturbo \
  --case long-synthesis --expert-cache-slots 16

# Per-phase profile (the phase table is printed by turbospark-check only)
TURBOSPARK_PHASES=1 TURBOSPARK_DISPATCH_PROFILE=1 cargo run --release -p turbospark-cli --bin turbospark-check -- \
  --model ~/.turbospark/models/text/minicpm-v47-35b-mxfp4.gturbo \
  --prompt "Explain how a hash map handles collisions, with an example." \
  --max-new 400 --temperature 0 --max-context 4096 --expert-cache-slots 16
```

Run on AC power with no other model process running. A tok/s footer from
`turbospark-check` compares to the frozen rows only at
`--expert-cache-slots 16`.
