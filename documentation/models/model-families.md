---
description: Deep dive into all 15 supported model families, architectural distinctions, and layer graphs.
icon: diagram-project
---

# Model Families and Architecture

TurboSpark dispatches inference based on the model's architectural family discriminator (`manifest.json -> arch.family`), never by file or tensor names. This prevents incorrect layer graph evaluation that could produce fluent but mathematically invalid outputs.

The engine implements 15 model family discriminators:

```rust
pub enum ModelFamily {
    Gemma4,
    QwenGdnMoe,
    DeepseekV4Flash,
    Llama,
    Qwen3Moe,
    GptOss,
    QwenGdnDense,
    MuseGlimmer,
    Qwen4Exp,
    Spark25,
    Qwen3Dense,
    MiniMaxM2,
    Qwen2Dense,
    Deepseek2,
    Qwen3Vl,
}
```

---

## 1. Gemma 4 (`gemma4`)
* **Format string**: `"gemma4"`
* **Architecture**: Fine-grained MoE with alternating Sliding Window Attention (SWA at 2048) and full attention, logit softcap, and sandwich normalization.
* **Key Checkpoints**: `gemma4` (26B-A4B Instruct, MLX INT4), `gemma4-iq3` (UD-Q3_K_M GGUF), `humanizer-12b` (Dense 12B, MLX INT8).
* **Execution**: Streams routed experts from SSD. Bounded Metal slot cache achieves a process memory peak of ~1.9 to 2.1 GB at 4k context.

---

## 2. QwenGdnMoe (`qwen36`)
* **Format string**: `"qwen36"` *(historical on-disk compatibility string preserved)*
* **Architecture**: Hybrid Gated-DeltaNet (GDN) recurrent linear attention with full attention every fourth layer. Features 256 routed experts (top-8) plus a shared expert, `head_dim` 256, and partial rotary factor 0.25 at theta 1e7.
* **Key Checkpoints**: `qwen36` (Qwen 3.6 35B-A3B), `ornith35b` (Ornith-1.5 35B-A3B), `minicpm-v47-35b-mxfp4` (MiniCPM-V 4.7 35B-A3B text trunk).
* **Execution**: Batched GEMV for resident matrix reads by default on INT4 installs, with routed experts evaluated on the streaming loop.

---

## 3. QwenGdnDense (`qwen35`)
* **Format string**: `"qwen35"` *(historical on-disk compatibility string preserved)*
* **Architecture**: Dense Gated-DeltaNet hybrid. 64 layers (48 GDN recurrent layers, 16 full-attention layers), hidden dimension 5120. Dense FFN (`mlp.gate`, `up`, `down`).
* **Key Checkpoints**: `qwen38-27b`, `qwen38-27b-mtp` (with MTP speculative head), `bonsai27b` (1-bit), `ternary27b` (2-bit), `bonsai2` (2-bit Hadamard-folded), `ornith9b` (Q8_0).
* **Execution**: Weights memory-mapped. Supports Steel GEMM/GEMV prefill kernels, native MTP speculative verification, and optional deepstack vision tower.

---

## 4. Qwen4Exp (`qwen4exp`)
* **Format string**: `"qwen4exp"` (Qwen 4 line, Qwen3.8-Flash-Next)
* **Architecture**: 48 layers with hyper-connections: the residual stream is 4x wider than `hidden_size` (`4 * 2048 = 8192`), and standard residual additions are replaced by gated inject/read matrices. Features sigmoid GDN output gate, hashed per-layer n-gram PLE embeddings (30.8% of checkpoint), and query-sparse attention indexer.
* **Key Checkpoints**: `qwen4-reap288` (288 pruned experts, MLX INT4), `qwen4exp-swift-iq2-xs` (512 experts, GGUF IQ2_XS).
* **Execution**: Fine-grained MoE streaming from SSD. Process peak ~1.60 GiB at 2k context on an M4 Max.

---

## 5. DeepSeek2 (`deepseek2`)
* **Format string**: `"deepseek2"`
* **Architecture**: Multi-Head Latent Attention (MLA) with fine-grained MoE. Compresses each token's key-value state into a single absorbed row per layer `[latent 512 ; rope-carried key 64]`, converting attention into MQA over the absorbed vector. 64 routed experts (top-6) plus a fused shared expert.
* **Key Checkpoints**: `dsv2lite-16b` (DeepSeek-V2-Lite-Chat 16B, GGUF Q8_0).
* **Execution**: MoE streaming from SSD with absorbed MLA attention cache.

---

## 6. Llama (`llama`)
* **Format string**: `"llama"`
* **Architecture**: Standard GQA decoder stack with full-head NeoX RoPE, SwiGLU, and RMSNorm (eps 1e-5). Covers dense models (Mistral 7B, Llama 2/3.x, TinyLlama) and coarse MoE models (Mixtral 8x7B) distinguished by `expert_count`.
* **Key Checkpoints**: `mistral7b` (Q4_K_M), `tinyllama` (Q6_K), `mixtral` (Q4_K_M).
* **Execution**: Dense models run memory-mapped. Mixtral requires ~55 GiB slot cache capacity due to coarse expert granularity (8 large experts).

---

## 7. Qwen3Moe (`qwen3moe`)
* **Format string**: `"qwen3moe"`
* **Architecture**: Fine-grained MoE on Llama-style flow: plain GQA, no shared expert, no softcap, full-head RoPE. Differs by applying per-head Q/K normalization before RoPE and RMSNorm eps 1e-6.
* **Key Checkpoints**: `qwen3moe` (Qwen3-30B-A3B, GGUF Q4_K_M).
* **Execution**: Streams small, fine-grained routed experts from SSD (~2.7 GB process peak @ 4k context).

---

## 8. Qwen3Dense (`qwen3dense`)
* **Format string**: `"qwen3dense"`
* **Architecture**: Dense Qwen 3 architecture (0.6B to 32B dense, Qwen3-Coder). Evaluated on the Llama flow with per-head Q/K normalization, RMSNorm eps 1e-6, and dense SwiGLU FFN.
* **Key Checkpoints**: Dense Qwen3 checkpoints, DeepSeek-R1-0528-Qwen3-8B.

---

## 9. Qwen2Dense (`qwen2`)
* **Format string**: `"qwen2"`
* **Architecture**: Dense Qwen 2 and Qwen 2.5 full-attention models. Unlike Qwen 3, it carries Q/K/V projection biases and does not perform per-head Q/K normalization.
* **Key Checkpoints**: `qwen25-7b-4bit` (MLX INT4), `qwen25-7b-q3km` (GGUF Q3_K_M), `qwen25-7b-q4km` (GGUF Q4_K_M).
* **Execution**: Memory-mapped execution with Q3_K and Q4_K custom Metal GEMV kernels.

---

## 10. GptOss (`gptOss`)
* **Format string**: `"gptOss"`
* **Architecture**: Fine-grained MoE with per-projection biases on Q/K/V/O and router, learned attention sinks (one logit per Q head added to softmax denominator), YaRN rope scaling, alternating 128-token SWA sliding window, and clamped SwiGLU `swish(gate) * (up + 1)`.
* **Key Checkpoints**: `gptoss-20b` (GGUF MXFP4).
* **Execution**: Streams 32 fine-grained routed experts from SSD (~5.4 GB process peak @ 8k context).

---

## 11. MuseGlimmer (`museGlimmer`)
* **Format string**: `"museGlimmer"`
* **Architecture**: Dense 52-layer GQA stack with centered layer norms (`1 + w`), alternating 3 sliding window / 1 full attention window, NoPE on full layers (layer_rope_theta = 0), unlearned per-head Q/K norms, and logit softcap 20.
* **Key Checkpoints**: `museglimmer` (Muse Glimmer 30B, MLX INT4).
* **Execution**: Dense memory-mapped text trunk execution.

---

## 12. Spark25 (`spark2_5`)
* **Format string**: `"spark2_5"`
* **Architecture**: Dense 36-layer hybrid stack (27 SWA layers at window 512, 9 full layers). Features per-class RoPE with partial factor (rotates leading 64 of 256 dims on full layers), headwise scalar sigmoid output gate, exact-erF GELU MLP, and fused QKV.
* **Key Checkpoints**: `spark25` (Spark-X2.5 4B, GGUF Q4_K_M).
* **Execution**: Dense memory-mapped execution.

---

## 13. Qwen3Vl (`qwen3_vl`)
* **Format string**: `"qwen3_vl"`
* **Architecture**: Dense 36-layer GQA stack, full rotary NeoX RoPE, per-head Q/K RMSNorm at eps 1e-6, tied embeddings. Supported as a multimodal model with depth-24 SigLIP2 vision tower and 3 deepstack mergers that inject visual tokens directly into trunk residual layers 0-2.
* **Key Checkpoints**: `qwen3vl-4b` (text trunk), `qwen3vl-4b-vision` (combined multimodal).
* **Execution**: Vision preprocessing via `crates/vision-io`, vision tower Metal encoding, and trunk LLM generation.

---

## 14. MiniMaxM2 (`minimax_m2`)
* **Format string**: `"minimax_m2"`
* **Architecture**: Full GQA with whole-projection Q/K normalization and sigmoid-routed MoE.
* **Status**: Implemented in engine; pending catalog promotion.

---

## 15. DeepSeek-V4-Flash (`deepseek_v4_flash`)
* **Format string**: `"deepseek_v4_flash"`
* **Architecture**: Scaffolded discriminator and layer graph in preparation for upcoming release.
