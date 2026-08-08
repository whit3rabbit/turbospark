# Supported Model Families & Architecture Detection

This document describes how `mrefrust` detects, registers, and executes supported large language model families, how automatic architecture detection works during GGUF and Hugging Face downloads, and provides a parity comparison against upstream engines like `llama.cpp`, `mlx-lm`, and `turbo-fieldfare`.

---

## 1. Automatic Architecture Detection

When given a Hugging Face URL, local `.gturbo` directory, or GGUF checkpoint, `mrefrust` detects the model architecture automatically before fetching large weight payloads.

```
                    +--------------------------------+
                    |  GGUF / HF Checkpoint Download |
                    +---------------+----------------+
                                    |
                                    v
                +----------------------------------------+
                | Read Metadata Header                   |
                | - GGUF: `general.architecture`         |
                | - HF: `config.json -> model_type`      |
                +-------------------+--------------------+
                                    |
                                    v
            +------------------------------------------------+
            |  `family_for_architecture(arch)`               |
            |  Map string -> `ModelFamily` enum              |
            +-----------------------+------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
    [ Known Family ]                                [ Unknown Family ]
  - Parse `ArchConfig`                       - Reject with error:
  - Transcode & Stream to `.gturbo`            `UnsupportedArchitecture`
  - Execute via Metal shaders                - (See `docs/NEW_MODEL.md`)
```

### Detection Strategy (GGUF vs. Hugging Face)
- **GGUF Checkpoints**: `mrefrust-repack` fetches the initial ~512 KB metadata header via `HttpRangeSource` and inspects `general.architecture`:
  - `"gemma4"` -> `ModelFamily::Gemma4`
  - `"qwen35moe"` -> `ModelFamily::Qwen36`
- **Hugging Face Safetensors**: `mrefrust-repack` fetches `config.json` and parses `model_type` or `architectures`:
  - `"gemma4"` -> `ModelFamily::Gemma4`
  - `"qwen2_moe"` / `"qwen3_5_moe"` -> `ModelFamily::Qwen36`

---

## 2. Complete Model Family Parity Matrix

The table below provides a comprehensive list of all major LLM architectures supported across `llama.cpp`, `mlx-lm`, `turbo-fieldfare`, and `mrefrust`.

| Model Family / GGUF `general.architecture` | Key Architectural Features | `mrefrust` (Rust) | `turbo-fieldfare` (Swift) | `llama.cpp` | `mlx-lm` | Peak RAM Footprint in `mrefrust` |
| --- | --- | :---: | :---: | :---: | :---: | ---: |
| **Gemma 4 26B-A4B** (`gemma4`) | SWA/Full Attention, MoE (128 experts, top-8), Tied Embeddings | **Full Support** | **Full Support** | Full Support | Full Support | **~2.1 GiB RAM** |
| **Qwen 3.6 35B-A3B** (`qwen35moe`) | Gated-DeltaNet Linear Attention + MoE (256 experts, top-8) | **Full Support** | **Full Support** | Full Support | Full Support | **~1.6 GiB RAM** |
| **DeepSeek V4 Flash** (`deepseek`, `deepseek2`) | Multi-head Latent Attention (MLA), DeepSeek MoE, Sinkhorn combine | *Scaffolded* | *Scaffolded* | Full Support | Full Support | *TBD* |
| **Llama 3 / 3.1 / 3.2 / 3.3 / Llama 2** (`llama`) | Standard Dense Transformer, GQA, RoPE frequency scaling | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **Mistral 7B / Mixtral 8x7B / 8x22B** (`mistral`, `mixtral`) | Sliding-Window Attention (SWA), MoE expert routing | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **Phi-2 / Phi-3 / Phi-3.5 / Phi-4** (`phi2`, `phi3`) | SuScaled RoPE, Partial RoPE, Block-sparse / Dense attention | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **Command-R / Command-R+** (`command-r`) | RAG / Tool-calling tuned architecture | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **Grok-1** (`grok`) | 314B MoE architecture (8 experts, top-2) | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **DBRX** (`dbrx`) | Fine-grained MoE (16 experts, top-4) | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **StarCoder / StarCoder2 / Stargate** (`starcoder`, `starcoder2`) | Specialized code generation Transformer | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **Falcon 7B / 40B / 180B** (`falcon`) | Multi-Query Attention (MQA), parallel attention/FFN | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **Baichuan / Baichuan2** (`baichuan`) | ALiBi / RoPE dense architecture | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **InternLM / InternLM2** (`internlm2`) | GQA, RoPE scaling | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **MiniCPM / MiniCPM3** (`minicpm`, `minicpm3`) | SwiGLU, Scale-depth RoPE | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **OLMo / OLMo 2** (`olmo`, `olmo2`) | Non-bias LayerNorm, SwiGLU | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **Exaone** (`exaone`) | GQA, SwiGLU | *Planned* | *Planned* | Full Support | Full Support | *TBD* |
| **GPT-2 / GPT-NeoX / MPT / Bloom** (`gpt2`, `gptneox`, `mpt`, `bloom`) | Legacy dense autoregressive Transformers | *Planned* | *Planned* | Full Support | Full Support | *TBD* |

---


## 3. How `llama.cpp` Handles Architectures vs. `mrefrust`

`llama.cpp` handles architecture discovery using a centralized enum and dynamic graph construction:

1. **Architecture Registry (`enum llm_arch`)**: `llama.cpp` defines enum variants for every supported family (e.g. `LLM_ARCH_LLAMA`, `LLM_ARCH_GEMMA2`, `LLM_ARCH_GEMMA4`, `LLM_ARCH_QWEN35MOE`).
2. **Metadata Key Lookup**: GGUF files declare `general.architecture`. `llama.cpp` matches this string against its lookup table.
3. **Graph Builder**: `llama.cpp` constructs a C++ compute graph dynamically based on `llm_arch`.
4. **Hardcoded Fallbacks**: Because GGUF metadata omits hyper-parameter behavioral flags (like sandwich norms or attention scaling formulas), `llama.cpp` hardcodes these in its internal graph builder per `llm_arch`.

### How `mrefrust` Implements This Strategy
`mrefrust` follows a clean, strongly-typed Rust implementation of the same pattern:
- **`ModelFamily` Enum** (`crates/model-io/src/arch_config.rs`): Defines supported discriminators (`Gemma4`, `Qwen36`, `DeepseekV4Flash`).
- **Baseline Specifications** (`crates/model-io/src/arch_baselines.rs`): Provides compile-time defaults for behavioral architecture flags missing from GGUF metadata.
- **Tensor Mapping Engine** (`crates/repack/src/gguf_names.rs`): Maps GGUF tensor naming conventions to canonical parameter names.
- **Dedicated Metal Forward Passes** (`crates/runtime/src/real_forward_*.rs`): Each family owns an optimized Metal execution flow tuned for its layer graph.

---

## 4. Extending Support to New Families

To add a new model family to `mrefrust`:
1. Register the new variant in `ModelFamily` (`crates/model-io/src/arch_config.rs`).
2. Add baseline specs in `arch_baselines.rs`.
3. Follow the 7-phase step-by-step checklist in [`docs/NEW_MODEL.md`](docs/NEW_MODEL.md).

---

## 5. Document References

- Architecture Bring-up Guide: [`docs/NEW_MODEL.md`](docs/NEW_MODEL.md)
- `.gturbo` Format Specification: [`docs/GTURBO.md`](docs/GTURBO.md)
- Benchmark Parity & Measurements: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
