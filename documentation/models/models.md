---
description: Discover catalog aliases, evidence status, and download instructions for supported models.
icon: list-check
---

# Supported Models

The TurboSpark catalog in `crates/catalog/src/models.json` is the source of truth for all verified models, aliases, and evidence statuses.

## Catalog Commands

```sh
# List all models in the catalog
turbospark-model list

# Get ranked recommendations for your Mac and context size
turbospark-model recommend --context 4096

# Inspect detailed metadata, checksums, and measured evidence for an alias
turbospark-model info gemma4

# Install a model
turbospark-model pull gemma4
```

## Evidence Levels

Every catalog model carries an evidence status:

* **`verified`**: The model has frozen quality test gates (e.g. perplexity, logit tolerance) and a frozen memory oracle reading on real Apple Silicon hardware.
* **`runs`**: The model installs cleanly and generates coherent output, but does not yet have a frozen quality gate row.
* **`caveat`**: The model runs, but has a documented operational limitation (e.g. high slot-cache memory requirements or unusual quantization).

---

## Catalog Models Overview

### Mixture-of-Experts (MoE) Models (Streaming from SSD)

MoE models stream routed expert weights on demand from SSD through a bounded Metal slot cache (typically 16 slots). This allows large models to run with a fraction of their on-disk size in system RAM.

| Alias | Model Name | Format | Quantization | Disk Size | Measured Process Peak | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| `gemma4` | Gemma 4 26B-A4B Instruct | MLX | INT4 (group 64) | ~13 GB | ~2.1 GB @ 4k ctx | verified |
| `gemma4-iq3` | Gemma 4 26B-A4B Instruct | GGUF | UD-Q3_K_M (IQ3 experts) | ~13 GB | ~1.9 GB @ 4k ctx | verified |
| `gemma4-gguf` | Gemma 4 26B-A4B Instruct | GGUF | Q8_0 | ~27 GB | - | runs |
| `qwen36` | Qwen 3.6 35B-A3B | MLX | INT4 (group 64) | ~18 GB | ~1.6 GB @ 4k ctx | verified |
| `qwen36-gguf` | Qwen 3.6 35B-A3B | GGUF | Q4_K_M | ~21 GB | - | runs |
| `ornith35b` | Ornith-1.5 35B-A3B | MLX | INT4 (group 64) | ~20 GB | - | verified |
| `ornith35b-gguf` | Ornith-1.5 35B-A3B | GGUF | Q8_0 | ~38 GB | - | runs |
| `minicpm-v47-35b-mxfp4` | MiniCPM-V 4.7 35B-A3B (Text trunk) | MLX | mxfp4 (group 32) | ~22 GB | ~1.5 GB @ 4k ctx | verified |
| `qwen4-reap288` | Qwen3.8-Flash-Next REAP-288 | MLX | INT4 (288 experts) | ~74 GB | ~2.5 GB @ 2k ctx | verified |
| `qwen4exp-swift-iq2-xs`| Swift-1.5 Qwen3.8-Flash-Next | GGUF | IQ2_XS (512 experts) | ~69 GB res. | ~1.60 GB @ 2k ctx | verified |
| `qwen3moe` | Qwen3-30B-A3B | GGUF | Q4_K_M | ~19 GB | ~2.7 GB @ 4k ctx | verified |
| `gptoss-20b` | gpt-oss 20B | GGUF | MXFP4 | ~12 GB | ~5.4 GB @ 8k ctx | verified |
| `dsv2lite-16b` | DeepSeek-V2-Lite-Chat 16B (MLA) | GGUF | Q8_0 | ~17 GB | ~4.1 GB @ 8k ctx | runs |
| `mixtral` | Mixtral 8x7B Instruct v0.1 | GGUF | Q4_K_M | ~29 GB | ~55 GB cache req. | caveat |

---

### Dense Models (Weights Memory-Mapped)

Dense models keep all layer weights in memory-mapped address space. Plan for the on-disk size plus KV-cache and scratch memory that scales with context window.

| Alias | Model Name | Format | Quantization | Disk Size | Notes | Status |
| --- | --- | --- | --- | ---: | --- | --- |
| `qwen38-27b` | Qwen3.8 27B Dense | MLX | INT4 (group 64) | ~15 GB | Hybrid DeltaNet + Attention, Steel GEMM prefill | verified |
| `qwen38-27b-mtp` | Qwen3.8 27B + MTP Head | MLX | INT4 (group 64) | ~15.4 GB | Native MTP speculative decoding (21-28 tok/s) | verified |
| `qwen38-27b-vision` | Qwen3.8 27B with Vision | MLX | INT4 + BF16 tower | ~16 GB | Combined text + vision tower | verified |
| `bonsai27b` | Bonsai 27B (1-bit) | MLX | Affine 1-bit (group 128) | ~3.9 GB | 1-bit quantization of Qwen3.8 line | runs |
| `ternary27b` | Ternary-Bonsai 27B (2-bit) | MLX | Affine 2-bit (group 128) | ~8 GB | 2-bit quantization | verified |
| `bonsai2` | Ternary-Bonsai 2 27B | MLX | 2-bit Hadamard-folded | ~8 GB | Hadamard transform preserves outliers | verified |
| `qwen3vl-4b` | Qwen3-VL 4B Instruct | MLX | INT4 (group 64) | ~2.3 GB | Text trunk; combined install adds vision tower | verified |
| `museglimmer` | Muse Glimmer 30B | MLX | INT4 (group 64) | ~16 GB | Centered norms, 52-layer hybrid stack | verified |
| `humanizer-12b` | Humanizer 12B (Dense Gemma 4) | MLX | INT8 (group 64) | ~12.7 GB | Raw completion engine for text rewriting | verified |
| `qwen25-7b-4bit` | Qwen2.5 7B Instruct | MLX | INT4 (group 64) | ~4.5 GB | Standard dense full-attention | verified |
| `qwen25-7b-q3km` | Qwen2.5 7B Instruct | GGUF | Q3_K_M | ~4.2 GB | Official single-file GGUF | verified |
| `ornith9b` | Ornith-1.5 9B | GGUF | Q8_0 | ~10 GB | High-precision dense Qwen variant | verified |
| `spark25` | Spark-X2.5 4B | GGUF | Q4_K_M | ~2.6 GB | Hybrid SWA, exact-erF GELU | runs |
| `mistral7b` | Mistral 7B Instruct v0.3 | GGUF | Q4_K_M | ~4.4 GB | Standard Llama architecture | runs |
| `tinyllama` | TinyLlama 1.1B Chat | GGUF | Q6_K | ~0.9 GB | Ultra-compact dense model | runs |

---

## Probing Unlisted Hugging Face Models

You can test any Hugging Face GGUF or MLX checkpoint before downloading weights using `probe`:

```sh
turbospark-model probe TheBloke/Llama-2-7B-Chat-GGUF --file llama-2-7b-chat.Q4_K_M.gguf
```

The probe inspects remote headers to determine:
1. Model family and architecture compatibility.
2. Quantization type (ensuring custom Metal dequantization kernels exist).
3. Expert granularity (for MoE models) and slot cache viability.
4. Tokenizer sidecars and chat template availability.
