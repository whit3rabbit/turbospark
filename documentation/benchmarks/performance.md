---
description: Measured tokens per second, memory footprints, and power efficiency across Apple Silicon chips.
icon: chart-line
---

# Benchmarks and Performance

All measurements below reflect real hardware runs on Apple Silicon devices. Benchmarks are conducted using the committed release harness (`crates/bench`) with isolated host conditions.

## LLM Inference Benchmarks (Apple M4 Max, 36 GB Unified Memory)

The table below lists measured `phys_footprint` process peaks and decode tokens per second at the stated context window. MoE models use 16 expert-cache slots.

| Model | Context | Quantization | Format | Measured Process Peak | Decode Tok/s | Install Size |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** | 4,096 | INT4 (group 64) | MLX | 2,175 MiB | 33.0 - 45.6 | ~13.0 GB |
| **Qwen 3.6 35B-A3B** | 4,096 | INT4 (group 64) | MLX | 1,610 MiB | 32.6 - 38.0 | ~18.0 GB |
| **MiniCPM-V 4.7 35B** | 4,096 | mxfp4 (group 32) | MLX | 1,491 MiB | 24.0 - 25.6 | ~22.0 GB |
| **Qwen3-30B-A3B** | 4,096 | Q4_K_M | GGUF | 2,748 MiB | 16.0 - 27.3 | ~18.6 GB |
| **Swift-1.5 Qwen3.8** | 2,048 | IQ2_XS | GGUF | 1,634 MiB | 12.2 - 12.8 | ~69 GB res. |
| **Qwen 3.8 27B Dense**| 4,096 | INT4 (group 64) | MLX | 661 MiB | 16.8 - 19.0 | ~15.2 GB |
| **Qwen 3.8 27B + MTP** | 4,096 | INT4 + MTP Head | MLX | 872 MiB | **21.9 - 28.7** | ~15.4 GB |
| **gpt-oss 20B** | 8,192 | MXFP4 | GGUF | 5,422 MiB | 23.4 - 31.3 | ~12.2 GB |
| **Humanizer 12B** | 8,192 | INT8 (group 64) | MLX | 774 - 924 MiB | 6.5 - 13.4 | ~12.7 GB |

*Footprint note*: For dense models, `phys_footprint` measures dirty heap and scratch allocations, not memory-mapped weight pages. For MoE models, `phys_footprint` accurately reflects the entire active working set because routed experts stream into pre-allocated slot buffers.

---

## Image Diffusion Timings (M4 Max)

| Model | Steps | Resolution | Engine | Generation Time | Peak Memory |
| --- | ---: | --- | --- | --- | ---: |
| **Z-Image Turbo (8-bit)** | 9 | 1024 x 1024 | Swift MLX | ~58 - 87 seconds | 12.8 GiB |
| **Qwen-Image 2.1 Turbo** | 8 | 1024 x 1024 | Swift MLX | ~96 seconds | 11.2 GiB |
| **Qwen-Image 2.1 Turbo** | 8 | 512 x 512 | Swift MLX | ~20 - 26 seconds | 8.4 GiB |

---

## Power Draw and Efficiency (Watts)

Measurements taken on Apple Silicon hardware using macOS `powermetrics` during sustained generation:

* **Idle Baseline**: ~1.2 - 2.5 Watts.
* **Dense Prefill Phase**: ~32 - 44 Watts (saturates memory bus and GPU cores).
* **MoE Token Generation**: ~14 - 22 Watts (highly energy efficient due to sparse memory access).
* **Audio Voice Synthesis**: ~8 - 15 Watts on Metal GPU.
