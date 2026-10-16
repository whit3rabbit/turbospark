---
description: Native speculative decoding with Multi-Token Prediction (MTP) heads and DFlash drafters.
icon: gauge-high
---

# Speculative Decoding and MTP

Speculative decoding accelerates token generation by drafting multiple speculative candidate tokens and verifying them in parallel with a single forward pass of the target model.

On Apple Silicon, where inference speed is primarily bound by unified memory bandwidth rather than compute ALU limits, verifying multiple tokens simultaneously yields measurable throughput increases without degrading output quality.

## Multi-Token Prediction (MTP) Heads

The primary speculative mechanism implemented in TurboSpark is **Native Multi-Token Prediction (MTP)**:

* **Architecture**: The model carries a dedicated MTP prediction head that predicts token `t + 1` concurrently with token `t`.
* **Block Speculation**: Candidates are proposed in blocks (typically block length 2).
* **Parallel Verification**: The main model's next forward pass evaluates candidate logits in parallel. Accepted tokens are emitted immediately; rejected candidates roll back to the first mismatch.

### Measured Performance Gains (Qwen 3.8 27B on M4 Max)

| Configuration | Context | Decoding Speed | Measured Footprint |
| --- | ---: | ---: | ---: |
| **Qwen 3.8 27B (Standard greedy decode)** | 4,096 | 14.7 - 17.9 tok/s | ~661 MiB |
| **Qwen 3.8 27B + MTP Head (Speculative block 2)** | 4,096 | **21.9 - 28.7 tok/s** | ~872 MiB |

*Result*: MTP speculation delivers an approximately **1.4x to 1.6x speedup** on real Apple Silicon hardware with zero loss in output fidelity.

---

## Running with Speculation

Install the MTP-enabled checkpoint:

```sh
turbospark-model pull qwen38-27b-mtp
```

Run inference with speculative decoding enabled:

```sh
turbospark-check --model qwen38-27b-mtp --chat
```

---

## DFlash Drafter Research

TurboSpark also includes experimental support for **DFlash** (a block-diffusion speculative drafter). While DFlash generates fast candidates, empirical benchmarks show that native MTP heads provide significantly higher token acceptance rates with lower memory overhead on Apple Silicon unified memory.
