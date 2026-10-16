---
description: Understanding MoE streaming, slot cache math, process memory footprint, and context window capacity.
icon: memory
---

# Memory and Capacity

One of TurboSpark's primary architectural advantages is fine-grained Mixture-of-Experts (MoE) streaming: routed expert weights remain on SSD until required by the active token, allowing 20B to 35B parameter models to run with a small bounded process memory footprint on Apple Silicon.

## Key Concepts: Footprint vs. Size

Understanding memory in TurboSpark requires distinguishing four metrics:

1. **On-Disk Size**: The compressed download or packed file size on SSD (e.g., ~18 GB for Qwen 3.6 35B INT4, ~69 GB for Swift-1.5 IQ2_XS).
2. **Process Peak Footprint (`phys_footprint`)**: The physical dirty memory allocated by the TurboSpark process during inference, measured by macOS task diagnostic counters.
3. **Working Set**: The total system memory including file-backed memory-mapped pages managed dynamically by the macOS virtual memory subsystem.
4. **Resident vs. Streamed Weights**: Dense layer weights (embeddings, attention projections, norms) stay resident in memory, while routed MoE experts stream through the slot cache.

---

## MoE Expert Streaming & Slot Cache

In an MoE model, only a fraction of total experts are activated per token (e.g., top-8 of 256 experts in Qwen 3.6, top-10 of 512 experts in Qwen 3.8-Flash-Next).

Instead of holding all experts in RAM:
* TurboSpark allocates a bounded **Metal Slot Cache** (default: 16 slots per layer).
* Active experts are loaded into cache slots via asynchronous memory-mapped reads.
* Experts are processed in router ranking order (reduction order).
* Unused slots are recycled using an LRU cache policy.

### Why Expert Granularity Matters

Slot cache size is determined by **expert granularity**, not overall model file size:

* **Fine-Grained MoE (Qwen 3.6 35B-A3B, Gemma 4 26B-A4B)**:
  * Each expert is small (~12 to 18 MiB per expert).
  * 16 slots per layer require only ~1.5 to 2.1 GiB of total GPU slot-cache capacity.
  * Runs smoothly on 16 GB and 36 GB Macs.

* **Coarse-Grained MoE (Mixtral 8x7B)**:
  * Each expert is massive (~3.5 GiB per expert).
  * 16 slots per layer would require ~55 GiB of cache capacity.
  * Consequently, Mixtral carries a `caveat` status because it cannot stream within typical consumer Mac RAM limits.

---

## Dense Models (Memory-Mapped)

Dense models (e.g. Qwen 3.8 27B, Bonsai 27B, Mistral 7B) do not stream experts. Every parameter is required for every token.

For dense models:
* Budget for the **full model size on disk** in unified memory.
* Plus **KV Cache memory**: scales linearly with context length and layer count.
* Plus **Scratch & Activation buffers**: depends on batch size and hidden dimensions.

The low process peak reported for dense models in benchmarks is a memory-leak sentinel: memory-mapped file pages do not count toward process dirty footprint until paged in by macOS.

---

## Context Window Scaling

As you increase the conversation context window, KV-cache memory expands:

| Model | 2,048 Tokens | 4,096 Tokens | 8,192 Tokens | 16,384 Tokens |
| --- | ---: | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** | ~1.6 GB | ~2.1 GB | ~3.2 GB | ~5.8 GB |
| **Qwen 3.6 35B-A3B** | ~1.2 GB | ~1.6 GB | ~2.6 GB | ~4.9 GB |
| **DeepSeek-V2 Lite 16B (MLA)** | ~2.2 GB | ~2.9 GB | ~4.1 GB | ~6.5 GB |
| **Qwen 3.8 27B Dense** | ~16.2 GB | ~16.9 GB | ~18.5 GB | ~21.8 GB |

*Note: DeepSeek-V2's absorbed Multi-Head Latent Attention (MLA) compresses the KV-cache to a single 576-half row per layer, significantly reducing context memory growth compared to standard MHA/GQA.*

---

## Load Guardrails & Memory Pressure Watcher

TurboSpark actively protects your Mac from system freezing or out-of-memory (OOM) kernel panics:

1. **AutoFit Floor**: When starting generation, TurboSpark checks current system free memory. If free memory is below safety thresholds, the runtime refuses allocation with a descriptive error rather than crashing.
2. **Pressure Watcher**: An asynchronous monitor tracks macOS dispatch memory pressure events (`DISPATCH_MEMORYPRESSURE_WARN` and `CRITICAL`). If pressure reaches critical levels, caches are purged and active generations stop cleanly.
3. **Load Guard Settings**: You can configure cache slot counts using `--expert-cache-slots <N>` to adapt memory consumption to available RAM.
