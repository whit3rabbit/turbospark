---
description: 1-bit Bonsai, 2-bit Hadamard-folded Ternary-Bonsai 2, and TurboQuant KV-cache quantization.
icon: binary
---

# Extreme Quantization

TurboSpark pushes local inference efficiency boundaries by supporting extreme sub-4-bit weights and KV-cache compression on Apple Silicon.

## The Bonsai Line: 1-Bit and 2-Bit Models

The Bonsai line represents research into running full-sized 27B parameter models in under 8 GB of unified memory:

| Checkpoint | Alias | Quantization | Size on Disk | Notes |
| --- | --- | --- | ---: | --- |
| **Bonsai 27B** | `bonsai27b` | Affine 1-bit (group 128) | ~3.9 GB | 1 bit per weight; fits on an 8 GB Mac |
| **Ternary-Bonsai 27B** | `ternary27b` | Affine 2-bit (group 128) | ~8.0 GB | 2 bits per weight; improved perplexity |
| **Ternary-Bonsai 2 27B** | `bonsai2` | 2-bit Hadamard-folded | ~8.0 GB | Newest; Hadamard transform suppresses outlier errors |

### Why Hadamard Folding Matters (Bonsai 2)

Standard 2-bit and 1-bit quantization suffers when specific weight matrix channels carry large outlier activations.

**Ternary-Bonsai 2** applies an orthogonal **Walsh-Hadamard Transform** to the projection matrices prior to quantization:
* The Hadamard matrix rotates the weight space, spreading activation spikes across all dimensions.
* Outlier values are flattened without changing mathematical projection results.
* The quantized model achieves substantially closer logit alignment and perplexity to unquantized FP16 baselines.

---

## TurboQuant KV-Cache Quantization

At long context lengths (e.g. 16k or 32k tokens), the Key-Value (KV) cache often consumes more memory than the model weights themselves.

**TurboQuant** compresses the KV-cache using dedicated quantized ring buffers:

* **Standard FP16 KV Cache**: 2 bytes per element per layer.
* **TurboQuant INT8**: 1 byte per element (~50% memory reduction).
* **TurboQuant INT4**: 0.5 bytes per element (~75% memory reduction).

### Enabling KV-Cache Quantization

Configure KV-cache quantization via environment variable or CLI flag:

```sh
# Set KV-cache quantization to INT8
export TURBOSPARK_KV_QUANT=int8
turbospark-server --model gemma4

# Or INT4 for maximum context memory reduction
export TURBOSPARK_KV_QUANT=int4
turbospark-server --model gemma4
```

TurboQuant derives the ring buffer layout directly from the model layer mask and caches compiled Metal pipeline state objects with function constants to prevent pipeline mismatches.
