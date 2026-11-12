---
description: The .gturbo model installation format, manifest schema, and tensor packing specification.
icon: box-archive
---

# The .gturbo Format Specification

TurboSpark installs and packages models into `.gturbo` directory bundles. The `.gturbo` format separates resident weights from streaming MoE weights, ensuring predictable memory mapping and fast startup times.

## Directory Structure

A complete `.gturbo` installation contains:

```text
model-name.gturbo/
|-- manifest.json          # Architectural metadata, tensor index, and hardware configuration
|-- resident.bin           # Continuous memory-mapped buffer containing dense attention and norms
|-- streamed.bin           # Packed routed expert tensors indexed by layer and expert ID (MoE only)
|-- tokenizer.json         # Fast Hugging Face tokenizer vocabulary and merges
|-- tokenizer_config.json  # Tokenizer special tokens and configuration
|-- chat_template.jinja    # Jinja chat formatting template
`-- receipt.json           # Cryptographic installation and verification receipt
```

---

## The `manifest.json` Schema

The `manifest.json` file is the contract between model packaging and the inference runtime:

```json
{
  "format_version": 1,
  "arch": {
    "family": "qwen36",
    "hidden_size": 2048,
    "num_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 152064,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000000.0,
    "expert_count": 256,
    "expert_used_count": 8,
    "expert_shared_count": 1
  },
  "quantization": {
    "dense": "int4_g64",
    "experts": "int4_g64"
  },
  "tensors": {
    "model.embed_tokens.weight": {
      "location": "resident",
      "offset": 0,
      "length": 155713536,
      "shape": [152064, 2048],
      "dtype": "int4_g64"
    }
  }
}
```

### Critical Invariants

* **`arch.family`**: Specifies the runtime family discriminator (`ModelFamily`). Decode flow is determined strictly by this field.
* **Separation of Concerns**: Dense layers live in `resident.bin` and are mapped with `mmap`. Routed experts live in `streamed.bin` and are loaded on demand into GPU cache slots.
