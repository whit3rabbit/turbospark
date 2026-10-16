---
description: Resolve model pull, Hugging Face probe, quantization, and installation errors.
icon: triangle-exclamation
---

# Model Install Troubleshooting

This guide addresses common issues encountered when probing, pulling, or running models in TurboSpark.

## Common Issues & Solutions

### 1. "Refused at repack: unsupported quantization"

**Symptom**: `turbospark-model pull` fails during the repack phase with an error stating that a tensor quantization arm is missing.

**Cause**: TurboSpark uses handcrafted Metal dequantization kernels. While standard formats (`INT4`, `Q4_0`, `Q4_K_M`, `Q8_0`, `MXFP4`, `IQ2_XS`, 1-bit, 2-bit affine) are supported, uncommon hybrid quantizations (such as checkpoints where some routed experts use `Q5_K` or `IQ1_S`) do not have matching GPU kernel arms.

**Solution**:
* Check `turbospark-model info <alias>` for recommended variants.
* For GGUF checkpoints, prefer official `Q4_K_M` or `Q8_0` single-file conversions.
* For MLX checkpoints, prefer standard 4-bit (`INT4 group 64`) or 8-bit conversions.

---

### 2. "Split GGUF shards not supported"

**Symptom**: Probing or pulling a multi-part GGUF (e.g. `model-00001-of-00004.gguf`) fails.

**Cause**: The single-file GGUF intake pipeline expects all tensor headers and weights to reside in a single contiguous addressable file.

**Solution**:
* Use the single-file conversion (often published by community quantizers like `mradermacher`).
* Or merge split shards into a single file with `llama.cpp`'s `gguf-split --merge`.

---

### 3. Missing Tokenizer Sidecars

**Symptom**: Model downloads successfully, but chat fails with `Missing tokenizer.json` or `Chat template not found`.

**Cause**: The model repository lacks standard Hugging Face tokenizer files.

**Solution**:
* TurboSpark catalog entries automatically pull sidecars from canonical parent repositories if the quantized repo omits them.
* If pulling an unlisted model via `--repo`, verify that the repository (or its upstream base model) contains `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`.

---

### 4. "Probe returned NOT_RUNNABLE"

**Symptom**: `turbospark-model probe <repo>` reports that the checkpoint cannot run.

**Cause**:
* The architecture is not one of the 15 supported model families.
* The expert count is zero for an expected MoE model, or expert granularity is too coarse for slot-cache streaming.
* Checkpoint uses unsupported rope scaling or positional embedding variants.

**Solution**: Check [Supported Model Families](model-families.md) for current architecture compatibility.

---

### 5. Download Interrupted or Incomplete Checksum

**Symptom**: `turbospark-model pull` stops unexpectedly or fails checksum verification.

**Solution**:
* Re-run `turbospark-model pull <alias>`. TurboSpark supports HTTP Range resumes and will continue downloading from the last completed chunk.
* To clear a corrupted partial download, remove the temporary directory under `~/.turbospark/tmp/`.

---

### 6. macOS Quarantine on Bundled Tools

**Symptom**: App or command fails to start with "developer cannot be verified".

**Solution**: Clear the quarantine attribute:

```sh
xattr -dr com.apple.quarantine /Applications/TurboSpark.app
xattr -dr com.apple.quarantine ~/.turbospark
```
