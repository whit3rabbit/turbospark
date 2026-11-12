---
description: "Portable embedding models (EmbeddingGemma, Snowflake arctic-embed, SigLIP2 NaFlex, CLAP) with pinned checkpoints, sentence-transformers pooling, and cosine, single-vector, and MaxSim scoring."
icon: project-diagram
---

# embeddings

`crates/embeddings` (package `turbospark-embeddings`) is TurboSpark's portable embedding layer: six model families plus the sentence-transformers post-processing they share (pooling, L2 and Matryoshka normalization, cosine and ColVision scoring). Everything is plain f32 on the CPU, `#![forbid(unsafe_code)]`, no Metal, no MLX runtime, and no macOS APIs. The family list follows the mlxrs `embeddings` module as the port plan; the numeric reference is Hugging Face `transformers` 4.57.1 and `sentence-transformers` 5.1.2 run on the pinned checkpoints.

## What it does

| Layer | Contents |
|---|---|
| Model families | One module and one cargo feature per family (all on by default), each with its own config parser, shape-checked weight loader, and parity tests |
| sentence-transformers front end | `pooling` and `st_config` (the Pooling module and `modules.json`), `normalize` (L2 and Matryoshka truncation), `similarity` (cosine, single-vector, MaxSim), and prompt tables from `config_sentence_transformers.json` |
| Checkpoint access | `weights::Weights` opens the safetensors shards of a directory and shape-checks every load; MLX affine layers dequantize at load while quantized embedding tables stay packed; `pins` holds the immutable revisions |
| Text front end | `tokenize::TextTokenizer` over the Hugging Face `tokenizers` crate, with sentence-transformers truncation (content cut before special tokens are re-attached) |

The families, all ports of pinned checkpoints:

| Module | Feature | Entry type | Checkpoints (see `crates/embeddings/MODELS.md`) |
|---|---|---|---|
| `embeddinggemma` | `embeddinggemma` | `EmbeddingGemma` | `google/embeddinggemma-300m` plus QAT variants and the mlx-community bf16 / 8-bit / 4-bit conversions |
| `siglip2` | `siglip2` | `Siglip2` | `google/siglip2-base-patch16-naflex`, the dual tower with variable-resolution (NaFlex) image input |
| `clap` | `clap` | `Clap` | `laion/clap-htsat-unfused`, pinned to the `refs/pr/3` safetensors conversion |
| `bert` | `bert` | `BertEmbedder` | Snowflake arctic-embed v1 / v1.5 / l on BERT and l-v2.0 on XLM-RoBERTa, plus the mlx-community l-v2.0 conversions |
| `gte` | `gte` | `Gte` | the rotary GTE encoder behind arctic-embed-m-v2.0 |
| `nomic_bert` | `nomic-bert` | `NomicBert` | the Nomic BERT encoder behind arctic-embed-m-long |

## Where it sits

Depends on:

- [audio](audio.md): the shared f32 kernels (`ops`, `nn`), MLX affine dequantization, and the mel and STFT front ends CLAP needs. Kernels are reused, never copied.
- [model-io](model-io.md): the safetensors reader (`SafetensorsFile`) and `ModelError`.
- `turbospark-vision-io`: image decode and PIL-compatible resize for the SigLIP2 vision tower.
- `serde`, `serde_json`, and the HF `tokenizers` crate.

Used by:

- [runtime](runtime.md), as a dev-dependency: its only in-workspace consumer today is the `embeddinggemma3_probe` example, which runs this crate's CPU port beside a Core ML export. The Core ML embedding pipeline served to the app lives in `crates/runtime/src/coreml/`, not here.

## Public API tour

### Embedders

Every text embedder exposes the same shape:

- `load(dir)`: reads `config.json`, the safetensors shards, and the sentence-transformers metadata. Both on-disk layouts are accepted where they exist: the upstream sentence-transformers directory and the mlx-community conversion.
- `embed(text)`, `embed_batch(texts, prompt_name)`, and `embed_ids(ids)` for pre-tokenized input. Each text runs unpadded, so a result never depends on its batch neighbors.
- `tokenize(text, prompt_name)`, `prompt_names()`, and `prompt(name)` for the checkpoint's prompt table.
- `with_truncate_dim(Some(d))` for Matryoshka outputs, with `dimension()` and `full_dimension()`.
- `max_seq_length()`, `config()`, and accessors for the underlying encoder.

`BertEmbedder` adds `embed_query` and `embed_document` wrappers, `matryoshka_dimensions()`, and `load_with_pooling` for an explicit pooling config (needed for MLX conversions that drop `1_Pooling/`). `Siglip2` adds `embed_image_file`, `preprocess`/`embed_pixels` for variable-resolution input, and `logit`, `logits`, and `probabilities`. `Clap` adds `embed_audio_file`, `embed_audio_with` (with a `LongAudioCrop` policy for long inputs), and `classify`.

### Shared machinery

- `pooling`: `pool` applies every mode a `PoolingConfig` enables, concatenated in reference order (cls, max, mean, mean_sqrt_len, weighted_mean, last_token); `pool_one` applies one mode. Exports `MASK_SUM_FLOOR` and `MAX_POOL_PAD`.
- `similarity`: `cosine_similarity`, `similarity_matrix`, `score_single_vector`, `score_multi_vector` (ColVision MaxSim), and `rank_descending` (stable, NaN sorts last). Sums accumulate in f64.
- `normalize`: `l2_normalize` and `truncate_and_normalize`, the Matryoshka op.
- `st_config`: `PoolingConfig`, `PromptConfig`, and the `modules.json` reader.
- `weights`: `Weights` shard access with `load` (shape-checked), `linear`, and `embedding` (returns an `EmbeddingTable` that dequantizes per gathered row).
- `pins`: `PINS`, the machine-readable table of `PinnedCheckpoint`s (`Family`, repo, full-commit revision, `WeightFormat`, upstream pairing, gated flag). `MODELS.md` must agree with it.

## Examples

The examples need a checkpoint directory on disk. `crates/embeddings/MODELS.md` lists every pin; the `google` repositories are gated behind the Gemma license. The first example is adapted from `crates/runtime/examples/embeddinggemma3_probe.rs`, simplified.

### Retrieve with EmbeddingGemma

```rust
use turbospark_embeddings::embeddinggemma::EmbeddingGemma;
use turbospark_embeddings::similarity_matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = EmbeddingGemma::load("models/embeddinggemma-300m-bf16")?
        .with_truncate_dim(Some(512))?;

    // The checkpoint's prompt table separates queries from documents.
    let query = model.embed_with_prompt("query", "how should I store coffee?")?;
    let docs = model.embed_batch(
        &[
            "Keep beans in an airtight jar away from light.",
            "Rust enums are sum types.",
        ],
        Some("document"),
    )?;

    // Row-major [1, 2] cosine matrix over the 512-dim vectors.
    let scores = similarity_matrix(&query, &docs, model.dimension())?;
    println!("relevance: {scores:?}");
    Ok(())
}
```

### Score images against captions with SigLIP2

```rust
use std::path::Path;

use turbospark_embeddings::siglip2::Siglip2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Siglip2::load("models/siglip2-base-patch16-naflex")?;

    // NaFlex embeds variable-resolution images; preprocess + embed_pixels
    // is the two-step form, embed_image_file does both.
    let image = model.embed_image_file(Path::new("cat.jpg"))?;
    let captions = ["a photo of a cat", "a photo of a dog"]
        .iter()
        .map(|text| model.embed_text(text))
        .collect::<Result<Vec<_>, _>>()?;

    // SigLIP scores each pair independently: one sigmoid per (image,
    // text) pair, so the values do not sum to one across captions.
    let probs = model.probabilities(&[image], &captions)?;
    println!("probabilities: {probs:?}");
    Ok(())
}
```

### Classify audio zero-shot with CLAP

```rust
use std::path::Path;

use turbospark_embeddings::clap::Clap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Clap::load("models/clap-htsat-unfused")?;

    // Loads the WAV, mixes to mono, resamples to the checkpoint's
    // sample rate, and embeds the HTSAT audio tower.
    let audio = model.embed_audio_file(Path::new("field.wav"))?;

    // Scored like the HF zero-shot pipeline and sorted by descending
    // probability (ties keep label order).
    let ranked = model.classify(&audio, &["dog barking", "piano music", "rain"])?;
    for (label, score) in ranked {
        println!("{score:.3}  {label}");
    }
    Ok(())
}
```

## Pins, fixtures, and tests

Every checkpoint is pinned to a full commit in `src/pins.rs`, and every MLX conversion names the upstream checkpoint it came from. Add the pin before adding a loader path for a new checkpoint, and never load a pickle. `tools/gen_<family>_fixtures.py` writes `testdata/<family>/` from the pinned reference with a `manifest.json` recording package versions, revisions, and file digests; regeneration is separate from tests, and tests never download.

The default suite runs tiny random-init models plus frontend parity, so no install is needed. The `#[ignore]`d suite runs the pinned real checkpoints, resolved from `TURBOSPARK_<FAMILY>_DIR` or `~/models/embeddings/<name>`, and skips when absent. It reports max abs diff and min cosine; do not widen a tolerance without a new measurement.

```sh
cargo test -p turbospark-embeddings
cargo test -p turbospark-embeddings --no-default-features
cargo test -p turbospark-embeddings --release -- --ignored --nocapture
cargo check --target x86_64-unknown-linux-gnu -p turbospark-embeddings
cargo clippy -p turbospark-embeddings --tests
```

## Gotchas

{% hint style="warning" %}
Checkpoint provenance is enforced by refusal, not coercion. CLAP is pinned to the `refs/pr/3` safetensors conversion because upstream `main` ships only a pickle (`pytorch_model.bin`), which is never loaded. Only MLX groupwise affine quantization is accepted; any other scheme fails at load. The mlx-community 8-bit and 4-bit EmbeddingGemma repos were converted from the QAT checkpoints, not the base model, so compare them against QAT goldens. The 5-bit and 6-bit EmbeddingGemma conversions are pinned but the loader refuses them until they pass the same golden and real-checkpoint tests as the 4-bit and 8-bit ports.
{% endhint %}

{% hint style="warning" %}
Keep the crate portable and the kernels shared. Fix a kernel in `turbospark-audio`, not in a private copy here; a family-specific variant is allowed only when the shared kernel's numeric contract differs from the reference and the module says why. Refuse config values a family has not implemented (activation, pooling mode, fusion, softcapping) instead of ignoring them. Keep the Linux cross-target check green: from a Mac it fails in `onig_sys` under `tokenizers` unless a Linux C compiler is set (the audio crate has the same caveat); CI's native Linux leg needs none of that.
{% endhint %}

## Related pages

- [Crate overview](overview.md)
- [audio](audio.md): the shared kernels and mel front ends this crate reuses
- [model-io](model-io.md): the safetensors reader underneath `weights`
- [runtime](runtime.md): the Core ML embedding pipeline and backend probes
- [Testing and verification](../developer/testing-and-verification.md)
