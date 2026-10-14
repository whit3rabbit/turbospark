---
description: "Portable text, image, and audio embedding models in turbospark-embeddings: EmbeddingGemma, Snowflake arctic-embed, SigLIP2 NaFlex, and CLAP, with pinned checkpoints."
icon: box
---
# turbospark-embeddings

turbospark-embeddings turns text, images, and audio into unit-length vectors you can compare with cosine similarity. It ships the three model families of the mlxrs `embeddings` module plus the Snowflake arctic-embed models, all checked against Hugging Face `transformers` and `sentence-transformers` on pinned checkpoints:

| Family | Input | Checkpoint | Output |
| --- | --- | --- | --- |
| EmbeddingGemma | text | `google/embeddinggemma-300m` and its MLX conversions | 768, Matryoshka 512/256/128 |
| arctic-embed (BERT, XLM-RoBERTa) | text | `Snowflake/snowflake-arctic-embed-{xs,s,m,m-v1.5,l,l-v2.0}` and the mlx-community l-v2.0 bf16, 8/6/4-bit conversions | 384, 768, or 1024, Matryoshka 256 for m-v1.5 and l-v2.0 |
| arctic-embed GTE | text, multilingual, up to 8192 tokens | `Snowflake/snowflake-arctic-embed-m-v2.0` | 768, Matryoshka 256 |
| arctic-embed Nomic BERT | text, up to 8192 tokens | `Snowflake/snowflake-arctic-embed-m-long` | 768 |
| SigLIP2 NaFlex | image and text | `google/siglip2-base-patch16-naflex` | 768 per tower, sigmoid image-text scores |
| CLAP HTSAT-unfused | 48 kHz audio and text | `laion/clap-htsat-unfused` (safetensors revision) | 512 per tower, zero-shot labels |

Everything runs as plain f32 on the CPU, builds off macOS, and forbids `unsafe`. There is no Metal path and no batching across inputs yet, so treat it as a correct reference, not a throughput engine. The exact repository revisions live in `crates/embeddings/MODELS.md` and `src/pins.rs`.

## Add the crate

The Cargo package is `turbospark-embeddings`. Rust code imports it as `turbospark_embeddings`. Each family is a default cargo feature (`embeddinggemma`, `bert`, `gte`, `nomic-bert`, `siglip2`, `clap`), so turn off default features and pick one if you only need one.

```toml
[dependencies]
turbospark-embeddings = { path = "../turbospark/crates/embeddings" }
```

Download the checkpoints yourself, at the pinned revisions. `google/embeddinggemma-300m` is gated, so accept its license on Hugging Face first. For CLAP use revision `refs/pr/3`, because the upstream `main` ships only a pickle and this crate does not load pickles.

## Smallest working example

The program below compiles and runs as written against a checkout of this repository, with the four checkpoints under `~/models/embeddings/`. The output shown after it comes from that run. The Gemma similarities and SigLIP2 probabilities match `sentence-transformers` and `transformers` on the same inputs to four decimals.

```rust
use std::path::PathBuf;

use turbospark_embeddings::bert::BertEmbedder;
use turbospark_embeddings::clap::Clap;
use turbospark_embeddings::embeddinggemma::EmbeddingGemma;
use turbospark_embeddings::siglip2::{sigmoid, Siglip2};
use turbospark_embeddings::{cosine_similarity, Result};
use turbospark_vision_io::Rgb8Image;

fn model(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
        .join("models/embeddings")
        .join(name)
}

fn main() -> Result<()> {
    // Text retrieval: query and documents use different prompts.
    let gemma = EmbeddingGemma::load(model("embeddinggemma-300m"))?;
    let query = gemma.embed_with_prompt("query", "Which planet is known as the Red Planet?")?;
    for doc in [
        "Mars is often called the Red Planet.",
        "Venus is the hottest planet in the solar system.",
    ] {
        let d = gemma.embed_with_prompt("document", doc)?;
        println!("gemma  {:.4}  {doc}", cosine_similarity(&query, &d));
    }

    // Snowflake arctic-embed l-v2.0, MLX 4-bit conversion. The query gets
    // the checkpoint's "query: " prompt, documents get none.
    let arctic = BertEmbedder::load(model("mlx-snowflake-arctic-embed-l-v2.0-4bit"))?;
    let query = arctic.embed_query("Which planet is known as the Red Planet?")?;
    for doc in [
        "Mars is often called the Red Planet.",
        "Venus is the hottest planet in the solar system.",
    ] {
        let d = arctic.embed_document(doc)?;
        println!("arctic {:.4}  {doc}", cosine_similarity(&query, &d));
    }

    // Image-text: a solid red 320x240 image against two captions.
    let siglip = Siglip2::load(model("siglip2-base-patch16-naflex"))?;
    let (w, h) = (320, 240);
    let red = Rgb8Image {
        width: w,
        height: h,
        data: [255u8, 0, 0].repeat(w * h),
    };
    let image = siglip.embed_image(&red)?;
    for caption in ["a solid red image", "a photo of a dog"] {
        let logit = siglip.logit(&image, &siglip.embed_text(caption)?)?;
        println!("siglip {:.4}  {caption}", sigmoid(logit));
    }

    // Audio-text: three seconds of a 440 Hz sine at 48 kHz.
    let clap = Clap::load(model("clap-htsat-unfused"))?;
    let pcm: Vec<f32> = (0..48_000 * 3)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
        .collect();
    let audio = clap.embed_audio(&pcm)?;
    for (label, p) in clap.classify(&audio, &["a sine wave tone", "a dog barking", "people talking"])? {
        println!("clap   {p:.4}  {label}");
    }
    Ok(())
}
```

Output:

```text
gemma  0.6111  Mars is often called the Red Planet.
gemma  0.3371  Venus is the hottest planet in the solar system.
arctic 0.7199  Mars is often called the Red Planet.
arctic 0.4929  Venus is the hottest planet in the solar system.
siglip 0.8511  a solid red image
siglip 0.0000  a photo of a dog
clap   1.0000  a sine wave tone
clap   0.0000  people talking
clap   0.0000  a dog barking
```

## EmbeddingGemma

`EmbeddingGemma::load(dir)` reads either the sentence-transformers layout of the upstream repository or an MLX conversion (F16 or affine-quantized), chosen from the tensor names.

- `embed_with_prompt(name, text)` prepends a prompt from `config_sentence_transformers.json`. Use `"query"` for search queries and `"document"` for the texts you search over. `prompt_names()` lists the rest (`STS`, `Clustering`, and so on). `embed(text)` uses no prompt.
- `with_truncate_dim(Some(256))` returns Matryoshka embeddings: the first 256 values, renormalized.
- Input longer than 2048 tokens is cut to 2048 with the end-of-sequence token kept, as `sentence-transformers` does.
- The quantized MLX repositories were converted from Google's QAT checkpoints, not the base model, so their vectors differ from the base model's more than quantization alone explains. See the family README before mixing them in one index.

## Snowflake arctic-embed

Three loaders cover the eight Snowflake checkpoints. Each reads the repository's sentence-transformers files (CLS pooling, Normalize, prompts, `max_seq_length`).

- `bert::BertEmbedder::load(dir)` handles `xs`, `s`, `m`, `m-v1.5`, `l` (BERT) and `l-v2.0` (XLM-RoBERTa, multilingual, 8192 tokens), plus the mlx-community l-v2.0 conversions. `embed_query` adds the checkpoint's query prompt and `embed_document` adds its document prompt, which is empty for these models.
- `gte::Gte::load(dir)` handles `m-v2.0`, and `nomic_bert::NomicBert::load(dir)` handles `m-long`. Use `embed_with_prompt("query", text)` for queries and `embed(text)` for documents.
- The MLX conversions ship no pooling config. The loader uses l-v2.0's CLS pooling only when the config and prompts match l-v2.0 exactly, and refuses otherwise. Call `BertEmbedder::load_with_pooling` to set it yourself.
- Quantization costs accuracy: against full-precision l-v2.0 the 8-bit conversion keeps cosine 0.9995 and the 4-bit one 0.90. Do not mix vectors from different conversions in one index.
- `m-v2.0` and `m-long` were verified against the Python code those repositories ship. The Rust ports run none of it.

## SigLIP2 NaFlex

`Siglip2::load(dir)` needs `config.json`, `preprocessor_config.json`, `model.safetensors`, and `tokenizer.json`.

- `embed_image(&Rgb8Image)` or `embed_image_file(path)` (PNG or JPEG) resizes to at most 256 patches at the image's own aspect ratio, using the same antialiased bilinear resize as the `transformers` fast processor.
- `embed_text(text)` pads to 64 tokens. Text is not lowercased, matching the reference.
- `logit(image, text)` is `exp(scale) * dot + bias`. Pass it through `sigmoid` for an independent per-pair probability. Scores across captions do not sum to one.

## CLAP

`Clap::load(dir)` needs `config.json`, `preprocessor_config.json`, `tokenizer.json`, and `model.safetensors`.

- `embed_audio(pcm)` takes mono f32 at 48 kHz. Clips under 10 s are tiled to 10 s as the reference does. Longer clips keep the first 10 s, and `embed_audio_with(pcm, LongAudioCrop::Center)` or `LongAudioCrop::Offset(n)` picks another window. The reference picks a random window, so it is not reproducible there.
- `embed_audio_file(path)` reads a WAV, mixes to mono, and resamples to 48 kHz with the workspace resampler, which is not the one a Python pipeline would use.
- `classify(audio, labels)` returns `(label, probability)` pairs sorted best first, a softmax over the labels.

## Shared helpers

`similarity` has `cosine_similarity`, `similarity_matrix`, and the ColVision scorers `score_single_vector` and `score_multi_vector` (MaxSim). `pooling` and `st_config` implement `sentence-transformers` pooling and module configs for any checkpoint you wire up yourself. `normalize::truncate_and_normalize` does Matryoshka truncation.
