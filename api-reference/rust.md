---
description: "Use TurboSpark Rust crates to inspect models and integrate generation."
icon: terminal
---

# Rust library

TurboSpark is a set of Rust crates. The model catalog and file tools can be used separately from inference. Real Metal inference requires macOS on Apple Silicon.

## Useful crates

| Crate | Use it for | Reference |
| --- | --- | --- |
| `turbospark-catalog` | List aliases, recommend models, probe checkpoints, and install models. | [docs.rs](https://docs.rs/turbospark-catalog/latest/turbospark_catalog/) |
| `turbospark-model-io` | Read and validate TurboSpark installs, architecture config, and resident indexes. | [docs.rs](https://docs.rs/turbospark-model-io/latest/turbospark_model_io/) |
| `turbospark-tokenizer` | Load tokenizer files, render/tokenize prompts, and decode output. | [docs.rs](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/) |
| `turbospark-selection` | Sampling and token-selection primitives. | [docs.rs](https://docs.rs/turbospark-selection/latest/turbospark_selection/) |
| `turbospark-runtime` | Prefill, decode, and the real model runner. | [docs.rs](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/) |
| `turbospark-server` | Embed the Axum-compatible local HTTP routes. | [source](https://github.com/whit3rabbit/turbospark/tree/main/crates/server) |

The lower-level `turbospark-core`, `turbospark-compute`, `turbospark-streaming`, `turbospark-invocation`, `turbospark-window-fit`, `turbospark-repack`, `turbospark-vision-io`, and `turbospark-image` crates support specialized integrations. Browse the [workspace](https://github.com/whit3rabbit/turbospark/tree/main/crates) when you need those boundaries.

## Read the catalog

Add the catalog crate:

```sh
cargo add turbospark-catalog
```

List the embedded aliases:

```rust
use turbospark_catalog::Catalog;

fn main() -> Result<(), String> {
    let catalog = Catalog::embedded()?;
    for entry in catalog.entries() {
        println!("{}: {}", entry.alias, entry.name);
    }
    Ok(())
}
```

Use `Catalog::load(&home)` to merge a local `models.json` override. `probe` checks a checkpoint's metadata and headers without downloading its weights. `recommend_catalog` ranks catalog entries for a machine and context; `install` performs the selected install plan. See the [catalog API](https://docs.rs/turbospark-catalog/latest/turbospark_catalog/) for the full argument and result types.

## Call the generation loop

`run_raw_completion` is for an integration that already has a `LogitProducer`, a matching `MfTokenizer`, tokenized prompt IDs, a `GenerationConfig`, the context limit, and vocabulary size:

```sh
cargo add turbospark-runtime turbospark-tokenizer
```

```rust
use turbospark_runtime::{run_raw_completion, RawDecodeProgress};

let result = run_raw_completion(
    &mut producer,
    &tokenizer,
    &prompt_ids,
    &config,
    max_context,
    vocab_size,
    |event| {
        if let RawDecodeProgress::Token { delta, .. } = event {
            print!("{delta}");
        }
    },
)?;
```

The returned `RawDecodeResult` includes token counts, stop reason, timing, and KV state. `TokenId` is signed 32-bit. Get IDs from the tokenizer for the same install; do not hard-code IDs from a fixture or another model.

This is not a `generate(prompt) -> String` wrapper. `RealForwardRunner` provides the model-backed producer on macOS, while custom producers can implement `LogitProducer`. Check the [runtime reference](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/) for runner setup and the [tokenizer reference](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/) for tokenizer APIs.

## Embed the HTTP server

The `turbospark-server` crate exposes `build_router`, `build_router_with_options`, and `ServerState`. Use these when embedding the HTTP routes in an existing Axum service. For a separate process, use `turbospark-server` from the [CLI guide](https://app.gitbook.com/s/7snUJsZfxrqIt5S95nxb/cli-and-api). The default listener is loopback; the server does not provide TLS.
