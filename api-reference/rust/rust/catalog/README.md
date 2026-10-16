---
description: "Curated model catalog, header-only Hugging Face probe, hardware-aware recommendations, and the install driver in turbospark-catalog."
icon: box
---

# turbospark-catalog

`turbospark-catalog` lists the models TurboSpark has run, probes any Hugging Face repository by reading its headers, ranks models against a machine's memory, and installs a checkpoint as a `.gturbo` directory. Reach for it when your tool needs to show, check, choose, or install TurboSpark models without shelling out to the `turbospark-model` command. The crate does not load weights, decode tokens, or allocate Metal buffers.

The models it installs run only on macOS with Apple Silicon. The crate contains no `unsafe` code (`#![forbid(unsafe_code)]`). Generation lives in [`turbospark-runtime`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/).

## Pages

| Page | Read it when |
| --- | --- |
| [Probe and recommend](probe-and-recommend.md) | You need to know whether a repository will run here, or which models fit a machine. |
| [Install](install.md) | You need to download, verify, and record a model, including multi-file GGUF sets. |

The full API is on [docs.rs](https://docs.rs/turbospark-catalog/latest/turbospark_catalog/).

## Install the crate

The package is `turbospark-catalog`, and the Rust import path is `turbospark_catalog`. Inside the TurboSpark workspace, the crate is imported under the short alias `catalog`. That alias is workspace-internal, so use the package name in your own project. Add the package from crates.io once the version you need is published, or use a path or git dependency from a checkout.

```toml
[dependencies]
turbospark-catalog = { path = "/path/to/turbospark/crates/catalog" }
# Needed for the LoadGuard and ExpertCacheSlots arguments of the recommendation API.
turbospark-model-io = { path = "/path/to/turbospark/crates/model-io" }
```

The recommendation functions take `LoadGuard` and `ExpertCacheSlots` from `turbospark_model_io`. If you never call them, you can leave that dependency out.

## Quick start

This program lists every row in the embedded table. It runs with no network access.

```rust
use turbospark_catalog::Catalog;

fn main() -> Result<(), String> {
    let catalog = Catalog::embedded()?;
    for entry in catalog.entries() {
        println!("{:<24} {:<10} {}", entry.alias, entry.status.as_str(), entry.name);
    }
    Ok(())
}
```

The embedded table is small on purpose. A row is added only after its checkpoint was streamed and generated on real hardware. [`Status`](https://docs.rs/turbospark-catalog/latest/turbospark_catalog/enum.Status.html) records how far that went: `verified` means a frozen benchmark row exists, `runs` means the model installs and produces coherent text, with no frozen row, and `caveat` means it runs but a note in the row disqualifies it for ordinary use. Anything the table does not list can still be probed.

## Local rows

`Catalog::load(home)` merges `models.json` from the directory `home` over the embedded table, matching by alias. A local row can add a model or repoint an existing alias without a rebuild. `Catalog::is_user_row(alias)` tells you which rows came from that file. Embedded and local rows pass the same validation, and a file with a `schema_version` other than `1` is refused.

## Store layout

The default store root is `$TURBOSPARK_HOME` when that variable is set, and `$HOME/.turbospark` otherwise. Text installs land at `models/text/<alias>.gturbo` under that root, and the record of installs is `installed.json` at the root. `Store::new(root)` uses an explicit root, which is useful in tests and in tools that keep models somewhere else.

## Network and credentials

Probe and install calls use a blocking HTTP client, so call them from a thread that may block. In an async program, run them through your runtime's blocking-task facility. Recommendations from the catalog table make no network calls.

Requests go to the Hugging Face endpoint. `$HF_ENDPOINT` sets it, the default is `https://huggingface.co`, and `set_hf_endpoint_override` changes it inside one process. `Client::new()` looks for a token in this order: `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`, the store's `hf_token` file, and then the standard Hugging Face cache file. `Client::with_token(Some(token))` checks an explicit token before those. A gated repository without a token fails with an error that names the missing credential.

## Errors

Most functions return `Result<T, String>`. The strings are for people to read, so do not parse them. Where a caller needs to recognize an outcome, the crate exports a constant for it, such as `INSTALL_CANCELLED`. `HubClient`, the live Hub search surface, returns its own `HubError`, which these pages do not cover.

A model that would not run is not an error. `probe` returns `Ok` with a report whose `verdict` is `Verdict::Refused`. Check the verdict as well as the `Result`.

## Where to go next

- [Probe and recommend](probe-and-recommend.md) covers the header gates, the probe report, and the fit arithmetic.
- [Install](install.md) covers the gate, the install walk, recording, cancellation, and GGUF shard sets.
- The catalog contract, including how a row earns its place, is in [docs/MODELS.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/MODELS.md).
- The on-disk format is in [docs/GTURBO.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/GTURBO.md).
- The `turbospark-model` command line tool uses this crate for its `list`, `probe`, `recommend`, and `pull` commands, so you can compare its output with yours.
