---
description: "The turbospark-catalog crate: the curated model table, the header-only Hugging Face probe, and the install driver that turns either into a .gturbo directory."
icon: database
---

# catalog

`crates/catalog` (package `turbospark-catalog`) is the catalog boundary:
it turns published artifacts into validated local installs, aliases, and
model-store entries. The crate is safe Rust (`#![forbid(unsafe_code)]`) and
portable: nothing here decodes, dispatches, or allocates a GPU buffer, so it
builds on every platform even though the models it installs only run on
macOS. Its verdicts are load-bearing. A catalog row controls what users can
download, open, and measure, so a convenient approximation is still a broken
row.

## What it does

The crate answers three questions, stated in `src/lib.rs`:

| Layer | Question | Where |
| --- | --- | --- |
| `Catalog` | What has been run here: the curated table, one row per repo and revision streamed on real hardware | `src/catalog.rs`, `src/entry.rs` |
| `probe` | What could be run here: architecture, block types, expert stride, tokenizer sidecars, from headers only (KB and seconds, never a download) | `src/probe/` |
| `install` | The walk: plan, gate, stream weights, write the `.gturbo`, record it | `src/install.rs`, `src/stream.rs` |
| `Store` | Where installs live and how a name resolves to one | `src/store.rs` |

Around that core sit the per-modality tables, each an embedded JSON database
read with `include_str!`: `models.json` (text trunks), `image_models.json`,
`music_models.json`, `speech_models.json`, `embedding_models.json`,
`ocr_models.json`, and, behind `src/audio_catalog.rs`, `audio_assets.json`
and `audio_families.json` (pinned audio profiles, fingerprints, and
provenance). `Catalog::load` merges an optional `$TURBOSPARK_HOME/models.json`
user override over the embedded table by alias and refuses a schema version
it does not recognize rather than best-effort parsing it.

## Where it sits

`turbospark-catalog` depends on `core`, `model-io` (sizing policies and the
`LoadGuard` the fit arithmetic shares with `open()`), `repack` (the actual
weight streaming), `tokenizer` (sidecar verification), `audio` (speech
descriptors), and `coreml` (Core ML profile checks). It is consumed by
`cli` (the `turbospark-model` surface), `server` (alias resolution for
`--model`, macOS only there), `ffi`, and `bench`.

## Public API tour

Root re-exports from `crates/catalog/src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `Catalog` | `src/catalog.rs` | `embedded()`, `load(home)`, `get(alias)`, `entries()`, `find(needle)`; `SCHEMA_VERSION` guards the JSON shape |
| `CatalogEntry`, `Source`, `Sidecars`, `Status`, `Measured`, `EntryKind` | `src/entry.rs` | One row: repo, revision, sidecar repo, byte counts, gate targets, per-chip measurements; `validate()` enforces the structural rules at load |
| `probe(client, repo, want_file, sidecar_repo)` | `src/probe/mod.rs` | The header-only verdict; `evaluate_gguf` / `evaluate_config` are the offline gates each format's rules live in |
| `ProbeReport`, `Verdict`, `TypeShare` | `src/probe/types.rs` | What a probe found and whether the checkpoint would run; the first refusal wins |
| `Client`, `RepoRef` | `src/hf.rs` | The Hugging Face client (KB-scale endpoints only) and `owner/name[@rev]` parsing |
| `Store`, `InstalledModel`, `ModelModality` | `src/store.rs` | The `~/.turbospark` layout, per-modality namespaces, install records, and alias resolution |
| `resolve_model_arg` | `src/store.rs` | The one resolution both `turbospark-check` and `turbospark-server` call |
| `InstallPlan`, `gate`, `install`, `install_with_byte_progress`, `install_variant`, `record`, `Installed` | `src/install.rs` | The install driver; `INSTALL_CANCELLED` is the constant a cancel flag produces |
| `recommend_catalog`, `fit`, `Machine`, `Recommendation`, `Fit` | `src/recommend/` | What this machine should run, ranked, under a named load guard |
| `resolve_hf_token`, `validate_hf_token` | `src/auth.rs` | Token resolution (explicit, env, store, `hf` cache) and `whoami-v2` validation |
| `ImageCatalog`, `AudioCatalog`, `MusicCatalogEntry`, `SpeechCatalogEntry`, `OcrCatalog`, `EmbeddingCatalog` | per-modality modules | The embedded tables and install flows for the non-text modalities |
| `HubClient`, `HubQuery`, `HubEntry` | `src/hub.rs` | Search and variant selection over Hugging Face, filtered through the same probe |

## Examples

### Resolve a `--model` argument the way the CLIs do

`resolve_model_arg` is shared by `turbospark-check` and `turbospark-server`,
so an install answers to one name in both. Adapted from
`crates/catalog/src/store.rs` and its `tests/store.rs`:

```rust
use turbospark_catalog::{resolve_model_arg, Store};

// A path stays a path; a bare alias resolves to its install directory.
let dir = resolve_model_arg("gemma4");

let store = Store::default_store()?;
for (alias, model) in store.installed() {
    println!("{alias}: {} ({})", model.path.display(), model.family);
}
```

An unresolvable name comes back unchanged, so the caller reports the usual
"no such install" even on a machine with no `HOME`.

### Probe, gate, install, record

The four-step flow `turbospark-model pull` runs, adapted from
`crates/cli/src/bin/model_cmd/mod.rs` (progress reporting simplified):

```rust
use turbospark_catalog::{gate, install_with_byte_progress, record, Catalog, Client, InstallPlan, Store};

let store = Store::default_store()?;
let catalog = Catalog::load(store.root())?;
let client = Client::new();
let entry = catalog.get("tinyllama").ok_or("no such alias")?;

let plan = InstallPlan::from_entry(entry);
let dir = store.install_path(&plan.alias);
let mut progress = |stage: &str| println!("[pull] {stage}");

// Sidecars are fetched and verified inside the gate and the walk.
let report = gate(&client, &plan, false, &mut progress)?;
match &report.verdict {
    verdict => println!("{verdict:?}"),
}

let installed = install_with_byte_progress(
    &plan,
    &dir,
    &client,
    &mut progress,
    None, // byte-progress callback; pass Some for a progress bar
    None, // cancel flag; the CLI passes None too
)?;
record(&store, &installed)?;
```

For an arbitrary repository rather than a curated row, build the plan with
`InstallPlan::from_probe` behind `probe()`'s verdict instead.

### Run the rot guard

The embedded table claims file lists, byte counts, and headers for
main-pinned artifacts. The ignored network test validates all three without
downloading any payload, and the crate's `AGENTS.md` says to run it after
changing a row:

```sh
cargo test -p turbospark-catalog
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture
```

## Gotchas

{% hint style="warning" %}
Sidecars are fetched, loaded, and exercised (a one-turn conversation is
rendered and encoded) BEFORE any weight byte moves. Every hand-rolled
repack test used to stream multi-GB weights first and fetch tokenizer files
after, which is how a 404 for `merges.txt` wasted a 20-minute stream on a
perfectly good install. Do not reorder this: the cheap failures happen
first by design.
{% endhint %}

{% hint style="warning" %}
`Store::resolve` prefers an existing directory over an alias, and that is
not a tie-break. A bare name that silently preferred an alias would run a
different model than the command line named, fluently and with no error.
An existing directory argument always wins, and an unresolvable name is
passed through unchanged. Modality namespaces (`models/text`, `models/image`,
`models/audio`) are resolved by separate functions on purpose; do not add a
modality to the text fallback order.
{% endhint %}

Two more rules worth knowing before editing a row: `download_bytes` is the
fingerprint for main-pinned artifacts, so a floating row's figures go stale
on re-upload (paste the rot guard's numbers back rather than widening its
tolerance), and `src/hf.rs` deliberately handles KB-scale endpoints only.
Multi-GB weights stream through `repack`'s ranged source, which owns the
chunking and retry ladder.

## Related pages

- [model-io](model-io.md)
- [repack](repack.md)
- [cli](cli.md)
- [Model Management CLI](../command-line-and-local-api/cli.md)
- [The model catalog](../models/models.md)
