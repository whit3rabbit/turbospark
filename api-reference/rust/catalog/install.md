---
description: "Install a catalog row or a probed repository as a verified .gturbo directory, record it in the store, and handle GGUF shard sets."
icon: box
---

# Install a model

An install turns a plan into a `.gturbo` directory. The flow has four steps: build a plan, gate it, run the install walk, and record the result. The walk fetches the tokenizer first, streams the weights, and then reads the finished directory back through the loaders the runtime uses. Use this page when your tool downloads models. To check only whether a model would run, go to [probe and recommend](probe-and-recommend.md) instead.

{% hint style="warning" %}
`install` and `install_with_byte_progress` do not run the gate. Call `gate` first, as the CLI does. Without it, the walk skips the probe's refusals and starts streaming. `install_variant` runs the gate itself, so it needs no separate call.
{% endhint %}

## The four steps

1. **Build a plan.** `InstallPlan::from_entry(entry)` takes a catalog row. `InstallPlan::from_probe(alias, &report, sidecars)` takes a repository you probed.
2. **Gate the plan.** `gate(&client, &plan, force, &mut progress)` probes the repository and returns its `ProbeReport`. A refusal returns `Err` with the reason and a hint to pass `force`. Pass `force = true` to install anyway.
3. **Install.** `install_with_byte_progress(&plan, &dir, &client, progress, byte_progress, cancel)` returns an `Installed`. `install` is the same walk with no byte callback and no cancel flag.
4. **Record.** `record(&store, &installed)` adds the install to the store's record.

The walk runs sidecars first on purpose. The source notes that a sidecar is a few megabytes and that the tokenizer either loads or does not within moments. A missing tokenizer file therefore fails before the weights start streaming, not after a long transfer.

## Install a catalog row

This program installs the TinyLlama row. It compiles against the crate. I did not run it, because the install downloads about 0.9 GB.

```rust
use turbospark_catalog::{gate, install_with_byte_progress, record, Catalog, Client, InstallPlan, Store};

fn main() -> Result<(), String> {
    let store = Store::default_store()?;
    let catalog = Catalog::load(store.root())?;
    let entry = catalog.get("tinyllama").ok_or("no tinyllama row in the catalog")?;

    let plan = InstallPlan::from_entry(entry);
    let client = Client::new();
    let dir = store.install_path(&plan.alias);
    let mut progress = |stage: &str| eprintln!("[pull] {stage}");

    // Gate first: a refusal costs one header read, not a stream.
    gate(&client, &plan, false, &mut progress)?;
    let installed = install_with_byte_progress(&plan, &dir, &client, &mut progress, None, None)?;
    record(&store, &installed)?;

    println!("installed {} at {}", installed.model.alias, installed.model.path.display());
    Ok(())
}
```

`Store::default_store()` reads `TURBOSPARK_HOME`, then `HOME`. `Store::install_path(alias)` returns `<root>/models/text/<alias>.gturbo`. The walk creates the directory when it is missing, but it does not check whether the directory already holds an install. The CLI checks for a `manifest.json` in the target and refuses. Make the same check in your tool, or you can overwrite an existing install.

## Install a probed repository

A repository that is not in the catalog takes the same steps with a different plan. Probe it, build the plan from the report, and then gate that plan.

- `InstallPlan::from_probe(alias, &report, sidecars)` takes its sidecar files from what the repository actually has, not from a list copied from a sibling checkpoint.
- Its `install_bytes` is the probed download size, because no catalog estimate exists.
- Its `status` is `unlisted`, and the install record keeps that value.

The plan fields are public, so you can read them before you install. The vision-tower and multi-token-prediction paths (`InstallPlan::for_vision_tower`, `mtp`, and `reuse_trunk_from`) are outside this page.

## Progress and cancel

- The progress callback is `FnMut(&str)`. It receives stage lines from the install driver and from the weight walk, so one callback covers the whole run.
- `ByteProgressCallback` is `Arc<dyn Fn(u64) + Send + Sync>`. Pass `Some(callback)` to receive byte counts as the transfer advances.
- `CancelFlag::new()` creates a flag, and `cancel()` sets it from any thread. Pass `Some(&flag)` to the walk. The walk checks the flag at each step and inside each ranged read, so a cancel takes effect within one chunk.
- A cancelled install returns `Err(INSTALL_CANCELLED)`. Compare against the constant, because the text may change.

## The result

`Installed` has two fields:

- `model`, an `InstalledModel` for the store record. It carries `alias`, `repo`, `revision`, `path` (absolute), `family`, `install_bytes` (measured on disk after the walk), `installed_on` (the install date as a string), `status`, `kind`, and `modality`. The record has no quantization field.
- `arch`, the `ArchConfig` the walk read back. It comes from `turbospark_model_io`.

`record` replaces any row with the same alias and modality. It also drops rows whose directories no longer exist. `Store::installed()` returns the text installs whose directories still exist, keyed by alias.

## GGUF shard sets

A large GGUF may be split across several files. Each file declares the shard count in its header, and the first shard declares its position as zero. Name shard 1 when you probe or install. The probe reads that header, finds the other shards, and totals their bytes. Naming any other shard is an error, and so is a shard count outside 1 to 1024.

This program probes the weights of a bundled sharded row by naming shard 1. It makes live requests. It compiles against the crate. The output below is from an earlier run and was not re-run for this audit.

```rust
use turbospark_catalog::{probe, Client, RepoRef, Verdict};

fn main() -> Result<(), String> {
    let client = Client::new();
    let repo = RepoRef::parse(
        "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF@b22d729eae29b5796f76fb70f91aef549b9fc52c",
    )?;
    // Name shard 1. Its header declares the set, and the probe reads the rest.
    let shard_one = "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf";
    let report = probe(&client, &repo, Some(shard_one), None)?;

    println!("file: {:?}", report.file);
    println!("download bytes: {:?}", report.download_bytes);
    match &report.verdict {
        Verdict::Runnable => println!("verdict: runnable"),
        Verdict::Refused(why) => println!("verdict: refused: {why}"),
    }
    Ok(())
}
```

Output from that run:

```text
file: Some("Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf")
download bytes: Some(68152167168)
verdict: refused: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF@b22d729eae29b5796f76fb70f91aef549b9fc52c has no tokenizer.json. ...
```

The byte total covers both shards, and it matches the `download_bytes` the catalog records for this row. The refusal is expected here, because this probe passed no sidecar repository. Pass one as `sidecar_repo` to clear it. The bundled row's sidecar repository currently fails the id check described under Errors, so update that row before you rely on it.

To list the variants in a repository, use `gguf_variants`. Each `.gguf` file appears with its quantization label, its size, and whether the crate has kernels for that label. Shard parts are counted in `sharded_skipped` and not listed, so a shard set never appears as a choice. Name its first shard to install it.

```rust
use turbospark_catalog::{gguf_variants, Client, RepoRef};

fn main() -> Result<(), String> {
    let client = Client::new();
    let repo = RepoRef::parse("Qwen/Qwen3-30B-A3B-GGUF")?;
    let found = gguf_variants(&client, &repo)?;

    for variant in &found.variants {
        let label = variant.quant_label.as_deref().unwrap_or("unlabeled");
        println!("{:<10} executable={:<5} {:?} {}", label, variant.executable, variant.bytes, variant.file);
    }
    println!("shard parts skipped: {}", found.sharded_skipped);
    Ok(())
}
```

Output from an earlier run (not re-run for this audit):

```text
Q8_0       executable=true  Some(32483931648) Qwen3-30B-A3B-Q8_0.gguf
Q6_K       executable=true  Some(25092531712) Qwen3-30B-A3B-Q6_K.gguf
Q5_K       executable=true  Some(21725580800) Qwen3-30B-A3B-Q5_K_M.gguf
Q4_K       executable=true  Some(18556685824) Qwen3-30B-A3B-Q4_K_M.gguf
unlabeled  executable=false Some(21080509952) Qwen3-30B-A3B-Q5_0.gguf
shard parts skipped: 0
```

The label is read from the filename, not the header. A name the crate does not cover, such as `Q5_0` above, appears as unlabeled, and the crate treats it as not executable. The Swift repository at the revision used in the shard example reports `shard parts skipped: 6`. It lists only two unlabeled auxiliary files, so none of its shard sets appears as a choice. `group_variants` returns the logical groups and a `ShardSetStatus` (`SingleFile`, `Complete`, `Incomplete`, or `Inconsistent`) if you need to explain a set to a user.

### Install a pinned variant

`VariantInstallPlan::from_hub_variant` binds one variant to a probe, and `install_variant` installs it. A variant must meet all of these conditions before the plan is built:

- The variant's installability is `Ready`.
- Its revision is a 40-character commit, not a floating name.
- The probe report is for exactly the same repository, is a runnable GGUF, and names a file in the variant.
- The variant is one recognized file group. A sharded group must be complete.
- The variant's total size equals the sum of its file sizes.

`HubClient::repo_variants` lists the `HubGgufVariant` values for a repository. `HubClient::new` takes the client, a catalog, a machine, a context, and a slot policy. This program compiles against the crate. I did not run it, because the Q4_K file it selects is about 18.6 GB.

```rust
use turbospark_catalog::{
    install_variant, probe, record, Catalog, Client, HubClient, Machine, RepoRef, Store,
    VariantInstallPlan, VariantInstallability,
};
use turbospark_model_io::{ExpertCacheSlots, LoadGuard};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<(), String> {
    let store = Store::default_store()?;
    let catalog = Catalog::load(store.root())?;
    let row = catalog.get("qwen3moe").ok_or("no qwen3moe row in the catalog")?;
    let sidecars = RepoRef::new(row.sidecar_repo(), row.sidecar_revision());

    let client = Client::new();
    let machine = Machine {
        physical_bytes: 36 * GIB,
        load_guard: LoadGuard::Relaxed,
        ..Machine::default()
    };
    let hub = HubClient::new(&client, catalog, machine, 8192, ExpertCacheSlots::Auto);

    // A variant install needs an immutable revision (a 40-character commit).
    let repo = RepoRef::parse("Qwen/Qwen3-30B-A3B-GGUF@e4d4bafdfb96a411a163846265362aceb0b9c63a")?;
    let variants = hub.repo_variants(&repo).map_err(|e| e.to_string())?;
    let variant = variants
        .iter()
        .find(|v| v.label.0 == "Q4_K" && v.installability == VariantInstallability::Ready)
        .ok_or("no ready Q4_K variant at this revision")?;

    // Probe the first file of the set. The probe reads the shard topology from its header.
    let file = variant
        .files
        .iter()
        .map(|f| f.name.as_str())
        .find(|name| !name.contains("-of-") || name.contains("-00001-of-"))
        .ok_or("variant has no first file")?
        .to_string();
    let report = probe(&client, &variant.repo, Some(&file), Some(&sidecars))?;

    let alias = "qwen3moe-q4km";
    let plan = VariantInstallPlan::from_hub_variant(alias, variant, &report, sidecars)?;
    let dir = store.install_path(alias);
    let mut progress = |stage: &str| eprintln!("[pull] {stage}");

    // install_variant runs the same gate as `gate`, so no separate call is needed.
    let installed = install_variant(&plan, &dir, &client, &mut progress)?;
    record(&store, &installed)?;
    println!("installed {alias} at {}", installed.model.path.display());
    Ok(())
}
```

The revision in that example is a commit that existed when this page was written. Replace it with a current commit from the repository before you run the program.

## Errors

- A refusal from `gate` reads `<repo> would not run here: <reason>`, followed by a line that says to re-run with `force`.
- A cancelled install returns `INSTALL_CANCELLED`.
- `from_hub_variant` returns a specific reason for each unmet condition, such as a non-immutable revision or a total size that does not match.
- A repository that Hugging Face renamed fails the crate's id check (`repository_id_matches_request`). The crate refuses the redirect on purpose. Update the catalog row to the current name, or pass the current name as the sidecar repository.
- A gated repository without a token fails with an error that names the missing credential. The [overview](README.md) lists the token lookup order.

## Other install surfaces

Audio, speech, image, and music models have their own install functions in the same crate, such as `install_music` and `install_speech_receipt`. They follow their own catalogs and receipts, so this page does not cover them.
