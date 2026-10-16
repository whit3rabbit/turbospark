---
description: "Check whether a Hugging Face repository will run in TurboSpark, and rank models by fit for a machine, without downloading weights."
icon: box
---

# Probe and recommend

Probing answers whether a checkpoint can run here. It reads the repository's file list and the headers of its weights, which costs kilobytes and a few requests, not a download. Recommending answers which models fit a machine, using the machine's memory and the checkpoint's shape. Probe before you start a large install, and feed the probe report into [install](install.md) when you proceed.

## Probe a catalog row

`probe_entry` probes a curated row at its own repository, revision, file, and sidecar repository. You do not rebuild that tuple by hand. The program below probes the TinyLlama row. It makes live requests to Hugging Face.

```rust
use turbospark_catalog::{probe_entry, Catalog, Client, Verdict};

fn main() -> Result<(), String> {
    let catalog = Catalog::embedded()?;
    let entry = catalog.get("tinyllama").ok_or("no tinyllama row in the catalog")?;

    let client = Client::new();
    let report = probe_entry(&client, entry)?;

    match &report.verdict {
        Verdict::Runnable => println!("{}: runnable", report.repo),
        Verdict::Refused(why) => println!("{}: refused: {why}", report.repo),
    }
    if let Some(bytes) = report.download_bytes {
        println!("download bytes: {bytes}");
    }
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    Ok(())
}
```

Output from an earlier run (not re-run for this audit):

```text
TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF@main: runnable
download bytes: 904385920
```

## Probe any repository

`probe` takes a repository, an optional file name, and an optional sidecar repository.

```rust
pub fn probe(
    client: &Client,
    repo: &RepoRef,
    want_file: Option<&str>,
    sidecar_repo: Option<&RepoRef>,
) -> Result<ProbeReport, String>
```

`RepoRef::parse("owner/name@revision")` builds the repository argument. A revision is optional and defaults to `main`. Pin a commit for anything you plan to install, because `main` can change between your probe and your install.

Three rules decide the rest of the call:

- A repository with exactly one `.gguf` file, or with safetensors files, needs no file name. A repository with several `.gguf` files is an error until you name one with `want_file`. The error lists the files on offer.
- A GGUF file carries llama.cpp's tokenizer, not a Hugging Face `tokenizer.json`. Pass `sidecar_repo` for a GGUF repository that has no tokenizer of its own. Without it, the probe refuses for the missing tokenizer.
- `sidecar_repo` is a different repository from the weights for every bundled GGUF row, and its revision is its own.

The refusal for a GGUF without a sidecar repository reads like this. It comes from a probe of a bundled sharded row's weights repository, with `sidecar_repo` left as `None`:

```text
verdict: refused: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF@b22d729eae29b5796f76fb70f91aef549b9fc52c has no tokenizer.json. This port loads an HF tokenizer, so a checkpoint without one needs --sidecar-repo pointing at the checkpoint this artifact was converted from.
```

The message says `--sidecar-repo` because it is written for the CLI. In Rust, the argument is `sidecar_repo`.

## The report

`ProbeReport` is the answer. The fields you will read most often are these.

| Field | Meaning |
| --- | --- |
| `verdict` | `Verdict::Runnable`, or `Verdict::Refused(reason)` with the first reason that failed. |
| `family` | The decode family, as `model_io::ModelFamily`, when the architecture has one. |
| `arch` | The checkpoint's `ArchConfig`, when a header declared enough to build one. |
| `download_bytes` | Total weight bytes on the wire, summed across a shard set. |
| `types` | GGUF block types with their tensor counts and byte shares. `bytes` is `None` for a type the crate cannot size. |
| `expert_stride` | Bytes of one routed expert in one layer, for a mixture-of-experts model. `None` for a dense model. |
| `trained_context` | The window the checkpoint declares it was trained at. `None` means the file declares none, not zero. |
| `sidecars_present`, `sidecars_missing` | Tokenizer and chat-template files the repository has, and the ones it lacks. |
| `warnings` | Advisory lines that do not block an install, such as expert-cache arithmetic. |

`ProbeReport::slot_cache_bytes()` returns the pinned-slot working set for 8, 16, and 32 slots, for mixture-of-experts models only.

The probe runs four gates, cheapest first, and keeps the first refusal:

1. **File list.** One request decides GGUF or safetensors, and reads which tokenizer sidecars exist.
2. **Architecture.** The family must have a decode flow in this port. A recognized architecture without one returns the registry's own description of what it would need.
3. **Quantization.** Every block type must have kernels here. F32, F16, and BF16 are transcoded when the install is written, so they pass.
4. **Expert granularity.** Reported for mixture-of-experts models and never a refusal. Whether a model fits is a memory question, which the recommendation layer answers.

A refusal stops an install. A warning does not.

## Recommend for a machine

`Machine` describes the memory and chip the fit is for. It derives `Default`, so set only the fields you have.

| Field | Meaning |
| --- | --- |
| `physical_bytes` | Installed memory. Every budget comes from this number. |
| `working_set_bytes` | What the Metal device reports it will hold. Advisory only, and `None` off macOS. |
| `load_guard` | The memory tier. `LoadGuard::Relaxed` is the default and matches the frozen rows. Use the same tier the loader will open with, or the recommendation can promise a fit the loader refuses. |
| `chip` | The chip brand string. A measured row applies only when this string contains the row's chip string. |

The offline call ranks the embedded table. It makes no network requests, and the program below runs without a network. The measured row for a chip gives an exact answer, and every other row reports `unknown`.

```rust
use turbospark_catalog::{recommend_catalog, Catalog, CatalogEntry, Machine};
use turbospark_model_io::{ExpertCacheSlots, LoadGuard};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<(), String> {
    let machine = Machine {
        physical_bytes: 36 * GIB,
        working_set_bytes: None,
        load_guard: LoadGuard::Relaxed,
        // Must match a `measured` row's chip string to use its peak.
        chip: "Apple M4 Max".to_string(),
    };

    let catalog = Catalog::embedded()?;
    let entries: Vec<&CatalogEntry> = catalog.entries().collect();
    let rows = recommend_catalog(&entries, &machine, 8192, ExpertCacheSlots::Auto);

    for rec in rows.iter().take(12) {
        println!(
            "{:<46} {:<28} {:?} install: {}",
            rec.name,
            rec.fit.verdict.as_str(),
            rec.fit.counted_source,
            rec.origin.install_target()
        );
    }
    Ok(())
}
```

Output from a run on a 36 GiB machine with the chip string above, shortened to four rows:

```text
Qwen2.5 7B Instruct (MLX INT4, group 64)       fits, fully resident         Measured install: qwen25-7b-4bit
gpt-oss 20B (GGUF MXFP4)                       fits, fully resident         Measured install: gptoss-20b
Qwen2.5 7B Instruct (GGUF Q3_K_M, official)    fits, fully resident         Measured install: qwen25-7b-q3km
Muse Glimmer 30B (MLX INT4, group 64)          fits, fully resident         Measured install: museglimmer
```

Rows report `unknown (probe it)` until something has read their header. The crate does not fill that gap from the architecture name. One architecture string covers checkpoints of different shapes, so a guess could describe the wrong model. Probe the row to get a real answer.

The ranking puts rows that fit first, then rows with more evidence, then the measured decode rate. Read `fit` and `evidence` on each row before you act on its position.

A measured peak applies only when two things are true. The row's chip string appears in `Machine::chip`, and the measurement was taken at the context you asked about. The embedded rows were measured at 2048, 4096, or 8192 tokens, so ask for the window a row was measured at to get its exact answer.

## Probe the whole table

`recommend_catalog_probed` reads the header of every model row before it ranks them. It runs four probes at a time, and it sends requests to Hugging Face. A row whose probe fails stays in the list as `unknown`, with the failure in `notes`. The call returns an error only when no header could be read at all.

`recommend_catalog_probed_with_progress` takes the same arguments plus a callback. The callback receives `(done, total)` on the calling thread after each batch of probes.

```rust
use turbospark_catalog::{recommend_catalog_probed_with_progress, Catalog, CatalogEntry, Client, Machine};
use turbospark_model_io::{ExpertCacheSlots, LoadGuard};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<(), String> {
    let machine = Machine {
        physical_bytes: 36 * GIB,
        load_guard: LoadGuard::Relaxed,
        chip: "Apple M4 Max".to_string(),
        ..Machine::default()
    };
    let catalog = Catalog::embedded()?;
    let entries: Vec<&CatalogEntry> = catalog.entries().collect();
    let client = Client::new();

    let rows = recommend_catalog_probed_with_progress(
        &entries,
        &client,
        &machine,
        8192,
        ExpertCacheSlots::Auto,
        |done, total| eprintln!("probed {done}/{total}"),
    )?;
    for rec in rows.iter().take(5) {
        println!("{:<46} {}", rec.name, rec.fit.verdict.as_str());
    }
    Ok(())
}
```

This program compiles against the crate. I did not run it, because it makes one request per model row.

## Discover popular repositories

`discover` lists popular GGUF repositories on Hugging Face, probes each one, and returns candidates in the same `Recommendation` shape as the catalog. Options come from `DiscoverOptions`, which has these fields and defaults:

- `scan`: how many repositories to consider. Default 20.
- `context`: the window to fit against. Default 4096.
- `slots`: the expert-cache policy. Default `ExpertCacheSlots::Auto`.
- `concurrency`: how many probes run at once. Default 8, clamped to between 1 and 16.

Every discovered row carries `Evidence::Discovered`, which sorts below every curated row that fits. Merge the two lists and call `rank_recommendations` once, rather than ranking each list and interleaving the results.

```rust
use turbospark_catalog::{discover, rank_recommendations, Client, DiscoverOptions, Machine};
use turbospark_model_io::LoadGuard;

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<(), String> {
    let machine = Machine {
        physical_bytes: 36 * GIB,
        load_guard: LoadGuard::Relaxed,
        chip: "Apple M4 Max".to_string(),
        ..Machine::default()
    };
    let options = DiscoverOptions { scan: 10, ..DiscoverOptions::default() };
    let mut rows = discover(&Client::new(), &machine, &options)?;
    rank_recommendations(&mut rows);
    for rec in rows.iter().take(5) {
        println!("{:<46} {}", rec.name, rec.fit.verdict.as_str());
    }
    Ok(())
}
```

This program compiles against the crate. I did not run it, because it makes network requests for every repository it scans.

## Fit verdicts

`Recommendation::fit` carries a `Fit` with the arithmetic and the verdict. The verdict is one of these values.

| `FitVerdict` | Meaning |
| --- | --- |
| `Resident` | Everything fits, including the install. |
| `Streams` | The engine's allocations fit. Expert weights stream from disk. This is the normal case for a streamed mixture-of-experts model, and it counts as a fit. |
| `Tight` | Allocations fit with under 10 percent to spare under the relaxed guard. The model runs, and other work on the machine competes with it. |
| `Refused` | The engine's allocations do not fit. The open fails rather than running slowly. |
| `Unknown` | Not enough is known to say. This is never a fit. |

`Fit::counted_source` says where the memory number came from. `Measured` means a frozen row for this chip at this context. `Estimated` means the crate computed it from the checkpoint's shape. `Unknown` means nothing has read the header, so the slot cache and KV terms are missing. The value is not zero.

`Fit::largest_context` is the largest window this machine affords. `context_ladder` prices a set of windows for one shape, and its rows are not linear. Do not multiply one window's cost by a factor to estimate another, because sliding-window layers stop growing at their cap.

## Errors

- `probe` returns `Err` for a request that cannot be made or read, and for a repository that the probe cannot choose a file from. A repository with no GGUF or safetensors file returns an error that says so.
- `probe_entry` returns `Err` when Hugging Face's response fails the crate's validation. One case is a sidecar repository that has been renamed. The response reports the new id, and the crate refuses the mismatch on purpose. Update the row to the current name, or pass the current name as `sidecar_repo`.
- A model that would not run is `Ok` with `Verdict::Refused`. Check the verdict.
