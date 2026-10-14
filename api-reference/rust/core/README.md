---
description: "Shared types, errors, runtime configuration, and steering modes in the turbospark-core crate."
icon: box
---

# turbospark-core

`turbospark-core` is the shared vocabulary for TurboSpark's Rust crates. Reach for it when you need the token ID type, the allowed values for a runtime knob, the logit view type, or the steering mode enum. It has no platform-specific dependencies and does not run inference by itself.

The package name is `turbospark-core`, and the import path is `turbospark_core`. Most sibling crates inside the TurboSpark workspace import it under the alias `foundation`, but `turbospark-audio` uses the package name. The alias is internal to the workspace, so use the package name in your own code.

## Add the crate

Add the package from crates.io once the version you need is published. Until then, depend on a checkout with a git or path dependency:

```toml
[dependencies]
turbospark-core = { git = "https://github.com/whit3rabbit/turbospark", package = "turbospark-core" }
```

The crate forbids `unsafe` code. Its only dependency is `half`, which provides the half-precision logit type.

## Types you will use first

Most integrations touch four names:

| Item | Kind | What it gives you |
| --- | --- | --- |
| `TokenId` | type alias for `i32` | A token ID at the width used across crate boundaries. |
| `LogitValue` | type alias for `half::f16` | One half-precision logit. |
| `LogitsView<'a>` | struct | A borrowed, read-only view over a slice of `LogitValue`. |
| `SteeringMode` | enum | The four directional steering edits. |

`TokenId` is signed 32-bit. The width matches the downstream buffer element type, so keep it at that width when a value crosses a crate boundary. The workspace rule is to resolve token IDs from a loaded tokenizer and never copy numbers from fixture JSON.

`LogitsView` wraps a borrowed slice and does not copy it. In the workspace, producers return logits. Sampling owns normalization, so do not apply softmax in both places. The example below shows the basic shapes.

```rust
use turbospark_core::runtime_config::ALLOWED_CHUNK_SIZES;
use turbospark_core::{LogitValue, LogitsView, SteeringMode, TokenId};

fn validate_chunk_size(requested: u32) -> Result<u32, String> {
    if ALLOWED_CHUNK_SIZES.contains(&requested) {
        Ok(requested)
    } else {
        Err(format!(
            "chunk size {requested} is not one of {ALLOWED_CHUNK_SIZES:?}"
        ))
    }
}

fn main() -> Result<(), String> {
    let token: TokenId = 42;
    let raw = [LogitValue::from_f32(0.5), LogitValue::from_f32(-1.25)];
    let logits = LogitsView::new(&raw);
    println!("token {token} has {} logits", logits.len());

    let chunk = validate_chunk_size(512)?;
    println!("chunk size {chunk} is allowed");

    let mode = SteeringMode::parse("renorm").ok_or("unknown steering mode")?;
    println!("mode {} can_grow={}", mode.as_str(), mode.can_grow());
    Ok(())
}
```

## Errors

The crate has one error type, `PrefillError`. It has one variant, `ChunkedRunnerDirty(String)`, which `PrefillChunkCommitState::require_clean` returns. The string names the operation and the uncommitted token range. `PrefillError` implements `Display` and `std::error::Error`.

Most functions here do not return `Result`. `SteeringMode::parse` returns an `Option`, and the resolve and planning functions return plain values. Validation is your job at the boundary.

{% hint style="warning" %}
`SteeringMode::parse` returns `None` for unknown input and never falls back to a default. Turn `None` into an error at your boundary, as the steering example below does. A silent fallback would measure the wrong edit and report it as the right one.
{% endhint %}

## Runtime configuration

The `runtime_config` module holds the allowed values and defaults for runtime knobs. The allowed sets are sorted, and const assertions in the crate fail the build if a default falls outside its set. The crate does not validate input for you, so check membership yourself or use `resolve_automatic_chunk_size` for chunk sizes.

| Constant | Type | Allowed values | Default |
| --- | --- | --- | --- |
| `ALLOWED_CACHE_SLOTS` | `[u32; 8]` | 8, 16, 24, 32, 48, 64, 96, 128 | `DEFAULT_CACHE_SLOTS` = 16 |
| `ALLOWED_CHUNK_SIZES` | `[u32; 8]` | 32, 64, 128, 256, 512, 1024, 2048, 4096 | `DEFAULT_CHUNK_SIZE` = 128 |
| `ALLOWED_SPECULATION_BLOCKS` | `RangeInclusive<u32>` | 1 through 15 | none |
| `DEFAULT_MAX_CONTEXT` | `u32` | any positive value | 4096 |

`DEFAULT_MAX_CONTEXT` has no allowed set. A context window is a per-token KV allocation, so the runtime checks it against available memory instead of this crate.

The speculation range is wider than the values most callers want. The source comment explains the bound and points to the measured optimum in the [MTP notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/MTP.md). Read that before you pick a block size.

## Steering modes

`SteeringMode` names the edit a directional steering kernel applies to one residual row. This crate defines the mode only. The CPU reference in [`turbospark-compute`](https://docs.rs/turbospark-compute/latest/turbospark_compute/) and the Metal dispatch in `turbospark-gpu` implement the edit.

| Variant | Effect on the row | Can grow the row |
| --- | --- | --- |
| `Ablate` (default) | Removes the steered direction. | no |
| `Add` | Adds the direction, scaled by alpha. | yes |
| `Clamp` | Pins the coefficient along the unit direction to a target. | yes |
| `Renorm` | Ablates, then rescales the row to its original norm. | no |

`Add` and `Clamp` can push values past the largest finite half-precision value, 65,504. The overflow arrives as infinity and then as NaN. A NaN score can look like a perfect result on a ranking metric, so check for non-finite output when you use a growing mode with a large alpha.

`SteeringMode::parse` accepts the lowercase names listed in `STEERING_MODE_NAMES`. The wire values are fixed in declaration order, so `as_u32` returns the same value the GPU kernel switches on.

```rust
use turbospark_core::{SteeringMode, STEERING_MODE_NAMES};

fn steering_mode_from_flag(flag: &str) -> Result<SteeringMode, String> {
    SteeringMode::parse(flag).ok_or_else(|| {
        format!("unknown steering mode {flag:?}, expected one of {STEERING_MODE_NAMES:?}")
    })
}

fn main() {
    println!("{:?}", steering_mode_from_flag("clamp"));
    println!("{:?}", steering_mode_from_flag("Ablate"));
}
```

## Other modules

- `chunk_sizing` and `prefill` cover chunk-size resolution and span planning. See [Chunking and prefill](chunking.md).
- `humanizer` holds the prompt wrapper and post-generation audit for the rewrite mode in the CLI. It is public, but it is a separate feature, and this page does not cover it. See the [docs.rs entry](https://docs.rs/turbospark-core/latest/turbospark_core/humanizer/).

## Where to go next

- [Chunking and prefill](chunking.md) for span planning and commit-state tracking.
- [Rust library overview](../../rust.md) for the other crates and how they fit together.
- [API reference on docs.rs](https://docs.rs/turbospark-core/latest/turbospark_core/) for every signature.
- [Crate guide](https://github.com/whit3rabbit/turbospark/blob/main/crates/core/AGENTS.md) for the rules that apply when changing this crate.
