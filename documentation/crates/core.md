---
description: "The shared leaf crate: token and logit primitives, runtime-knob allowed sets, chunked-prefill planning, steering modes, and the humanizer rewrite flow."
icon: cube
---

# core

`crates/core` (package `turbospark-core`) is the leaf crate every other
TurboSpark crate depends on. It owns the types that cross crate boundaries
(token IDs, half-precision logits), the single home for runtime-knob allowed
value sets and defaults, chunked-prefill span planning, the steering edit
modes shared by the CPU reference and the Metal kernel, and the humanizer
rewrite flow's pure prompt and audit helpers. It holds no runtime, no I/O,
and no platform code, and it compiles with `#![forbid(unsafe_code)]`.

## What it does

| Module | Responsibility |
| --- | --- |
| `primitives` | `TokenId` (signed 32-bit), `LogitValue` (IEEE-754 binary16 via the `half` crate), `LogitsView` |
| `runtime_config` | Allowed numeric sets and fixed defaults for cache slots, chunk sizes, speculation blocks, and context |
| `chunk_sizing` | Automatic prompt-processing chunk-size resolution |
| `prefill` | Chunked-prefill span planning and commit-state (dirty runner) tracking |
| `steering` | The four directional-steering edit modes (`Ablate`, `Add`, `Clamp`, `Renorm`) |
| `humanizer` | Byte-exact humanizer rewrite prompt, sampling recipe constants, and the post-generation audit |

Everything re-exported at the crate root is intentionally small: the point of
the crate is that a fact is declared exactly once here and every other crate
reads it from here.

## Where it sits

`turbospark-core` depends on no other TurboSpark crate (its only dependency
is the `half` crate for `f16`). All 17 sibling crates depend on it, usually
under the alias `foundation` so references are not confused with the standard
library `core` crate in the extern prelude:

```toml
foundation = { package = "turbospark-core", path = "../core", version = "0.6.5" }
```

Follow the same alias convention if you add a crate.

## Public API tour

Key re-exports from `crates/core/src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `TokenId` | `src/primitives.rs` | Token id interchange type, signed `i32`, matching the downstream buffer element width |
| `LogitValue` | `src/primitives.rs` | One half-precision logit element (`half::f16`) |
| `LogitsView<'a>` | `src/primitives.rs` | Borrowed view over a contiguous logits slice (`new`, `as_slice`, `len`, `is_empty`) |
| `ALLOWED_CACHE_SLOTS` | `src/runtime_config.rs` | `[8, 16, 24, 32, 48, 64, 96, 128, 192, 256]` |
| `ALLOWED_CHUNK_SIZES` | `src/runtime_config.rs` | `[32, 64, 128, 256, 512, 1024, 2048, 4096]` |
| `ALLOWED_SPECULATION_BLOCKS` | `src/runtime_config.rs` | Inclusive range `1..=15` |
| `DEFAULT_CACHE_SLOTS`, `DEFAULT_CHUNK_SIZE`, `DEFAULT_MAX_CONTEXT` | `src/runtime_config.rs` | `16`, `128`, `4096` |
| `resolve_automatic_chunk_size`, `InputLength` | `src/chunk_sizing.rs` | Three-state resolution of an `Auto` chunk request |
| `prefill_chunk_spans`, `PrefillChunkSpan`, `PrefillChunkCommitState`, `PrefillError`, `MAX_CHUNK_TOKENS` | `src/prefill.rs` | Chunked-prefill planning and dirty-runner guards |
| `SteeringMode`, `STEERING_MODE_NAMES` | `src/steering.rs` | The four edit modes; discriminants are pinned MSL wire values |
| `build_humanize_prompt`, `audit_rewrite`, `RewriteAudit` | `src/humanizer.rs` | Humanizer completion wrapper and rewrite audit |

The file `crates/core/src/runtime_config.rs` deserves a special mention: it
carries compile-time assertions that each default is a member of its own
allowed set and that the sets are sorted ascending, so a default cannot
silently drift out of range.

## Examples

### Wrap logits for a downstream sampler

`LogitsView` is the read-only shape every sampling consumer takes. The
element type is `half::f16`, exposed as `LogitValue`.

```rust
use foundation::{LogitValue, LogitsView};

let values: Vec<LogitValue> = [0.5f32, -1.0, 2.0]
    .iter()
    .map(|&v| LogitValue::from_f32(v))
    .collect();
let view = LogitsView::new(&values);

assert_eq!(view.len(), 3);
assert!(!view.is_empty());
// Hand `view` to a consumer that takes `LogitsView<'_>`.
let slice: &[LogitValue] = view.as_slice();
assert_eq!(slice[2].to_f32(), 2.0);
```

(Simplified from how `crates/selection` and `crates/runtime` build views
before calling the sampler.)

### Resolve an automatic chunk size

`resolve_automatic_chunk_size` picks the smallest allowed size that covers a
known input length, falls back to the largest allowed size past the ceiling,
and uses the documented default when the length is not known yet. Values are
from the crate's own unit tests in `crates/core/src/chunk_sizing.rs`.

```rust
use foundation::{resolve_automatic_chunk_size, InputLength};

// Unknown length: the documented default (128).
assert_eq!(
    resolve_automatic_chunk_size(InputLength::Unknown),
    foundation::DEFAULT_CHUNK_SIZE
);
// Known length: the smallest allowed size that covers it.
assert_eq!(resolve_automatic_chunk_size(InputLength::Known(1)), 32);
assert_eq!(resolve_automatic_chunk_size(InputLength::Known(33)), 64);
// Past every allowed size: the largest allowed size (4096).
assert_eq!(
    resolve_automatic_chunk_size(InputLength::Known(1 << 20)),
    foundation::ALLOWED_CHUNK_SIZES[7]
);
```

### Plan a chunked prefill and guard a dirty runner

`prefill_chunk_spans` splits a prompt into spans of at most
`MAX_CHUNK_TOKENS`; `PrefillChunkCommitState` refuses to reuse a runner that
wrote KV rows for a chunk without committing them. Adapted from the unit
tests in `crates/core/src/prefill.rs`.

```rust
use foundation::{prefill_chunk_spans, PrefillChunkCommitState};

let spans = prefill_chunk_spans(10, 0, 4);
assert_eq!(spans.len(), 3);
assert_eq!(spans[0].token_count, 4);
assert_eq!(spans[2].token_count, 2); // remainder chunk
assert_eq!(spans[2].start_position, 8);

let mut state = PrefillChunkCommitState::new();
state.mark_dirty(spans[0].start_position, spans[0].token_count);
assert!(state.is_dirty());
// Reusing the runner now is an error naming the in-flight range.
assert!(state.require_clean("prefill").is_err());
state.mark_committed();
assert!(state.require_clean("prefill").is_ok());
```

## Gotchas

{% hint style="warning" %}
Depend on the crate under an alias (`foundation = { package =
"turbospark-core" }`), never as a plain `core` dependency. The crate's own
docs warn that an unaliased name collides with the standard library `core`
crate in the extern prelude.
{% endhint %}

{% hint style="warning" %}
Runtime knobs have exactly one allowed-value set and one default, and they
live only in `runtime_config.rs`. If you add or change a knob, update the
parser, validation, serialization, and tests together; do not copy a numeric
range into a second crate.
{% endhint %}

Two more rules from the crate's `AGENTS.md` worth keeping in mind: token IDs
crossing a crate boundary are always `turbospark_core::TokenId` (signed
32-bit), and the `SteeringMode` discriminants are wire values the MSL kernel
switches on, pinned by a GPU test, so their order must never be reordered.

## Related pages

- [Crate reference overview](overview.md)
- [compute](compute.md)
- [selection](selection.md)
- [Building from Source](../developer/building-from-source.md)
- [Directional Steering](../advanced/directional-steering.md)
