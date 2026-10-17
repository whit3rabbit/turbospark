---
description: "Host-side sampling: a validated shaping configuration, penalty and truncation filters, and the seeded draw that turns logits into exactly one token."
icon: filter
---

# selection

`crates/selection` (package `turbospark-selection`) owns token selection:
turning a score vector over the vocabulary into exactly one candidate id
under a validated shaping configuration. It applies repetition, presence, and
frequency penalties, composes top-p, top-k, and min-p truncation, reweights
by temperature, and draws with a seeded or clock-derived uniform value. It
also materializes the shaped distribution that speculative decoding needs for
exact rejection sampling. Normalization lives here and only here: the crate
takes raw logits and applies the softmax exactly once.

## What it does

| Responsibility | Where |
| --- | --- |
| The selection pipeline (penalties, truncation, reweight, draw) | `src/choose.rs` (`select`, `shaped_distribution`) |
| Validated shaping configuration and selection errors | `src/shaping.rs` (`ShapingConfig`, `SelectionError`) |
| The materialized distribution and the speculative residual | `src/distribution.rs` (`ShapedDistribution`, `residual`) |
| Repetition and presence/frequency penalties | `src/penalty.rs` |
| Top-k, top-p, and min-p truncation filters | `src/truncation.rs` |
| Seeded step-value derivation | `src/derive.rs` |

The full probability vector is never materialized per token: candidates are
ranked by unnormalized `exp(s - max)` values and normalized probabilities
are computed on the fly only for the ranked prefixes the pipeline touches.
Reusable scratch buffers are kept in a thread-local so a per-token call does
not re-allocate multi-megabyte vectors at real vocabulary sizes.

## Where it sits

`turbospark-selection` depends only on `turbospark-core` (aliased
`foundation`), from which it takes `LogitsView`, `LogitValue`, and `TokenId`.
Its consumers are `runtime` (both decode loops, which pass the generated
count as `position`), `cli`, `server`, `ffi`, and `bench`.

## Public API tour

Root re-exports from `crates/selection/src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `select(scores, config, history, position)` | `src/choose.rs` | Select one `TokenId`; `Result<TokenId, SelectionError>` |
| `shaped_distribution(...)` | `src/choose.rs` | The distribution `select` samples from, materialized |
| `ShapingConfig::new(temperature, top_k, top_p, repetition_penalty, seed)` | `src/shaping.rs` | Build and validate a config; builder methods add penalties and min-p |
| `ShapingConfig::is_deterministic()` | `src/shaping.rs` | Exact `temperature == 0.0` decision |
| `MAX_TOP_K` | `src/shaping.rs` | Inclusive rank-truncation bound, 256; `top_k = 0` disables |
| `DEFAULT_REPETITION_PENALTY` | `src/shaping.rs` | 1.0, the identity |
| `ShapedDistribution` | `src/distribution.rs` | Support plus probabilities; `probability(token)`, `draw(seed, position)`, `point_mass(token)` |
| `residual(p, q)` | `src/distribution.rs` | Normalized `max(p - q, 0)`; `None` when p is covered |
| `SelectionError` | `src/shaping.rs` | Distinguishable descriptive error with a public `reason` field |

Validation is at construction time and refuses: non-finite or negative
temperature, `top_k` above `MAX_TOP_K`, `top_p` outside `(0, 1]`, and
non-finite or non-positive repetition penalty. The builder methods refuse
presence and frequency penalties outside `[-2, 2]` and a min-p threshold
outside `[0, 1)`; `min_p` of exactly 1.0 is refused rather than read as
"greedy".

## Examples

### Greedy selection

At the deterministic temperature the pipeline short-circuits to argmax over
the raw scores (ties break to the lowest index), ignoring penalties,
truncation, and the seed.

```rust
use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, ShapingConfig};

let scores: Vec<LogitValue> = [1.0f32, 3.2, 0.4]
    .iter()
    .map(|&v| LogitValue::from_f32(v))
    .collect();
let config = ShapingConfig::new(0.0, 64, Some(0.95), 1.0, Some(7)).unwrap();

let view = LogitsView::new(&scores);
let history: Vec<TokenId> = vec![];
let token = select(view, &config, &history, 0).unwrap();
assert_eq!(token, 1);
```

### Seeded sampled selection

From `crates/selection/tests/determinism.rs`: the same seed, scores, history,
and position always agree, and replaying a sequence of positions reproduces
element by element.

```rust
use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{select, ShapingConfig};

let scores: Vec<LogitValue> = [1.0f32, 1.05, 0.9, 1.1, 0.95, 1.0, 0.8, 1.2]
    .iter()
    .map(|&v| LogitValue::from_f32(v))
    .collect();
let config = ShapingConfig::new(0.8, 0, None, 1.0, Some(7)).unwrap();
let history: Vec<TokenId> = vec![];

let view = LogitsView::new(&scores);
let a = select(view, &config, &history, 3).unwrap();
let b = select(LogitsView::new(&scores), &config, &history, 3).unwrap();
assert_eq!(a, b);
```

Penalties and extra truncation join through the config's builder methods:

```rust
let config = ShapingConfig::new(0.8, 40, Some(0.9), 1.1, Some(7))
    .unwrap()
    .with_presence_penalty(0.3)
    .unwrap()
    .with_frequency_penalty(0.05)
    .unwrap()
    .with_min_p(0.05)
    .unwrap();
```

### The shaped distribution for speculative decoding

`shaped_distribution` exposes the same surviving set and reweighted
probabilities `select` draws from; `residual` computes the corrected-token
distribution for Leviathan/Chen rejection sampling. The draw identity is
pinned by `crates/selection/tests/rejection_sampling.rs`.

```rust
use foundation::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{residual, select, shaped_distribution, ShapingConfig};

fn logits(values: &[f32]) -> Vec<LogitValue> {
    values.iter().map(|&v| LogitValue::from_f32(v)).collect()
}

let target_scores = logits(&[0.1, 2.0, 1.4]);
let draft_scores = logits(&[0.1, 1.7, 1.5]);
let config = ShapingConfig::new(0.7, 0, None, 1.0, Some(9)).unwrap();
let history: Vec<TokenId> = vec![];

let p = shaped_distribution(LogitsView::new(&target_scores), &config, &history, 0).unwrap();
let q = shaped_distribution(LogitsView::new(&draft_scores), &config, &history, 0).unwrap();

// THE IDENTITY (pinned by tests/rejection_sampling.rs): `select` returns
// exactly what the materialized distribution's draw returns for the same
// seed and position.
let drawn = p.draw(Some(9), 0);
assert_eq!(
    select(LogitsView::new(&target_scores), &config, &history, 0).unwrap(),
    drawn as TokenId
);

// Speculative acceptance compares p(x) / q(x) over these same shaped
// distributions; on rejection the correction comes from the normalized
// residual max(p - q, 0).
let proposal = q.draw(Some(9), 0);
let accept = (p.probability(proposal) / q.probability(proposal)).min(1.0);
let token = if accept >= 1.0 {
    proposal
} else {
    match residual(&p, &q) {
        Some(r) => r.draw(Some(9), 0),
        // p covered by q: every proposal is accepted in the limit.
        None => p.draw(Some(9), 0),
    }
};
let _ = token;
```

(The acceptance comparison itself is the caller's job; the crate guarantees
both sides see the same shaped distributions, penalties and truncation
included.)

## Gotchas

{% hint style="warning" %}
Determinism is an exact `temperature == 0.0` decision. A tiny nonzero
temperature still exercises the sampling path, and the greedy arm ignores
penalties, truncation, and the seed, so greedy runs cannot expose a
distribution bug. Test both paths, as the crate's `AGENTS.md` requires.
{% endhint %}

{% hint style="warning" %}
`position` doubles as the generated-token count: presence and frequency
penalties apply to the last `position` entries of `history`, clamped
defensively. Callers in `crates/runtime` pass the generated count; passing
something else quietly changes which tokens get penalized.
{% endhint %}

## Related pages

- [core](core.md)
- [Speculative Decoding](../advanced/speculative-decoding.md)
- [The CLI](../command-line-and-local-api/cli.md)
- [The Local Server](../command-line-and-local-api/server.md)
