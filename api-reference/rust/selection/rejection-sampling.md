---
description: "Use shaped_distribution, ShapedDistribution, and residual to build speculative-decoding acceptance tests over the distribution select samples from."
icon: scale-balanced
---

# Rejection sampling

Use these functions when you run a draft model and a target model. They supply the accept-or-correct step from speculative decoding. Skip this page if you only need ordinary sampling.

`shaped_distribution` returns the same surviving tokens and probabilities that `select` samples from. That lets your acceptance test compare the distributions the sampler uses. It avoids a second approximation that drifts from the sampler.

The crate does not run the acceptance loop or own a random number generator. Both stay in your code. You supply the uniform values, or you use `ShapedDistribution::draw` when you want the exact token `select` would return. The method comes from [Leviathan et al.](https://arxiv.org/abs/2211.17192) and [Chen et al.](https://arxiv.org/abs/2302.01318). The source doc comment in [`distribution.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/selection/src/distribution.rs) cites both.

## The three pieces

- [`shaped_distribution`](https://docs.rs/turbospark-selection/latest/turbospark_selection/fn.shaped_distribution.html) takes the same arguments as `select` and returns a `ShapedDistribution`. Call it once for the target logits and once for the draft logits, with the same `ShapingConfig`.
- [`ShapedDistribution`](https://docs.rs/turbospark-selection/latest/turbospark_selection/struct.ShapedDistribution.html) holds a support of candidate IDs and their probabilities. Its methods are `support()`, `probability(token)`, `draw(seed, position)`, `new`, and `point_mass`. `probability` returns `0.0` for any token outside the support.
- [`residual`](https://docs.rs/turbospark-selection/latest/turbospark_selection/fn.residual.html) takes the target `p` and the draft `q`, and returns the normalized `max(p - q, 0)` as an `Option`. It returns `None` when the residual mass is effectively zero, which happens when `p` and `q` match. In that case, draw from `p` directly.

Candidate IDs in `ShapedDistribution` are `u32`, while `TokenId` is `i32`, so your code needs an explicit cast between them. Vocabulary indices are never negative, so `as TokenId` does not change the value.

## Example

This step accepts or corrects one proposed token. It takes two caller-supplied uniform values, so the crate's seed does not drive the acceptance decision. The signatures were checked against source. This audit did not compile or run the example.

```rust
use turbospark_core::{LogitValue, LogitsView, TokenId};
use turbospark_selection::{residual, shaped_distribution, ShapedDistribution, ShapingConfig};

/// Walk the support in order and return the first token whose cumulative
/// probability passes `u`, where `u` is in [0, 1) from the caller's RNG.
fn draw_with(dist: &ShapedDistribution, u: f64) -> u32 {
    let mut cumulative = 0.0;
    for &token in dist.support() {
        cumulative += dist.probability(token);
        if u < cumulative {
            return token;
        }
    }
    *dist.support().last().expect("support is non-empty")
}

fn speculative_step(
    target: &[LogitValue],
    draft: &[LogitValue],
    config: &ShapingConfig,
    history: &[TokenId],
    position: u64,
    u_draft: f64,
    u_accept: f64,
    u_correct: f64,
) -> Result<TokenId, Box<dyn std::error::Error>> {
    let p = shaped_distribution(LogitsView::new(target), config, history, position)?;
    let q = shaped_distribution(LogitsView::new(draft), config, history, position)?;

    let proposal = draw_with(&q, u_draft);
    let q_x = q.probability(proposal);
    let p_x = p.probability(proposal);
    if q_x > 0.0 && u_accept < (p_x / q_x).min(1.0) {
        return Ok(proposal as TokenId);
    }
    let corrected = match residual(&p, &q) {
        Some(r) => draw_with(&r, u_correct),
        None => draw_with(&p, u_correct),
    };
    Ok(corrected as TokenId)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ShapingConfig::new(1.0, 0, Some(0.9), 1.0, Some(1))?;
    let target: Vec<LogitValue> = [3.0f32, 1.0, 0.5, -1.0]
        .into_iter()
        .map(LogitValue::from_f32)
        .collect();
    let draft: Vec<LogitValue> = [1.0f32, 2.5, 0.0, -0.5]
        .into_iter()
        .map(LogitValue::from_f32)
        .collect();
    let token = speculative_step(&target, &draft, &config, &[], 0, 0.3, 0.9, 0.5)?;
    println!("step: {token}");
    Ok(())
}
```

Use one `ShapingConfig` for both sides, because the acceptance test only means something when `p` and `q` are shaped the same way that `select` shapes its sampling distribution. If you deliberately shape the draft differently, document that choice next to the call.

## Matching `select`

For a seeded run, `shaped_distribution(...).draw(seed, position)` returns the same token as `select` with the same scores, history, position, and config. The crate's tests check this identity. Use `draw` only when you need that exact match, and otherwise draw with your own uniform values. The reason is that `draw` derives its value from `(seed, position)` alone. Calling `draw` twice at the same position therefore returns the same token both times, which can correlate a proposal with its correction if you reuse the position.

## Edge cases

- At temperature `0.0`, `shaped_distribution` returns a point mass on the argmax. `probability` of any other token is `0.0`, so the target accepts only that token.
- `ShapedDistribution::new` panics if the support and probability vectors have different lengths.
- `ShapedDistribution::draw` panics on an empty support. Values from `shaped_distribution` and `residual` are never empty.
- `residual` returns `None` when `p` and `q` are the same distribution, or when `q` covers `p` almost everywhere. When `p` equals `q`, every proposal is accepted and no correction is needed. In the near-covered case the caller draws from `p` instead, as the example does.

## Next steps

- [Overview](README.md): the `select` pipeline, errors, and the `position` rule.
- [Shaping configuration](shaping-config.md): ranges and defaults for each setting.
