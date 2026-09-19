# turbospark-selection

Host-side sampling and token selection.

## Read first

- [Detailed module guide](../../.claude/docs/modules/selection.md)
- [Testing rules](../../docs/TESTING.md)

## Rules

- Inputs are logits. Apply normalization exactly once in the sampler.
- Deterministic mode is an exact `temperature == 0.0` decision. A tiny
  nonzero temperature still exercises sampling.
- Preserve top-k, top-p, temperature, seed, and special-token behavior.
- Test both greedy and sampled paths. Greedy cannot expose distribution errors.

## Checks

```sh
cargo test -p turbospark-selection
```
