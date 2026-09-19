# turbospark-runtime

Model opening, family dispatch, forward passes, state, sessions, and memory
policy.

## Read first

- [Detailed module guide](../../.claude/docs/modules/runtime.md)
- [Load guard](../../docs/LOAD_GUARD.md)
- [Family checklist](../../docs/NEW_MODEL.md)
- [Model gates](../../.claude/docs/model-gates.md)

## Rules

- Keep `#![forbid(unsafe_code)]`.
- Select the decode flow from `ArchConfig.family`, never from tensor names.
  New family support must trace the enum and persisted string through every
  consumer.
- Forward producers return logits. The sampler owns softmax, temperature,
  top-k, and top-p.
- Reset every recurrent, convolutional, KV, and speculative state on session
  reset. A fluent output can still be wrong after stale state.
- Dispatch routed experts in router ranking order. That order is reduction
  order.
- Use an inner Metal autorelease pool for repeated forward passes and preserve
  KV ring capacity in pipeline specialization.
- Memory guard decisions must distinguish resident allocations, expert slot
  capacity, KV, and streamed on-disk weights.
- Real-model validation needs greedy, sampled, and memory-oracle coverage. A
  compile or synthetic fixture is not a runtime parity claim.

## Checks

```sh
cargo test -p turbospark-runtime
cargo fmt --check
```

Read the family-specific gate page before changing a decode path or model
family.
