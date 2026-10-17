# turbospark-runtime

Model opening, family dispatch, forward passes, state, sessions, and memory
policy.

## Read first

- [Detailed module guide](../../.claude/docs/modules/runtime.md)
- [Load guard](../../docs/LOAD_GUARD.md)
- [Family checklist](../../docs/NEW_MODEL.md)
- [Speech to text](../../docs/SPEECH_TO_TEXT.md)
- [Whisper runner](../../docs/WHISPER.md)
- [Audio models](../audio/MODELS.md)
- [Audio music generation](../../docs/audio/music.md)
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
- MegaTTS3 sessions open on Metal by default on macOS; the portable CPU
  reference path runs only as an opt-in fallback when Metal cannot open. Memory
  admission must size for Metal device allocations.
- Real-model validation needs greedy, sampled, and memory-oracle coverage. A
  compile or synthetic fixture is not a runtime parity claim.
- Qwen MoE batched prefill (`qwenGdnMoe`) batches resident weight reads by
  default on INT4 installs, while routed experts keep the per-token loop.
  Reduction-order drift flips routing near-ties, held to a 1e-2 nats mean KL
  probe bound with top-1 agreement >= 50% (the dense bound is 1e-3), see
  `crates/bench/tests/qwen36_prefill_arm_probe.rs`. The family covers Qwen 3.6
  and Ornith 35B. MiniCPM-V 4.7 (BF16 resident matrices) keeps the per-token
  path. `TURBOSPARK_BATCHED_GEMV=0` opts out to the byte-identical per-token
  path. See `docs/BATCHED_PREFILL.md`.

## Checks

```sh
cargo test -p turbospark-runtime
cargo fmt --check
```

Read the family-specific gate page before changing a decode path or model
family.
