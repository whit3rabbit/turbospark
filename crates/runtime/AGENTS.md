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
- The ANE+Steel prefill split (`families/qwen/ane_split.rs`) is opt-in twice:
  cargo feature `ane-prefill` and `TURBOSPARK_ANE_PREFILL` (or via FFI/Swift
  `AnePrefillMode`). When set to `Auto`, it activates on Apple Silicon M4+ /
  A18+ (`hardware_supports_ane_prefill`) for dense Qwen, falling back to
  Steel with a warning if programs are missing or hardware is older. The
  staged layer functions it calls (`batched_layers.rs`, `prefill_layers.rs`)
  must keep the default arm's dispatch order. The default rows of
  `qwen38_prefill_arm_probe` and the greedy md5 are the check. A split
  micro-batch redoes on Steel from a GDN snapshot after any failure. Anything
  new it mutates in place must be undone there too. It widens the micro-batch
  to 2048 only while no MTP head or DFlash2 drafter is open. See
  `docs/SPEEDUP_PLAYBOOK.md`.
- Qwen batched prefill, speculative verify, and MTP priming encode their small
  per-row operations (q/gate split, q/k norm, RoPE, output gate, FFN
  activation, residual adds and norms) as a single multi-row dispatch across
  all rows. Each row computes the same bits.
  `TURBOSPARK_QWEN_BATCHED_ROW_OPS=0` restores the per-row form.
- Music 3 zero-tolerance MLX matches are device-bound. The GitHub macOS
  runner's GPU ("Apple Paravirtual device") returns different cos and sin than
  the M4 Max that produced the pinned references, so `music3_rope` moves in the
  last bit (`fast::` and `precise::` alike) and the `affine4` tiny profile in
  `tests/music3_metal.rs` flips a bf16 rounding. That comparison runs only on
  `Apple M4 Max` (or with `TURBOSPARK_MUSIC3_ORACLE_ANY_DEVICE=1`); other GPUs
  still check reset, finiteness and sizes. Do not widen the tolerance or chase
  it in the quantized kernels (explicit `fma` changed nothing). Details:
  `docs/audio/minimax_music3.md`.

## Checks

```sh
cargo test -p turbospark-runtime
cargo fmt --check
```

Read the family-specific gate page before changing a decode path or model
family.
