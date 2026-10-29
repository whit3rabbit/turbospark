# turbospark-runtime

Model opening, family dispatch, forward passes, state, sessions, and memory
policy.

## Read first

- [Detailed module guide](../../.claude/docs/modules/runtime.md)
- [Load guard](../../docs/LOAD_GUARD.md)
- [Family checklist](../../docs/NEW_MODEL.md)
- [Flint family](../../docs/FLINT.md)
- [Speech to text](../../docs/SPEECH_TO_TEXT.md)
- [Whisper runner](../../docs/WHISPER.md)
- [Audio models](../audio/MODELS.md)
- [Audio music generation](../../docs/audio/music.md)
- [Model gates](../../.claude/docs/model-gates.md)
- [Kernel modes](../../docs/KERNEL_MODES.md): read before adding an
  alternative kernel path behind a switch.

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
  path. `TURBOSPARK_ROUTED_BATCH=1` (opt-in) also batches the routed half on
  top of the batched arm, bit-identical to its per-token routed loop. It binds
  each sub-batch's expert union, not every slot, so it serves more than 32
  slots, and route slots there are BINDING indices. See
  `docs/BATCHED_PREFILL.md`.
- Flint (`ModelFamily::Flint`) has its own flow in `families/flint/`: learned
  attention sinks, per-head QK norm before RoPE, NoPE by layer mask, an INT8
  sigmoid router, an ungated shared expert, and a dense lead. No other family's
  flow has more than two of those, so a neighbour's flow would decode fluently
  and wrongly. Decode streams the 256 routed experts (top-6) through the expert
  slot cache; mapped expert residency is refused at open. Prefill runs the
  layer-batched chunk driver by default; `TURBOSPARK_FLINT_CHUNKED_PREFILL=0`
  (read once) restores the sequential produce path, which repeats the
  same dispatches in the same order and must stay byte-identical to
  the driver. A flint prefill chunk refuses `TURBOSPARK_BATCHED_GEMV` and
  `TURBOSPARK_ROUTED_BATCH` by name. Every flint install is rebalanced at
  intake by a fixed 2^6 up/down rebalance (`crates/repack/src/flint_rebalance.rs`)
  that keeps the SwiGLU product inside FP16, and the flow relies on it. FP16
  range is the family's open risk; `TURBOSPARK_FLINT_DEBUG` and
  `TURBOSPARK_FLINT_ROUTER_DUMP` are the readback diagnostics. See
  `docs/FLINT.md` and `docs/FLINT_PHASE0.md`.
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
- The dense qwen chunked prefill primes an open drafter only through
  `prefill_chunk_priming_drafter`: the MTP head per row, the DFlash2 drafter
  from aux captures (`Priming::Dflash` in `families/qwen/prefill.rs`). The
  DFlash2 variant must end in the token walk's state: cursor and
  `capture_base` on the last prompt row, `capture_rows = 1`, that row at
  capture row 0. Its context write is sliced at 16 rows because
  `encode_gemm_any` caps the batch there. `TURBOSPARK_DFLASH_CHUNKED_PREFILL=0`
  keeps the walk. `real_forward_qwen35_dflash_chunked.rs` is the check.
- Greedy fixed-depth DFlash2 rounds verify a tree by default
  (`TURBOSPARK_DFLASH_TREE=0` keeps the chain). The tree spec rides
  `BatchedScratch::tree` for one `dflash_verify_tree` call and is read only
  by the attention and linear cores. Any new per-row buffer
  that the round-end reads (rollback, fold, next context write) must join
  the gather in `families/qwen/dflash_tree.rs`, or a branching round commits
  another branch's rows. `real_forward_qwen35_dflash_tree.rs` is the check;
  its fixture is the wide-head one because the tree kernels need head_dim
  128/256 and a GDN key dim of 128.
- Music 3 zero-tolerance MLX matches are device-bound. The GitHub macOS
  runner's GPU ("Apple Paravirtual device") returns different cos and sin than
  the M4 Max that produced the pinned references, so `music3_rope` moves in the
  last bit (`fast::` and `precise::` alike) and the `affine4` tiny profile in
  `tests/music3_metal.rs` flips a bf16 rounding. That comparison runs only on
  `Apple M4 Max` (or with `TURBOSPARK_MUSIC3_ORACLE_ANY_DEVICE=1`); other GPUs
  still check reset, finiteness and sizes. Do not widen the tolerance or chase
  it in the quantized kernels (explicit `fma` changed nothing). Details:
  `docs/audio/minimax_music3.md`.
- `TURBOSPARK_QWEN_FUSED_FFN=1` activates fused INT4 FFN dispatches (gate+up+act
  GEMV and down+epilogue GEMV) for Qwen dense and MoE shared expert decode.
  Must remain bit-exact with the sequential separate GEMV and activation dispatches.
- `TURBOSPARK_INT4_MMA8` (opt-in, experimental, read once) routes INT4
  projections of at least 4096 output rows through the small-M matrix-unit
  GEMM: `1` in both decode and verify, `verify` in batched calls only. Decode
  and verify stay bit-exact against each other, but decode numerics change, so
  frozen digests do not hold. `TURBOSPARK_INT4_MMA8_MIN_ROWS` overrides the
  row threshold for tests. Only qwen35 dense is gated. See
  `docs/INT4_SMALL_M_MMA.md`.
- `TURBOSPARK_GDN_LAZY_TAPE=1` records rank-1 conv and delta updates during
  speculative verify and folds them asynchronously at round commit, bypassing
  recurrent state snapshots on clean state. `checkpoint_round` snapshots recurrent
  state only when the trunk was dirtied. Eager advances take deferred snapshots.
  Checkpoints remain valid across speculative rollbacks prior to commit. On
  verify failure, `discard_pending` restores clean state. See `docs/KAPITSA_KERNELS.md`.

## Checks

```sh
cargo test -p turbospark-runtime
cargo fmt --check
```

Read the family-specific gate page before changing a decode path or model
family.
