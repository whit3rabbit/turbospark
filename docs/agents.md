# Agent Guidelines and Notes

This page points to the primary project instructions and records the MoE
prefill batching rules for agents working on `turbospark-runtime`.

- Root instructions: [`../AGENTS.md`](../AGENTS.md)
- Runtime crate instructions: [`../crates/runtime/AGENTS.md`](../crates/runtime/AGENTS.md)
- Batched prefill architecture: [`BATCHED_PREFILL.md`](BATCHED_PREFILL.md)
- Verification reference: [`../.claude/docs/verification.md`](../.claude/docs/verification.md)
- Model gates: [`../.claude/docs/model-gates.md`](../.claude/docs/model-gates.md)

## MoE batched prefill rules

- The Qwen MoE chunked prefill (`qwenGdnMoe`: Qwen 3.6, Ornith 35B) batches its
  resident weight reads by default on INT4 installs, with routed experts keeping
  the per-token loop.
- Batched reduction-order drift flips routing near-ties. The probe gate bound is
  1e-2 nats mean KL, not the dense 1e-3 (`qwen36_prefill_arm_probe`).
- `TURBOSPARK_BATCHED_GEMV=0` restores the byte-identical per-token path.
- MiniCPM-V 4.7 (BF16 resident matrices) keeps the per-token path.
- Ornith 35B real-install checks are owed.
