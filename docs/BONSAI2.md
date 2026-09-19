# Bonsai-2: the Hadamard-folded checkpoint line

The home for what the Bonsai-2 line is, the weight-basis contract its
checkpoints ship under, and what this engine implements to run them. Read it
before touching `families/qwen/`'s transform call sites, the manifest's
`hadamard` section, or quoting any number from this line.

## What the line is

`prism-ml/Ternary-Bonsai-2-27B` (2026-09-17) is the second generation of
prism-ml's ternary compression of the same base model this repo already runs
twice: `text_config` parses to `qwen_gdn_dense_27b()` exactly, as Bonsai-27B's,
Ternary-Bonsai-27B's and Qwen3.8-27B's do. Fourth checkpoint, one architecture,
`families/qwen/`'s dense half. No shape key moved; the offline proof is
`every_published_checkpoint_parses_to_one_baseline` in `crates/repack`'s
`qwen35_config.rs`, now over all four configs.

What IS new is the basis the weights live in, and one config rename:

- Root `model_type` is `prism_hadamard_qwen35` -- unknown to `SUPPORTED_HF`.
  Family resolution rides the two-step probe falling through to
  `text_config.model_type` (`qwen3_5_text`), which is what
  `config_json_resolution`'s root-then-text_config order exists for. Pinned by
  `bonsai2_resolves_only_through_the_text_config`.
- `mtp_num_hidden_layers: 0` and `components.mtp: false`: no drafter. The
  original Ternary-Bonsai declared `mtp_num_hidden_layers: 1` and its
  conversion still dropped `mtp.*`; Bonsai-2 does not even declare one. The
  GGUF repo carries none either. So the batched-int2 speculation arm stays
  GATED on a drafter artifact existing (a known reopening condition, recorded
  in ROADMAP P3.3), and `RealForwardRunner`'s open path additionally refuses
  the combination of a folded trunk and an open drafter outright.

## The weight-basis contract

`PACK-RUNTIME.md` in the repo states it in one line: "Ordinary MLX loaders do
not apply the required transforms." Every one of the checkpoint's 402 packed
modules -- embedding, lm_head, and every attention, GDN and FFN projection --
is stored as

    W' = W * diag(signs) * H_block / sqrt(block)

where `H_block` is independent Hadamard butterflies over `block`-sized
segments of the INPUT axis (`block: 1024`, one width for the whole
checkpoint, validated against the bundled runtime's set 512/1024/2048/4096)
and `signs` is a +/-1 vector spanning the full input width. The runtime's
job is the matching ACTIVATION transforms:

- forward, `y = H(signs * x) / sqrt(block)`, on every folded entry's input;
- inverse, `y = signs * (H x) / sqrt(block)`, on the embedding's dequantized
  rows (the one module stored with its OUTPUT rotated).

The residual stream therefore stays in the ORIGINAL basis everywhere, which
is also what makes the vision-tower blit arm safe: a tower row is an
original-basis activation, and the transforms happen only at weight-input
boundaries. The bundled GGUF runtime (`runtime.py`) is the reference for the
order of operations: signs first for the forward, signs last for the inverse,
`1/sqrt(block)` on every path. The GPU kernel is
`crates/gpu/src/shaders/hadamard.metal`, its butterfly copied from
`attention_tq.metal`'s `tq_rht`, and its parity tests compare against
`compute::kv_quant`'s `rht_forward`/`rht_inverse` -- the same two maps, per
block segment.

Two facts pinned off the real checkpoint that the install format leans on:

- **Sign vectors are shared per input width.** All 5120-wide folded modules
  carry a byte-identical sign vector; likewise 6144 (o_proj input and GDN
  out_proj input) and 17408 (down_proj input). `read_hadamard` ASSERTS the
  equality per width while streaming, so a future checkpoint with genuinely
  per-module signs fails the walk loudly instead of running on the wrong
  vector. This is also why one forward dispatch per shared input point is
  correct: q/k/v and in_proj_qkv/in_proj_z split one transformed norm.
- **`in_proj_a`/`in_proj_b` are unquantized F32** here, where every earlier
  checkpoint quantized them (they are absent from the `modules` list, hence
  unfolded, hence they read the RAW norm). Their install entries are the
  walk's BF16 tag, and reading that tag for a MATRIX is the `DTYPE_RAW_BF16`
  arm `encode_gemv_any` gained (`crates/gpu/src/gemv_bf16.rs`). The tag was
  honoured for norms and the embedding before; this line is the first with a
  live matrix arm.

## What the install carries

- `hadamard.bin`: the sign vectors, one per distinct width (114,976 bytes =
  (5120 + 6144 + 17408) F32 on the real checkpoint), written by the walk and
  listed in `manifest.files`.
- `manifest.hadamard`: `{block, signs: [{width, offset, bytes}], folded:
  [names], inverse: [names]}`. Optional; absent means no transform, which is
  the correct default for every install written before the contract.
  `model_io` validates the section's shape at load; the runtime refuses a
  section whose family has no folded-weight support, so a folded checkpoint
  cannot silently open under a flow that would skip the transforms.
- The walk (`write_qwen_gdn_dense_install_streamed_with_hadamard`) refuses a
  checkpoint carrying `.signs` tensors without a parsed contract, so the
  contract-less walk cannot silently write a sign-less install of a folded
  checkpoint. The catalog streamer parses the contract off the same
  `config.json` and routes to the right walk automatically.

## Where the runtime transforms

`RealQwenState.hadamard` (a `HadamardPlan`) is built at open from the
manifest section; `None` on every pre-contract install, which makes every
consumer site byte-identical to its pre-contract code. The dense flow
transforms:

1. the embedding's dequantized row (inverse, into the residual stream) --
   decode and chunked-prefill embed loops, the vision blit arm untouched;
2. the input-layernorm output (forward, shared by q/k/v and
   in_proj_qkv/in_proj_z; in_proj_a/b read the raw norm);
3. the post-attention norm output (forward, shared by gate/up);
4. the gated attention output before `o_proj`, the gated GDN output before
   `out_proj`, and the FFN activation before `down_proj` (forward, each at
   its own call site, selected per NAME);
5. the final norm before a folded head.

Selection is per-NAME via the manifest's folded list, not per-family, so an
unfolded entry always reads the raw activation regardless of what the rest
of the checkpoint does.

## The boundaries (what is deliberately not wired)

- `TURBOSPARK_BATCHED_GEMV` refuses a folded install by name: the M-row
  inputs would need the transforms the per-token encoders carry, and the
  batched GEMM has no BF16 arm for the unquantized pair anyway. Refusal, not
  a silently untransformed batch -- the museglimmer precedent.
- A folded trunk with an open drafter is refused at open: the drafter's own
  weight inputs would need the same transforms, and no folded drafter
  artifact exists to wire against. This is the concrete form of the int2
  speculation gate; the reopening condition is a drafter artifact (an mtp.*
  or DFlash2 conversion) for a folded checkpoint, which does not exist.
- The MoE half of the flow has no folded support; `HadamardPlan::build`
  refuses a routed install that carries a hadamard section.

## Gates and evidence

- `crates/gpu/tests/hadamard_fwht_parity.rs`: the kernel against the CPU
  reference at the real widths, both directions, multi-row, block 512, plus
  the inverse-is-not-forward check (the two directions are different maps;
  a mutation that fuses them is a bug, not an optimization).
- `crates/gpu/tests/bf16_gemv_parity.rs`: the unquantized-matrix GEMV
  against a CPU reference reading the same bf16 bits.
- `crates/repack/tests/qwen35_config.rs`: the offline config proof, now over
  four checkpoints, plus the hadamard-contract parser cases.
- `crates/repack/tests/bonsai2_checkpoint_network.rs`: the walk against the
  real checkpoint (header pinned by SHA; the repo ships no index.json).
- `crates/bench/tests/bonsai2_{quality_gate,memory_oracle}.rs`: the frozen
  gates; a healthy row here is proof the activation transforms are right,
  since a missing transform reads as fluent output with a catastrophic
  perplexity (the Gemma sandwich-norm failure shape).
