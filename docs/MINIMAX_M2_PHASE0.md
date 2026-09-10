# MiniMax-M2: Phase 0

Status, 2026-09-10: GGUF recognition only. No family baseline, tensor intake,
decode flow, or verified install exists here. The planned registry row keeps
`probe` and `pull` refused and names the missing behavior.

## Witness and scope

The header-only probe read `general.architecture = minimax-m2` from
[Unsloth's first Q4_K_M shard](https://huggingface.co/unsloth/MiniMax-M2-GGUF/blob/main/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf).
The existing `arch_registry_network` test re-reads this witness with every
other planned row. The source revision is floating `main`.

```sh
cargo run -p turbospark-cli --bin turbospark-model -- probe \
  unsloth/MiniMax-M2-GGUF \
  --file Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf \
  --sidecar-repo MiniMaxAI/MiniMax-M2
```

Start with GGUF intake. The original checkpoint uses FP8 block quantization;
that is a separate source-format project. Do not infer M2.1 or later model
support from this witness. MTP and vision are outside this initial scope.

## Checkpoint facts

The [publisher's config](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json)
was read on 2026-09-10. HF spells the type `minimax_m2`.

| Property | Value |
| --- | --- |
| Trunk layers / hidden width | 62 / 3072 |
| Attention | Full attention throughout; no sliding window |
| Q heads / KV heads / head width | 48 / 8 / 128 |
| Rotary width / theta | 64 / 5000000 |
| Experts / selected / expert FFN width | 256 / 8 / 1536 |
| Shared expert width | 0 |
| RMS epsilon | 1e-6 |
| Vocabulary / trained context | 200064 / 196608 |
| Embedding head | Untied |

The explicit head width matters: hidden width divided by Q heads is 64,
which is not this checkpoint's 128. `mlp_intermediate_size` is not the routed
expert width. The eight selected experts determine the minimum slot count.

## Layer contract

Two independent implementations agree:
[MLX MiniMax](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py)
and [llama.cpp MiniMax-M2](https://github.com/ggml-org/llama.cpp/blob/master/src/models/minimax-m2.cpp).
These links were inspected on 2026-09-10 and float with upstream.

Attention uses separate bias-free Q/K/V projections. Learned Q/K RMS norms
span the entire projected vectors, before the heads are reshaped: widths
6144 and 1024. They are not per-head norms. V is not normalized. Rotate the
leading 64 coordinates of each 128-wide head using split-half pairing;
attention scales by `1/sqrt(128)`.

Each block is pre-norm attention plus residual, then pre-norm routed SwiGLU
plus residual. There are no sandwich norms, shared expert, or output gate.
The final learned RMS norm feeds an untied linear head returning raw logits.

MLX makes the routing distinction explicit:

```text
scores = sigmoid(router(x))
selected = top8(scores + correction_bias)
weights = scores[selected] / sum(scores[selected])
output = sum(weights * selected_expert_outputs)
```

The correction bias changes selection only. Putting it into the weights, or
substituting softmax, changes the model. Preserve router ranking when reducing
expert outputs, following this repo's existing deterministic slot contract.

## Remaining bring-up gates

1. Inventory every tensor and block type across all GGUF shards. Pin the
   Q/K norm layouts and routing-bias name against those actual tensors.
2. Calculate slot bytes from the actual expert block types and padding,
   then resident and KV bytes at the intended context. No footprint estimate
   or streaming-fit verdict is established by this recognition change.
3. Add the manifest family and baseline only with complete behavioral
   fields. Map tensors and verify the resulting install structurally.
4. Implement and independently parity-test whole-projection normalization,
   partial RoPE wiring, and selection-bias sigmoid routing. Reuse existing
   primitives only where those tests prove equivalence.
5. Wire the decode flow and reset behavior. Resolve EOS and chat framing
   from the checkpoint sidecars. Run greedy and sampled real-model smokes,
   the family memory oracle, and quality gate before claiming execution.
6. Add a catalog row only after that exact artifact installs and generates
   here, following [NEW_MODEL.md](NEW_MODEL.md).

The offline registry regression exercises `arch_from_gguf` with an
architecture-only header: MiniMax must be recognized and refused before
shape parsing. It was observed failing without the row and passing with it.
