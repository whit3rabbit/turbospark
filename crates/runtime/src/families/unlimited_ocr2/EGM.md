# Unlimited OCR2 Family

## Overview
Decode flow for the `unlimited_ocr2` family: a DeepSeek-V2-shaped MoE decoder with DASH block-sparse attention over page2d RoPE, plus an image tower for page inputs. Each layer commits twice, once so the host can read the normed row and route, and once so it can read the expert gate. Prefill uses the same per-token path as decode. `state.rs` allocates the per-install and per-generation router state, and `moe.rs` encodes the routed and shared experts.

## Interface
- `layer_tensor` [mod.rs:layer_tensor]: canonical `language_model.model.layers.N.` tensor name
- `RealUnlimitedOcr2State` [state.rs:RealUnlimitedOcr2State]: `build`, `reset`, `set_prompt`, `clear_prompt`, `angles_for`
- `produce_real_unlimited_ocr2` [mod.rs:produce_real_unlimited_ocr2]: per-token decode on `RealForwardRunner`
- `set_unlimited_ocr2_prompt`, `attach_unlimited_ocr2_pages`, `clear_unlimited_ocr2_prompt` [mod.rs]: install or drop the image prompt
- `encode_unlimited_ocr2_page`, `unlimited_ocr2_image_ids`, `unlimited_ocr2_last_route`, `unlimited_ocr2_view_stages` [mod.rs]: tower encode, ids, and parity-gate probes
- `encode_layer_moe` [moe.rs:encode_layer_moe]: shared expert, softmax over all experts with top-k and no renorm, routed phases 1 and 2, residual add
- Subdirectories: None

## Invariants
- `build` refuses another family, sandwich norms, router scaling, embedding scaling, output gate, gated shared expert, K=V, a softcap, any non-full layer mask, mismatched head geometry, a dense lead count of 0 (`num_dense_leading_layers`; layer 0 is read as dense), no MoE layers (lead count at or above layer count), a non-positive dense lead intermediate size, and `top_k` of zero or above `MAX_STREAMED_EXPERTS` [state.rs:build]
- Required tensors are named at open so a gap fails early [state.rs:build]
- Router tensor sizes are checked against the expected element counts [state.rs:build]
- Position must equal `kv.position()` and token must be < vocab [mod.rs:produce_unlimited_ocr2_inner]
- Inside the prompt, the fed token must match the installed prompt id [mod.rs:produce_unlimited_ocr2_inner]
- `set_prompt` requires exactly one embedding row per image token [state.rs:set_prompt]
- There is no prefix reuse or rollback; `reset` rewinds to empty and every generation prefills [state.rs:reset]
- Slot order is the router's ranking order [moe.rs:encode_layer_moe]
- Logits are raw, with no softcap; vocab must match the caller's buffer [mod.rs:produce_unlimited_ocr2_inner]

## Side Effects
- Metal pass encoding, two or more commits per layer, GPU buffer writes for angles, positions, bias and count [mod.rs:produce_unlimited_ocr2_inner, route_layer]
- Advances the KV cache [mod.rs:produce_unlimited_ocr2_inner]
- Updates `phases` timing and expert counters [mod.rs, moe.rs]
- Pread expert streaming, or mapped-residency offsets [moe.rs:encode_layer_moe]
- Opens the vision tower file lazily on the first image [mod.rs:take_unlimited_ocr2_tower]
- Loads DASH config from the install directory [state.rs:build]

## Verification
- Run: `cargo test -p turbospark-runtime --test real_forward_unlimited_ocr2` (synthetic install; a name filter does not select it)
