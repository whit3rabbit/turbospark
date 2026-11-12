# arch_config (model-io)

## Overview
The architecture description types: the `ModelFamily` enum with its canonical persisted strings, `ArchConfig`, and the family-dependent sub-configs (RoPE scaling, linear attention, vision, compressed attention, MLA, hyper-connections, PLE).

## Interface
- `ModelFamily` with `as_str`, `parse` [arch_config/family.rs:ModelFamily]
- `ArchConfig` with `decode_int4_gemv_shapes`, `decode_int8_gemv_shapes`, `layer_is_full`, `layer_is_linear`, `layer_is_csa`, `layer_is_mla`, `layer_is_hca`, `layer_is_compressed`, `layer_is_short_conv`, `layer_is_hash_routed`, `has_*_layers` [arch_config/config.rs:ArchConfig]
- `RopeScalingConfig`, `LinearAttentionConfig`, `VisionConfig`, `VisionTowerKind`, `CompressedAttentionConfig`, `MlaConfig`, `HyperConnectionConfig`, `PleConfig`, `MAX_VISION_DEEPSTACK_MERGERS` [arch_config/sub_configs.rs]

## Invariants
- `ModelFamily::as_str` and `parse` are the persisted family strings and must round-trip [arch_config/family.rs:ModelFamily::parse]
- At most `MAX_VISION_DEEPSTACK_MERGERS` (3) deepstack mergers [arch_config/sub_configs.rs:MAX_VISION_DEEPSTACK_MERGERS]
- Layer-kind predicates are driven by the layer mask, not tensor names [arch_config/config.rs:layer_is_full]

## Side Effects
- None

## Verification
- Run: `cargo test -p turbospark-model-io`
