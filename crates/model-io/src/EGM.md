# model-io (src)

## Overview
Model install layout for `turbospark-model-io`: manifest and architecture validation, packed-expert layout, resident tensor index, SHA-256 verification, install and speech receipts, load-guard and context/expert-cache sizing policy, KV quantization geometry, mmap-backed resident buffers and safetensors access. A narrow amount of `unsafe` is allowed (`mmap`, `statvfs`, `sysconf`).

## Interface
- Crate root re-exports [lib.rs:pub use]
- Architecture: `ArchConfig`, `ModelFamily`, `known_architecture`, `all_known_architectures` (see children below)
- Manifest: `Manifest`, `load_manifest`, `validate_manifest`, `peek_family`, `REQUIRED_FILES` (see children below)
- Integrity: `ModelIntegrityPolicy`, `VerifiedInstallReceipt`, `load_install_receipt`, `validate_install_receipt`, `validate_install_receipt_manifest_binding` [install_receipt.rs:ModelIntegrityPolicy]; `hash_file`, `hash_data`, `verify_file`, `hash_file_observed` [sha256.rs:hash_file]
- Layout and residency: `PackedExpertsLayout`, `load_packed_experts_layout` [packed_experts_layout.rs]; `ResidentIndex`, `load_resident_index`, `INDEX_MAGIC`, `HEADER_BYTES`, `ENTRY_BYTES`, `TRAILER_BYTES`, `INDEX_FORMAT_VERSION` [resident_index.rs]; `ResidentBuffer` [resident_buffer.rs:ResidentBuffer]; `SafetensorsFile` [safetensors.rs:SafetensorsFile]
- Policy: `resolve_max_context`, `committed_bytes`, `kv_bytes_for_context`, `CONTEXT_*` constants [context_policy.rs]; `ExpertResidency`, `ExpertCacheSlots`, `HEADROOM_*` constants [expert_cache_policy.rs]; `LoadGuard`, `LoadPolicy`, `GuardBudget` [load_guard.rs]; `KvQuant`, `kv_layer_strides` [kv_quant.rs]
- Probes: `available_bytes` [disk_space.rs:available_bytes], `probe_cgroup_memory`, `parse_cgroup_limit` [cgroup.rs]
- Sidecars and model-specific config: `DashConfig` [dash_config.rs], `EncoderConfig` [encoder_config.rs], `WhisperConfig`, `WhisperSpecialTokens` [whisper_config.rs], `SpeechFamily` [speech_family.rs], `SpeechInstallReceipt` [speech_receipt.rs], `SteeringSet`, `LayerDirection` [steering_set.rs], `NgramTableLayout`, `load_ngram_table_layout`, `ple_ngram_rows`, `NgramContext` [ngram_table.rs, ngram_hash.rs], `SidecarRecord`, `load_vision_sidecar` [vision_sidecar.rs], `trunk_role::{has_marker, read, write_headless}` [trunk_role.rs]; file-name constants `SPEECH_RECEIPT_FILENAME`, `SIDECAR_RECORD_FILE`, `DASH_SIDECAR` [lib.rs:pub use], `TRUNK_ROLE_FILE` [trunk_role.rs]
- `ModelError` [error.rs:ModelError]; `TokenId` re-export [lib.rs:TokenId]
- Children: [arch_baselines](./arch_baselines/EGM.md), [arch_config](./arch_config/EGM.md), [manifest](./manifest/EGM.md)

## Invariants
- Unsafe is limited to `mmap` in `resident_buffer` and `safetensors`, plus the single `statvfs` call and the `sysconf(_SC_PAGESIZE)` call [resident_buffer.rs:unsafe_map] [safetensors.rs:SafetensorsFile] [disk_space.rs:available_bytes] (statvfs); the `sysconf` call is in [resident_buffer.rs:page_size_bytes]
- Metadata reads are size-capped by reading one byte past the cap [bounded_read.rs:read_capped]
- Receipts are not tamper protection; the receipt lives in the directory it describes [install_receipt.rs:ModelIntegrityPolicy]
- Index layout constants live here and repack imports them; readers refuse newer versions [resident_index.rs:INDEX_FORMAT_VERSION] [error.rs:ModelError::UnsupportedIndexFormat]
- Expert strides must be a multiple of 4096 bytes, a floor and not the OS page size (16 KiB on Apple Silicon; the writer aligns to 16 KiB) [manifest/mod.rs:page_size_bytes] [error.rs:ModelError::ExpertStrideNotPageAligned]
- Token ids cross crates as `TokenId` [lib.rs:TokenId]

## Side Effects
- Reads manifests, receipts, indexes and safetensors from the install directory; memory-maps weight files [resident_buffer.rs:ResidentBuffer] [safetensors.rs:SafetensorsFile]
- Writes `speech` receipt, DASH sidecar and trunk role marker files [speech_receipt.rs:SpeechInstallReceipt::write_to_dir] [dash_config.rs:DashConfig::write_sidecar] [trunk_role.rs:write_headless]
- Calls `statvfs` and `sysconf` through libc [disk_space.rs:available_bytes] [resident_buffer.rs:page_size_bytes]
- Reads `/sys/fs/cgroup/memory.max` and `memory.high` on any platform (no cfg gate; absent files give `None`) [cgroup.rs:probe_self]

## Verification
- Run: `cargo test -p turbospark-model-io`
