# catalog src

## Overview
Source of `turbospark-catalog`: three layers that answer three questions.
The curated `Catalog` says what has been run here, the `probe` says what could
be run (header-only), and `install` performs the walk that turns either into a
`.gturbo` directory. Beside them sit per-modality pinned catalogs (audio,
speech, embedding, OCR, image, music), the install `Store`, the Hugging Face
`Client` and `HubClient`, and the library, probe, and recommend subsystems.
Nothing here decodes, dispatches, or allocates a Metal buffer, so it builds on
every platform. The crate forbids unsafe code [src/lib.rs:forbid(unsafe_code)].

## Interface
- Crate re-exports are the public surface [src/lib.rs].
- Curated table: `Catalog`, `SCHEMA_VERSION`, `CatalogEntry`, `Source`, `Status`, `Measured`, `Sidecars` [src/catalog.rs:Catalog, src/entry.rs:CatalogEntry].
- Hugging Face: `Client`, `RepoRef`, `RepoFile`, `hf_endpoint`, `set_hf_endpoint_override` [src/hf.rs:Client]; tokens: `resolve_hf_token`, `validate_hf_token` [src/auth.rs:resolve_hf_token]; hub search: `HubClient`, `HubQuery`, `HubEntry` [src/hub.rs:HubClient]; metadata validation: `HubMetadataValidator` [src/hub_validation.rs:HubMetadataValidator].
- Compatibility: `check_repo`, `arch_hint`, `RepoCheck` [src/check.rs:check_repo]. Quant labels: `quant_label`, `group_variants` [src/quant.rs:group_variants].
- Install: `install`, `install_variant`, `install_with_byte_progress`, `gate`, `record`, `InstallPlan`, `INSTALL_CANCELLED`, `OWNED_SIDECAR_NAMES` [src/install.rs:install]. Streaming transport [src/stream.rs]. Pinned dirs: `PinnedAsset`, `PinnedDirReceipt` [src/pinned_dir.rs:PinnedAsset].
- Store: `Store`, `InstalledModel`, `default_root`, `resolve_model_arg`, `relocate_default_store` [src/store.rs:Store, src/store.rs:default_root].
- Modalities: `AudioCatalog` [src/audio_catalog.rs:AudioCatalog], `EmbeddingCatalog` [src/embedding.rs:EmbeddingCatalog], `OcrCatalog` [src/ocr.rs:OcrCatalog], `ImageCatalog` [src/image.rs:ImageCatalog], `install_music` [src/music.rs:install_music], `install_speech_receipt`, `probe_speech_dir`, `verify_speech_install` [src/speech.rs:verify_speech_install], `resolve_vision_sidecar` [src/vision.rs:resolve_vision_sidecar], Drama 3 conversion install [src/drama3_install.rs].
- Subdirectories: [probe](./probe/EGM.md), [recommend](./recommend/EGM.md), [library](./library/EGM.md).

## Invariants
- Embedded catalog JSON must match `SCHEMA_VERSION`, else load fails [src/catalog.rs:SCHEMA_VERSION].
- Embedded data files: models.json, tags.json, embedding_models.json, audio_assets.json, audio_families.json, image_models.json, ocr_models.json, music_models.json, speech_models.json [src/catalog.rs:EMBEDDED, src/audio_catalog.rs:ASSETS, src/speech.rs:EMBEDDED].
- Store root is the process-global override set by `relocate_default_store`, else `$TURBOSPARK_HOME`, else `$HOME/.turbospark` [src/store.rs:default_root].
- Hub metadata is untrusted and validated without owning any HTTP [src/hub_validation.rs: module docs].
- Cancellation surfaces as the exact error string `install cancelled` [src/install.rs:INSTALL_CANCELLED].
- Audio rows are pinned profiles whose pins come from a resolved commit with per-file SHA-256 [src/audio_catalog.rs: module docs, examples/audio_pin.rs].

## Side Effects
- Network to the Hugging Face endpoint (probe, file lists, downloads, token validation) [src/hf.rs:Client, src/auth.rs:validate_hf_token].
- Disk: writes model installs, install records, and registry files under the store root [src/install.rs:record, src/store.rs:Store].
- Reads `TURBOSPARK_HOME` and `HOME` [src/store.rs:default_root].

## Verification
- Run: `cargo test -p turbospark-catalog`
