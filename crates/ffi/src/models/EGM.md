# ffi models

## Overview
Portable model-management surface behind the C ABI: catalog, installed listing, probe, Hugging Face search, install pipeline, hardware-fit recommendations, family capabilities, control vectors. Nothing here decodes, so it works on non-macOS hosts.

## Interface
- Catalog: [catalog.rs:catalog_json], [catalog.rs:installed_json]; images: [image_install.rs:catalog_json], [image_install.rs:install].
- Install: [install.rs:TS_INSTALL_STAGE], [install.rs:TS_INSTALL_BYTES], [install.rs:cancel_active_installs], [install.rs:refuse_while_installing], [install.rs:pause_active_installs], [install.rs:resume_active_installs], [install.rs:cancel_active_move].
- Probe and hub: [probe.rs:probe_json], [probe.rs:repo_variants_json], [hub.rs] (`hub_search_json`, per-repo compatibility check).
- Fit: [fit.rs:context_ladder_json], [fit.rs:recommend_json_with_progress].
- Capabilities: [capabilities.rs:kv_quant_supported], [capabilities.rs:family_capabilities_json].
- Control vectors: [control_vector.rs:control_vector_info_json].

## Invariants
- Capability predicates are answered once in Rust so hosts do not restate family sets [capabilities.rs] (module doc).
- Hub search results are ADVISORY; list limit defaults 20, max 50, minimum query 2 chars [hub.rs:DEFAULT_LIMIT], [hub.rs:MAX_LIMIT], [hub.rs:MIN_QUERY_CHARS].
- Store moves and library changes are refused while installs or a model move are running, with a fixed lock order (installs, then move) [install.rs:refuse_while_installing], [install.rs:refuse_while_installing_for].

## Side Effects
- Network: Hugging Face list and file requests, downloads into the model store [hub.rs], [install.rs], [probe.rs].
- Disk: writes install directories and reads manifests [install.rs], [image_install.rs].
- Global install registry used to cancel active installs [install.rs:cancel_active_installs].

## Verification
- Run: `cargo test -p turbospark-ffi`
