# catalog recommend

## Overview
Answers "what should this machine run?". Ranks the curated model rows (vision-tower rows are dropped, not fit candidates), and
optionally Hugging Face at large, by whether a candidate fits in memory and how
strong the evidence behind it is. The default arm touches no network.

## Interface
- `fit`, `context_ladder`, `Shape`, `CountedSource`, `Fit`, `FitVerdict`, `LadderRung` [src/recommend/fit.rs:fit, src/recommend/fit.rs:context_ladder, src/recommend/fit.rs:Fit, src/recommend/fit.rs:FitVerdict].
- `rank`, `name_params_hint`, `Evidence`, `Key` [src/recommend/rank.rs:rank, src/recommend/rank.rs:name_params_hint, src/recommend/rank.rs:Evidence].
- `discover`, `DiscoverOptions`, `gguf_variants`, `gguf_variants_from_files`, `GgufVariant`, `GgufVariants` [src/recommend/discover.rs:discover, src/recommend/discover.rs:gguf_variants].
- `Machine`, `Origin`, `ThroughputBand`, `Recommendation` [src/recommend/mod.rs:Machine, src/recommend/mod.rs:Origin, src/recommend/mod.rs:Recommendation].
- `recommend_catalog`, `recommend_catalog_probed`, `recommend_catalog_probed_with_progress`, `rank_recommendations`, `probe_entry`, `from_entry` [src/recommend/mod.rs:recommend_catalog, src/recommend/mod.rs:recommend_catalog_probed, src/recommend/mod.rs:rank_recommendations].

## Invariants
- Evidence outranks everything except fitting: an unrun discovered repo sorts below a curated row [src/recommend/rank.rs: module docs, `Evidence::of`].
- Fit is judged on a working set, not `phys_footprint` or bits-per-weight [src/recommend/fit.rs: module docs].
- A discovered row is refused for the same reason as the probe, since every candidate passes through the unchanged probe [src/recommend/discover.rs: module docs].
- The `Tight` threshold is a fraction of the budget [src/recommend/fit.rs:TIGHT_FRACTION].
- Context ladder windows are fixed [src/recommend/fit.rs:LADDER_WINDOWS].
- Quantization choice picks the largest `.gguf` at or under the physical budget. `QUANT_LADDER` only filters (a file needs a ladder rank) and breaks size ties. Sharded files (`-of-0`) are excluded. When nothing fits it falls back to the smallest runnable file [src/recommend/discover.rs:choose_quantization].

## Side Effects
- `recommend_catalog` performs none. The `_probed` variants and `discover` use `Client` network requests [src/recommend/mod.rs:probe_entry, src/recommend/discover.rs:discover].

## Verification
- Run: `cargo test -p turbospark-catalog`
