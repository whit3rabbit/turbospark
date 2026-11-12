# GGUF Header Parser

## Overview
Parses the GGUF v3 header (24-byte preamble, metadata key/values, tensor table) from the leading bytes of a file into a `GgufHeader`. It is pure and bounds-checked: input comes from an untrusted (often ranged, partial) fetch, and a `TooShort` error reports the absolute offset needed so a caller can fetch exactly that much and retry. `ggml.rs` holds the type-id to name and block-size tables used to size tensors.

## Interface
- `parse_header` [parser.rs:parse_header]: `(leading_bytes, max_bytes) -> GgufHeader`, validating magic, version, counts, dims, alignment
- `GgufHeader` [types.rs:GgufHeader]: version, metadata map, tensor map, alignment, `data_region_start`; helpers `absolute_range`, `metadata_str/u64/f64`, `architecture`
- `GgufTensorInfo` [types.rs:GgufTensorInfo]: `ggml_type`, `dims`, `offset`; `element_count`, `byte_size`
- `GgufValue` [types.rs:GgufValue]: metadata value enum with `as_u64/as_f64/as_str/as_bool/as_array`
- `GgufHeaderError` [types.rs:GgufHeaderError]: TooShort, BadMagic, UnsupportedVersion, HeaderTooLarge, BadValueType, InvalidUtf8, ArrayTooDeep, DuplicateKey/Tensor, BadDimensions, BadAlignment, MisalignedTensor, UnsupportedType, UnknownType, RaggedTensor, Overflow
- `ggml_type_name` [ggml.rs:ggml_type_name]: id to human name, verbatim from `ggml.h`
- `ggml_type_block` [ggml.rs:ggml_type_block]: id to `(elements per block, bytes per block)`, deliberately partial
- Constants `DEFAULT_ALIGNMENT` (32), `DEFAULT_MAX_HEADER_BYTES` (64 MiB), `MAX_ARRAY_DEPTH` (64), `SUPPORTED_VERSION` (3) [types.rs]

## Invariants
- Only version 3 and magic `GGUF` are accepted [parser.rs:parse_header]
- Every read is bounds-checked against both the buffer and `max_bytes`; no read past either [parser.rs:Cursor::take]
- Counts are rejected before allocation when the minimum possible size (32 B per tensor, 13 B per KV) exceeds `max_bytes` [parser.rs:parse_header]
- Array nesting is capped at `MAX_ARRAY_DEPTH` to prevent stack overflow, and array length is capped by `max_bytes` [parser.rs:Cursor::value]
- Tensors need 1..=4 non-zero dims; duplicate keys and tensor names are errors [parser.rs:parse_header]
- Alignment must be a nonzero power of two and every tensor offset a multiple of it [parser.rs:parse_header]
- `byte_size` and `absolute_range` use checked arithmetic and reject element counts not divisible by the block size [types.rs:GgufTensorInfo::byte_size, GgufHeader::absolute_range]
- `ggml_type_block` lists only types whose block size was verified; unlisted types error rather than guess (a wrong constant would silently misalign later tensors). A row buys parsing, not execution [ggml.rs:ggml_type_block]

## Side Effects
- None (pure functions over byte slices)

## Verification
- Run: `cargo test -p turbospark-repack`
