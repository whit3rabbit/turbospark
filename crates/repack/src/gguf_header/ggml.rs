//! Helper functions mapping ggml type IDs to names and block configurations.

/// Human-readable name for a ggml type id, for error messages. Ids and gaps
/// are verbatim from `ggml/include/ggml.h`'s `enum ggml_type` (4 and 5 were
/// removed types and stay unused).
pub fn ggml_type_name(id: u32) -> Option<&'static str> {
    Some(match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        _ => return None,
    })
}

/// `(elements per block, bytes per block)` for a ggml type.
///
/// DELIBERATELY PARTIAL. Only types whose block size was read off the ggml
/// spec or source are listed; everything else returns `None` and parses to
/// [`crate::gguf_header::types::GgufHeaderError::UnsupportedType`], which names the type. A guessed
/// constant here would not fail loudly, it would silently misalign every
/// tensor after the first one of that type. Adding a type is one row plus
/// the source citation.
///
/// A row here buys PARSING, not execution: whether an install of that type
/// runs is decided separately by [`model_io::EXECUTABLE_GGUF_TYPES`] and by
/// `RealForwardRunner::open` (AGENTS.md Gotcha 29). Q2_K/Q3_K/Q5_K and the
/// three IQ types are listed so a mixed sub-4-bit file can be HEADER-PROBED
/// for scoping (ROADMAP Phase S); none of the six has a kernel. The IQ rows
/// in particular size a tensor without being able to READ one: IQ3_XXS and
/// IQ4_XS decode through codebooks, which is a different job from every
/// affine and K-quant unpacker in this port.
///
/// The K-quant and IQ rows were read out of ggml itself rather than off a
/// spec page, which is one command against brew's `llama.cpp` (b10310 here):
///
/// ```text
/// // cc -I/opt/homebrew/include x.c -L/opt/homebrew/lib -lggml -lggml-base
/// for (int i = 0; i < GGML_TYPE_COUNT; i++)
///     printf("%s %lld %lld\n", ggml_type_name(i),
///            (long long) ggml_blck_size(i), (long long) ggml_type_size(i));
/// ```
pub fn ggml_type_block(id: u32) -> Option<(u64, u64)> {
    Some(match id {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        18 => (256, 98),  // IQ3_XXS
        20 => (32, 18),   // IQ4_NL
        23 => (256, 136), // IQ4_XS
        // ROADMAP M5 Phase 0. Parse-only, like the six above it: gpt-oss
        // ships its routed experts as MXFP4 and nothing else does, so
        // without this row the candidate's expert table sizes to ZERO and
        // the survey reports the model as 1.8 GiB of Q8_0 bystanders. That
        // is exactly the UNSIZED failure Phase S hit from the other side
        // (AGENTS.md Gotcha 29), and it ranks the type carrying ~85% of the
        // weights last in a share column.
        39 => (32, 17), // MXFP4
        24 => (1, 1),   // I8
        25 => (1, 2),   // I16
        26 => (1, 4),   // I32
        27 => (1, 8),   // I64
        28 => (1, 8),   // F64
        30 => (1, 2),   // BF16
        _ => return None,
    })
}
