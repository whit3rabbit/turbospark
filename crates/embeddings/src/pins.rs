//! Pinned checkpoints for every ported family.
//!
//! Each MLX conversion is paired with the Hugging Face checkpoint its
//! model card says it was converted from (the quantized EmbeddingGemma
//! repos come from the QAT checkpoints, not the base model), and every
//! repository is pinned to an immutable commit.
//! A floating `main` can change weights under a frozen parity fixture.
//! Verification status lives in `MODELS.md`, not here: this table only
//! says where the bytes are.

/// The embedding families this crate implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// Gemma3 bidirectional text encoder + mean pooling + Dense head.
    EmbeddingGemma,
    /// SigLIP2 NaFlex dual tower (text + variable-resolution vision).
    Siglip2Naflex,
    /// CLAP HTSAT-unfused dual tower (RoBERTa text + HTSAT audio).
    Clap,
    /// BERT and XLM-RoBERTa sentence encoders (Snowflake arctic-embed).
    Bert,
    /// Alibaba GTE ("new") encoder: BERT with rotary positions
    /// (arctic-embed-m-v2.0). Reference code ships in the repository.
    Gte,
    /// Nomic BERT: rotary positions, SwiGLU, dynamic NTK scaling
    /// (arctic-embed-m-long). Reference code ships in the repository.
    NomicBert,
}

/// Weight encoding of a pinned checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Unquantized safetensors (F32, F16, or BF16; loaded to f32).
    Dense,
    /// MLX groupwise affine quantization.
    MlxAffine { bits: u32, group_size: usize },
}

/// One immutable checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedCheckpoint {
    pub family: Family,
    pub repo: &'static str,
    /// Full 40-hex commit. For CLAP this is the `refs/pr/3` safetensors
    /// conversion commit, because the upstream `main` ships only a pickle.
    pub revision: &'static str,
    pub format: WeightFormat,
    /// For an MLX conversion, the Hugging Face checkpoint it came from.
    /// `None` for an upstream checkpoint.
    pub converted_from: Option<&'static str>,
    /// Hugging Face requires accepting a license before download.
    pub gated: bool,
}

const fn upstream(
    family: Family,
    repo: &'static str,
    revision: &'static str,
    gated: bool,
) -> PinnedCheckpoint {
    PinnedCheckpoint {
        family,
        repo,
        revision,
        format: WeightFormat::Dense,
        converted_from: None,
        gated,
    }
}

const fn mlx(
    family: Family,
    repo: &'static str,
    revision: &'static str,
    format: WeightFormat,
    from: &'static str,
) -> PinnedCheckpoint {
    PinnedCheckpoint {
        family,
        repo,
        revision,
        format,
        converted_from: Some(from),
        gated: false,
    }
}

const fn q(bits: u32) -> WeightFormat {
    WeightFormat::MlxAffine {
        bits,
        group_size: 64,
    }
}

const GEMMA: &str = "google/embeddinggemma-300m";
const GEMMA_QAT4: &str = "google/embeddinggemma-300m-qat-q4_0-unquantized";
const GEMMA_QAT8: &str = "google/embeddinggemma-300m-qat-q8_0-unquantized";
const ARCTIC_L_V2: &str = "Snowflake/snowflake-arctic-embed-l-v2.0";

/// Every pinned checkpoint. Upstream entries come first per family.
pub const PINS: &[PinnedCheckpoint] = &[
    upstream(
        Family::EmbeddingGemma,
        GEMMA,
        "57c266a740f537b4dc058e1b0cda161fd15afa75",
        true,
    ),
    upstream(
        Family::EmbeddingGemma,
        GEMMA_QAT4,
        "7740c61c3ad98a9deb8423a20388ab6c3c5c45f2",
        true,
    ),
    upstream(
        Family::EmbeddingGemma,
        GEMMA_QAT8,
        "7b5b24595322ab0ea4d08827066860a6df8cb0aa",
        true,
    ),
    mlx(
        Family::EmbeddingGemma,
        "mlx-community/embeddinggemma-300m-bf16",
        "2f420efb007317a1ef2aa1608f87ef24226e6807",
        WeightFormat::Dense,
        GEMMA,
    ),
    mlx(
        Family::EmbeddingGemma,
        "mlx-community/embeddinggemma-300m-8bit",
        "c8316c9e35cf8830541b04d315398c67faa6e497",
        q(8),
        GEMMA_QAT8,
    ),
    mlx(
        Family::EmbeddingGemma,
        "mlx-community/embeddinggemma-300m-6bit",
        "e03111914aba981cacdd4a4c3a920d5064d9e33a",
        q(6),
        GEMMA_QAT8,
    ),
    mlx(
        Family::EmbeddingGemma,
        "mlx-community/embeddinggemma-300m-5bit",
        "717843cb884a1ee6ca617ee6b84ab26546106583",
        q(5),
        GEMMA_QAT8,
    ),
    mlx(
        Family::EmbeddingGemma,
        "mlx-community/embeddinggemma-300m-4bit",
        "5d9ef074df3957afc5c77127f208fddbc3c54187",
        q(4),
        GEMMA_QAT4,
    ),
    mlx(
        Family::EmbeddingGemma,
        "mlx-community/embeddinggemma-300m-qat-q4_0-unquantized-bf16",
        "41b4b86661761a34f2bc275661ba67319dd83409",
        WeightFormat::Dense,
        GEMMA_QAT4,
    ),
    mlx(
        Family::EmbeddingGemma,
        "mlx-community/embeddinggemma-300m-qat-q8_0-unquantized-bf16",
        "2aeb9554ff157247ed06a41e61e8339e7cae9c22",
        WeightFormat::Dense,
        GEMMA_QAT8,
    ),
    upstream(
        Family::Siglip2Naflex,
        "google/siglip2-base-patch16-naflex",
        "b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
        false,
    ),
    upstream(
        Family::Clap,
        "laion/clap-htsat-unfused",
        "79b58ed25fc00386262a2bea4b19fd21dc4310a0",
        false,
    ),
    upstream(
        Family::Bert,
        ARCTIC_L_V2,
        "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
        false,
    ),
    upstream(
        Family::Bert,
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        "e58a8f756156a1293d763f17e3aae643474e9b8a",
        false,
    ),
    upstream(
        Family::Bert,
        "Snowflake/snowflake-arctic-embed-l",
        "d8fb21ca8d905d2832ee8b96c894d3298964346b",
        false,
    ),
    upstream(
        Family::Bert,
        "Snowflake/snowflake-arctic-embed-m",
        "fc74610d18462d218e312aa986ec5c8a75a98152",
        false,
    ),
    upstream(
        Family::Bert,
        "Snowflake/snowflake-arctic-embed-s",
        "e596f507467533e48a2e17c007f0e1dacc837b33",
        false,
    ),
    upstream(
        Family::Bert,
        "Snowflake/snowflake-arctic-embed-xs",
        "d8c86521100d3556476a063fc2342036d45c106f",
        false,
    ),
    mlx(
        Family::Bert,
        "mlx-community/snowflake-arctic-embed-l-v2.0-bf16",
        "3da10e38e7875d6501adc14665c95b6708646121",
        WeightFormat::Dense,
        ARCTIC_L_V2,
    ),
    mlx(
        Family::Bert,
        "mlx-community/snowflake-arctic-embed-l-v2.0-8bit",
        "98033f618b7503c73aa857bc0656f763a1587e73",
        q(8),
        ARCTIC_L_V2,
    ),
    mlx(
        Family::Bert,
        "mlx-community/snowflake-arctic-embed-l-v2.0-6bit",
        "c8125c01b2142f0065f2494c123a7df46f9ef08e",
        q(6),
        ARCTIC_L_V2,
    ),
    mlx(
        Family::Bert,
        "mlx-community/snowflake-arctic-embed-l-v2.0-4bit",
        "430b52bb3dcb9ad2404b1ac2b60f2fdc00f26033",
        q(4),
        ARCTIC_L_V2,
    ),
    upstream(
        Family::Gte,
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "95c2741480856aa9666782eb4afe11959938017f",
        false,
    ),
    upstream(
        Family::NomicBert,
        "Snowflake/snowflake-arctic-embed-m-long",
        "92d97331f1f4b6a366c1f161354b9f3390cc219f",
        false,
    ),
];

/// Pins for one family, upstream first.
pub fn pins_for(family: Family) -> impl Iterator<Item = &'static PinnedCheckpoint> {
    PINS.iter().filter(move |p| p.family == family)
}

/// Looks up a pin by repository id.
pub fn pin(repo: &str) -> Option<&'static PinnedCheckpoint> {
    PINS.iter().find(|p| p.repo == repo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revisions_are_full_commits() {
        for p in PINS {
            assert_eq!(p.revision.len(), 40, "{}", p.repo);
            assert!(
                p.revision
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "{}",
                p.repo
            );
        }
    }

    #[test]
    fn every_conversion_names_a_pinned_upstream() {
        for p in PINS {
            if let Some(from) = p.converted_from {
                let up =
                    pin(from).unwrap_or_else(|| panic!("{} upstream {from} not pinned", p.repo));
                assert_eq!(up.family, p.family);
                assert!(up.converted_from.is_none());
            }
        }
    }

    #[test]
    fn every_family_has_an_upstream_pin() {
        for family in [
            Family::EmbeddingGemma,
            Family::Siglip2Naflex,
            Family::Clap,
            Family::Bert,
            Family::Gte,
            Family::NomicBert,
        ] {
            let first = pins_for(family).next().expect("family pinned");
            assert!(first.converted_from.is_none());
        }
    }

    #[test]
    fn repos_are_unique() {
        for (i, a) in PINS.iter().enumerate() {
            assert!(PINS[i + 1..].iter().all(|b| b.repo != a.repo), "{}", a.repo);
        }
    }
}
