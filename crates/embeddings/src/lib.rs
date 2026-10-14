//! Portable embedding models and the sentence-transformers post-processing
//! they share.
//!
//! The model families follow the mlxrs `embeddings` module (findit-studio)
//! as the list of what to port, with the Hugging Face `transformers` and
//! `sentence-transformers` implementations as the numeric reference:
//!
//! - [`embeddinggemma`]: `google/embeddinggemma-300m`, a Gemma3
//!   bidirectional text encoder with mean pooling and a Dense head. Also
//!   loads the mlx-community bf16 and affine-quantized conversions.
//! - [`siglip2`]: `google/siglip2-base-patch16-naflex`, the SigLIP2 dual
//!   tower with variable-resolution (NaFlex) image input.
//! - [`clap`]: `laion/clap-htsat-unfused`, the CLAP dual tower (RoBERTa
//!   text, HTSAT audio).
//! - [`bert`]: BERT and XLM-RoBERTa sentence encoders, covering the
//!   Snowflake arctic-embed v1, v1.5, and l-v2.0 checkpoints and the
//!   mlx-community l-v2.0 conversions.
//! - [`gte`]: the GTE rotary encoder behind arctic-embed-m-v2.0.
//! - [`nomic_bert`]: the Nomic BERT encoder behind arctic-embed-m-long.
//!
//! Shared pieces: [`pooling`] and [`st_config`] (sentence-transformers
//! modules), [`normalize`] (L2 and Matryoshka truncation), [`similarity`]
//! (cosine, ColVision single-vector and MaxSim scoring), [`weights`]
//! (shape-checked safetensors access with MLX dequantization), and
//! [`pins`] (the immutable checkpoint revisions every family is tested
//! against).
//!
//! Everything is plain f32 on the CPU and builds off macOS. Kernels come
//! from `turbospark-audio`'s portable ops so a fix lands once.

#![forbid(unsafe_code)]

pub mod error;
pub mod normalize;
pub mod pins;
pub mod pooling;
pub mod similarity;
pub mod st_config;
pub mod tokenize;
pub mod weights;

#[cfg(test)]
pub(crate) mod testutil;

#[cfg(feature = "bert")]
pub mod bert;
#[cfg(feature = "clap")]
pub mod clap;
#[cfg(feature = "embeddinggemma")]
pub mod embeddinggemma;
#[cfg(feature = "gte")]
pub mod gte;
#[cfg(feature = "nomic-bert")]
pub mod nomic_bert;
#[cfg(feature = "siglip2")]
pub mod siglip2;

pub use error::EmbeddingError;
pub use normalize::{l2_normalize, truncate_and_normalize};
pub use pins::{Family, PinnedCheckpoint, WeightFormat, PINS};
pub use pooling::{pool, PoolingMode};
pub use similarity::{
    cosine_similarity, score_multi_vector, score_single_vector, similarity_matrix,
};
pub use st_config::PoolingConfig;

pub type Result<T> = std::result::Result<T, EmbeddingError>;
