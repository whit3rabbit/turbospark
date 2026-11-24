//! C ABI entry points organized by domain.

mod core;
mod daemon;
mod embedding;
mod generate;
mod image;
mod libraries;
mod models;
mod ocr;
#[cfg(target_os = "macos")]
mod quantize;
#[cfg(not(target_os = "macos"))]
#[path = "quantize_unsupported.rs"]
mod quantize;
mod server;
mod session;
#[cfg(target_os = "macos")]
mod stt;
#[cfg(not(target_os = "macos"))]
#[path = "stt_unsupported.rs"]
mod stt;
mod tokenizer;

pub use core::*;
pub use daemon::*;
pub use embedding::*;
pub use generate::*;
pub use image::*;
pub use libraries::*;
pub use models::*;
pub use ocr::*;
pub use quantize::*;
pub use server::*;
pub use session::*;
pub use stt::*;
pub use tokenizer::*;

mod audio;
pub use audio::*;
mod audio_stream;
pub use audio_stream::*;
mod enhance_stream;
pub use enhance_stream::*;
