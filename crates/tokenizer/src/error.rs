//! Error taxonomy for the tokenizer crate. Ported from the error enums in
//! `Tokenization/Tokenizer.swift` and `Tokenization/GemmaToolCallParser.swift`.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    MissingSpecialToken(String),
    InvalidChatTemplate(String),
    UnsupportedForDialect(String),
    /// A caller-supplied pair of vision marker ids (`vision_start`,
    /// `image_pad`, from a `VisionConfig`) does not hold up against THIS
    /// tokenizer: either id fails to resolve to a real token, the two
    /// collide, or the checkpoint's own chat template does not place exactly
    /// one of each around a single image. See
    /// [`crate::MfTokenizer::verify_image_markers`].
    InvalidVisionMarkers(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::MissingSpecialToken(t) => {
                write!(f, "tokenizer missing required special token: {t}")
            }
            TokenizerError::InvalidChatTemplate(detail) => {
                write!(f, "invalid chat messages: {detail}")
            }
            TokenizerError::UnsupportedForDialect(operation) => write!(
                f,
                "operation is not supported for this tokenizer's chat dialect: {operation}"
            ),
            TokenizerError::InvalidVisionMarkers(detail) => {
                write!(f, "invalid vision marker ids: {detail}")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallParserError {
    Malformed,
    UnknownTool(String),
    Oversized,
}

impl fmt::Display for ToolCallParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolCallParserError::Malformed => write!(f, "malformed tool call"),
            ToolCallParserError::UnknownTool(name) => write!(f, "unknown tool: {name}"),
            ToolCallParserError::Oversized => write!(f, "tool call exceeds the size limit"),
        }
    }
}

impl std::error::Error for ToolCallParserError {}
