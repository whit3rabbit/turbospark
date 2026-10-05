//! File extension classification, shared by the importer and the picker.
//!
//! ONE list. The app asks this crate (through `ts_audio_capabilities_json`)
//! instead of keeping its own copy, so a codec added here appears in the
//! file picker without a second edit that could drift.

/// What the importer should do with a file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileClass {
    /// A decoder here reads it.
    Supported,
    /// Recognized as audio, but nothing here decodes it. Refused with a
    /// sentence instead of a generic "could not extract text".
    Refused,
    /// Not audio at all; the document path handles it.
    NotAudio,
}

/// Extensions the enabled symphonia readers and codecs decode.
///
/// `ogg`/`oga` are listed because Ogg Vorbis decodes; an Ogg file carrying
/// Opus opens and then fails with [`crate::AudioError::Unsupported`] at the
/// codec, which names the problem just as well.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "aac", "aif", "aifc", "aiff", "caf", "flac", "m4a", "mka", "mp3", "oga", "ogg", "wav", "wave",
];

/// Audio formats with no decoder in this build.
pub const REFUSED_EXTENSIONS: &[&str] = &["opus", "webm", "weba", "wma", "amr", "ac3"];

/// Classifies a file extension, case-insensitively, without a leading dot.
pub fn classify_extension(extension: &str) -> FileClass {
    let lowered = extension.trim_start_matches('.').to_ascii_lowercase();
    if SUPPORTED_EXTENSIONS.contains(&lowered.as_str()) {
        FileClass::Supported
    } else if REFUSED_EXTENSIONS.contains(&lowered.as_str()) {
        FileClass::Refused
    } else {
        FileClass::NotAudio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_case_insensitive_and_ignores_the_dot() {
        assert_eq!(classify_extension("WAV"), FileClass::Supported);
        assert_eq!(classify_extension(".m4a"), FileClass::Supported);
        assert_eq!(classify_extension("Opus"), FileClass::Refused);
        assert_eq!(classify_extension("pdf"), FileClass::NotAudio);
        assert_eq!(classify_extension(""), FileClass::NotAudio);
    }

    #[test]
    fn no_extension_is_both_supported_and_refused() {
        for ext in SUPPORTED_EXTENSIONS {
            assert!(!REFUSED_EXTENSIONS.contains(ext), "{ext} is in both lists");
        }
    }
}
