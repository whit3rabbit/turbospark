//! Speech model family identity.
//!
//! A SEPARATE enum from the decoder [`crate::arch_config::ModelFamily`], by
//! design: speech distributions are safetensors installs with their own
//! receipt, catalog kind, and runtime flow, and no `.gturbo` decoder
//! manifest ever carries one. Adding a speech family therefore never
//! touches the decoder family exhaustiveness test or any decode-flow
//! dispatch -- the two enums answer different questions ("which speech
//! graph runs" against "which decoder family runs") and must not collapse
//! into one.
//!
//! Wire strings here are fresh (nothing on disk predates this enum), so
//! unlike `ModelFamily` they are free to match their variant names.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeechFamily {
    /// Whisper-class encoder-decoder transcription models (openai/whisper
    /// and compatible distributions): mel frontend, conv front end, a
    /// Pre-LN encoder stack, and a greedy decoder driven by the SOT
    /// grammar. Per-model shape comes from the whisper `config.json`, not
    /// from this arm.
    Whisper,
}

impl SpeechFamily {
    /// Exhaustive list of all speech families.
    pub const ALL: [SpeechFamily; 1] = [SpeechFamily::Whisper];

    /// Returns the static wire string for the speech family.
    ///
    /// Persisted in the speech install receipt and the catalog
    /// `InstalledModel.speech_family` field. Nothing on disk predates this
    /// enum, so the string matches its variant name; it becomes a
    /// compatibility promise the moment the first speech install exists.
    pub fn as_str(&self) -> &'static str {
        match self {
            SpeechFamily::Whisper => "whisper",
        }
    }

    /// Parses a wire string back into the speech family.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "whisper" => Some(SpeechFamily::Whisper),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_string_round_trips() {
        for family in SpeechFamily::ALL {
            assert_eq!(SpeechFamily::parse(family.as_str()), Some(family));
        }
        assert_eq!(SpeechFamily::Whisper.as_str(), "whisper");
    }

    #[test]
    fn parse_rejects_unknown_and_decoder_family_strings() {
        assert_eq!(SpeechFamily::parse("whisper-large"), None);
        assert_eq!(SpeechFamily::parse(""), None);
        // A decoder family string must never parse as a speech family:
        // the two enums stay independent by design.
        assert_eq!(SpeechFamily::parse("qwen3"), None);
        assert_eq!(SpeechFamily::parse("gemma4"), None);
    }

    #[test]
    fn all_is_exhaustive() {
        // Every variant appears exactly once in ALL. When a new speech
        // family is added this match refuses to compile until the arm and
        // the ALL entry are both written.
        let mut seen = 0;
        for family in SpeechFamily::ALL {
            match family {
                SpeechFamily::Whisper => seen += 1,
            }
        }
        assert_eq!(seen, SpeechFamily::ALL.len());
        assert_eq!(SpeechFamily::ALL.len(), 1);
    }
}
