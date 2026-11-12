//! SegLST hypotheses and frozen meeting references.
//!
//! SegLST is the list-of-segments JSON NeMo's multitalker driver writes:
//! `[{session_id, speaker, words, start_time, end_time}, ...]`. References use
//! the frozen evaluation manifest's per-word layout. Both readers are strict:
//! a missing field or an unknown session is an error, never a silent skip.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

/// One hypothesis segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Clip id the segment belongs to.
    pub session_id: String,
    /// Hypothesis speaker label, such as `speaker_0`.
    pub speaker: String,
    /// Segment start in seconds from the clip start.
    pub start_time: f64,
    /// Segment end in seconds from the clip start.
    pub end_time: f64,
    /// Space-separated words.
    pub words: String,
}

/// One reference word.
#[derive(Debug, Clone, PartialEq)]
pub struct RefWord {
    /// Reference speaker label (AMI channel letter).
    pub speaker: String,
    /// Word start in seconds from the clip start.
    pub start: f64,
    /// Word end in seconds from the clip start.
    pub end: f64,
    /// Word text as annotated.
    pub text: String,
}

/// One reference speaker turn, used for DER.
#[derive(Debug, Clone, PartialEq)]
pub struct RefSegment {
    /// Reference speaker label.
    pub speaker: String,
    /// Turn start in seconds from the clip start.
    pub start: f64,
    /// Turn end in seconds from the clip start.
    pub end: f64,
}

/// One frozen evaluation clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    /// Clip id, `<meeting>_<start>_<end>`.
    pub id: String,
    /// AMI meeting id.
    pub meeting: String,
    /// Offset of the clip in the meeting, seconds.
    pub source_offset: f64,
    /// Clip length, seconds.
    pub duration: f64,
    /// SHA-256 of the 16 kHz PCM_16 clip WAV.
    pub audio_sha256: String,
    /// Reference words.
    pub words: Vec<RefWord>,
    /// Reference speaker turns.
    pub segments: Vec<RefSegment>,
}

/// A malformed or inconsistent input file.
#[derive(Debug)]
pub struct FormatError(pub String);

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FormatError {}

fn err(msg: impl Into<String>) -> FormatError {
    FormatError(msg.into())
}

fn field<'a>(v: &'a Value, key: &str, ctx: &str) -> Result<&'a Value, FormatError> {
    v.get(key)
        .ok_or_else(|| err(format!("{ctx}: missing field `{key}`")))
}

fn string(v: &Value, key: &str, ctx: &str) -> Result<String, FormatError> {
    match field(v, key, ctx)? {
        Value::String(s) => Ok(s.clone()),
        // NeMo writes speakers as strings, but integer labels must not be dropped.
        Value::Number(n) => Ok(n.to_string()),
        other => Err(err(format!("{ctx}: `{key}` is not a string: {other}"))),
    }
}

fn number(v: &Value, key: &str, ctx: &str) -> Result<f64, FormatError> {
    field(v, key, ctx)?
        .as_f64()
        .ok_or_else(|| err(format!("{ctx}: `{key}` is not a number")))
}

fn read_json(path: &Path) -> Result<Value, FormatError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| err(format!("{}: {e}", path.display())))
}

/// Parses a SegLST document.
pub fn parse_seglst(doc: &Value) -> Result<Vec<Segment>, FormatError> {
    let items = doc
        .as_array()
        .ok_or_else(|| err("SegLST must be a JSON list"))?;
    items
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let ctx = format!("segment {i}");
            Ok(Segment {
                session_id: string(s, "session_id", &ctx)?,
                speaker: string(s, "speaker", &ctx)?,
                start_time: number(s, "start_time", &ctx)?,
                end_time: number(s, "end_time", &ctx)?,
                words: string(s, "words", &ctx)?,
            })
        })
        .collect()
}

/// Reads a SegLST file.
pub fn read_seglst(path: &Path) -> Result<Vec<Segment>, FormatError> {
    parse_seglst(&read_json(path)?)
}

/// Parses the frozen references document (`testdata/frozen_e/references.json`).
pub fn parse_references(doc: &Value) -> Result<Vec<Clip>, FormatError> {
    let items = doc
        .as_array()
        .ok_or_else(|| err("references must be a JSON list"))?;
    items
        .iter()
        .map(|c| {
            let id = string(c, "id", "clip")?;
            let words = field(c, "reference_words", &id)?
                .as_array()
                .ok_or_else(|| err(format!("{id}: reference_words is not a list")))?
                .iter()
                .map(|w| {
                    Ok(RefWord {
                        speaker: string(w, "speaker", &id)?,
                        start: number(w, "start", &id)?,
                        end: number(w, "end", &id)?,
                        text: string(w, "text", &id)?,
                    })
                })
                .collect::<Result<_, FormatError>>()?;
            let segments = field(c, "reference_segments", &id)?
                .as_array()
                .ok_or_else(|| err(format!("{id}: reference_segments is not a list")))?
                .iter()
                .map(|s| {
                    Ok(RefSegment {
                        speaker: string(s, "speaker", &id)?,
                        start: number(s, "start", &id)?,
                        end: number(s, "end", &id)?,
                    })
                })
                .collect::<Result<_, FormatError>>()?;
            Ok(Clip {
                meeting: string(c, "meeting", &id)?,
                source_offset: number(c, "source_offset", &id)?,
                duration: number(c, "duration", &id)?,
                audio_sha256: string(c, "audio_sha256", &id)?,
                words,
                segments,
                id,
            })
        })
        .collect()
}

/// Reads the frozen references file.
pub fn read_references(path: &Path) -> Result<Vec<Clip>, FormatError> {
    parse_references(&read_json(path)?)
}

/// Groups segments by session, refusing any session not in `clips`. This is
/// the frozen scorer's guard against cross-clip attribution mistakes.
pub fn group_by_session(
    segments: Vec<Segment>,
    clips: &[Clip],
) -> Result<BTreeMap<String, Vec<Segment>>, FormatError> {
    let mut out: BTreeMap<String, Vec<Segment>> = BTreeMap::new();
    for s in segments {
        if !clips.iter().any(|c| c.id == s.session_id) {
            return Err(err(format!(
                "unexpected SegLST session id {:?}",
                s.session_id
            )));
        }
        out.entry(s.session_id.clone()).or_default().push(s);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seglst_rejects_missing_fields_and_unknown_sessions() {
        let doc: Value = serde_json::from_str(
            r#"[{"session_id":"a","speaker":"speaker_0","start_time":0.0,"end_time":1.0}]"#,
        )
        .unwrap();
        assert!(parse_seglst(&doc).unwrap_err().0.contains("`words`"));

        let doc: Value = serde_json::from_str(
            r#"[{"session_id":"b","speaker":0,"start_time":0,"end_time":1,"words":"hi"}]"#,
        )
        .unwrap();
        let segs = parse_seglst(&doc).unwrap();
        assert_eq!(segs[0].speaker, "0");
        let clip = Clip {
            id: "a".into(),
            meeting: "m".into(),
            source_offset: 0.0,
            duration: 1.0,
            audio_sha256: String::new(),
            words: vec![],
            segments: vec![],
        };
        assert!(group_by_session(segs, &[clip])
            .unwrap_err()
            .0
            .contains("unexpected SegLST session id"));
    }
}
