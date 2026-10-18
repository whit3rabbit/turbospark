//! Speaker-attributed meeting transcription scoring for the Frozen E port.
//!
//! The frozen reference is NVIDIA NeMo's Multitalker Parakeet + Nemotron-3
//! diarization run on twelve 120 s AMI IS clips. `testdata/frozen_e/` holds
//! those clips' references and the frozen E hypothesis (text only). The
//! scorer here must reproduce the frozen counts exactly before any native
//! result is compared against them.

pub mod cpwer;
pub mod seglst;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use cpwer::{cpwer, tokens, CpWer, TooManySpeakers};
pub use seglst::{
    group_by_session, read_references, read_seglst, Clip, FormatError, RefSegment, RefWord, Segment,
};

/// The six clips matched across every Frozen E comparison arm.
pub const MATCHED_SIX: [&str; 6] = [
    "IS1000a_00510_00630",
    "IS1001a_00390_00510",
    "IS1002b_00880_01000",
    "IS1003a_00320_00440",
    "IS1004a_00140_00260",
    "IS1005a_00440_00560",
];

/// Directory of the committed Frozen E fixtures.
pub fn frozen_e_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/frozen_e")
}

/// One clip's row in a panel score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipScore {
    /// Clip id.
    pub id: String,
    /// The clip's cpWER result.
    pub score: CpWer,
}

/// Summed panel score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelScore {
    /// Per-clip rows in the order requested.
    pub clips: Vec<ClipScore>,
    /// Summed errors.
    pub errors: usize,
    /// Summed reference tokens.
    pub ref_tokens: usize,
}

impl PanelScore {
    /// Errors over reference tokens.
    pub fn cpwer(&self) -> f64 {
        self.errors as f64 / self.ref_tokens as f64
    }
}

/// Scoring failure.
#[derive(Debug)]
pub enum ScoreError {
    /// Malformed input.
    Format(FormatError),
    /// A clip has more speakers than the frozen scorer allows.
    Speakers(String, TooManySpeakers),
    /// A requested clip id is not in the references.
    UnknownClip(String),
}

impl std::fmt::Display for ScoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Format(e) => write!(f, "{e}"),
            Self::Speakers(id, e) => write!(f, "{id}: {e}"),
            Self::UnknownClip(id) => write!(f, "unknown clip id {id:?}"),
        }
    }
}

impl std::error::Error for ScoreError {}

impl From<FormatError> for ScoreError {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}

/// Scores `ids` (all reference clips when empty). A clip with no hypothesis
/// segments scores as all deletions, as in the frozen scorer.
pub fn score_panel(
    clips: &[Clip],
    hypothesis: Vec<Segment>,
    ids: &[&str],
) -> Result<PanelScore, ScoreError> {
    let grouped = group_by_session(hypothesis, clips)?;
    let wanted: Vec<&str> = if ids.is_empty() {
        clips.iter().map(|c| c.id.as_str()).collect()
    } else {
        ids.to_vec()
    };
    let mut rows = Vec::with_capacity(wanted.len());
    for id in wanted {
        let clip = clips
            .iter()
            .find(|c| c.id == id)
            .ok_or_else(|| ScoreError::UnknownClip(id.to_owned()))?;
        let segs = grouped.get(id).map(Vec::as_slice).unwrap_or(&[]);
        let score = cpwer(&clip.words, segs).map_err(|e| ScoreError::Speakers(id.to_owned(), e))?;
        rows.push(ClipScore {
            id: id.to_owned(),
            score,
        });
    }
    Ok(PanelScore {
        errors: rows.iter().map(|r| r.score.errors).sum(),
        ref_tokens: rows.iter().map(|r| r.score.ref_tokens).sum(),
        clips: rows,
    })
}

/// Reads `frozen_counts.json`: clip id to `(errors, reference tokens)`.
pub fn read_frozen_counts(path: &Path) -> Result<BTreeMap<String, (usize, usize)>, FormatError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| FormatError(format!("{}: {e}", path.display())))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| FormatError(format!("{}: {e}", path.display())))?;
    let clips = doc
        .get("clips")
        .and_then(|c| c.as_object())
        .ok_or_else(|| FormatError("frozen_counts.json: missing `clips`".into()))?;
    clips
        .iter()
        .map(|(id, v)| {
            let get = |k: &str| {
                v.get(k)
                    .and_then(|n| n.as_u64())
                    .map(|n| n as usize)
                    .ok_or_else(|| FormatError(format!("{id}: missing `{k}`")))
            };
            Ok((id.clone(), (get("cpwer_errors")?, get("reference_tokens")?)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// References, the frozen E hypothesis, and the frozen per-clip counts.
    type Frozen = (Vec<Clip>, Vec<Segment>, BTreeMap<String, (usize, usize)>);

    fn frozen() -> Frozen {
        let dir = frozen_e_dir();
        (
            read_references(&dir.join("references.json")).unwrap(),
            read_seglst(&dir.join("E_seglst.json")).unwrap(),
            read_frozen_counts(&dir.join("frozen_counts.json")).unwrap(),
        )
    }

    #[test]
    fn seglst_reads_frozen_schema() {
        let (clips, segs, counts) = frozen();
        assert_eq!(clips.len(), 12);
        assert_eq!(segs.len(), 356);
        assert_eq!(counts.len(), 12);
        let grouped = group_by_session(segs, &clips).unwrap();
        assert_eq!(grouped.len(), 12);
        assert_eq!(grouped["IS1000a_00510_00630"].len(), 31);
    }

    #[test]
    fn cpwer_reproduces_frozen_e_per_clip_counts() {
        let (clips, segs, counts) = frozen();
        let panel = score_panel(&clips, segs, &[]).unwrap();
        for row in &panel.clips {
            assert_eq!(
                (row.score.errors, row.score.ref_tokens),
                counts[&row.id],
                "{}",
                row.id
            );
        }
        assert_eq!((panel.errors, panel.ref_tokens), (355, 3814));
    }

    #[test]
    fn cpwer_reproduces_frozen_e_160_of_1740() {
        let (clips, segs, _) = frozen();
        let panel = score_panel(&clips, segs, &MATCHED_SIX).unwrap();
        assert_eq!((panel.errors, panel.ref_tokens), (160, 1740));
    }

    #[test]
    fn missing_hypothesis_scores_as_deletions() {
        let (clips, _, _) = frozen();
        let panel = score_panel(&clips, vec![], &["IS1004a_00140_00260"]).unwrap();
        assert_eq!((panel.errors, panel.ref_tokens), (219, 219));
        assert!(matches!(
            score_panel(&clips, vec![], &["nope"]),
            Err(ScoreError::UnknownClip(_))
        ));
    }
}
