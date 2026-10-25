//! The `reference-bracket-v1` prompt as frame rows.
//!
//! Port of `Model.build_frames` in mlx-audio `fish_drama3/drama3.py` at
//! [`super::REFERENCE_COMMIT`]. Rows are `[token, c0, ..., c{N-1}]`: text
//! rows carry zeros in the code columns, audio rows are
//! `[c0 + semantic_start, c0..c{N-1}]`. Every string segment is tokenized on
//! its own, because tokenizing the joined prompt changes the ids at the
//! seams. The system block depends only on the references, so the trunk and
//! tower state after it can be reused across chunks and calls.

use tokenizers::Tokenizer;

use super::chunking::{reference_transcript, validate_text, with_speaker_tag};
use super::config::Drama3Config;
use crate::{Result, SpeechError};

/// One utterance with its codes, `codes[book][frame]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    pub text: String,
    pub codes: Vec<Vec<i32>>,
}

/// A built prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Prompt {
    /// Frame rows, `1 + num_codebooks` wide.
    pub rows: Vec<Vec<i32>>,
    /// How many leading rows form the system block.
    pub system_rows: usize,
}

enum Segment<'a> {
    Text(String),
    Codes(&'a [Vec<i32>]),
}

fn check_codes(cfg: &Drama3Config, codes: &[Vec<i32>], location: &str) -> Result<()> {
    let books = cfg.tower.num_codebooks;
    let size = cfg.tower.vocab_size as i32;
    let bad = |why: String| {
        Err(SpeechError::Input {
            why: format!("{location}: {why}"),
        })
    };
    if codes.len() != books {
        return bad(format!(
            "codes must be [{books}, frames]; got {} books",
            codes.len()
        ));
    }
    let frames = codes[0].len();
    if frames == 0 {
        return bad("codes are empty".into());
    }
    if codes.iter().any(|book| book.len() != frames) {
        return bad("codebooks differ in length".into());
    }
    if codes.iter().flatten().any(|&c| c < 0 || c >= size) {
        return bad(format!("codes must be in [0, {size})"));
    }
    Ok(())
}

/// Builds the prompt for `target`, in the voice of `references`, after the
/// `history` turns.
pub fn build_frames(
    tokenizer: &Tokenizer,
    cfg: &Drama3Config,
    references: &[Utterance],
    history: &[Utterance],
    target: &str,
) -> Result<Prompt> {
    validate_text(target, "text")?;
    let mut segments: Vec<Segment<'_>> = vec![Segment::Text("<|im_start|>system\n".into())];
    if references.is_empty() {
        segments.push(Segment::Text("convert the provided text to speech".into()));
    } else {
        segments.push(Segment::Text(
            "convert the provided text to speech reference to the following:".into(),
        ));
        for (i, reference) in references.iter().enumerate() {
            let location = format!("reference {}", i + 1);
            validate_text(&reference.text, &location)?;
            check_codes(cfg, &reference.codes, &location)?;
            segments.push(Segment::Text(format!(
                "\n\n[Reference {}]\nText: {}\nSpeech: ",
                i + 1,
                reference_transcript(&reference.text)
            )));
            segments.push(Segment::Codes(&reference.codes));
        }
    }
    segments.push(Segment::Text("<|im_end|>\n".into()));
    let system_segments = segments.len();
    for (i, turn) in history.iter().enumerate() {
        let location = format!("history {}", i + 1);
        validate_text(&turn.text, &location)?;
        check_codes(cfg, &turn.codes, &location)?;
        segments.push(Segment::Text("<|im_start|>user\n".into()));
        segments.push(Segment::Text(with_speaker_tag(&turn.text)));
        segments.push(Segment::Text("<|im_end|>\n".into()));
        segments.push(Segment::Text("<|im_start|>assistant\n<|voice|>".into()));
        segments.push(Segment::Codes(&turn.codes));
        segments.push(Segment::Text("<|im_end|>\n".into()));
    }
    segments.push(Segment::Text("<|im_start|>user\n".into()));
    segments.push(Segment::Text(with_speaker_tag(target)));
    segments.push(Segment::Text("<|im_end|>\n".into()));
    segments.push(Segment::Text("<|im_start|>assistant\n<|voice|>".into()));

    let width = cfg.tower.frame_width();
    let mut rows = Vec::new();
    let mut system_rows = 0;
    for (index, segment) in segments.iter().enumerate() {
        let before = rows.len();
        match segment {
            Segment::Text(text) => {
                let encoding =
                    tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| SpeechError::Input {
                            why: format!("tokenizing {text:?}: {e}"),
                        })?;
                for &id in encoding.get_ids() {
                    let mut row = vec![0i32; width];
                    row[0] = i32::try_from(id).map_err(|_| SpeechError::Input {
                        why: format!("token id {id} does not fit i32"),
                    })?;
                    rows.push(row);
                }
            }
            Segment::Codes(codes) => {
                for frame in 0..codes[0].len() {
                    let mut row = Vec::with_capacity(width);
                    row.push(codes[0][frame] + cfg.semantic_start_token_id);
                    row.extend(codes.iter().map(|book| book[frame]));
                    rows.push(row);
                }
            }
        }
        if index < system_segments {
            system_rows += rows.len() - before;
        }
    }
    Ok(Prompt { rows, system_rows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::fish_drama3::tower::TowerConfig;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/fish_drama3")
            .join(name)
    }

    fn ints(v: &serde_json::Value) -> Vec<i32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    }

    fn utterances(v: &serde_json::Value) -> Vec<Utterance> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|u| Utterance {
                text: u[0].as_str().unwrap().to_string(),
                codes: u[1].as_array().unwrap().iter().map(ints).collect(),
            })
            .collect()
    }

    fn setup() -> (serde_json::Value, Tokenizer, Drama3Config) {
        let text = std::fs::read_to_string(fixture("prompt.json")).unwrap();
        let golden: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(golden["commit"], super::super::REFERENCE_COMMIT);
        let tokenizer = Tokenizer::from_file(fixture("prompt_tokenizer.json")).unwrap();
        let vocab = golden["vocab"].as_u64().unwrap() as usize;
        let begin = golden["semantic_begin"].as_i64().unwrap() as i32;
        let cfg = Drama3Config {
            tower: TowerConfig {
                dim: 64,
                n_head: 1,
                n_local_heads: 1,
                head_dim: 64,
                intermediate_size: 64,
                vocab_size: vocab,
                num_codebooks: golden["num_codebooks"].as_u64().unwrap() as usize,
                shared_enrichment_layers: 1,
                vq_backbone_codebooks: 1,
                norm_eps: 1e-6,
                rope_base: 1e7,
                rotary_percent: 0.25,
            },
            semantic_start_token_id: begin,
            semantic_end_token_id: begin + vocab as i32 - 1,
            eos_token_id: 2,
            hidden_size: 64,
        };
        (golden, tokenizer, cfg)
    }

    #[test]
    fn frames_and_system_block_match_the_reference() {
        let (golden, tokenizer, cfg) = setup();
        cfg.check_tokenizer(&tokenizer)
            .expect("fixture tokenizer agrees");
        for (i, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
            let prompt = build_frames(
                &tokenizer,
                &cfg,
                &utterances(&case["references"]),
                &utterances(&case["history"]),
                case["target"].as_str().unwrap(),
            )
            .unwrap();
            let want: Vec<Vec<i32>> = case["frames"]
                .as_array()
                .unwrap()
                .iter()
                .map(ints)
                .collect();
            assert_eq!(prompt.rows, want, "case {i}");
            assert_eq!(
                prompt.system_rows,
                case["system_rows"].as_u64().unwrap() as usize,
                "case {i}"
            );
        }
    }

    /// Opt-in: the real 248k-entry tokenizer, which the repo does not carry.
    /// `TURBOSPARK_FISH_DRAMA3_MODEL` names a Drama 3 install (its
    /// `config.json` and `tokenizer.json`). Byte-level BPE sees what the tiny
    /// whitespace tokenizer cannot: the transcript tag moving ahead of its
    /// white space, and every segment seam.
    #[test]
    fn real_tokenizer_frames_match_the_reference() {
        let Ok(dir) = std::env::var("TURBOSPARK_FISH_DRAMA3_MODEL") else {
            eprintln!("skipping: TURBOSPARK_FISH_DRAMA3_MODEL not set");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let root: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let cfg = Drama3Config::from_json(&root).unwrap();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        cfg.check_tokenizer(&tokenizer).unwrap();
        let golden: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture("prompt_real.json")).unwrap())
                .unwrap();
        assert_eq!(golden["commit"], super::super::REFERENCE_COMMIT);
        for (i, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
            let prompt = build_frames(
                &tokenizer,
                &cfg,
                &utterances(&case["references"]),
                &utterances(&case["history"]),
                case["target"].as_str().unwrap(),
            )
            .unwrap();
            let want: Vec<Vec<i32>> = case["frames"]
                .as_array()
                .unwrap()
                .iter()
                .map(ints)
                .collect();
            assert_eq!(prompt.rows, want, "case {i}");
            assert_eq!(
                prompt.system_rows,
                case["system_rows"].as_u64().unwrap() as usize,
                "case {i}"
            );
        }
    }

    #[test]
    fn refuses_bad_codes_and_text() {
        let (_, tokenizer, cfg) = setup();
        let ok = vec![vec![1, 2]; 3];
        let good = |codes: Vec<Vec<i32>>| Utterance {
            text: "Hi.".into(),
            codes,
        };
        assert!(build_frames(&tokenizer, &cfg, &[good(ok.clone())], &[], "Hallo.").is_ok());
        for codes in [
            vec![vec![1, 2]; 2],                   // too few books
            vec![vec![]; 3],                       // empty
            vec![vec![1, 2], vec![1], vec![1, 2]], // ragged
            vec![vec![1, 16]; 3],                  // out of range
            vec![vec![-1, 2]; 3],                  // negative
        ] {
            assert!(
                build_frames(&tokenizer, &cfg, &[good(codes.clone())], &[], "Hallo.").is_err(),
                "{codes:?}"
            );
        }
        assert!(build_frames(&tokenizer, &cfg, &[], &[], "x <|im_start|> y").is_err());
        let mut wrong = cfg.clone();
        wrong.eos_token_id = 3;
        assert!(wrong.check_tokenizer(&tokenizer).is_err());
    }
}
