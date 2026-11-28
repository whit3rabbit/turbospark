//! Prompt assembly parity against upstream's own helpers
//! (`testdata/ace_step15_prompt/cases.json`, written by
//! `tools/gen_ace_step15_prompt_fixtures.py`). Strings are checked by
//! default; token ids need the checkpoint tokenizer
//! (TURBOSPARK_ACE_STEP15_TEXT points at `Qwen3-Embedding-0.6B/`).

use std::path::{Path, PathBuf};

use super::prompt::{caption_prompt, lyric_prompt, Metas, PromptTokenizer};

fn cases() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/ace_step15_prompt/cases.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn metas(c: &serde_json::Value) -> Metas {
    Metas {
        bpm: c["bpm"].as_u64().map(|b| b as u32),
        key_scale: c["key_scale"].as_str().unwrap().to_string(),
        time_signature: c["time_signature"].as_str().unwrap().to_string(),
        duration_seconds: c["duration"].as_f64(),
    }
}

#[test]
fn prompts_match_upstream_helpers() {
    let doc = cases();
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let m = metas(c);
        assert_eq!(m.render(), c["metas"].as_str().unwrap(), "{name}: metas");
        let caption = caption_prompt(
            c["instruction"].as_str().unwrap(),
            c["caption"].as_str().unwrap(),
            &m,
        );
        assert_eq!(
            caption,
            c["caption_prompt"].as_str().unwrap(),
            "{name}: caption prompt"
        );
        let lyrics = lyric_prompt(
            c["lyrics"].as_str().unwrap(),
            c["language"].as_str().unwrap(),
        );
        assert_eq!(
            lyrics,
            c["lyric_prompt"].as_str().unwrap(),
            "{name}: lyric prompt"
        );
    }
}

#[test]
fn task_instructions_match_upstream() {
    use super::prompt::{Task, DEFAULT_DIT_INSTRUCTION};
    let doc = cases();
    let t = &doc["task_instructions"];
    assert_eq!(
        Task::Text2Music.instruction(),
        t["text2music"].as_str().unwrap()
    );
    assert_eq!(Task::Cover.instruction(), t["cover"].as_str().unwrap());
    assert_eq!(Task::Repaint.instruction(), t["repaint"].as_str().unwrap());
    assert_eq!(
        DEFAULT_DIT_INSTRUCTION,
        doc["default_dit_instruction"].as_str().unwrap()
    );
}

#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_TEXT"]
fn token_ids_match_upstream_tokenizer() {
    let dir = PathBuf::from(std::env::var("TURBOSPARK_ACE_STEP15_TEXT").unwrap());
    let tok = PromptTokenizer::load(&dir).unwrap();
    let doc = cases();
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let ids = |k: &str| -> Vec<i32> {
            c[k].as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect()
        };
        let cap = tok.caption(c["caption_prompt"].as_str().unwrap()).unwrap();
        assert_eq!(cap.ids, ids("caption_ids"), "{name}: caption ids");
        assert!(!cap.truncated, "{name}: caption should fit");
        let lyr = tok.lyrics(c["lyric_prompt"].as_str().unwrap()).unwrap();
        assert_eq!(lyr.ids, ids("lyric_ids"), "{name}: lyric ids");
        let untruncated = c["lyric_untruncated_len"].as_u64().unwrap() as usize;
        assert_eq!(
            lyr.truncated,
            untruncated > lyr.ids.len(),
            "{name}: truncation flag"
        );
    }
}
