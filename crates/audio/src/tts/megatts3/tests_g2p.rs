//! Tests for the G2P model glue.
//!
//! `testdata/megatts3_text/tiny_g2p/` holds a seeded tiny Qwen2 saved by HF
//! transformers plus a small byte-level BPE tokenizer carrying the real
//! tokenizer's added tokens (`<BOT>`, `<BOS>`, `<Reserved_TTS_n>`);
//! `tiny_g2p_cases.json` records the tokenizer ids and the exact
//! `generate(do_sample=True, top_k=1, eos_token_id=...)` outputs. The real
//! checkpoint is gated in with TURBOSPARK_G2P_DIR (the converted `g2p/`
//! directory) and TURBOSPARK_G2P_GOLDEN (the `g2p_real.json` from
//! `--real-g2p`); both are skipped when unset. Tests never download.

use std::path::{Path, PathBuf};

use super::g2p::{generation_input, phones_from_ids, phones_from_output, G2p, MAX_NEW_TOKENS};

fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/megatts3_text/tiny_g2p")
}

fn cases() -> serde_json::Value {
    let text =
        std::fs::read_to_string(tiny_dir().parent().unwrap().join("tiny_g2p_cases.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn ints(v: &serde_json::Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect()
}

#[test]
fn tiny_tokenizer_and_greedy_decode_match_transformers() {
    let mut g2p = G2p::open(&tiny_dir()).unwrap();
    let c = cases();
    assert_eq!(
        g2p.speech_start() as i64,
        c["speech_start"].as_i64().unwrap()
    );
    for case in c["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let txt_ids = g2p.text_ids(text).unwrap();
        assert_eq!(txt_ids, ints(&case["txt_ids"]), "token ids for {text:?}");
        let input = ints(&case["input_ids"]);
        let eos = case["eos"].as_i64().unwrap() as i32;
        let max_new = case["max_new"].as_u64().unwrap() as usize;
        let got = g2p.generate(&input, max_new, eos).unwrap();
        assert_eq!(
            got,
            ints(&case["output"]),
            "generate for {text:?} eos {eos}"
        );
    }
}

#[test]
fn generation_stops_at_eos_and_at_the_budget() {
    let mut g2p = G2p::open(&tiny_dir()).unwrap();
    let c = cases();
    let stopped = c["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| {
            ints(&case["output"]).len()
                < ints(&case["input_ids"]).len() + case["max_new"].as_u64().unwrap() as usize
        })
        .count();
    assert!(
        stopped >= 3,
        "fixtures should include EOS stops, found {stopped}"
    );
    // A zero budget returns the prompt unchanged.
    let input = ints(&c["cases"][0]["input_ids"]);
    assert_eq!(g2p.generate(&input, 0, 1).unwrap(), input);
}

#[test]
fn generate_validates_inputs() {
    let mut g2p = G2p::open(&tiny_dir()).unwrap();
    assert!(g2p.generate(&[], 4, 1).is_err());
    assert!(g2p.generate(&[-1], 4, 1).is_err());
    assert!(g2p.generate(&[1_000_000], 4, 1).is_err());
    assert!(g2p.generate(&vec![1; 3000], 4, 1).is_err());
}

#[test]
fn model_without_room_for_speech_tokens_is_rejected() {
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::{AddedToken, Tokenizer};

    // The speech tokens run to speech_start + 801. A tokenizer that puts
    // `<Reserved_TTS_0>` past the tiny model's 1155-entry vocabulary must fail.
    let weights = crate::lm::Weights::open(&tiny_dir()).unwrap();
    let good = Tokenizer::from_file(tiny_dir().join("tokenizer.json")).unwrap();
    assert!(G2p::load(good, &weights).is_ok());

    let mut vocab: HashMap<String, u32> = (0..5000).map(|i| (format!("w{i}"), i)).collect();
    vocab.insert("<unk>".to_string(), 5000);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut far = Tokenizer::new(model);
    far.with_pre_tokenizer(Some(Whitespace {}));
    far.add_special_tokens(&[AddedToken::from("<Reserved_TTS_0>", true)]);
    assert!(far.encode("<Reserved_TTS_0>", true).unwrap().get_ids()[0] >= 5000);
    assert!(G2p::load(far, &weights).is_err());
}

#[test]
fn phone_mapping_matches_split_ph() {
    // Plain ids keep tone 3; Chinese ids fold phone and tone.
    let (ph, tone) = phones_from_ids(&[145, 12, 173, 200, 201, 206, 207, 788]).unwrap();
    assert_eq!(ph, vec![145, 12, 173, 2, 3, 3, 4, 100]);
    assert_eq!(tone, vec![3, 3, 3, 15, 4, 15, 4, 15]);
    for bad in [
        vec![],
        vec![0],
        vec![-3],
        vec![174],
        vec![199],
        vec![789],
        vec![801],
    ] {
        assert!(phones_from_ids(&bad).is_err(), "{bad:?} accepted");
    }
}

#[test]
fn prompt_ends_with_the_sil_speech_token() {
    assert_eq!(generation_input(&[5, 6, 7], 1000), vec![5, 6, 7, 1145]);
}

#[test]
fn output_slicing_keeps_sil_and_drops_the_last_token() {
    let ssi = 1000;
    let txt = [5, 6, 7];
    // prompt (text + sil), then 12, a Chinese id, and the EOS (800 + 1 + ssi).
    let out = [5, 6, 7, 1145, 1012, 1200, 1801];
    let (ph, tone) = phones_from_output(&out, txt.len(), ssi).unwrap();
    assert_eq!(ph, vec![145, 12, 2]);
    assert_eq!(tone, vec![3, 3, 15]);
    // Without an EOS (budget exhausted) the last real token is dropped anyway.
    let (ph, _) = phones_from_output(&[5, 6, 7, 1145, 1012, 1020], txt.len(), ssi).unwrap();
    assert_eq!(ph, vec![145, 12]);
    // Nothing generated after the prompt, or tokens below the speech base, are errors.
    assert!(phones_from_output(&[5, 6, 7, 1145], txt.len(), ssi).is_err());
    assert!(phones_from_output(&[5, 6, 7, 1145, 3, 1801], txt.len(), ssi).is_err());
}

#[test]
fn budget_matches_the_reference_cap() {
    assert_eq!(MAX_NEW_TOKENS, 256);
}

/// Real converted checkpoint against the reference's own `g2p()`.
#[test]
fn real_checkpoint_matches_transformers() {
    let (Some(dir), Some(golden)) = (
        std::env::var_os("TURBOSPARK_G2P_DIR"),
        std::env::var_os("TURBOSPARK_G2P_GOLDEN"),
    ) else {
        eprintln!("skipped: TURBOSPARK_G2P_DIR / TURBOSPARK_G2P_GOLDEN not set");
        return;
    };
    let mut g2p = G2p::open(Path::new(&dir)).unwrap();
    let text = std::fs::read_to_string(golden).unwrap();
    let g: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        g2p.speech_start() as i64,
        g["speech_start"].as_i64().unwrap()
    );
    for r in g["results"].as_array().unwrap() {
        let sentence = r["text"].as_str().unwrap();
        assert_eq!(
            g2p.text_ids(sentence).unwrap(),
            ints(&r["txt_ids"]),
            "token ids for {sentence:?}"
        );
        let (ph, tone) = g2p.phonemize(sentence).unwrap();
        println!("real: {sentence:?}: {} phones", ph.len());
        assert_eq!(ph, ints(&r["ph"]), "phones for {sentence:?}");
        assert_eq!(tone, ints(&r["tone"]), "tones for {sentence:?}");
    }
}
