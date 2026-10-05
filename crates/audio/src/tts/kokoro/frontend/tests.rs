use super::*;

fn vocab() -> PhonemeVocabulary {
    let config = serde_json::from_str(include_str!("../testdata/config.json")).unwrap();
    PhonemeVocabulary::from_config(&config, 178, 178, 512).unwrap()
}

#[test]
fn frontend_independent_python_corpus() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../testdata/frontend-python.json")).unwrap();
    let frontend = EnglishFrontend::new();
    for case in fixture["results"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        // These reference POS errors have separate English semantic assertions.
        if text.contains("read every") || text.contains("bass guitar") {
            continue;
        }
        let segments = frontend
            .prepare(&SynthesisRequest::new(text), &vocab())
            .unwrap();
        let actual = segments
            .iter()
            .map(|s| s.phonemes())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(actual, case["phonemes"].as_str().unwrap(), "input: {text}");
    }
}

#[test]
fn frontend_sentence_limits_and_word_boundaries() {
    let f = EnglishFrontend::new();
    let text = format!("{} End. Next sentence!", "hello ".repeat(250));
    let chunks = f.prepare(&SynthesisRequest::new(&text), &vocab()).unwrap();
    assert!(chunks.len() > 2);
    assert_eq!(
        chunks
            .iter()
            .map(|s| s.text())
            .collect::<Vec<_>>()
            .join(" "),
        text.trim()
    );
    for s in &chunks {
        assert!((1..=510).contains(&s.phonemes().chars().count()));
        assert_eq!(s.ids().len(), s.phonemes().chars().count() + 2);
        assert_eq!(s.ids().first(), Some(&0));
        assert_eq!(s.ids().last(), Some(&0));
        assert!(
            s.phonemes().ends_with('O')
                || s.phonemes().ends_with('.')
                || s.phonemes().ends_with('!')
        );
    }
    assert!(f
        .prepare(&SynthesisRequest::new("q".repeat(300)), &vocab())
        .is_err());
}

#[test]
fn frontend_style_rows_count_characters_and_pack_geometry() {
    let a =
        SynthesisSegment::from_phonemes("h\u{259}l\u{2c8}O, w\u{2c8}\u{25c}\u{279}ld!", &vocab())
            .unwrap();
    assert_eq!(a.style_row(), 13);
    let b = SynthesisSegment::from_phonemes("a", &vocab()).unwrap();
    assert_eq!(b.style_row(), 0);
    let last = SynthesisSegment::from_phonemes(&"a".repeat(510), &vocab()).unwrap();
    assert_eq!(last.style_row(), 509);
    let values: Vec<_> = (0..510)
        .flat_map(|row| std::iter::repeat(row as f32).take(256))
        .collect();
    let pack = VoicePack::from_values(Voice::AfHeart, &[510, 1, 256], values).unwrap();
    assert_eq!(pack.style_for(&a), &[13.0; 256]);
    assert_eq!(pack.style_for(&b), &[0.0; 256]);
    assert_eq!(pack.style_for(&last), &[509.0; 256]);
    for shape in [&[509, 1, 256][..], &[510, 256][..], &[510, 2, 128][..]] {
        assert!(VoicePack::from_values(Voice::AfHeart, shape, vec![0.0; 510 * 256]).is_err());
    }
    assert!(
        VoicePack::from_values(Voice::AfHeart, &[510, 1, 256], vec![0.0; 510 * 256 - 1]).is_err()
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = vec![0.0; 510 * 256];
        values[510 * 256 - 1] = bad;
        assert!(VoicePack::from_values(Voice::AfHeart, &[510, 1, 256], values).is_err());
    }
}

#[test]
fn frontend_rejects_invalid_inputs_and_vocab_without_loss() {
    let f = EnglishFrontend::new();
    let v = vocab();
    for text in [
        "",
        " \n\t",
        "hello\0world",
        "hello \u{1f600}",
        "\u{4f60}\u{597d}",
        "3:45",
        "[hello](/xyz/)",
    ] {
        assert!(
            f.prepare(&SynthesisRequest::new(text), &v).is_err(),
            "{text:?}"
        );
    }
    let mut r = SynthesisRequest::new("hello");
    r.language = "en-GB".into();
    assert!(f.prepare(&r, &v).unwrap_err().to_string().contains("en-US"));
    r.language = "en-US".into();
    for speed in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        r.speed = speed;
        assert!(f.prepare(&r, &v).is_err());
    }
    assert_eq!(SynthesisRequest::new("hello").speed, 1.0);
    for phones in ["", "a\u{2603}", &"a".repeat(511)] {
        assert!(SynthesisSegment::from_phonemes(phones, &v).is_err());
    }
    let mut map = HashMap::new();
    map.insert('a', 178);
    assert!(PhonemeVocabulary::new(map.clone(), 178, 179, 512).is_err());
    assert!(PhonemeVocabulary::new(map, 179, 178, 512).is_err());
    assert!(PhonemeVocabulary::new(HashMap::new(), 178, 178, 1).is_err());
    let small = PhonemeVocabulary::new([('a', 1)].into_iter().collect(), 2, 2, 4).unwrap();
    assert!(SynthesisSegment::from_phonemes("aaa", &small).is_err());
    let malformed = serde_json::json!({"vocab":{"aa":1}});
    assert!(PhonemeVocabulary::from_config(&malformed, 178, 178, 512).is_err());
}

#[test]
fn frontend_context_uses_sounds_and_semantic_pos() {
    let f = EnglishFrontend::new();
    let v = vocab();
    let joined = |text| {
        f.prepare(&SynthesisRequest::new(text), &v)
            .unwrap()
            .into_iter()
            .map(|s| s.phonemes)
            .collect::<Vec<_>>()
            .join(" ")
    };
    assert_eq!(joined("the hour"), "\u{f0}i \u{2c8}W\u{259}\u{279}");
    assert_eq!(
        joined("the university"),
        "\u{f0}\u{259} j\u{2cc}un\u{259}v\u{2c8}\u{25c}\u{279}s\u{259}Ti"
    );
    assert!(joined("I read every day.").contains("\u{279}\u{2c8}id"));
    assert!(joined("The bass guitar plays.").contains("b\u{2c8}As"));
    assert_ne!(joined("the record"), joined("Please record"));
}
