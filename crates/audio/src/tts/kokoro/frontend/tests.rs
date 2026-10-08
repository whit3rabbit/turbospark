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

#[test]
fn frontend_plain_text_independent_regressions() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../testdata/frontend-contract.json")).unwrap();
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let mut failures = Vec::new();
    for case in fixture["results"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected = case["phonemes"].as_str().unwrap();
        match frontend.prepare(&SynthesisRequest::new(text), &vocab) {
            Ok(segments) => {
                let actual = segments
                    .iter()
                    .map(|s| s.phonemes())
                    .collect::<Vec<_>>()
                    .join(" ");
                if actual != expected {
                    failures.push(format!("{text:?}: expected {expected:?}, got {actual:?}"));
                }
            }
            Err(error) => failures.push(format!("{text:?}: unexpected {error}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_terminal_punctuation_stays_with_sentence() {
    let frontend = EnglishFrontend::new();
    for (text, expected) in [
        ("Hello?! Next.", vec!["Hello?!", "Next."]),
        ("Hello... Next.", vec!["Hello...", "Next."]),
        ("Hello!\" Next.", vec!["Hello!\"", "Next."]),
        (
            "He said (\"Hello!\"). Next.",
            vec!["He said (\"Hello!\").", "Next."],
        ),
    ] {
        let segments = frontend
            .prepare(&SynthesisRequest::new(text), &vocab())
            .unwrap();
        assert_eq!(
            segments.iter().map(|s| s.text()).collect::<Vec<_>>(),
            expected,
            "{text}"
        );
    }
}

#[test]
fn frontend_terminal_groups_stay_with_word_at_context_limit() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let prefix = vec!["hello"; 84].join(" ");
    let boundary = format!("{prefix} he a a");
    let bare = frontend
        .prepare(&SynthesisRequest::new(&boundary), &vocab)
        .unwrap();
    assert_eq!(bare.len(), 1);
    assert_eq!(bare[0].phonemes().chars().count(), 510);
    for suffix in ["!", "?!", "!)", "!\"", "?!\")", "...)."] {
        let text = format!("{boundary}{suffix} Next.");
        let segments = frontend
            .prepare(&SynthesisRequest::new(&text), &vocab)
            .unwrap();
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.text())
                .collect::<Vec<_>>(),
            [
                format!("{prefix} he a"),
                format!("a{suffix}"),
                "Next.".into()
            ],
            "terminal group at context limit: {suffix:?}"
        );
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.text())
                .collect::<Vec<_>>()
                .join(" "),
            text
        );
        for segment in segments {
            let count = segment.phonemes().chars().count();
            assert!((1..=510).contains(&count));
            assert_eq!(segment.style_row(), count - 1);
            assert_eq!(segment.ids().len(), count + 2);
        }
    }
    for (opening, middle, closing) in [
        ("(", "a a", "?!)"),
        ("\"", "a a", "?!\""),
        ("(\"", "he", "?!\")"),
    ] {
        let before_word = format!("{opening}{prefix} {middle}");
        let boundary = format!("{before_word} a");
        let bare = frontend
            .prepare(&SynthesisRequest::new(&boundary), &vocab)
            .unwrap();
        assert_eq!(bare[0].phonemes().chars().count(), 510);
        let text = format!("{boundary}{closing} Next.");
        let segments = frontend
            .prepare(&SynthesisRequest::new(&text), &vocab)
            .unwrap();
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.text())
                .collect::<Vec<_>>(),
            [before_word, format!("a{closing}"), "Next.".into()],
            "closing delimiters at context limit: {text}"
        );
        assert!(segments
            .iter()
            .all(|segment| segment.phonemes().chars().count() <= 510));
    }
}

#[test]
fn frontend_rejects_unsplittable_terminal_group() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let fitting = format!("hello{}", "!".repeat(505));
    let segments = frontend
        .prepare(&SynthesisRequest::new(&fitting), &vocab)
        .unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].phonemes().chars().count(), 510);
    let oversized = format!("hello{}", "!".repeat(506));
    let error = frontend
        .prepare(&SynthesisRequest::new(oversized), &vocab)
        .unwrap_err();
    assert!(matches!(error, SpeechError::Input { .. }));
    assert!(error.to_string().contains("terminal punctuation"));
}

#[test]
fn frontend_every_accepted_punctuation_run_stays_with_word_at_limit() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let prefix = format!("{} he a", vec!["hello"; 84].join(" "));
    let boundary = format!("{prefix} a");
    assert_eq!(
        frontend
            .prepare(&SynthesisRequest::new(&boundary), &vocab)
            .unwrap()[0]
            .phonemes()
            .chars()
            .count(),
        510
    );
    let mut failures = Vec::new();
    for suffix in [
        ".",
        "!",
        "?",
        ";",
        ":",
        ",",
        ")",
        "\"",
        "\u{2014}",
        "\u{2026}",
        "\u{2026}!)",
        "\u{2026}!\")",
        ",;:\u{2014}\u{2026}?!\")",
        "-",
        "\u{2013}",
        "\u{201c}",
        "\u{201d}",
    ] {
        let text = format!("{boundary}{suffix}");
        let segments = frontend
            .prepare(&SynthesisRequest::new(&text), &vocab)
            .unwrap();
        let actual = segments.iter().map(|s| s.text()).collect::<Vec<_>>();
        let expected = [prefix.clone(), format!("a{}", normalize::text(suffix))];
        if actual != expected {
            failures.push(format!(
                "suffix {suffix:?}: expected {expected:?}, got {actual:?}"
            ));
        }
        assert!(segments.iter().all(|s| {
            let count = s.phonemes().chars().count();
            count <= 510 && s.ids().len() == count + 2 && s.style_row() == count - 1
        }));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_every_accepted_punctuation_group_obeys_exact_fit_and_refusal() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let mut failures = Vec::new();
    for mark in [
        ".", "!", "?", ";", ":", ",", ")", "\"", "\u{2014}", "\u{2026}",
    ] {
        let fitting = format!("hello{}", mark.repeat(505));
        let segments = frontend
            .prepare(&SynthesisRequest::new(&fitting), &vocab)
            .unwrap();
        assert_eq!(segments.len(), 1, "exact-fit {mark:?}");
        assert_eq!(segments[0].phonemes().chars().count(), 510);
        let oversized = format!("hello{}", mark.repeat(506));
        match frontend.prepare(&SynthesisRequest::new(oversized), &vocab) {
            Err(SpeechError::Input { why }) if why.contains("punctuation") => {}
            other => failures.push(format!(
                "oversized {mark:?}: expected input error, got {other:?}"
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_opening_delimiters_reserve_the_following_word() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let prefix = format!("{} he a", vec!["hello"; 84].join(" "));
    for body in ["(hello)", "(\"hello\")", "( hello )"] {
        let text = format!("{prefix}{body}");
        let segments = frontend
            .prepare(&SynthesisRequest::new(&text), &vocab)
            .unwrap();
        assert_eq!(
            segments.iter().map(|s| s.text()).collect::<Vec<_>>(),
            [prefix.as_str(), body]
        );
    }
    for opening in ["(", "\""] {
        let fitting = format!("{opening}hello{}", ")".repeat(504));
        let segments = frontend
            .prepare(&SynthesisRequest::new(fitting), &vocab)
            .unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].phonemes().chars().count(), 510);
        let oversized = format!("{opening}hello{}", ")".repeat(505));
        assert!(matches!(
            frontend.prepare(&SynthesisRequest::new(oversized), &vocab),
            Err(SpeechError::Input { .. })
        ));
    }
}

#[test]
fn frontend_pause_marks_preserve_sentence_boundaries_and_phones() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    for pause in [";", ":", ",", "\u{2014}", "\u{2026}"] {
        let text = format!("Hello{pause} Next.");
        let segments = frontend
            .prepare(&SynthesisRequest::new(&text), &vocab)
            .unwrap();
        assert_eq!(segments.len(), 1, "pause {pause:?}");
        assert_eq!(segments[0].text(), text);
        assert_eq!(
            segments[0].phonemes(),
            format!("h\u{259}l\u{2c8}O{pause} n\u{2c8}\u{25b}kst.")
        );
    }
    for (text, expected) in [
        ("Hello! (Next.)", ["Hello!", "(Next.)"]),
        ("Hello!(Next.)", ["Hello!", "(Next.)"]),
        ("Hello! \"Next.\"", ["Hello!", "\"Next.\""]),
    ] {
        let segments = frontend
            .prepare(&SynthesisRequest::new(text), &vocab)
            .unwrap();
        assert_eq!(
            segments.iter().map(|s| s.text()).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn frontend_adjacent_sentence_opening_quotes_stay_forward() {
    // Directional chunking is a contract test. The independent Python corpus
    // cannot pronounce some of these no-space composites as lexical tokens.
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let mut failures = Vec::new();
    for (text, expected) in [
        ("Hello!\"Next.\"", ["Hello!", "\"Next.\""]),
        ("Hello!\u{201c}Next.\u{201d}", ["Hello!", "\"Next.\""]),
        ("Hello?!\"Next.\"", ["Hello?!", "\"Next.\""]),
        ("Hello...\"Next.\"", ["Hello...", "\"Next.\""]),
        ("Hello!\"(Next.)\"", ["Hello!", "\"(Next.)\""]),
        ("Hello!\"((Next.))\"", ["Hello!", "\"((Next.))\""]),
        ("Hello!(\"Next.\")", ["Hello!", "(\"Next.\")"]),
        ("Hello! \"Next.\"", ["Hello!", "\"Next.\""]),
        ("Hello!\" Next.", ["Hello!\"", "Next."]),
        ("\"Hello!\" Next.", ["\"Hello!\"", "Next."]),
        ("Hello!\"\" Next.", ["Hello!\"\"", "Next."]),
    ] {
        let segments = frontend
            .prepare(&SynthesisRequest::new(text), &vocab)
            .unwrap();
        let actual = segments.iter().map(|s| s.text()).collect::<Vec<_>>();
        if actual != expected {
            failures.push(format!("{text:?}: expected {expected:?}, got {actual:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_opening_quotes_stay_forward_at_context_limit() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let prefix = format!("{} said", vec!["hello"; 84].join(" "));
    let bare = frontend
        .prepare(&SynthesisRequest::new(&prefix), &vocab)
        .unwrap();
    assert_eq!(bare.len(), 1);
    assert_eq!(bare[0].phonemes().chars().count(), 508);
    let mut failures = Vec::new();
    for mark in ["", ",", ":", ";", "\u{2014}", "\u{2026}", "!", "?", "."] {
        for gap in ["", " "] {
            for (opening, closing) in [("\"", "\""), ("\u{201c}", "\u{201d}")] {
                let before = format!("{prefix}{mark}");
                let text = format!("{before}{gap}{opening}hello{closing}");
                let segments = frontend
                    .prepare(&SynthesisRequest::new(&text), &vocab)
                    .unwrap();
                let actual = segments.iter().map(|s| s.text()).collect::<Vec<_>>();
                let expected = [before.as_str(), "\"hello\""];
                if actual != expected {
                    failures.push(format!(
                        "mark {mark:?}, gap {gap:?}, quote {opening:?}: expected {expected:?}, got {actual:?}"
                    ));
                    continue;
                }
                assert_eq!(
                    segments[0].phonemes(),
                    format!("{}{mark}", bare[0].phonemes())
                );
                assert_eq!(segments[1].phonemes(), "\u{201c}h\u{259}l\u{2c8}O\u{201d}");
                assert_eq!(segments[1].style_row(), 6);
                assert_eq!(
                    format!("{}{gap}{}", actual[0], actual[1]),
                    normalize::text(&text)
                );
                for segment in segments {
                    let count = segment.phonemes().chars().count();
                    assert!((1..=510).contains(&count));
                    assert_eq!(segment.style_row(), count - 1);
                    assert_eq!(segment.ids().len(), count + 2);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_nested_quote_groups_obey_context_fit_and_refusal() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let prefix = format!("{} said,", vec!["hello"; 84].join(" "));
    let mut failures = Vec::new();
    for (body, count) in [
        ("\"hello\"", 7),
        ("\"(hello)\"", 9),
        ("(\"hello\")", 9),
        ("\"((hello))\"", 11),
        ("(\"(hello)\")", 11),
        ("((\"hello\"))", 11),
        ("\"( hello )\"", 11),
        ("\u{201c}(hello)\u{201d}", 9),
        ("(\u{201c}hello\u{201d})", 9),
    ] {
        let normalized = normalize::text(body);
        let text = format!("{prefix}{body}");
        let segments = frontend
            .prepare(&SynthesisRequest::new(&text), &vocab)
            .unwrap();
        let actual = segments.iter().map(|s| s.text()).collect::<Vec<_>>();
        let expected = [prefix.as_str(), normalized.as_str()];
        if actual != expected {
            failures.push(format!(
                "nested {body:?}: expected {expected:?}, got {actual:?}"
            ));
        } else {
            assert_eq!(segments[0].phonemes().chars().count(), 509);
            assert_eq!(segments[1].phonemes().chars().count(), count);
        }

        let fitting = PhonemeVocabulary::new(vocab.map.clone(), 178, 178, count + 2).unwrap();
        let text = format!("a,{body}");
        let segments = frontend
            .prepare(&SynthesisRequest::new(&text), &fitting)
            .unwrap();
        let actual = segments.iter().map(|s| s.text()).collect::<Vec<_>>();
        if actual != ["a,", normalized.as_str()] {
            failures.push(format!("exact-fit {body:?}: got {actual:?}"));
        } else {
            assert_eq!(segments[1].phonemes().chars().count(), count);
            assert_eq!(segments[1].style_row(), count - 1);
            assert_eq!(segments[1].ids().len(), count + 2);
        }
        let too_small = PhonemeVocabulary::new(vocab.map.clone(), 178, 178, count + 1).unwrap();
        if !matches!(
            frontend.prepare(&SynthesisRequest::new(&text), &too_small),
            Err(SpeechError::Input { .. })
        ) {
            failures.push(format!("oversized forward group {body:?} was accepted"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_spaced_punctuation_keeps_word_and_context_bound() {
    let frontend = EnglishFrontend::new();
    let vocab = vocab();
    let prefix = format!("{} he a", vec!["hello"; 84].join(" "));
    let mut failures = Vec::new();
    for suffix in [
        " !",
        " ?",
        " ;",
        " :",
        " ,",
        " )",
        " \u{2014}",
        " \u{2026}",
        " \u{2026} !)",
    ] {
        let text = format!("{prefix} a{suffix}");
        let segments = frontend
            .prepare(&SynthesisRequest::new(text), &vocab)
            .unwrap();
        let actual = segments.iter().map(|s| s.text()).collect::<Vec<_>>();
        let expected = [prefix.clone(), format!("a{suffix}")];
        if actual != expected {
            failures.push(format!(
                "spaced {suffix:?}: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    for mark in ["!", "\u{2026}", ")"] {
        let fitting = format!("hello {}", mark.repeat(504));
        let segments = frontend
            .prepare(&SynthesisRequest::new(fitting), &vocab)
            .unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].phonemes().chars().count(), 510);
        let oversized = format!("hello {}", mark.repeat(505));
        if !matches!(
            frontend.prepare(&SynthesisRequest::new(oversized), &vocab),
            Err(SpeechError::Input { .. })
        ) {
            failures.push(format!("spaced oversized {mark:?} was accepted"));
        }
    }
    for (opening, closing) in [("(", ")"), ("\"", "\"")] {
        let prefix = format!("{opening}{} a a", vec!["hello"; 84].join(" "));
        let text = format!("{prefix} a {closing}");
        let segments = frontend
            .prepare(&SynthesisRequest::new(text), &vocab)
            .unwrap();
        let actual = segments.iter().map(|s| s.text()).collect::<Vec<_>>();
        let expected = [prefix, format!("a {closing}")];
        if actual != expected {
            failures.push(format!(
                "spaced closer {closing:?}: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_malformed_decimal_returns_error() {
    let error = EnglishFrontend::new()
        .prepare(&SynthesisRequest::new("1.2,345"), &vocab())
        .unwrap_err();
    assert!(matches!(error, SpeechError::Input { .. }));
}

#[test]
fn frontend_spelling_refuses_to_drop_nonletters() {
    let error = EnglishFrontend::new()
        .prepare(&SynthesisRequest::new("qzx'wv"), &vocab())
        .unwrap_err();
    assert!(matches!(error, SpeechError::Unsupported { .. }));
}
