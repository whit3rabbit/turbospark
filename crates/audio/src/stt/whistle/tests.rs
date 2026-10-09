use super::*;

fn reference(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(expected).all(|v| v.is_finite()));
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max <= tolerance, "max error {max} exceeds {tolerance}");
}

#[test]
fn cq_and_hadamard_mlp_match_pinned_release_oracle() {
    let archive = cact::Archive::parse(include_bytes!("fixtures/ops.cact").to_vec()).unwrap();
    assert_eq!(archive.len(), 18);
    let matrix = archive.quantized(0).unwrap();
    close(
        &matrix.row(0),
        &reference(include_bytes!("fixtures/q_row.f32")),
        1e-6,
    );
    let x: Vec<f32> = (0..512)
        .map(|i| ((i as f64 * 0.1).sin() * 0.3) as f32)
        .collect();
    close(
        &matrix.project(&x),
        &reference(include_bytes!("fixtures/q_project.f32")),
        3e-6,
    );
    let permutations = [
        layers::permutation(&archive, 14).unwrap(),
        layers::permutation(&archive, 15).unwrap(),
    ];
    let mlp = layers::HadamardMlp::load(&archive, 1).unwrap();
    close(
        &mlp.forward(&x, &permutations[0], &permutations[1]),
        &reference(include_bytes!("fixtures/hada.f32")),
        4e-6,
    );
}

#[test]
fn frontend_matches_pinned_release_oracle() {
    let archive = cact::Archive::parse(include_bytes!("fixtures/ops.cact").to_vec()).unwrap();
    let frontend =
        frontend::Frontend::new(layers::floats(&archive, 16, &[257, 80]).unwrap()).unwrap();
    let pcm: Vec<f32> = (0..16_000)
        .map(|i| (0.2 * (std::f64::consts::TAU * 440.0 * i as f64 / 16_000.0).sin()) as f32)
        .collect();
    close(
        &frontend.features(&pcm).unwrap(),
        &reference(include_bytes!("fixtures/tone_features.f32")),
        0.0002,
    );
}

#[test]
fn tokenizer_roundtrips_utf8_resolves_languages_and_refuses_invalid_ids() {
    let archive = cact::Archive::parse(include_bytes!("fixtures/ops.cact").to_vec()).unwrap();
    let tokenizer = tokenizer::Tokenizer::parse(archive.raw(17).unwrap(), 8199).unwrap();
    for text in [
        "Hello world!",
        "Gr\u{fc}\u{df}e, Fran\u{e7}ais",
        "\u{1f680}",
        "  leading and trailing  ",
    ] {
        assert_eq!(tokenizer.decode(&tokenizer.encode(text)).unwrap(), text);
    }
    for language in LANGUAGES {
        let marker = format!("<|{language}|>");
        assert_eq!(
            tokenizer.encode(&marker),
            [tokenizer.marker(&marker).unwrap()]
        );
    }
    assert!(tokenizer.marker("<|ja|>").is_err());
    assert!(tokenizer.decode(&[-1]).is_err());
    assert!(tokenizer.decode(&[8199]).is_err());
    assert!(tokenizer::Tokenizer::parse(&archive.raw(17).unwrap()[..23], 8199).is_err());
    let mut incomplete = archive.raw(17).unwrap().to_vec();
    incomplete.pop();
    assert!(tokenizer::Tokenizer::parse(&incomplete, 8199).is_err());
}

#[test]
fn refuses_operator_fixture_as_a_whistle_checkpoint() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stt/whistle/fixtures/ops.cact");
    assert!(matches!(
        Whistle::load(&path),
        Err(SpeechError::BadConfig { .. })
    ));
    assert!(finite(&[f32::INFINITY]).is_err());
}

#[test]
#[ignore = "requires pinned checkpoint and local speech.wav / jfk.wav; see README"]
fn checkpoint_transcribes_synthetic_and_natural_speech() {
    let root = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_WHISTLE_REFERENCE").expect("TURBOSPARK_WHISTLE_REFERENCE"),
    );
    let model = Whistle::load(&root.join("whistle.cact")).unwrap();
    for (name, expected) in [
        ("speech.wav", "The quick brown fox jumps over the lazy dog. This is a test of speech recognition."),
        ("jfk.wav", "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."),
    ] {
        let waveform = crate::read_wav_f32(&root.join(name)).unwrap();
        assert_eq!((waveform.sample_rate,waveform.channels),(16_000,1));
        for language in [Some("en"), None] {
            let transcript = model.transcribe(&waveform.samples,language,100).unwrap();
            assert_eq!(transcript.text,expected,"{name}, language={language:?}");
            assert_eq!(transcript.language,"en");
            assert!(!transcript.truncated);
        }
        let capped = model.transcribe(&waveform.samples,Some("en"),1).unwrap();
        assert_eq!(capped.tokens.len(),1);
        assert!(capped.truncated);
    }
    let encoded = model.encode(&[0.0; 16_000]).unwrap();
    assert_eq!(
        (encoded.frames, encoded.width, encoded.values.len()),
        (13, 512, 13 * 512)
    );
    assert!(encoded.values.iter().all(|v| v.is_finite()));
    let silent = model.transcribe(&[0.0; 160], None, 20).unwrap();
    assert!(silent.text.is_empty() && silent.tokens.is_empty() && !silent.truncated);
    assert!(model.transcribe(&[0.0; 160], Some("ja"), 20).is_err());
    for cap in [0, 319] {
        assert!(model.transcribe(&[0.0; 160], None, cap).is_err());
    }
}
