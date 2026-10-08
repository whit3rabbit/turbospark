#[test]
fn language_without_emitted_metadata_stays_absent() {
    use turbospark_audio::stt::qwen3_asr::reported_language;
    assert_eq!(reported_language("Plain text transcript"), None);
    assert_eq!(reported_language("language None<asr_text>"), None);
    assert_eq!(
        reported_language("language French<asr_text>Bonjour"),
        Some("French".into())
    );
}
