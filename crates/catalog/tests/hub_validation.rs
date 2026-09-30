use turbospark_catalog::{HubMetadataValidator, HubValidationError, HubValidationLimits};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

fn make_validator(limits: HubValidationLimits) -> HubMetadataValidator {
    HubMetadataValidator::new(limits)
}

#[test]
fn search_fixture_keeps_valid_rows_and_names_invalid_entries_and_rules() {
    let validator = make_validator(HubValidationLimits::default());
    let body = format!(
        r#"[
            {{"id":"owner/good-model","sha":"{REVISION}","downloads":12,"likes":3}},
            {{"id":"owner/../escape","sha":"{REVISION}"}},
            {{"id":"owner/bad-revision","sha":"main"}},
            {{"id":"owner/bad-count","sha":"{REVISION}","downloads":-1}}
        ]"#
    );

    let report = validator.validate_search_payload(body.as_bytes()).unwrap();

    assert_eq!(report.valid.len(), 1);
    assert_eq!(report.valid[0].repo_id, "owner/good-model");
    assert_eq!(report.valid[0].revision, REVISION);
    assert_eq!(report.rejected.len(), 3);
    assert_eq!(report.rejected[0].entry, "owner/../escape");
    assert_eq!(report.rejected[0].rule, "repository_id_shape");
    assert_eq!(report.rejected[1].entry, "owner/bad-revision");
    assert_eq!(report.rejected[1].rule, "immutable_revision");
    assert_eq!(report.rejected[2].entry, "owner/bad-count");
    assert_eq!(report.rejected[2].rule, "nonnegative_count");
}

#[test]
fn repository_fixture_isolates_unsafe_paths_and_invalid_sizes() {
    let limits = HubValidationLimits {
        max_file_size_bytes: 100,
        ..HubValidationLimits::default()
    };
    let validator = make_validator(limits);
    let body = format!(
        r#"{{"id":"owner/model","sha":"{REVISION}","siblings":[
            {{"rfilename":"model-Q4_K_M.gguf","size":80}},
            {{"rfilename":"config.json","size":4}},
            {{"rfilename":"tokenizer.json"}},
            {{"rfilename":"../outside.gguf","size":4}},
            {{"rfilename":"/absolute.gguf","size":4}},
            {{"rfilename":"nested\\escape.gguf","size":4}},
            {{"rfilename":"huge.bin","size":101}},
            {{"rfilename":"bad-size.bin","size":"large"}}
        ]}}"#
    );

    let report = validator
        .validate_repository_payload(body.as_bytes())
        .unwrap();

    assert_eq!(report.valid.repo_id, "owner/model");
    assert_eq!(report.valid.revision, REVISION);
    assert_eq!(
        report
            .valid
            .files
            .iter()
            .map(|file| (file.path.as_str(), file.size_bytes))
            .collect::<Vec<_>>(),
        vec![
            ("model-Q4_K_M.gguf", Some(80)),
            ("config.json", Some(4)),
            ("tokenizer.json", None)
        ]
    );
    assert_eq!(report.rejected.len(), 5);
    assert_eq!(report.rejected[0].entry, "../outside.gguf");
    assert_eq!(report.rejected[0].rule, "safe_relative_path");
    assert_eq!(report.rejected[1].entry, "/absolute.gguf");
    assert_eq!(report.rejected[1].rule, "safe_relative_path");
    assert_eq!(report.rejected[2].entry, "nested\\escape.gguf");
    assert_eq!(report.rejected[2].rule, "safe_relative_path");
    assert_eq!(report.rejected[3].entry, "huge.bin");
    assert_eq!(report.rejected[3].rule, "file_size_within_limit");
    assert_eq!(report.rejected[4].entry, "bad-size.bin");
    assert_eq!(report.rejected[4].rule, "nonnegative_size");
}

#[test]
fn url_component_and_windows_path_syntax_are_rejected() {
    let validator = make_validator(HubValidationLimits::default());
    let body = format!(
        r#"{{"id":"owner/model","sha":"{REVISION}","siblings":[
            {{"rfilename":"nested/config.json","size":4}},
            {{"rfilename":"file?download=true","size":4}},
            {{"rfilename":"file#fragment","size":4}},
            {{"rfilename":"encoded%2e%2e/path","size":4}},
            {{"rfilename":"C:/outside.gguf","size":4}}
        ]}}"#
    );

    let report = validator
        .validate_repository_payload(body.as_bytes())
        .unwrap();

    assert_eq!(report.valid.files.len(), 1);
    assert_eq!(report.rejected.len(), 4);
    assert!(report
        .rejected
        .iter()
        .all(|entry| entry.rule == "safe_relative_path"));
}

#[test]
fn malformed_json_and_wrong_top_level_shapes_fail_closed() {
    let validator = make_validator(HubValidationLimits::default());

    assert_eq!(
        validator.validate_search_payload(b"[{"),
        Err(HubValidationError::InvalidResponse {
            entry: "<response>".to_string(),
            rule: "valid_json"
        })
    );
    assert_eq!(
        validator.validate_search_payload(br#"{"results":[]}"#),
        Err(HubValidationError::InvalidResponse {
            entry: "<response>".to_string(),
            rule: "search_array_shape"
        })
    );
    assert_eq!(
        validator.validate_repository_payload(br#"{"siblings":[]}"#),
        Err(HubValidationError::InvalidResponse {
            entry: "<repository>".to_string(),
            rule: "repository_metadata_shape"
        })
    );
}

#[test]
fn response_and_entry_count_limits_refuse_oversized_payloads() {
    let limits = HubValidationLimits {
        max_response_bytes: 20,
        max_search_entries: 1,
        max_repo_files: 1,
        ..HubValidationLimits::default()
    };
    let validator = make_validator(limits);

    assert!(matches!(
        validator.validate_search_payload(br#"[{"id":"owner/model"}]"#),
        Err(HubValidationError::OversizedResponse { .. })
    ));

    let limits = HubValidationLimits {
        max_search_entries: 1,
        ..HubValidationLimits::default()
    };
    let validator = make_validator(limits);
    let body =
        format!(r#"[{{"id":"owner/a","sha":"{REVISION}"}},{{"id":"owner/b","sha":"{REVISION}"}}]"#);
    assert_eq!(
        validator.validate_search_payload(body.as_bytes()),
        Err(HubValidationError::InvalidResponse {
            entry: "<response>".to_string(),
            rule: "entry_count_within_limit"
        })
    );

    let limits = HubValidationLimits {
        max_repo_files: 1,
        ..HubValidationLimits::default()
    };
    let validator = make_validator(limits);
    let body = format!(
        r#"{{"id":"owner/model","sha":"{REVISION}","siblings":[{{"rfilename":"a"}},{{"rfilename":"b"}}]}}"#
    );
    assert_eq!(
        validator.validate_repository_payload(body.as_bytes()),
        Err(HubValidationError::InvalidResponse {
            entry: "owner/model".to_string(),
            rule: "file_count_within_limit"
        })
    );
}

#[test]
fn checked_variant_totals_refuse_overflow_and_preserve_unknown_sizes() {
    let validator = make_validator(HubValidationLimits {
        max_file_size_bytes: u64::MAX,
        max_file_set_size_bytes: u64::MAX,
        ..HubValidationLimits::default()
    });
    let body = format!(
        r#"{{"id":"owner/model","sha":"{REVISION}","siblings":[
            {{"rfilename":"a.gguf","size":18446744073709551615}},
            {{"rfilename":"b.gguf","size":1}}
        ]}}"#
    );
    let report = validator
        .validate_repository_payload(body.as_bytes())
        .unwrap();
    assert!(matches!(
        validator.checked_file_total(&report.valid.files),
        Err(HubValidationError::InvalidResponse {
            rule: "checked_size_total",
            ..
        })
    ));

    let unknown = vec![turbospark_catalog::HubFileMetadata {
        path: "unsized.gguf".to_string(),
        size_bytes: None,
    }];
    assert_eq!(validator.checked_file_total(&unknown), Ok(None));

    let empty_gguf = vec![turbospark_catalog::HubFileMetadata {
        path: "empty.gguf".to_string(),
        size_bytes: Some(0),
    }];
    assert_eq!(
        validator.checked_file_total(&empty_gguf),
        Err(HubValidationError::InvalidResponse {
            entry: "empty.gguf".to_string(),
            rule: "nonzero_gguf_size"
        })
    );

    let size_limited = make_validator(HubValidationLimits {
        max_file_size_bytes: 100,
        max_file_set_size_bytes: 9,
        ..HubValidationLimits::default()
    });
    let files = vec![
        turbospark_catalog::HubFileMetadata {
            path: "first.gguf".to_string(),
            size_bytes: Some(4),
        },
        turbospark_catalog::HubFileMetadata {
            path: "second.gguf".to_string(),
            size_bytes: Some(6),
        },
    ];
    assert_eq!(
        size_limited.checked_file_total(&files),
        Err(HubValidationError::InvalidResponse {
            entry: "second.gguf".to_string(),
            rule: "file_set_size_within_limit"
        })
    );
}

#[test]
fn cached_payloads_can_be_revalidated_by_the_same_boundary() {
    let validator = make_validator(HubValidationLimits::default());
    let cached = format!(r#"[{{"id":"owner/model","sha":"{REVISION}"}}]"#);

    let first = validator
        .validate_search_payload(cached.as_bytes())
        .unwrap();
    let second = validator
        .validate_search_payload(cached.as_bytes())
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(second.valid.len(), 1);

    let hostile_cached = format!(
        r#"{{"id":"owner/model","sha":"{REVISION}","siblings":[{{"rfilename":"../../outside.gguf","size":10}}]}}"#
    );
    for _ in 0..2 {
        let reread = validator
            .validate_repository_payload(hostile_cached.as_bytes())
            .unwrap();
        assert!(reread.valid.files.is_empty());
        assert_eq!(reread.rejected.len(), 1);
        assert_eq!(reread.rejected[0].entry, "../../outside.gguf");
        assert_eq!(reread.rejected[0].rule, "safe_relative_path");
    }
}
