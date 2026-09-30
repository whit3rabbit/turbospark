use turbospark_catalog::{
    group_variants, quant_label, QuantLabel, RepoFile, ShardSetIssue, ShardSetStatus,
};

fn file(name: &str) -> RepoFile {
    RepoFile {
        name: name.to_string(),
        size: None,
    }
}

#[test]
fn labels_are_canonical_and_independent_of_model_family() {
    let text_label = quant_label("Qwen3-8B-q4_k_m");
    let non_text_label = quant_label("z-image-turbo.q4_k_m.gguf");
    assert_eq!(text_label, Some(QuantLabel("Q4_K_M".to_string())));
    assert_eq!(non_text_label, text_label);
    assert_eq!(
        quant_label("audio-model-IQ4_XS"),
        Some(QuantLabel("IQ4_XS".to_string()))
    );
    // Recognition is filename taxonomy, not a claim that a kernel can run it.
    assert_eq!(
        quant_label("unsupported-Q5_0.gguf"),
        Some(QuantLabel("Q5_0".to_string()))
    );
}

#[test]
fn groups_plain_variants_and_complete_shards_deterministically() {
    let variants = group_variants(&[
        file("Qwen3-8B-Q5_0.gguf"),
        file("Qwen3-8B-Q4_K_M-00002-of-00002.gguf"),
        file("README.md"),
        file("Qwen3-8B-Q4_K_M-00001-of-00002.gguf"),
        file("Qwen3-8B-f16.gguf"),
        file("unrecognized-weights.gguf"),
    ]);

    assert_eq!(variants.len(), 3);
    let q4 = variants
        .iter()
        .find(|variant| variant.label == QuantLabel("Q4_K_M".to_string()))
        .expect("Q4 variant");
    assert_eq!(
        q4.files
            .iter()
            .map(|file| file.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "Qwen3-8B-Q4_K_M-00001-of-00002.gguf",
            "Qwen3-8B-Q4_K_M-00002-of-00002.gguf"
        ]
    );
    assert_eq!(q4.shard_set, ShardSetStatus::Complete { expected_count: 2 });
    assert_eq!(
        variants
            .iter()
            .map(|variant| variant.label.0.as_str())
            .collect::<Vec<_>>(),
        vec!["F16", "Q4_K_M", "Q5_0"]
    );
    for label in ["F16", "Q5_0"] {
        let plain = variants
            .iter()
            .find(|variant| variant.label.0 == label)
            .expect("plain variant");
        assert_eq!(plain.shard_set, ShardSetStatus::SingleFile);
    }
}

#[test]
fn only_gguf_extensions_join_a_variant_case_insensitively() {
    let variants = group_variants(&[
        file("model-Q4_K_M.safetensors"),
        file("model-Q4_K_M.bin"),
        file("model-Q4_K_M.GGUF"),
    ]);
    assert_eq!(variants.len(), 1);
    assert_eq!(variants[0].label, QuantLabel("Q4_K_M".to_string()));
    assert_eq!(variants[0].files, vec![file("model-Q4_K_M.GGUF")]);
    assert_eq!(variants[0].shard_set, ShardSetStatus::SingleFile);
}

#[test]
fn malformed_shard_suffixes_are_ignored_and_leave_valid_siblings_incomplete() {
    assert!(group_variants(&[file("model-Q4_K_M-00001-of-x.gguf")]).is_empty());
    assert!(group_variants(&[file("model-Q4_K_M-x-of-00002.gguf")]).is_empty());

    let ordinary = group_variants(&[file("mixture-of-experts-Q4_K_M.gguf")]);
    assert_eq!(ordinary.len(), 1);
    assert_eq!(ordinary[0].shard_set, ShardSetStatus::SingleFile);

    let variants = group_variants(&[
        file("model-Q4_K_M-00001-of-00002.gguf"),
        file("model-Q4_K_M-00002-of-x.gguf"),
    ]);
    assert_eq!(variants.len(), 1);
    assert_eq!(
        variants[0].files,
        vec![file("model-Q4_K_M-00001-of-00002.gguf")]
    );
    assert_eq!(
        variants[0].shard_set,
        ShardSetStatus::Incomplete {
            expected_count: 2,
            present_indices: vec![1]
        }
    );

    let variants = group_variants(&[
        file("model-Q4_K_M-00001-of-00002.gguf"),
        file("model-Q4_K_M-x-of-00002.gguf"),
    ]);
    assert_eq!(variants.len(), 1);
    assert_eq!(
        variants[0].files,
        vec![file("model-Q4_K_M-00001-of-00002.gguf")]
    );
    assert_eq!(
        variants[0].shard_set,
        ShardSetStatus::Incomplete {
            expected_count: 2,
            present_indices: vec![1]
        }
    );
}

#[test]
fn an_incomplete_shard_set_reports_present_indices() {
    let variants = group_variants(&[
        file("model-Q4_K_M-00003-of-00003.gguf"),
        file("model-Q4_K_M-00001-of-00003.gguf"),
    ]);
    assert_eq!(variants.len(), 1);
    assert_eq!(
        variants[0].shard_set,
        ShardSetStatus::Incomplete {
            expected_count: 3,
            present_indices: vec![1, 3]
        }
    );
}

#[test]
fn recognizes_dot_and_underscore_shard_separators() {
    let dotted = group_variants(&[file("model.q4_k_m.00001-of-00001.gguf")]);
    assert_eq!(dotted.len(), 1);
    assert_eq!(dotted[0].label, QuantLabel("Q4_K_M".to_string()));
    assert_eq!(
        dotted[0].shard_set,
        ShardSetStatus::Complete { expected_count: 1 }
    );

    let underscored = group_variants(&[
        file("audio_model_Q4_K_M_00001-of-00002.gguf"),
        file("audio_model_Q4_K_M_00002-of-00002.gguf"),
    ]);
    assert_eq!(underscored.len(), 1);
    assert_eq!(
        underscored[0].shard_set,
        ShardSetStatus::Complete { expected_count: 2 }
    );
}

#[test]
fn duplicate_conflicting_and_out_of_range_shards_are_inconsistent() {
    let duplicate = group_variants(&[
        file("model-Q4_K_M-1-of-2.gguf"),
        file("model-Q4_K_M-00001-of-02.gguf"),
    ]);
    assert_eq!(
        duplicate[0].shard_set,
        ShardSetStatus::Inconsistent {
            issues: vec![ShardSetIssue::DuplicateIndex { index: 1 }]
        }
    );

    let conflicting = group_variants(&[
        file("model-Q4_K_M-00001-of-00002.gguf"),
        file("model-Q4_K_M-00002-of-00003.gguf"),
    ]);
    assert_eq!(
        conflicting[0].shard_set,
        ShardSetStatus::Inconsistent {
            issues: vec![ShardSetIssue::ConflictingCounts {
                declared_counts: vec![2, 3]
            }]
        }
    );

    let out_of_range = group_variants(&[
        file("model-Q4_K_M-00001-of-00002.gguf"),
        file("model-Q4_K_M-00003-of-00002.gguf"),
    ]);
    assert_eq!(
        out_of_range[0].shard_set,
        ShardSetStatus::Inconsistent {
            issues: vec![ShardSetIssue::UnexpectedIndex {
                index: 3,
                declared_count: 2
            }]
        }
    );
}

#[test]
fn shards_split_across_quant_labels_are_not_reported_as_separate_complete_sets() {
    let variants = group_variants(&[
        file("model-Q4_K_M-00001-of-00002.gguf"),
        file("model-Q5_K_M-00002-of-00002.gguf"),
    ]);
    assert_eq!(variants.len(), 2);
    for variant in variants {
        assert_eq!(
            variant.shard_set,
            ShardSetStatus::Inconsistent {
                issues: vec![ShardSetIssue::MixedVariantKey]
            }
        );
    }

    let complete = group_variants(&[
        file("model-Q4_K_M-00001-of-00002.gguf"),
        file("model-Q4_K_M-00002-of-00002.gguf"),
        file("model-Q5_K_M-00001-of-00002.gguf"),
        file("model-Q5_K_M-00002-of-00002.gguf"),
    ]);
    assert_eq!(complete.len(), 2);
    assert!(complete
        .iter()
        .all(|variant| variant.shard_set == ShardSetStatus::Complete { expected_count: 2 }));
}

#[test]
fn unrecognized_gguf_filenames_do_not_create_variants() {
    assert_eq!(quant_label("Qwen3-8B-quantized"), None);
    assert!(group_variants(&[
        file("README.md"),
        file("weights.gguf"),
        file("model-unknown-label.gguf")
    ])
    .is_empty());
}
