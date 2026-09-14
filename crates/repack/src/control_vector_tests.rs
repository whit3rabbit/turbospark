use super::*;

/// Blocks 1..=`layers`, because block 0 has no name in this format.
fn dirs(layers: usize, hidden: usize) -> BTreeMap<usize, Vec<f32>> {
    (1..=layers)
        .map(|l| {
            (
                l,
                (0..hidden)
                    .map(|i| ((l * hidden + i) as f32 * 0.37).sin())
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn a_written_vector_reads_back_with_the_same_values() {
    let want = dirs(4, 16);
    let bytes = write_control_vector(&want, "qwen35", Some(SteeringMode::Ablate)).expect("writes");
    let set = parse_control_vector(&bytes).expect("parses");

    assert_eq!(set.hidden, 16);
    assert_eq!(set.layers.len(), 5, "spanned to the highest block, 4");
    assert_eq!(set.covered_layers(), 4);
    assert_eq!(set.declared_mode, Some(SteeringMode::Ablate));
    assert_eq!(set.declared_arch.as_deref(), Some("qwen35"));
    assert!(set.layer(0).is_none(), "block 0 is not expressible");
    for (l, w) in &want {
        let got = set.layer(*l).expect("block present");
        assert_eq!(&got.values, w, "block {l}");
    }
}

/// The round trip above cannot see a shift that the writer and reader
/// make together, so this pins the wire name against the block index
/// llama.cpp resolves it to. `direction.1` is block 1: its loader puts
/// `direction.N` at buffer offset `n_embd * (N - 1)` and its applier
/// reads block `il` from `n_embd * (il - 1)`.
///
/// This is the assertion that was WRONG for the life of the module, so it
/// is stated against the reference's arithmetic rather than against the
/// writer beside it.
#[test]
fn a_written_direction_lands_on_the_block_llama_cpp_would_apply_it_to() {
    let mut want = BTreeMap::new();
    want.insert(7usize, vec![1.0f32; 4]);
    let bytes = write_control_vector(&want, "qwen35", None).expect("writes");
    let header = parse_header(&bytes, bytes.len() as u64).expect("valid GGUF");

    let names: Vec<&str> = header.tensors.keys().map(String::as_str).collect();
    assert_eq!(names, vec!["direction.7"], "block 7 is direction.7, not 8");

    let set = parse_control_vector(&bytes).expect("parses");
    assert!(set.layer(7).is_some(), "reads back to the same block");
    assert!(set.layer(6).is_none() && set.layer(8).is_none());
}

/// The reader and the writer beside it share an author, so a round trip
/// alone cannot say the file matches llama.cpp's layout. This checks the
/// three things that layout actually requires, against the BYTES.
#[test]
fn the_written_bytes_match_the_documented_layout() {
    let bytes = write_control_vector(&dirs(3, 8), "qwen35", None).expect("writes");
    let header = parse_header(&bytes, bytes.len() as u64).expect("valid GGUF");

    let names: Vec<&str> = header.tensors.keys().map(String::as_str).collect();
    assert!(
        names.contains(&"direction.1") && names.contains(&"direction.3"),
        "names are ONE-based: {names:?}"
    );
    assert!(
        !names.contains(&"direction.0"),
        "a zero index is rejected by llama.cpp by name"
    );
    for (name, info) in &header.tensors {
        assert_eq!(info.ggml_type, GGML_TYPE_F32, "{name} must be F32");
        assert_eq!(info.dims.len(), 1, "{name} must be one-dimensional");
    }
}

/// Under llama.cpp's numbering `direction.N` IS block N. This test
/// asserted `direction.1 == block 0` from the first commit and was the
/// off-by-one in its purest form.
#[test]
fn direction_one_is_block_one() {
    assert_eq!(layer_index("direction.1", 1).unwrap(), 1);
    assert_eq!(layer_index("direction.64", 1).unwrap(), 64);
}

/// Legacy files this port wrote before the correction. They keep meaning
/// what they meant when the frozen rows were measured against them.
#[test]
fn a_layer_base_of_zero_reads_the_old_way() {
    assert_eq!(layer_index("direction.1", 0).unwrap(), 0);
    assert_eq!(layer_index("direction.64", 0).unwrap(), 63);
}

/// The two conventions must be DISTINGUISHABLE, or honouring the key is
/// theatre: a test whose fixture reads the same under both would pass
/// against a reader that ignored `layer_base` entirely.
#[test]
fn the_two_conventions_differ_by_exactly_one_block() {
    for n in ["direction.1", "direction.9", "direction.64"] {
        let new = layer_index(n, 1).unwrap();
        let old = layer_index(n, 0).unwrap();
        assert_eq!(new, old + 1, "{n} must shift by one, not by zero");
    }
}

#[test]
fn a_zero_index_is_refused_rather_than_shifted() {
    for base in [0, 1] {
        assert_eq!(
            layer_index("direction.0", base),
            Err(ControlVectorError::ZeroLayerIndex),
            "base {base}"
        );
    }
}

/// `direction.N`'s block index must not be allowed to size a `Vec` off an
/// attacker-controlled tensor name: `read_set` allocates `highest + 1`
/// entries, so `direction.18446744073709551614` (which used to abort with a
/// capacity overflow) and a merely huge-but-finite index must both be
/// refused rather than attempted.
#[test]
fn an_absurdly_large_layer_index_is_refused_rather_than_allocated() {
    assert!(matches!(
        layer_index("direction.18446744073709551614", 1),
        Err(ControlVectorError::LayerIndexTooLarge { .. })
    ));
    assert!(matches!(
        layer_index(&format!("direction.{}", MAX_LAYER_INDEX + 2), 1),
        Err(ControlVectorError::LayerIndexTooLarge { .. })
    ));
    // The boundary itself must still be accepted.
    assert!(layer_index(&format!("direction.{MAX_LAYER_INDEX}"), 1).is_ok());
}

/// The end-to-end path: a file naming an absurd block must fail to parse
/// rather than panic while sizing the layer vector.
#[test]
fn parsing_a_vector_with_an_absurd_layer_index_fails_cleanly() {
    let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let builder = crate::GgufBuilder::new().tensor(
        "direction.18446744073709551615",
        GGML_TYPE_F32,
        &[8],
        data,
    );
    assert!(matches!(
        parse_control_vector(&builder.build().0),
        Err(ControlVectorError::LayerIndexTooLarge { .. })
    ));
}

/// The path loader is used directly by the public control-vector inspector,
/// so it must bound bytes before allocating from a caller-selected file.
#[test]
fn an_oversized_control_vector_file_is_refused_before_parsing() {
    let path = std::env::temp_dir().join(format!(
        "turbospark-control-vector-limit-{}",
        std::process::id()
    ));
    let file = std::fs::File::create(&path).expect("create sparse fixture");
    file.set_len(MAX_CONTROL_VECTOR_BYTES + 1)
        .expect("size sparse fixture");

    let result = load_control_vector(&path);
    std::fs::remove_file(&path).expect("remove sparse fixture");

    assert!(matches!(
        result,
        Err(ControlVectorError::FileTooLarge {
            max_bytes: MAX_CONTROL_VECTOR_BYTES,
            ..
        })
    ));
}

#[test]
fn a_foreign_tensor_name_is_refused() {
    assert!(matches!(
        layer_index("blk.0.attn_q.weight", 1),
        Err(ControlVectorError::BadTensorName { .. })
    ));
    assert!(matches!(
        layer_index("direction.middle", 1),
        Err(ControlVectorError::BadTensorName { .. })
    ));
}

/// A vector carrying no `turbospark.layer_base` is a FOREIGN vector --
/// llama.cpp stamps no metadata at all -- so the default has to be its
/// convention. Defaulting to this port's old one would silently shift
/// every published `repeng` vector by a block.
#[test]
fn an_absent_layer_base_reads_as_llama_cpps_convention() {
    let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let builder = crate::GgufBuilder::new().tensor("direction.3", GGML_TYPE_F32, &[8], data);
    let set = parse_control_vector(&builder.build().0).expect("parses");

    assert!(set.layer(3).is_some(), "direction.3 is block 3");
    assert!(set.layer(2).is_none(), "not block 2");
}

/// Same bytes, one metadata key apart, landing a block apart. This is the
/// end-to-end form of the discrimination check above.
#[test]
fn the_stamped_key_moves_where_a_direction_lands() {
    let with_base = |base: u32| {
        let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let builder = crate::GgufBuilder::new()
            .metadata_u32(LAYER_BASE_KEY, base)
            .tensor("direction.5", GGML_TYPE_F32, &[8], data);
        parse_control_vector(&builder.build().0).expect("parses")
    };

    assert!(with_base(1).layer(5).is_some());
    assert!(with_base(0).layer(4).is_some());
    assert!(with_base(0).layer(5).is_none());
}

/// Refused rather than defaulted. A file numbering from 2 was written
/// against something nothing here implements, and reading it as either
/// known convention steers every block an unknown distance off.
#[test]
fn an_unknown_layer_base_is_refused() {
    let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let builder = crate::GgufBuilder::new()
        .metadata_u32(LAYER_BASE_KEY, 2)
        .tensor("direction.1", GGML_TYPE_F32, &[8], data);
    assert_eq!(
        parse_control_vector(&builder.build().0),
        Err(ControlVectorError::UnknownLayerBase { base: 2 })
    );
}

/// A PRESENT key this reader cannot read is not an ABSENT key, and
/// collapsing the two is the one way left for this surface to misplace a
/// direction in silence.
///
/// GGUF metadata carries no schema, so the value's type is the writer's
/// choice: `metadata_u64` answers `None` for a string, a float, a bool, an
/// array and a negative signed integer alike. Reading that `None` as
/// "absent" hands the file llama.cpp's convention -- which is the right
/// default for a foreign vector precisely BECAUSE llama.cpp stamps
/// nothing, and the wrong one for a file that stamped `"0"` and meant it.
/// Every block would land one place off, with no error, which is the
/// failure the whole numbering correction exists to end.
///
/// Refused for the same reason `an_unknown_layer_base_is_refused` refuses
/// a 2: the file is stating a convention, and this reader cannot tell
/// which one.
#[test]
fn a_layer_base_this_reader_cannot_read_is_refused_rather_than_defaulted() {
    let unreadable = |b: crate::GgufBuilder| {
        let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        parse_control_vector(&b.tensor("direction.1", GGML_TYPE_F32, &[8], data).build().0)
    };

    assert!(
        matches!(
            unreadable(crate::GgufBuilder::new().metadata_str(LAYER_BASE_KEY, "0")),
            Err(ControlVectorError::MalformedLayerBase { .. })
        ),
        "a string layer_base must not read as an absent one"
    );

    // The control: the key really is optional, and its absence is still
    // llama.cpp's convention rather than an error. Without this the fix
    // above could be "refuse whenever the key does not parse", which
    // would reject every published vector in the ecosystem.
    assert!(
        unreadable(crate::GgufBuilder::new()).is_ok(),
        "an absent key is still the default, not a refusal"
    );
}

/// Dropping it silently would write a file missing an edit the caller
/// asked for, which is the failure this surface exists to avoid.
#[test]
fn writing_a_block_zero_direction_is_refused_rather_than_dropped() {
    let mut d = BTreeMap::new();
    d.insert(0usize, vec![1.0f32; 8]);
    d.insert(1usize, vec![1.0f32; 8]);
    assert_eq!(
        write_control_vector(&d, "qwen35", None),
        Err(ControlVectorError::LayerZeroNotExpressible)
    );
}

/// A sparse file is the NORMAL case, not a damaged one: steering a narrow
/// band of layers is what the research recommends, so a vector covering
/// only layers 30-32 must load with the rest absent rather than zeroed.
#[test]
fn a_sparse_file_leaves_uncovered_layers_absent() {
    let mut builder = crate::GgufBuilder::new().metadata_str("general.architecture", "qwen35");
    for l in [3usize, 5] {
        let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        builder = builder.tensor(&format!("direction.{l}"), GGML_TYPE_F32, &[8], data);
    }
    let set = parse_control_vector(&builder.build().0).expect("parses");

    assert_eq!(set.layers.len(), 6, "sized to the highest covered layer");
    assert_eq!(set.covered_layers(), 2);
    assert!(set.layer(3).is_some() && set.layer(5).is_some());
    for l in [0usize, 1, 2, 4] {
        assert!(set.layer(l).is_none(), "layer {l} should be absent");
    }
}

/// An F16 direction is the same width as nothing else here and would
/// decode to garbage; llama.cpp refuses it by name and so does this.
#[test]
fn a_non_f32_direction_is_refused() {
    let builder = crate::GgufBuilder::new().tensor("direction.1", 1, &[8], vec![0u8; 16]);
    assert!(matches!(
        parse_control_vector(&builder.build().0),
        Err(ControlVectorError::BadTensorShape { .. })
    ));
}

#[test]
fn directions_of_different_widths_are_refused() {
    let builder = crate::GgufBuilder::new()
        .tensor("direction.1", GGML_TYPE_F32, &[8], vec![0u8; 32])
        .tensor("direction.2", GGML_TYPE_F32, &[16], vec![0u8; 64]);
    assert!(matches!(
        parse_control_vector(&builder.build().0),
        Err(ControlVectorError::RaggedWidths { .. })
    ));
}

/// An ordinary GGUF is a readable container and NOT a control vector.
/// It has to fail by name rather than load as an empty set.
#[test]
fn a_gguf_that_is_not_a_control_vector_is_refused() {
    let builder = crate::GgufBuilder::new().metadata_str("general.architecture", "llama");
    assert_eq!(
        parse_control_vector(&builder.build().0),
        Err(ControlVectorError::NoDirections)
    );
}

/// The mode is a HINT the file may omit. Absent must mean "the caller
/// decides", never a silent default to one of the three edits.
#[test]
fn an_absent_mode_is_none_rather_than_a_default() {
    let bytes = write_control_vector(&dirs(2, 8), "qwen35", None).expect("writes");
    let set = parse_control_vector(&bytes).expect("parses");
    assert_eq!(set.declared_mode, None);
}

/// GLP's `project` mode is semantically different from this port's additive
/// control-vector modes. Refuse it at the shared parser so the CLI, server,
/// Swift settings and the model-sidebar downloader cannot silently interpret
/// a published refusal vector as `ablate`.
#[test]
fn a_projective_vector_is_refused_rather_than_treated_as_ablation() {
    let data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let builder = crate::GgufBuilder::new()
        .metadata_str("glp.mode", "project")
        .tensor("direction.1", GGML_TYPE_F32, &[8], data);

    assert_eq!(
        parse_control_vector(&builder.build().0),
        Err(ControlVectorError::UnsupportedMode {
            key: "glp.mode".to_string(),
            value: "project".to_string(),
        })
    );
}
