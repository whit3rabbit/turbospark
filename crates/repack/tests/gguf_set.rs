use turbospark_repack::{
    gguf_shard_names, parse_gguf_header, GgufBuilder, GgufHeader, GgufSet, GgufValue,
    MemoryRangeSource, RangeSource,
};

struct Bytes(Vec<u8>);
impl RangeSource for Bytes {
    fn read_range(&self, a: u64, b: u64) -> Result<Vec<u8>, turbospark_repack::DownloadError> {
        MemoryRangeSource::new(&self.0).read_range(a, b)
    }
}

fn shard(no: u32, name: &str) -> (GgufHeader, Bytes, u64) {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "minimax-m2")
        .metadata_u32("split.no", no)
        .metadata_u32("split.count", 2)
        .metadata_u32("split.tensors.count", 2)
        .tensor(name, 0, &[1], (no as f32 + 1.0).to_le_bytes().to_vec())
        .build();
    let len = bytes.len() as u64;
    (parse_gguf_header(&bytes, len).unwrap(), Bytes(bytes), len)
}
fn pair() -> Vec<(GgufHeader, Bytes, u64)> {
    vec![shard(0, "blk.0.a"), shard(1, "blk.0.b")]
}
fn error(v: Vec<(GgufHeader, Bytes, u64)>) -> String {
    GgufSet::new(v).err().expect("must reject")
}

#[test]
fn tensors_in_one_layer_can_live_in_different_shards() {
    let set = GgufSet::new(pair()).unwrap();
    for (name, expected) in [("blk.0.a", 1f32), ("blk.0.b", 2f32)] {
        let (start, end) = set.header.absolute_range(name).unwrap().unwrap();
        assert_eq!(set.read_range(start, end).unwrap(), expected.to_le_bytes());
    }
    assert_eq!(
        gguf_shard_names("dir/m-00001-of-00002.gguf", &shard(0, "a").0).unwrap(),
        ["dir/m-00001-of-00002.gguf", "dir/m-00002-of-00002.gguf"]
    );
}
#[test]
fn missing_shard_is_refused() {
    let mut v = pair();
    v.pop();
    assert_eq!(error(v), "incomplete GGUF shard set");
}
#[test]
fn duplicate_tensor_is_refused() {
    assert_eq!(
        error(vec![shard(0, "a"), shard(1, "a")]),
        "duplicate GGUF tensor: a"
    );
}
#[test]
fn inconsistent_split_number_is_refused() {
    assert_eq!(
        error(vec![shard(0, "a"), shard(0, "b")]),
        "inconsistent GGUF split metadata"
    );
}
#[test]
fn inconsistent_architecture_is_refused() {
    let mut v = pair();
    v[1].0.metadata.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    assert_eq!(error(v), "inconsistent GGUF metadata: general.architecture");
}
#[test]
fn metadata_first_seen_in_a_later_shard_must_stay_consistent() {
    let mut v = vec![shard(0, "a"), shard(1, "b"), shard(2, "c")];
    for s in &mut v {
        s.0.metadata.insert("split.count".into(), GgufValue::U32(3));
        s.0.metadata
            .insert("split.tensors.count".into(), GgufValue::U32(3));
    }
    v[1].0
        .metadata
        .insert("general.name".into(), GgufValue::String("first".into()));
    v[2].0
        .metadata
        .insert("general.name".into(), GgufValue::String("second".into()));
    assert_eq!(error(v), "inconsistent GGUF metadata: general.name");
}
#[test]
fn out_of_file_tensor_is_refused() {
    let mut v = pair();
    v[1].2 -= 1;
    assert_eq!(error(v), "tensor blk.0.b exceeds shard 2 length");
}
#[test]
fn empty_shard_must_still_contain_its_header() {
    let mut first = shard(0, "a");
    first
        .0
        .metadata
        .insert("split.tensors.count".into(), GgufValue::U32(1));
    let (bytes, _) = GgufBuilder::new()
        .metadata_u32("split.no", 1)
        .metadata_u32("split.count", 2)
        .metadata_u32("split.tensors.count", 1)
        .build();
    let len = bytes.len() as u64;
    let header = parse_gguf_header(&bytes, len).unwrap();
    let truncated = header.data_region_start - 1;
    assert_eq!(
        error(vec![first, (header, Bytes(bytes), truncated)]),
        "GGUF header exceeds shard 2 length"
    );
}
#[test]
fn declared_tensor_total_is_checked() {
    let mut v = pair();
    for s in &mut v {
        s.0.metadata
            .insert("split.tensors.count".into(), GgufValue::U32(3));
    }
    assert_eq!(
        error(v),
        "GGUF tensor count disagrees with split.tensors.count"
    );
}
#[test]
fn invalid_split_filename_is_refused() {
    assert!(gguf_shard_names("m-00002-of-00002.gguf", &shard(0, "a").0)
        .unwrap_err()
        .contains("does not name shard 1"));
}

#[test]
fn partial_split_metadata_cannot_masquerade_as_a_single_file() {
    let mut v = vec![shard(0, "a")];
    v[0].0.metadata.remove("split.count");
    assert_eq!(
        gguf_shard_names("m-00001-of-00002.gguf", &v[0].0).unwrap_err(),
        "split metadata requires split.count"
    );
    assert_eq!(error(v), "split metadata requires split.count");
}

#[test]
fn minimax_zero_experts_is_refused_before_dividing_tensor_bytes() {
    let (bytes, _) = turbospark_repack::build_synthetic_minimax_gguf();
    let mut h = parse_gguf_header(&bytes, bytes.len() as u64).unwrap();
    h.metadata
        .insert("minimax-m2.expert_count".into(), GgufValue::U32(0));
    for (name, tensor) in &mut h.tensors {
        if name.ends_with("_exps.weight") {
            *tensor.dims.last_mut().unwrap() = 0;
        }
    }
    let error = turbospark_repack::minimax_gguf_sizing(&h)
        .unwrap_err()
        .to_string();
    assert_eq!(
        error,
        "GGUF minimax-m2.expert_count: MiniMax expert count must be positive"
    );
}

#[test]
fn minimax_router_precision_contract_applies_to_direct_repacking() {
    let (bytes, _) = turbospark_repack::build_synthetic_minimax_gguf();
    for name in ["blk.0.ffn_gate_inp.weight", "blk.0.exp_probs_b.bias"] {
        let mut h = parse_gguf_header(&bytes, bytes.len() as u64).unwrap();
        h.tensors.get_mut(name).unwrap().ggml_type = 1;
        let error =
            turbospark_repack::orchestrate_gguf_checkpoint(&h, &MemoryRangeSource::new(&bytes))
                .err()
                .expect("F16 routing tensors must be refused")
                .to_string();
        assert_eq!(
            error,
            format!("tensor {name}: MiniMax router and correction bias must be F32")
        );
    }
}

#[test]
fn minimax_missing_output_head_is_not_treated_as_tied() {
    let (bytes, _) = turbospark_repack::build_synthetic_minimax_gguf();
    let mut header = parse_gguf_header(&bytes, bytes.len() as u64).unwrap();
    assert!(header.tensors.remove("output.weight").is_some());
    assert_eq!(
        turbospark_repack::arch_from_gguf(&header),
        Err(turbospark_repack::GgufConfigError::MissingTensor {
            name: "output.weight".into(),
        })
    );
}
