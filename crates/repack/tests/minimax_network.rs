//! Pinned header witness, no checkpoint weights are downloaded.
use turbospark_repack::{fetch_gguf_header, minimax_gguf_sizing, GgufSet, HttpRangeSource};

#[test]
#[ignore = "network: reads three MiniMax Q4_K_M headers"]
fn minimax_three_shard_contract() {
    let lengths = [49_386_415_584, 49_712_985_440, 39_242_984_096];
    let shards = lengths.into_iter().enumerate().map(|(i, len)| {
        let url = format!("https://huggingface.co/unsloth/MiniMax-M2-GGUF/resolve/06e952eab9e8e136847e9df067a0032582696778/Q4_K_M/MiniMax-M2-Q4_K_M-{:05}-of-00003.gguf", i + 1);
        let source = HttpRangeSource::new(url);
        let header = fetch_gguf_header(&source).unwrap();
        (header, source, len)
    }).collect();
    let set = GgufSet::new(shards).unwrap();
    assert_eq!(set.header.tensors.len(), 809);
    assert_eq!(
        turbospark_repack::arch_from_gguf(&set.header).unwrap(),
        model_io::minimax_m2()
    );
    let size = minimax_gguf_sizing(&set.header).unwrap();
    assert_eq!(size.max_expert_stride, 9_191_424);
    assert_eq!(size.kv_8192_bytes, 2_080_374_784);
    for (kind, count) in [(0, 373), (12, 375), (14, 61)] {
        assert_eq!(
            set.header
                .tensors
                .values()
                .filter(|t| t.ggml_type == kind)
                .count(),
            count
        );
    }
    println!(
        "download={} {size:?} install_allowance={}",
        set.bytes,
        size.install_bytes()
    );
}
