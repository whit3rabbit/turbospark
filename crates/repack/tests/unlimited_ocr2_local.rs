//! Repack a checkpoint already on disk through the real streamed walk.
//! `#[ignore]`d: it needs the 2.46 GB file and writes as much again.
//!
//! ```sh
//! TURBOSPARK_UOCR2_CHECKPOINT=~/models/Unlimited-OCR-2-4bit \
//! TURBOSPARK_UOCR2_INSTALL_DIR=~/models/Unlimited-OCR-2-4bit.gturbo \
//!   cargo test -p turbospark-repack --release --test unlimited_ocr2_local -- --ignored
//! ```

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Mutex;

use turbospark_repack::{
    canonicalize_unlimited_ocr2_header, fetch_safetensors_header, parse_gemma4_quantization,
    parse_unlimited_ocr2_config, write_unlimited_ocr2_install_streamed, DownloadError,
    Gemma4Shards, RangeSource,
};

struct FileSource(Mutex<std::fs::File>);

impl RangeSource for FileSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let mut file = self.0.lock().unwrap();
        let mut bytes = vec![0u8; (end_exclusive - start) as usize];
        file.seek(SeekFrom::Start(start)).unwrap();
        file.read_exact(&mut bytes).unwrap();
        Ok(bytes)
    }
}

#[test]
#[ignore = "needs the real checkpoint on disk (TURBOSPARK_UOCR2_CHECKPOINT)"]
fn repacks_the_local_checkpoint() {
    let source_dir = PathBuf::from(std::env::var("TURBOSPARK_UOCR2_CHECKPOINT").unwrap());
    let out = PathBuf::from(std::env::var("TURBOSPARK_UOCR2_INSTALL_DIR").unwrap());
    std::fs::create_dir_all(&out).unwrap();

    let config = std::fs::read_to_string(source_dir.join("config.json")).unwrap();
    let arch = parse_unlimited_ocr2_config(&config).unwrap();
    let quant = parse_gemma4_quantization(&config).unwrap();
    let root: serde_json::Value = serde_json::from_str(&config).unwrap();

    let file = std::fs::File::open(source_dir.join("model.safetensors")).unwrap();
    let source = FileSource(Mutex::new(file));
    let mut header = fetch_safetensors_header(&source).unwrap();
    canonicalize_unlimited_ocr2_header(&mut header).unwrap();
    let shards = Gemma4Shards::single(&header, &source);

    write_unlimited_ocr2_install_streamed(
        &out,
        &arch,
        "mlx-community/Unlimited-OCR-2-4bit",
        std::slice::from_ref(&header),
        &shards,
        &quant,
        &root["sparse_attention_config"],
        |line| eprintln!("{line}"),
    )
    .unwrap();

    for sidecar in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "config.json",
    ] {
        std::fs::copy(source_dir.join(sidecar), out.join(sidecar)).unwrap();
    }
    assert!(out.join(model_io::DASH_SIDECAR).exists());
}
