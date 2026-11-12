//! Streams a Fish Audio Drama 3 checkpoint's shards into a local Drama
//! install, for bring-up and measurement before the catalog owns the install:
//!
//! ```sh
//! cargo run --release -p turbospark-repack --example fish_drama3_intake -- \
//!     <config.json> <shard-url-prefix> <shard-count> <out-dir>
//! ```
//!
//! `<shard-url-prefix>` ends with `/` and names an immutable revision, e.g.
//! `https://huggingface.co/mlx-community/fish-audio-drama-3-4bit/resolve/cc639e27aba7edb73429d3e3806f471dcd79d316/`.
//! Shards are read by range and never staged whole: the trunk lands in
//! `<out-dir>/trunk/` and the tower and semantic head in
//! `<out-dir>/tts_side.safetensors`. The root files (`config.json`,
//! `tokenizer.json`, `codec.safetensors`, `LICENSE`, `NOTICE`) are the
//! caller's, fetched and hash-checked separately. Shard bytes are not
//! hash-checked here (range reads never see a whole file); the catalog
//! install path is where that policy lives.

use std::path::PathBuf;
use std::time::Instant;

use turbospark_repack::{
    canonicalize_drama3_header, fetch_safetensors_header, write_fish_drama3_install_streamed,
    Gemma4Shards, HttpRangeSource, RangeSource,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let config = PathBuf::from(args.next().ok_or("config.json path")?);
    let prefix = args.next().ok_or("shard url prefix")?;
    let count: usize = args.next().ok_or("shard count")?.parse()?;
    let out = PathBuf::from(args.next().ok_or("out dir")?);
    if !prefix.ends_with('/') {
        return Err("the shard url prefix must end with /".into());
    }
    let config_text = std::fs::read_to_string(&config)?;
    std::fs::create_dir_all(&out)?;

    let started = Instant::now();
    let sources: Vec<HttpRangeSource> = (1..=count)
        .map(|i| HttpRangeSource::new(format!("{prefix}model-{i:05}.safetensors")))
        .collect();
    let mut headers = Vec::with_capacity(count);
    for source in &sources {
        let mut header = fetch_safetensors_header(source)?;
        canonicalize_drama3_header(&mut header)?;
        headers.push(header);
    }
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(&sources)
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    )?;
    let model_id = prefix.trim_end_matches('/').to_string();
    let arch =
        write_fish_drama3_install_streamed(&out, &config_text, &model_id, &shards, |stage| {
            eprintln!("[{:7.1}s] {stage}", started.elapsed().as_secs_f64());
        })?;
    eprintln!(
        "done in {:.1}s: {} layers, {} experts, hidden {}",
        started.elapsed().as_secs_f64(),
        arch.num_layers,
        arch.num_experts,
        arch.hidden_size
    );
    Ok(())
}
