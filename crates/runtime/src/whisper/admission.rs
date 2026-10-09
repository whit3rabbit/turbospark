//! Admission for checkpoint-backed Whisper opens, before owned weight allocation.

use std::io::Read;
use std::path::Path;

use model_io::speech_receipt::{SpeechInstallReceipt, SPEECH_RECEIPT_FILENAME};
use model_io::whisper_config::WhisperConfig;
use model_io::LoadGuard;

pub(super) const CONFIG_LIMIT: u64 = 1 << 20;
pub(super) const TOKENIZER_LIMIT: u64 = 16 << 20;
const MAX_OPEN_BYTES: u64 = 24 << 30;

pub(super) fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|file| file.take(limit + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{} exceeds {limit} byte limit", path.display()));
    }
    Ok(bytes)
}

pub(super) fn host_budget() -> Result<u64, String> {
    let resident = gpu::process_footprint()
        .ok_or("Whisper memory admission cannot measure process footprint")?;
    budget(gpu::physical_memory(), resident, gpu::memory_pressure_raw())
}

fn budget(physical: u64, resident: u64, pressure: i64) -> Result<u64, String> {
    // A missing pressure reading must not silently authorize a large load.
    if physical == 0 || pressure != 1 {
        return Err(
            "Whisper memory admission requires a normal host memory pressure reading".into(),
        );
    }
    Ok(LoadGuard::default()
        .available(physical, resident)
        .min(MAX_OPEN_BYTES))
}

pub(super) fn estimated_open_bytes(config: &WhisperConfig, checkpoint_bytes: u64) -> u64 {
    // The receipt's CPU estimate predates Metal. Keep its wire meaning stable,
    // but reserve another copy for GPU weights/caches plus the quadratic scores.
    // Include mapped source pages and bounded tokenizer/allocator overhead.
    config
        .memory_ceiling_bytes()
        .saturating_mul(2)
        .saturating_add(
            (config.num_attention_heads as u64)
                .saturating_mul(config.max_source_positions as u64)
                .saturating_mul(config.max_source_positions as u64)
                .saturating_mul(4),
        )
        .saturating_add(checkpoint_bytes)
        .saturating_add(64 << 20)
}

pub(super) fn admit(
    config: &WhisperConfig,
    checkpoint_bytes: u64,
    available: u64,
) -> Result<(), String> {
    let estimated = estimated_open_bytes(config, checkpoint_bytes);
    let available = available.min(MAX_OPEN_BYTES);
    if estimated > available {
        return Err(format!("Whisper memory admission refused: {estimated} estimated bytes exceeds {available} available bytes; unload another model or use a smaller Whisper checkpoint"));
    }
    Ok(())
}

pub(super) fn receipt(dir: &Path, config: &WhisperConfig) -> Result<SpeechInstallReceipt, String> {
    let bytes = read_bounded(&dir.join(SPEECH_RECEIPT_FILENAME), CONFIG_LIMIT)?;
    let receipt: SpeechInstallReceipt =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid speech receipt: {e}"))?;
    receipt.validate().map_err(|e| e.to_string())?;
    if receipt.speech_family != "whisper"
        || receipt.n_mels != config.n_mels
        || receipt.memory_ceiling_bytes != config.memory_ceiling_bytes()
    {
        return Err(
            "speech receipt family, mel width, or memory estimate disagrees with Whisper config"
                .into(),
        );
    }
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        let entry = receipt
            .files
            .get(name)
            .ok_or_else(|| format!("speech receipt is missing {name}"))?;
        let meta = std::fs::metadata(dir.join(name)).map_err(|e| format!("stat {name}: {e}"))?;
        if !meta.is_file() || meta.len() != entry.size {
            return Err(format!("speech receipt size mismatch for {name}"));
        }
        if entry.sha256.len() != 64 || !entry.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("speech receipt has invalid SHA-256 for {name}"));
        }
    }
    Ok(receipt)
}

pub(super) fn verify_bytes(
    receipt: &SpeechInstallReceipt,
    name: &str,
    bytes: &[u8],
) -> Result<(), String> {
    let entry = &receipt.files[name];
    if entry.size != bytes.len() as u64
        || !entry
            .sha256
            .eq_ignore_ascii_case(&model_io::hash_data(bytes))
    {
        return Err(format!("speech receipt checksum mismatch for {name}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_reserves_host_and_existing_process_memory_and_fails_closed() {
        assert_eq!(budget(16 << 30, 2 << 30, 1).unwrap(), 10 << 30);
        assert_eq!(budget(4 << 30, 1, 1).unwrap(), 0);
        assert_eq!(budget(128 << 30, 0, 1).unwrap(), MAX_OPEN_BYTES);
        for pressure in [0, 2, 4, 9] {
            assert!(budget(16 << 30, 0, pressure).is_err());
        }
        assert!(budget(0, 0, 1).is_err());
    }

    #[test]
    fn admission_refuses_one_byte_over_budget_and_cannot_disable_the_hard_cap() {
        let config = WhisperConfig::from_json_str(super::super::tests::TINY_CONFIG_JSON).unwrap();
        let estimated = estimated_open_bytes(&config, 1024);
        assert!(admit(&config, 1024, estimated).is_ok());
        assert!(admit(&config, 1024, estimated - 1).is_err());
        assert!(admit(&config, 1025, estimated).is_err());
        assert!(admit(&config, MAX_OPEN_BYTES, u64::MAX).is_err());
    }
}
