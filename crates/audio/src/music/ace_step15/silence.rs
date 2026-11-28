//! The checkpoint's silence latent, read from `silence_latent.pt` without
//! unpickling.
//!
//! The file is a torch zip archive whose entries are all stored
//! (uncompressed). The tensor is fully described by `data.pkl`, a fixed
//! 166-byte `_rebuild_tensor_v2` record for a contiguous float32 storage
//! shaped `[1, 64, 15000]` at offset 0, and its bytes are the stored entry
//! `data/0`. This reader never interprets the pickle: it requires
//! `data.pkl` to equal those exact bytes, `byteorder` to be `little`, and
//! the payload to be exactly 3,840,000 bytes, then copies the floats. The
//! catalog install has already checked the whole file's SHA-256 against
//! its pin; a file that differs here is refused, never guessed at. A
//! `silence_latent.safetensors` written by `tools/convert_ace_step15_silence.py`
//! is accepted as well.

use std::path::Path;

use crate::{Result, SpeechError};

/// Channels and frames of the shipped silence latent.
pub const CHANNELS: usize = 64;
pub const FRAMES: usize = 15_000;

/// `data.pkl` of the pinned `silence_latent.pt`, byte for byte.
const EXPECTED_PICKLE: [u8; 166] = [
    0x80, 0x02, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x2e, 0x5f, 0x75, 0x74, 0x69, 0x6c, 0x73, 0x0a,
    0x5f, 0x72, 0x65, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x5f,
    0x76, 0x32, 0x0a, 0x71, 0x00, 0x28, 0x28, 0x58, 0x07, 0x00, 0x00, 0x00, 0x73, 0x74, 0x6f, 0x72,
    0x61, 0x67, 0x65, 0x71, 0x01, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x0a, 0x46, 0x6c, 0x6f, 0x61,
    0x74, 0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x0a, 0x71, 0x02, 0x58, 0x01, 0x00, 0x00, 0x00,
    0x30, 0x71, 0x03, 0x58, 0x03, 0x00, 0x00, 0x00, 0x63, 0x70, 0x75, 0x71, 0x04, 0x4a, 0x00, 0xa6,
    0x0e, 0x00, 0x74, 0x71, 0x05, 0x51, 0x4b, 0x00, 0x4b, 0x01, 0x4b, 0x40, 0x4d, 0x98, 0x3a, 0x87,
    0x71, 0x06, 0x4a, 0x00, 0xa6, 0x0e, 0x00, 0x4d, 0x98, 0x3a, 0x4b, 0x01, 0x87, 0x71, 0x07, 0x89,
    0x63, 0x63, 0x6f, 0x6c, 0x6c, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x0a, 0x4f, 0x72, 0x64,
    0x65, 0x72, 0x65, 0x64, 0x44, 0x69, 0x63, 0x74, 0x0a, 0x71, 0x08, 0x29, 0x52, 0x71, 0x09, 0x74,
    0x71, 0x0a, 0x52, 0x71, 0x0b, 0x2e,
];

fn bad(why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: "silence_latent.pt".into(),
        why: why.into(),
    }
}

fn u16_at(b: &[u8], at: usize) -> Result<usize> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]) as usize)
        .ok_or_else(|| bad("truncated zip structure"))
}

fn u32_at(b: &[u8], at: usize) -> Result<usize> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or_else(|| bad("truncated zip structure"))
}

/// The stored (uncompressed) entries of a zip archive, as name and data
/// range. Compressed entries, ZIP64 archives and inconsistent headers are
/// refused.
fn stored_entries(b: &[u8]) -> Result<Vec<(String, std::ops::Range<usize>)>> {
    const EOCD: u32 = 0x0605_4b50;
    const CENTRAL: u32 = 0x0201_4b50;
    const LOCAL: u32 = 0x0403_4b50;
    // The end-of-central-directory record is 22 bytes plus a comment of up
    // to 65535 bytes.
    let low = b.len().saturating_sub(22 + 0xFFFF);
    let eocd = (low..b.len().saturating_sub(21))
        .rev()
        .find(|&i| u32_at(b, i).ok() == Some(EOCD as usize))
        .ok_or_else(|| bad("not a zip archive"))?;
    let count = u16_at(b, eocd + 10)?;
    let cd_size = u32_at(b, eocd + 12)?;
    let cd_offset = u32_at(b, eocd + 16)?;
    if count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF {
        return Err(bad("ZIP64 archives are not expected here"));
    }
    if cd_offset.checked_add(cd_size).is_none_or(|end| end > eocd) {
        return Err(bad("central directory out of bounds"));
    }
    let mut at = cd_offset;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if u32_at(b, at)? != CENTRAL as usize {
            return Err(bad("bad central directory entry"));
        }
        let method = u16_at(b, at + 10)?;
        let compressed = u32_at(b, at + 20)?;
        let size = u32_at(b, at + 24)?;
        let name_len = u16_at(b, at + 28)?;
        let extra_len = u16_at(b, at + 30)?;
        let comment_len = u16_at(b, at + 32)?;
        let local = u32_at(b, at + 42)?;
        let name = b
            .get(at + 46..at + 46 + name_len)
            .ok_or_else(|| bad("truncated entry name"))?;
        let name = std::str::from_utf8(name)
            .map_err(|_| bad("entry name is not UTF-8"))?
            .to_string();
        if method != 0 || compressed != size {
            return Err(bad(format!("entry {name} is compressed")));
        }
        if u32_at(b, local)? != LOCAL as usize {
            return Err(bad(format!("entry {name} has no local header")));
        }
        let start = local + 30 + u16_at(b, local + 26)? + u16_at(b, local + 28)?;
        let end = start
            .checked_add(size)
            .filter(|&e| e <= cd_offset)
            .ok_or_else(|| bad(format!("entry {name} out of bounds")))?;
        out.push((name, start..end));
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(out)
}

/// Reads a torch-saved silence latent and returns it time-major,
/// `[FRAMES, CHANNELS]` (upstream's `.transpose(1, 2)`).
pub fn read_pt(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let entries = stored_entries(&bytes)?;
    let entry = |name: &str| -> Result<&[u8]> {
        let mut found = entries.iter().filter(|(n, _)| n == name);
        let (_, range) = found
            .next()
            .ok_or_else(|| bad(format!("missing entry {name}")))?;
        if found.next().is_some() {
            return Err(bad(format!("duplicate entry {name}")));
        }
        Ok(&bytes[range.clone()])
    };
    if entry("silence_latent/data.pkl")? != EXPECTED_PICKLE {
        return Err(bad(
            "tensor record differs from the pinned [1, 64, 15000] float32 latent",
        ));
    }
    if entry("silence_latent/byteorder")? != b"little" {
        return Err(bad("storage is not little-endian"));
    }
    let data = entry("silence_latent/data/0")?;
    if data.len() != CHANNELS * FRAMES * 4 {
        return Err(bad(format!(
            "storage holds {} bytes, expected {}",
            data.len(),
            CHANNELS * FRAMES * 4
        )));
    }
    let mut out = vec![0.0f32; CHANNELS * FRAMES];
    for (i, c) in data.chunks_exact(4).enumerate() {
        let (ch, t) = (i / FRAMES, i % FRAMES);
        out[t * CHANNELS + ch] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored zip with `entries`, as torch writes it (local headers,
    /// central directory, end record, no compression).
    fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in entries {
            let offset = out.len() as u32;
            let le16 = |v: u16| v.to_le_bytes();
            let le32 = |v: u32| v.to_le_bytes();
            out.extend(le32(0x0403_4b50));
            out.extend([0u8; 14]); // version, flags, method 0, time, date, crc
            out.extend(le32(data.len() as u32));
            out.extend(le32(data.len() as u32));
            out.extend(le16(name.len() as u16));
            out.extend(le16(0));
            out.extend(name.as_bytes());
            out.extend(*data);
            central.extend(le32(0x0201_4b50));
            central.extend([0u8; 16]); // versions, flags, method 0, time, date, crc
            central.extend(le32(data.len() as u32));
            central.extend(le32(data.len() as u32));
            central.extend(le16(name.len() as u16));
            central.extend([0u8; 12]); // extra, comment, disk, attrs
            central.extend(le32(offset));
            central.extend(name.as_bytes());
        }
        let cd_offset = out.len() as u32;
        out.extend(&central);
        out.extend(0x0605_4b50u32.to_le_bytes());
        out.extend([0u8; 4]);
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend((central.len() as u32).to_le_bytes());
        out.extend(cd_offset.to_le_bytes());
        out.extend([0u8; 2]);
        out
    }

    fn write(bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ace_silence_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "{}.pt",
            bytes.len() ^ (bytes.first().copied().unwrap_or(0) as usize)
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn payload() -> Vec<u8> {
        (0..CHANNELS * FRAMES)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect()
    }

    #[test]
    fn reads_the_pinned_layout_time_major() {
        let data = payload();
        let path = write(&zip(&[
            ("silence_latent/data.pkl", &EXPECTED_PICKLE),
            ("silence_latent/byteorder", b"little"),
            ("silence_latent/data/0", &data),
            ("silence_latent/version", b"3\n"),
        ]));
        let out = read_pt(&path).unwrap();
        // Channel-major index ch * FRAMES + t lands at t * CHANNELS + ch.
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], FRAMES as f32);
        assert_eq!(out[CHANNELS], 1.0);
        assert_eq!(out[5 * CHANNELS + 3], (3 * FRAMES + 5) as f32);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn refuses_anything_else() {
        let data = payload();
        let mut pickle = EXPECTED_PICKLE;
        pickle[120] ^= 1;
        let cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "other pickle",
                zip(&[
                    ("silence_latent/data.pkl", &pickle),
                    ("silence_latent/byteorder", b"little"),
                    ("silence_latent/data/0", &data),
                ]),
            ),
            (
                "big endian",
                zip(&[
                    ("silence_latent/data.pkl", &EXPECTED_PICKLE),
                    ("silence_latent/byteorder", b"big"),
                    ("silence_latent/data/0", &data),
                ]),
            ),
            (
                "short payload",
                zip(&[
                    ("silence_latent/data.pkl", &EXPECTED_PICKLE),
                    ("silence_latent/byteorder", b"little"),
                    ("silence_latent/data/0", &data[..data.len() - 4]),
                ]),
            ),
            (
                "missing payload",
                zip(&[
                    ("silence_latent/data.pkl", &EXPECTED_PICKLE),
                    ("silence_latent/byteorder", b"little"),
                ]),
            ),
            ("not a zip", b"not a zip archive at all".to_vec()),
        ];
        for (name, bytes) in cases {
            let path = write(&bytes);
            assert!(read_pt(&path).is_err(), "{name}: should be refused");
            let _ = std::fs::remove_file(path);
        }
        // A compressed entry (method 8) is refused even with valid bytes.
        let mut compressed = zip(&[("silence_latent/data.pkl", &EXPECTED_PICKLE)]);
        let cd = compressed.len() - 22 - (46 + "silence_latent/data.pkl".len());
        compressed[cd + 10] = 8;
        let path = write(&compressed);
        assert!(
            read_pt(&path).is_err(),
            "compressed entry should be refused"
        );
        let _ = std::fs::remove_file(path);
    }

    /// The real checkpoint file against the trusted conversion
    /// (`tools/convert_ace_step15_silence.py`, torch `weights_only`).
    #[test]
    #[ignore = "needs TURBOSPARK_ACE_STEP15_DIT with silence_latent.pt and .safetensors"]
    fn real_file_matches_the_trusted_conversion() {
        let dir = std::path::PathBuf::from(std::env::var("TURBOSPARK_ACE_STEP15_DIT").unwrap());
        let ours = read_pt(&dir.join("silence_latent.pt")).unwrap();
        let file = turbospark_model_io::safetensors::SafetensorsFile::open(
            &dir.join("silence_latent.safetensors"),
        )
        .unwrap();
        let want = file.load_as_f32("silence_latent").unwrap();
        assert_eq!(
            ours, want,
            "pickle-free read must equal the torch conversion bit for bit"
        );
    }
}
