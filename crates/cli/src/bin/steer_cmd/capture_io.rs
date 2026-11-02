//! Reader for the residual-capture file pair `TURBOSPARK_RESID_CAPTURE`
//! writes (`crates/runtime/src/resid_capture.rs`): a small JSON header
//! beside a raw little-endian f32 sidecar laid out
//! `[snapshot][layer][hidden]`.
//!
//! This is the same pair `scripts/extract_direction.py` reads; a Rust reader
//! means the extraction no longer needs Python. The header's `positions`
//! list is carried through for the sanity line the extract verb prints
//! (one position per prompt, varying with prompt length is expected; all
//! zeros would mean the capture keyed on the wrong pass).

use std::path::Path;

use compute::steering_extract::CaptureSet;

/// One side's captures, read and validated.
pub fn read_capture(header_path: &Path) -> Result<CaptureSet, String> {
    let header_bytes = std::fs::read(header_path)
        .map_err(|e| format!("capture header {}: {e}", header_path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| format!("capture header {}: {e}", header_path.display()))?;
    let layers = header
        .get("layers")
        .and_then(|v| v.as_u64())
        .ok_or("capture header has no layers count")? as usize;
    let hidden = header
        .get("hidden")
        .and_then(|v| v.as_u64())
        .ok_or("capture header has no hidden count")? as usize;
    let snapshots = header
        .get("snapshots")
        .and_then(|v| v.as_u64())
        .ok_or("capture header has no snapshots count")? as usize;
    let dtype = header
        .get("dtype")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if dtype != "float32" {
        return Err(format!(
            "capture header declares dtype {dtype:?}; only float32 sidecars are read"
        ));
    }
    if snapshots == 0 {
        // The capture writes NOTHING when no non-prefill pass ran, so an
        // explicit zero means the header predates that guard. Either way a
        // zero-snapshot set would average into nothing.
        return Err(format!(
            "capture {} holds no snapshots; the family may not feed the capture, \
             or no prompt reached the last-token pass",
            header_path.display()
        ));
    }
    let data_name = header
        .get("data")
        .and_then(|v| v.as_str())
        .ok_or("capture header has no data file name")?;
    let data_path = header_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(data_name);
    let raw = std::fs::read(&data_path)
        .map_err(|e| format!("capture sidecar {}: {e}", data_path.display()))?;
    let want = snapshots
        .checked_mul(layers)
        .and_then(|v| v.checked_mul(hidden))
        .ok_or("capture dimensions overflow")?;
    if raw.len() != want * 4 {
        return Err(format!(
            "capture sidecar {} holds {} bytes, header describes {} floats \
             ({snapshots} x {layers} x {hidden})",
            data_path.display(),
            raw.len(),
            want
        ));
    }
    let mut rows = Vec::with_capacity(snapshots);
    let row_len = layers * hidden;
    for s in 0..snapshots {
        let start = s * row_len;
        let row = raw[start * 4..(start + row_len) * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<f32>>();
        rows.push(row);
    }
    CaptureSet::new(layers, hidden, rows)
}

/// The `positions` list from the header, for the sanity print.
pub fn read_positions(header_path: &Path) -> Vec<usize> {
    let Ok(bytes) = std::fs::read(header_path) else {
        return Vec::new();
    };
    let Ok(header) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    header
        .get("positions")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-written header plus sidecar must round-trip into a
    /// `CaptureSet`, and a length mismatch must be an error rather than a
    /// silent short read.
    #[test]
    fn capture_roundtrip_and_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-steer-capture-io-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let header = dir.join("p.json");
        let data = dir.join("p.f32");
        std::fs::write(
            &header,
            r#"{
  "layers": 2,
  "hidden": 3,
  "snapshots": 2,
  "dtype": "float32",
  "layout": "[snapshot][layer][hidden]",
  "data": "p.f32",
  "positions": [7, 11]
}
"#,
        )
        .expect("write header");
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut raw = Vec::new();
        for v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&data, &raw).expect("write sidecar");

        let set = read_capture(&header).expect("reads");
        assert_eq!(set.layers, 2);
        assert_eq!(set.hidden, 3);
        assert_eq!(set.rows.len(), 2);
        assert_eq!(set.rows[0], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(read_positions(&header), vec![7, 11]);

        raw.pop();
        std::fs::write(&data, &raw).expect("truncate sidecar");
        let err = read_capture(&header).expect_err("truncated sidecar refused");
        assert!(err.contains("holds"), "error names the length: {err}");

        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
