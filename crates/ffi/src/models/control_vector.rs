//! Control vector inspection and parsing.

use serde::Serialize;

/// What a `.gguf` control vector declares, read from the file alone.
///
/// **A SHAPE MATCH IS NOT A SEMANTIC MATCH, and this reports the shape only.**
/// `SteeringSet::validate` refuses a width or layer-count mismatch, and refuses
/// nothing else: a vector extracted for a different checkpoint of the same
/// hidden size opens, steers, and changes behaviour in a direction nobody
/// asked for, silently (`docs/OBLITERATION.md`, "Running someone else's
/// vector"). A caller rendering these fields owes the user that sentence.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ControlVectorInfo {
    /// The hidden size every direction in the file declares. Must equal the
    /// install's own `arch.hiddenSize` or `open` refuses the set.
    hidden: usize,
    /// How many blocks actually carry a direction.
    covered_layers: usize,
    /// Lowest and highest block index covered, 0-based in THIS port's
    /// convention after `turbospark.layer_base` has been honoured. A file
    /// written by llama.cpp or by this port since 2026-08-24 starts at 1,
    /// because block 0 is not expressible (`crates/repack`'s
    /// `LayerZeroNotExpressible`).
    min_layer: Option<usize>,
    max_layer: Option<usize>,
    /// The span `layers` covers, which is `max_layer + 1` and is what the
    /// install's own `arch.numLayers` is compared against.
    spanned_layers: usize,
    /// The mode the file itself declares, used when the caller names none.
    declared_mode: Option<String>,
    /// The architecture string the file was extracted against, when it
    /// carries one. **Advisory only** -- nothing validates against it, which
    /// is exactly why it is worth showing.
    declared_arch: Option<String>,
}

/// Reads a control vector's header and reports what it covers, with no model
/// open and no session.
///
/// Goes through `repack::control_vector::load_control_vector`, the ONE parser
/// that decides what a vector means, rather than letting a GUI reimplement the
/// GGUF layout. A vector is ~1.3 MB, so this is milliseconds.
pub fn control_vector_info_json(path: &str) -> Result<String, String> {
    let set = repack::control_vector::load_control_vector(std::path::Path::new(path))
        .map_err(|e| format!("{e}"))?;
    // `covered_layers()` is a COUNT, so the indices come from the vector
    // itself. Both are wanted: the count says how much of the model is
    // steered and the range says WHICH part, and a user comparing a file
    // against an install needs the second (a 31-of-32 vector starting at
    // block 1 is llama.cpp's convention working correctly, not a gap).
    let covered: Vec<usize> = set
        .layers
        .iter()
        .enumerate()
        .filter_map(|(l, d)| d.as_ref().map(|_| l))
        .collect();
    let info = ControlVectorInfo {
        hidden: set.hidden,
        covered_layers: set.covered_layers(),
        min_layer: covered.first().copied(),
        max_layer: covered.last().copied(),
        spanned_layers: set.layers.len(),
        declared_mode: set.declared_mode.map(|m| m.as_str().to_string()),
        declared_arch: set.declared_arch.clone(),
    };
    serde_json::to_string(&info).map_err(|e| e.to_string())
}

#[cfg(test)]
mod control_vector_info_tests {
    use super::control_vector_info_json;
    use std::collections::BTreeMap;

    /// Writes a control vector through `repack`'s own writer, so the test
    /// reads what this engine really produces rather than a hand-rolled GGUF.
    fn write_vector(dir: &std::path::Path, hidden: usize, blocks: &[usize]) -> std::path::PathBuf {
        let mut directions: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        for (n, &b) in blocks.iter().enumerate() {
            directions.insert(b, (0..hidden).map(|i| (i + n + 1) as f32).collect());
        }
        let bytes = repack::control_vector::write_control_vector(&directions, "test-arch", None)
            .expect("the writer should accept these directions");
        let path = dir.join("d.gguf");
        std::fs::write(&path, bytes).expect("write");
        path
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ts-cv-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The reported shape is what the FILE carries, not what the caller
    /// hoped. A host compares `hidden` against its install's own
    /// `arch.hiddenSize`, so a wrong width here is a compatibility check that
    /// passes on a vector `open` will refuse.
    #[test]
    fn the_reported_width_and_coverage_come_from_the_file() {
        let dir = scratch("shape");
        let path = write_vector(&dir, 64, &[1, 2, 3, 7]);
        let json = control_vector_info_json(path.to_str().unwrap()).expect("readable");
        let v: serde_json::Value = serde_json::from_str(&json).expect("json");

        assert_eq!(v["hidden"], 64);
        assert_eq!(v["coveredLayers"], 4);
        assert_eq!(v["minLayer"], 1);
        assert_eq!(v["maxLayer"], 7);
        // The SPAN, not the count: a gapped vector covers 4 blocks across 8,
        // and it is the span that gets compared against the model's layer
        // count. Reporting only the count would call an 8-block vector on a
        // 5-layer model compatible.
        assert_eq!(v["spannedLayers"], 8);
        assert_eq!(v["declaredArch"], "test-arch");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **BLOCK 0 IS NOT EXPRESSIBLE**, so a well-formed file starts at 1 and
    /// a host must not read "31 of 32" as a gap. Pinned here because the
    /// natural reading of a minLayer of 1 is that something is missing.
    #[test]
    fn a_vector_starts_at_block_one_because_block_zero_cannot_be_written() {
        let dir = scratch("base");
        let path = write_vector(&dir, 8, &[1, 2]);
        let json = control_vector_info_json(path.to_str().unwrap()).expect("readable");
        let v: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(v["minLayer"], 1);

        let mut zero: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        zero.insert(0, vec![1.0; 8]);
        assert!(
            repack::control_vector::write_control_vector(&zero, "test-arch", None).is_err(),
            "the writer must refuse block 0, which is what makes minLayer 1 normal"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same reading against a REAL vector, which a synthetic one cannot
    /// stand in for: the fixtures above are written by this port's own writer
    /// on this run, so they cannot catch a convention that drifted between
    /// the writer and a file already on disk.
    ///
    /// `#[ignore]`d and env-gated, like every other real-artifact test here.
    ///
    /// ```sh
    /// TURBOSPARK_STEERING_VECTOR=~/models/steering-vectors/ocean-legacy-layerbase0.gguf \
    ///   cargo test -p turbospark-ffi --lib -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn a_real_control_vector_reports_its_own_shape() {
        let Ok(path) = std::env::var("TURBOSPARK_STEERING_VECTOR") else {
            eprintln!("set TURBOSPARK_STEERING_VECTOR to run this");
            return;
        };
        let json = control_vector_info_json(&path).expect("a real vector should read");
        eprintln!("{json}");
        let v: serde_json::Value = serde_json::from_str(&json).expect("json");
        // Nothing here asserts a SPECIFIC width: the point is that the fields
        // are populated from the file rather than defaulted, and a hardcoded
        // 5120 would make this a test about one vector on one machine.
        assert!(v["hidden"].as_u64().unwrap_or(0) > 0, "{json}");
        assert!(v["coveredLayers"].as_u64().unwrap_or(0) > 0, "{json}");
        assert!(
            v["spannedLayers"].as_u64().unwrap_or(0) >= v["coveredLayers"].as_u64().unwrap_or(0),
            "the span can never be smaller than the count: {json}"
        );
    }

    /// An unreadable path is an ERROR naming itself, never an empty or
    /// default-looking reading. A host that got `{"hidden":0}` back would
    /// render a width mismatch against every model instead of "that file is
    /// not a control vector".
    #[test]
    fn a_path_that_is_not_a_control_vector_is_refused_rather_than_reported_empty() {
        let dir = scratch("bad");
        let path = dir.join("not-a-vector.gguf");
        std::fs::write(&path, b"this is not a gguf").expect("write");
        assert!(control_vector_info_json(path.to_str().unwrap()).is_err());
        assert!(control_vector_info_json(dir.join("missing.gguf").to_str().unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
