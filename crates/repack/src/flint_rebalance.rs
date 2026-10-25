//! Flint's FP16 range fix: move a power of two from every `up_proj` into the
//! matching `down_proj`.
//!
//! **THE PROBLEM IS REAL AND MEASURED, NOT HYPOTHETICAL.** On the published
//! `flint-int4`, the SwiGLU product `silu(gate) * up` reaches about 97,000 at
//! a layer-46 expert on an ordinary English prompt ("Write a detailed
//! explanation of how a bill becomes a law in the United States."), against
//! an FP16 maximum of 65,504. The product is stored FP16 between phase 1 and
//! phase 2 of the routed pair (and between the shared expert's and the dense
//! lead's two GEMVs), so it overflows to `inf`, the weighted sum becomes NaN
//! in every element, and the very next RMS norm turns the whole stream to NaN.
//! The llama.cpp port needed adaptive input scaling for the same reason.
//!
//! **THE FIX IS EXACT IN THE MATH.** `up` is linear, so dividing its scales and
//! biases by `2^K` divides `up(x)` by `2^K`, and so the stored product by
//! `2^K`; `down` is linear too, so multiplying ITS scales and biases by `2^K`
//! puts the result back. The gate is untouched because `silu` is not linear.
//! Net, `down'(silu(g) * up'(x)) == down(silu(g) * up(x))`. A BF16 scale is
//! shifted by adding to its exponent field, which is exact (no mantissa bit
//! moves), and the transform is invertible bit for bit.
//!
//! WHY NOT A RUNTIME SCALE: the routing weights could carry `2^K` (phase 2
//! multiplies by them), but only for the routed experts; the shared expert and
//! the dense lead have no such handle. Folding both halves into the weights
//! covers all three with no runtime constant, no manifest field and no kernel
//! change. The cost is that an install must be built (or migrated) with it:
//! an install without it overflows on prompts like the one above. The
//! transform is the install's format, recorded in `docs/FLINT_PHASE0.md` and
//! `DEVIATIONS.md`.
//!
//! WHY `K = 6`: the measured peak (97k) needs at least 2^1; 64 gives about 43x
//! headroom over the FP16 limit for tokens not yet seen. The cost is
//! subnormal precision for products below `6.1e-5 * 64 = 3.9e-3`, which are
//! negligible next to the large ones that dominate a down-projection's sum.

use std::path::Path;

/// Binary exponent steps moved from `up_proj` to `down_proj`. See above.
pub const FLINT_SWIGLU_SHIFT: i32 = 6;

/// Which half of a SwiGLU an FFN tensor is, for the shift's sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwigluRole {
    Up,
    Down,
}

impl SwigluRole {
    /// The exponent delta this role takes for a shift of `steps`: `up` divides
    /// by `2^steps`, `down` multiplies by it.
    pub fn delta(self, steps: i32) -> i32 {
        match self {
            SwigluRole::Up => -steps,
            SwigluRole::Down => steps,
        }
    }

    /// The role of a RESIDENT tensor's name, or `None`. Matches the dense
    /// lead (`mlp.{up,down}_proj`) and the shared expert
    /// (`mlp.shared_expert.{up,down}_proj`), never a routed expert (those live
    /// in `packed_experts/`, not the resident index).
    pub fn of_resident(name: &str) -> Option<Self> {
        if name.ends_with(".mlp.up_proj.weight")
            || name.ends_with(".mlp.shared_expert.up_proj.weight")
        {
            Some(SwigluRole::Up)
        } else if name.ends_with(".mlp.down_proj.weight")
            || name.ends_with(".mlp.shared_expert.down_proj.weight")
        {
            Some(SwigluRole::Down)
        } else {
            None
        }
    }
}

/// Adds `delta` to a BF16 value's exponent. Exact. Zero, subnormal, infinity
/// and NaN are returned unchanged, as is any value whose shifted exponent
/// would leave `1..=254`.
pub fn shift_bf16_exponent(bits: u16, delta: i32) -> u16 {
    let exponent = i32::from((bits >> 7) & 0xFF);
    if exponent == 0 || exponent == 255 {
        return bits;
    }
    let shifted = exponent + delta;
    if !(1..=254).contains(&shifted) {
        return bits;
    }
    (bits & 0x807F) | ((shifted as u16) << 7)
}

/// [`shift_bf16_exponent`] over little-endian BF16 bytes. Returns how many
/// non-zero values were left alone for range.
pub fn shift_bf16_bytes(bytes: &mut [u8], delta: i32) -> usize {
    let mut left_alone = 0;
    for pair in bytes.chunks_exact_mut(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let shifted = shift_bf16_exponent(bits, delta);
        if shifted == bits && bits & 0x7FFF != 0 {
            left_alone += 1;
        }
        pair.copy_from_slice(&shifted.to_le_bytes());
    }
    left_alone
}

/// [`shift_bf16_exponent`] over a BF16 plane held as `u16` bits.
pub fn shift_bf16_plane(bits: &mut [u16], delta: i32) {
    for value in bits {
        *value = shift_bf16_exponent(*value, delta);
    }
}

/// The file that records an install's NET shift, so a migration cannot be
/// applied twice and a reader can tell a rebalanced install from the
/// checkpoint's own bytes. Absent means a net shift of zero.
pub const MARKER_FILE: &str = "flint_rebalance.json";

/// The net shift an install carries, from [`MARKER_FILE`] (zero when absent).
pub fn recorded_shift(dir: &Path) -> Result<i32, String> {
    let path = dir.join(MARKER_FILE);
    if !path.exists() {
        return Ok(0);
    }
    let value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).map_err(|e| format!("reading {MARKER_FILE}: {e}"))?,
    )
    .map_err(|e| format!("parsing {MARKER_FILE}: {e}"))?;
    value["netSteps"]
        .as_i64()
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| format!("{MARKER_FILE} has no integer netSteps"))
}

/// Records `net` in [`MARKER_FILE`], removing the file when it is zero.
pub fn record_shift(dir: &Path, net: i32) -> Result<(), String> {
    let path = dir.join(MARKER_FILE);
    if net == 0 {
        return match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("removing {MARKER_FILE}: {e}")),
        };
    }
    let body = serde_json::json!({
        "netSteps": net,
        "what": "binary exponent steps moved from every up_proj to the matching down_proj \
                 (scales and biases); see crates/repack/src/flint_rebalance.rs",
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&body).unwrap())
        .map_err(|e| format!("writing {MARKER_FILE}: {e}"))
}

/// What an in-place pass touched.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RebalanceStats {
    pub resident_tensors: usize,
    pub expert_blobs: usize,
    pub left_alone: usize,
}

/// Applies a shift of `steps` (negative to undo) to an existing Flint install
/// IN PLACE: every dense/shared `up`/`down` companion plane in
/// `model_weights.bin` and every routed expert's `up_*`/`down_*` scales and
/// biases in `packed_experts/layer_*.bin`.
///
/// This is the migration for an install built before the transform existed,
/// and the inverse the tests use. It is its own inverse when called with
/// `-steps`, exactly, unless `left_alone` is non-zero.
pub fn rebalance_install_in_place(dir: &Path, steps: i32) -> Result<RebalanceStats, String> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut stats = RebalanceStats::default();

    let weights_path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&weights_path).map_err(|e| e.to_string())?;
    let mut weights = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&weights_path)
        .map_err(|e| format!("opening {}: {e}", weights_path.display()))?;
    for entry in index.entries.values() {
        let Some(role) = SwigluRole::of_resident(&entry.name) else {
            continue;
        };
        for (offset, size) in [
            (entry.scale_offset, entry.scale_size),
            (entry.bias_offset, entry.bias_size),
        ] {
            let mut buf = vec![0u8; size as usize];
            weights
                .seek(SeekFrom::Start(offset))
                .and_then(|_| weights.read_exact(&mut buf))
                .map_err(|e| format!("reading {}: {e}", entry.name))?;
            stats.left_alone += shift_bf16_bytes(&mut buf, role.delta(steps));
            weights
                .seek(SeekFrom::Start(offset))
                .and_then(|_| weights.write_all(&buf))
                .map_err(|e| format!("writing {}: {e}", entry.name))?;
        }
        stats.resident_tensors += 1;
    }

    let layout_path = dir.join("packed_experts/layout.json");
    let layout: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&layout_path).map_err(|e| format!("reading layout.json: {e}"))?,
    )
    .map_err(|e| format!("parsing layout.json: {e}"))?;
    let layers = layout["layers"]
        .as_array()
        .ok_or("layout.json has no layers")?;
    for layer in layers {
        let file = layer["file"].as_str().ok_or("layer entry has no file")?;
        let path = dir.join("packed_experts").join(file);
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| format!("opening {}: {e}", path.display()))?;
        for expert in layer["experts"].as_array().ok_or("layer has no experts")? {
            let base = expert["offset"].as_u64().ok_or("expert has no offset")?;
            for (tensor, role) in [
                ("up_scales", SwigluRole::Up),
                ("up_biases", SwigluRole::Up),
                ("down_scales", SwigluRole::Down),
                ("down_biases", SwigluRole::Down),
            ] {
                let t = &expert["tensors"][tensor];
                let offset = base + t["offset"].as_u64().ok_or("tensor has no offset")?;
                let size = t["size"].as_u64().ok_or("tensor has no size")? as usize;
                let mut buf = vec![0u8; size];
                f.seek(SeekFrom::Start(offset))
                    .and_then(|_| f.read_exact(&mut buf))
                    .map_err(|e| format!("reading {tensor} in {file}: {e}"))?;
                stats.left_alone += shift_bf16_bytes(&mut buf, role.delta(steps));
                f.seek(SeekFrom::Start(offset))
                    .and_then(|_| f.write_all(&buf))
                    .map_err(|e| format!("writing {tensor} in {file}: {e}"))?;
            }
            stats.expert_blobs += 1;
        }
    }
    record_shift(dir, recorded_shift(dir)? + steps)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    #[test]
    fn a_shift_is_exact_and_invertible() {
        for v in [1.0f32, -0.0371, 3.4e-4, 123.5, -7.0e-3, 1.0e20, 1.0e-20] {
            let bits = bf16(v);
            let down = shift_bf16_exponent(bits, -6);
            assert_eq!(
                f32::from_bits(u32::from(down) << 16),
                f32::from_bits(u32::from(bits) << 16) / 64.0
            );
            assert_eq!(shift_bf16_exponent(down, 6), bits, "{v}");
        }
    }

    #[test]
    fn zero_and_non_finite_values_are_left_alone() {
        for bits in [0x0000u16, 0x8000, 0x7F80, 0xFF80, 0x7FC0] {
            assert_eq!(shift_bf16_exponent(bits, -6), bits);
            assert_eq!(shift_bf16_exponent(bits, 6), bits);
        }
    }

    #[test]
    fn a_shift_that_leaves_the_exponent_range_is_refused_not_wrapped() {
        // Exponent 3 minus 6 would wrap; the value is left as it was.
        let tiny = (3u16) << 7;
        assert_eq!(shift_bf16_exponent(tiny, -6), tiny);
        let mut bytes = tiny.to_le_bytes().to_vec();
        assert_eq!(shift_bf16_bytes(&mut bytes, -6), 1);
    }

    #[test]
    fn up_divides_and_down_multiplies() {
        assert_eq!(SwigluRole::Up.delta(6), -6);
        assert_eq!(SwigluRole::Down.delta(6), 6);
    }

    #[test]
    fn only_dense_and_shared_names_are_resident_swiglu_tensors() {
        let n = |s: &str| format!("language_model.model.layers.3.{s}");
        assert_eq!(
            SwigluRole::of_resident(&n("mlp.up_proj.weight")),
            Some(SwigluRole::Up)
        );
        assert_eq!(
            SwigluRole::of_resident(&n("mlp.down_proj.weight")),
            Some(SwigluRole::Down)
        );
        assert_eq!(
            SwigluRole::of_resident(&n("mlp.shared_expert.up_proj.weight")),
            Some(SwigluRole::Up)
        );
        assert_eq!(
            SwigluRole::of_resident(&n("mlp.shared_expert.down_proj.weight")),
            Some(SwigluRole::Down)
        );
        // The gate is untouched: silu is not linear.
        assert_eq!(SwigluRole::of_resident(&n("mlp.gate_proj.weight")), None);
        assert_eq!(
            SwigluRole::of_resident(&n("mlp.shared_expert.gate_proj.weight")),
            None
        );
        assert_eq!(SwigluRole::of_resident(&n("self_attn.o_proj.weight")), None);
        assert_eq!(SwigluRole::of_resident(&n("mlp.gate.weight")), None);
    }
}
