//! The install's weight inventory, read off its resident index.
//!
//! `ArchConfig::decode_int4_gemv_shapes` reconstructs projection shapes from
//! config fields, and that misses whatever a family does differently (MLA's
//! latent projections, Spark's fused q|k|v, a dense lead layer's width). The
//! resident index lists every tensor the runtime will bind, with its dtype
//! tag and logical shape, so it is the inventory by construction. Routed
//! experts live in `packed_experts/`, not in the index; their shapes still
//! come from `ArchConfig`.
//!
//! The kernel benches take their matmul shapes from here, the verify ladder
//! its per-shape multiplicity, and coverage its layout warnings.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

/// Resident dtype tags (`real_forward_dispatch::encode_gemv_any`).
pub const DTYPE_INT4_AFFINE: u8 = 4;
pub const DTYPE_INT8_AFFINE: u8 = 5;
pub const DTYPE_INT1_AFFINE: u8 = 15;
pub const DTYPE_INT2_AFFINE: u8 = 16;

pub fn dtype_name(tag: u8) -> String {
    match tag {
        DTYPE_INT4_AFFINE => "int4_affine".into(),
        DTYPE_INT8_AFFINE => "int8_affine".into(),
        DTYPE_INT1_AFFINE => "int1_affine".into(),
        DTYPE_INT2_AFFINE => "int2_affine".into(),
        other => format!("dtype_{other}"),
    }
}

/// One distinct matmul shape of one dtype: how many tensors have it, and
/// a few of their names with the layer index stripped.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Shape {
    pub dtype: u8,
    pub dtype_name: String,
    /// Output rows (the GEMV's M in `y = W x`).
    pub rows: usize,
    /// Input width.
    pub cols: usize,
    /// Tensors of this shape in the install, across layers. One decode token
    /// runs each of them once.
    pub count: usize,
    pub roles: Vec<String>,
}

/// `model.layers.17.self_attn.q_proj.weight` -> `self_attn.q_proj`.
pub fn role(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let start = parts
        .iter()
        .position(|p| p.parse::<usize>().is_ok())
        .map_or(0, |i| i + 1);
    let end = if parts.last() == Some(&"weight") {
        parts.len() - 1
    } else {
        parts.len()
    };
    parts[start.min(end)..end].join(".")
}

/// Distinct 2-D weight shapes by dtype, from (name, dtype, rows, cols)
/// rows. Portable so the grouping is tested without an install.
pub fn group(tensors: impl IntoIterator<Item = (String, u8, usize, usize)>) -> Vec<Shape> {
    let mut m: BTreeMap<(u8, usize, usize), (usize, Vec<String>)> = BTreeMap::new();
    for (name, dtype, rows, cols) in tensors {
        if rows == 0 || cols == 0 || !name.ends_with(".weight") {
            continue;
        }
        let e = m.entry((dtype, rows, cols)).or_default();
        e.0 += 1;
        let r = role(&name);
        if !e.1.contains(&r) && e.1.len() < 6 {
            e.1.push(r);
        }
    }
    m.into_iter()
        .map(|((dtype, rows, cols), (count, roles))| Shape {
            dtype,
            dtype_name: dtype_name(dtype),
            rows,
            cols,
            count,
            roles,
        })
        .collect()
}

/// The install's matmul shapes, or why they could not be read.
#[cfg(target_os = "macos")]
pub fn load(dir: &Path) -> Result<Vec<Shape>, String> {
    let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
        .map_err(|e| format!("{}: resident index: {e}", dir.display()))?;
    Ok(group(index.entries.into_values().filter_map(|e| {
        // Logical shape, rank padded to 4: a matrix is (rows, cols, 1, 1).
        let (a, b, c, d) = e.shape;
        (c <= 1 && d <= 1 && b > 1).then_some((e.name, e.dtype, a as usize, b as usize))
    })))
}

#[cfg(not(target_os = "macos"))]
pub fn load(dir: &Path) -> Result<Vec<Shape>, String> {
    Err(format!(
        "{}: the resident index reader is macOS-only",
        dir.display()
    ))
}

/// Shapes of one dtype.
pub fn of_dtype(shapes: &[Shape], dtype: u8) -> Vec<&Shape> {
    shapes.iter().filter(|s| s.dtype == dtype).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_strip_layer_and_suffix() {
        assert_eq!(
            role("model.layers.17.self_attn.q_proj.weight"),
            "self_attn.q_proj"
        );
        assert_eq!(role("lm_head.weight"), "lm_head");
        assert_eq!(role("mtp.fc.weight"), "mtp.fc");
    }

    #[test]
    fn groups_by_dtype_and_shape_and_counts_layers() {
        let t = (0..3)
            .map(|l| {
                (
                    format!("model.layers.{l}.mlp.up_proj.weight"),
                    4u8,
                    17408,
                    5120,
                )
            })
            .chain([
                ("model.layers.0.mlp.gate_proj.weight".into(), 4, 17408, 5120),
                ("model.layers.0.router.weight".into(), 5, 64, 5120),
                ("model.norm.weight".into(), 4, 0, 5120),
                ("model.layers.0.mlp.up_proj.scales".into(), 4, 17408, 80),
            ]);
        let g = group(t);
        assert_eq!(g.len(), 2, "{g:#?}");
        assert_eq!((g[0].dtype, g[0].count), (4, 4));
        assert_eq!(g[0].roles, vec!["mlp.up_proj", "mlp.gate_proj"]);
        assert_eq!(g[1].dtype_name, "int8_affine");
        assert_eq!(of_dtype(&g, DTYPE_INT8_AFFINE).len(), 1);
    }
}
