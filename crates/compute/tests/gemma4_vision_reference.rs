//! The Gemma 4 vision reference against mlx-vlm, stage by stage.
//!
//! The fixture is a tiny random tower run through mlx-vlm's own modules
//! (`scripts/gemma4_vision_oracle.py tower`). Stages are compared in order so
//! a failure names the first one that diverged. The grid is 9 wide and 6
//! high, so an x/y swap in the rope, the position table, or the pooler
//! cannot pass.

// Random weights can land near a named constant (one reads as 2/pi), which
// is a coincidence in a fixture, not an approximation. Inner attribute for
// the reason `kv_quant.rs` gives.
mod oracle {
    #![allow(clippy::approx_constant)]
    include!("generated/gemma4_vision_oracle.rs");
}

use turbospark_compute::vision_gemma4::{
    gemma4_vision_forward, Gemma4VisionBlock, Gemma4VisionDims, Gemma4VisionWeights,
};

fn dims() -> Gemma4VisionDims {
    Gemma4VisionDims {
        hidden: oracle::HIDDEN,
        intermediate: oracle::INTERMEDIATE,
        heads: oracle::HEADS,
        head_dim: oracle::HEAD_DIM,
        patch_dim: oracle::PATCH_DIM,
        pos_table_rows: oracle::POS_TABLE_ROWS,
        pool: oracle::POOL,
        text_hidden: oracle::TEXT_HIDDEN,
        rope_theta: oracle::ROPE_THETA,
        eps: oracle::EPS,
    }
}

fn role<'a>(block: &'a [&'a [f32]], name: &str) -> &'a [f32] {
    let i = oracle::BLOCK_ROLES
        .iter()
        .position(|r| *r == name)
        .unwrap_or_else(|| panic!("fixture has no role {name}"));
    block[i]
}

fn weights() -> Gemma4VisionWeights<'static> {
    let blocks = oracle::BLOCK_WEIGHTS
        .iter()
        .map(|b| Gemma4VisionBlock {
            ln_in: role(b, "ln_in"),
            q: role(b, "q"),
            k: role(b, "k"),
            v: role(b, "v"),
            q_norm: role(b, "q_norm"),
            k_norm: role(b, "k_norm"),
            o: role(b, "o"),
            ln_post_attn: role(b, "ln_post_attn"),
            ln_pre_ffw: role(b, "ln_pre_ffw"),
            gate: role(b, "gate"),
            up: role(b, "up"),
            down: role(b, "down"),
            ln_post_ffw: role(b, "ln_post_ffw"),
        })
        .collect();
    Gemma4VisionWeights {
        input_proj: oracle::W_INPUT_PROJ,
        pos_table: oracle::W_POS_TABLE,
        blocks,
        std_bias: oracle::W_STD_BIAS,
        std_scale: oracle::W_STD_SCALE,
        proj: oracle::W_PROJ,
    }
}

fn positions() -> Vec<(usize, usize)> {
    oracle::POSITIONS
        .chunks_exact(2)
        .map(|p| (p[0], p[1]))
        .collect()
}

/// Cosine and worst absolute error, NaN-safe: a non-finite value fails
/// rather than vanishing in a max.
fn compare(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let (mut dot, mut ng, mut nw, mut worst) = (0f64, 0f64, 0f64, 0f32);
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite() && w.is_finite(), "{name}: non-finite value");
        dot += f64::from(*g) * f64::from(*w);
        ng += f64::from(*g) * f64::from(*g);
        nw += f64::from(*w) * f64::from(*w);
        worst = worst.max((g - w).abs());
    }
    let cos = dot / (ng.sqrt() * nw.sqrt());
    // Both sides are FP32 over the same graph; the slack is summation order
    // and mlx's fused rms_norm and SDPA. 1e-4 absolute on values of order 1-10.
    assert!(cos > 0.999_999, "{name}: cosine {cos}");
    assert!(worst < 1e-4, "{name}: worst abs diff {worst}");
}

#[test]
fn every_stage_matches_mlx_vlm() {
    let (gw, _) = oracle::GRID_WH;
    let s = gemma4_vision_forward(oracle::PIXELS, &positions(), gw, &weights(), &dims());
    compare("patch_embed", &s.patch_embed, oracle::OUT_PATCH_EMBED);
    compare("block_0", &s.blocks[0], oracle::OUT_BLOCK_0);
    compare("block_1", &s.blocks[1], oracle::OUT_BLOCK_1);
    compare("tower", &s.tower, oracle::OUT_TOWER);
    compare("embedded", &s.embedded, oracle::OUT_EMBEDDED);
}

/// Running only the real patches is the same function as the reference's
/// padded run: padding is masked as keys, zeroed before pooling, and its
/// empty soft token stripped.
#[test]
fn padding_does_not_change_the_pooled_rows() {
    compare("padded tower", oracle::PADDED_OUT_TOWER, oracle::OUT_TOWER);
}

/// The fixture must be able to tell x from y, or the stage checks prove
/// nothing about the axial order.
#[test]
fn the_fixture_grid_is_not_square() {
    let (w, h) = oracle::GRID_WH;
    assert_ne!(w, h);
    assert_eq!(oracle::BLOCK_WEIGHTS.len(), oracle::BLOCKS);
    // The padded case must carry padding, or it proves nothing.
    const _: () = assert!(oracle::PADDING > 0);
}
