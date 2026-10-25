//! Measurement hooks the lab drives (`crates/lab`, evaluator items L1 and
//! L4 in `docs/MLX_SERVE_KERNELS.md`). Both are OFF unless their variable is
//! set, read once, and both are wired only where their effect is defined:
//! the dense FFN layers of the qwen and llama decode loops. Every other
//! flow prints no engagement line for them, which is how `tslab` refuses
//! an arm that would otherwise measure the default under a probe label.
//!
//! - `TURBOSPARK_DISPATCH_PROBE=N`: N in-place multiplies by exactly 1.0 on
//!   the FFN activation, between the activation and the down projection, in
//!   every layer. They sit on the dependency chain (the down projection
//!   reads their output), so the slope of time per token against N is the
//!   price of one chained dispatch on this host. Bit-exact on every finite
//!   f16 (`crates/gpu/tests/dispatch_probe_bits.rs`).
//! - `TURBOSPARK_STANDIN=attn,mlp,gdn`: skip the named blocks in the
//!   sequential decode trunk and leave the residual stream as it was, a zero
//!   delta at the stream's own scale. Base time minus stand-in time prices
//!   the block in place. Output is wrong by design: never quotable, never
//!   gated, and the speculative loop's acceptance moves with it.

use std::sync::OnceLock;

use crate::real_forward_types::RealForwardError;

/// `TURBOSPARK_DISPATCH_PROBE`, read once; 0 when unset or unparsable.
pub(crate) fn dispatch_probe_count() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("TURBOSPARK_DISPATCH_PROBE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Encode the probe's multiplies on `buf[..count]` (f16, in place).
pub(crate) fn encode_dispatch_probe(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    buf: (&gpu::MetalBuffer, u64),
    count: usize,
) -> Result<(), RealForwardError> {
    let n = dispatch_probe_count();
    if n == 0 {
        return Ok(());
    }
    foundation::engaged!("dispatch_probe");
    for _ in 0..n {
        gpu::encode_scalar_mul(context, pass, buf, 1.0, count as u32)
            .map_err(RealForwardError::Gpu)?;
    }
    Ok(())
}

/// A block class `TURBOSPARK_STANDIN` can skip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Block {
    /// Full (softmax) attention: projections, cache write, attention, o_proj.
    Attn,
    /// The dense FFN: gate, up, activation, down.
    Mlp,
    /// A gated-DeltaNet linear-attention block.
    Gdn,
}

fn standin_set() -> &'static [Block] {
    static SET: OnceLock<Vec<Block>> = OnceLock::new();
    SET.get_or_init(|| {
        std::env::var("TURBOSPARK_STANDIN")
            .unwrap_or_default()
            .split(',')
            .filter_map(|b| match b.trim() {
                "attn" => Some(Block::Attn),
                "mlp" => Some(Block::Mlp),
                "gdn" => Some(Block::Gdn),
                _ => None,
            })
            .collect()
    })
}

/// Whether to skip `block` in this layer; prints its engagement line.
pub(crate) fn standin(block: Block) -> bool {
    if !standin_set().contains(&block) {
        return false;
    }
    match block {
        Block::Attn => foundation::engaged!("standin_attn"),
        Block::Mlp => foundation::engaged!("standin_mlp"),
        Block::Gdn => foundation::engaged!("standin_gdn"),
    }
    true
}
