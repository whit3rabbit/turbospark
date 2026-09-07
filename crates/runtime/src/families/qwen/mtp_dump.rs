//! Debug dump facilities for MTP speculative decoding stages.

use std::path::Path;

use foundation::LogitValue;

use crate::real_forward::RealForwardRunner;

impl RealForwardRunner {
    /// Writes the head's surviving intermediates for `scripts/mtp_bisect.py`.
    ///
    /// **Which tensors these are is decided by what OUTLIVES the pass, not by
    /// what would be nicest to have.** One command buffer runs the whole step,
    /// so anything overwritten downstream is gone by the time the host can
    /// read it: `fc`'s raw output is clobbered when the attention residual
    /// adds into `scratch.x`. That one is recoverable offline -- the script
    /// recomputes it from `concat` and the `fc` weights -- so nothing is lost
    /// and the step keeps its single-commit shape. What survives is enough to
    /// bisect: `concat` is the exact input, `moe_x` is the post-attention
    /// norm (so it brackets `fc` AND attention), `x` is the block output
    /// after the FFN residual, and `normed` is after the head's own norm.
    pub(crate) fn dump_mtp_stage(
        &self,
        dir: &Path,
        token: i32,
        position: usize,
        hidden: usize,
        vocab: usize,
        has_logits: bool,
    ) {
        let _ = std::fs::create_dir_all(dir);
        let qwen = match self.real_qwen.as_ref() {
            Some(q) => q,
            None => return,
        };
        let mtp = match self.real_mtp.as_ref() {
            Some(m) => m,
            None => return,
        };
        let write = |name: &str, buf: &gpu::MetalBuffer, len: usize| {
            let mut host = vec![LogitValue::from_f32(0.0); len];
            gpu::read_buffer_f16_into(buf, 0, &mut host);
            let bytes: Vec<u8> = host
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            let _ = std::fs::write(dir.join(name), bytes);
        };
        write("concat.f16", &mtp.concat, 2 * hidden);
        write("moe_x.f16", &qwen.moe_x, hidden);
        write("block_out.f16", &self.scratch.x, hidden);
        write("post_norm.f16", &self.scratch.normed, hidden);
        if has_logits {
            write("logits.f16", &self.scratch.logits, vocab);
        }
        let _ = std::fs::write(
            dir.join("meta.json"),
            format!(
                "{{\"token\": {token}, \"position\": {position}, \
                 \"hidden\": {hidden}, \"vocab\": {vocab}}}\n"
            ),
        );
    }
}
