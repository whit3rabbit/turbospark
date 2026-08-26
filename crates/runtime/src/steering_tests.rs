use super::*;

/// THE PER-LAYER BLOCKS MUST PARTITION THE BUFFER, and nothing about the
/// generated logits can tell you whether they do.
///
/// A batched pass writes M floats from `coeff_offset(layer)`. If the
/// stride were narrower than a block -- one float per layer, which is
/// what it was before the batched path existed -- layer L's write would
/// land inside layer L+1's block, and at the LAST layer it would run off
/// the end of the buffer entirely. Neither shows up downstream: the
/// coefficients are an output-only measurement surface, every later
/// layer's own write happens to overwrite the spill, and a short GPU
/// buffer overrun lands in page slack rather than faulting.
///
/// So this is the only place the layout can be checked at all, and it is
/// arithmetic: no GPU, no install, microseconds.
#[test]
fn the_coefficient_blocks_partition_the_buffer() {
    for layers in [1usize, 2, 4, 64] {
        let size = SteeringState::coeff_bytes(layers);
        for l in 0..layers {
            let base = SteeringState::coeff_offset(l);
            assert!(
                base + (MAX_STEER_ROWS * 4) as u64 <= size,
                "a full block at layer {l} of {layers} runs past the {size}-byte buffer; \
                 a batched steer there would write off the end of it"
            );
            if l + 1 < layers {
                assert_eq!(
                    SteeringState::coeff_offset(l + 1) - base,
                    (MAX_STEER_ROWS * 4) as u64,
                    "layer {l}'s block overlaps layer {}'s, so a batched write of M rows \
                     would land in the next layer's coefficients",
                    l + 1
                );
            }
        }
    }
}
