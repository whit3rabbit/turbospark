// Q2_K follows the GGML block_q2_K layout. This source is concatenated after
// dequant_q4_k.metal, matching the K-quant source assembly convention.

constant constexpr uint kQ2_KBlockElems = 256;
constant constexpr uint kQ2_KBlockBytes = 84;
constant constexpr uint kRowsPerTGQ2_K = 8;

static inline uint q2_k_row_bytes(uint n) {
    return n / kQ2_KBlockElems * kQ2_KBlockBytes;
}

static inline float dequant_q2_k_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    float acc = 0.0f;
    const uint segment = lane / 16;
    const uint segment_lane = lane % 16;
    for (uint b = 0; b < N / kQ2_KBlockElems; ++b) {
        device const uint8_t* blk = W_row + b * kQ2_KBlockBytes;
        const ushort d_raw = ushort(blk[80]) | (ushort(blk[81]) << 8);
        const ushort min_raw = ushort(blk[82]) | (ushort(blk[83]) << 8);
        const float d = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(min_raw));
        for (uint half_block = 0; half_block < 2; ++half_block) {
            for (uint group = 0; group < 4; ++group) {
                const uint shift = 2 * group;
                const uint8_t sc = blk[half_block * 8 + group * 2 + segment];
                const float dl = d * float(sc & 15);
                const float ml = dmin * float(sc >> 4);
                const uint8_t q = blk[16 + half_block * 32 + segment * 16 + segment_lane];
                const uint at = b * 256 + half_block * 128 + group * 32 + segment * 16 + segment_lane;
                acc = fma(dl * float((q >> shift) & 3) - ml, float(x[at]), acc);
            }
        }
    }
    return simd_sum(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q2_k_gemv_simd(
    device const uint8_t* W [[buffer(0)]],
    device const half* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGQ2_K + sg_idx;
    if (row >= M) return;
    const float acc = dequant_q2_k_row_simd(W + row * q2_k_row_bytes(N), x, N, lane);
    if (lane == 0) y[row] = half(acc);
}
