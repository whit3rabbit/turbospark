// PORT-LOCAL, not vendored: the Swift engine decodes one token at a time
// and has no batched verify, so there is no upstream kernel to mirror.
//
// `dequant_int4_gemm_simd` is `dequant_int4_gemv_simd` with B right-hand
// sides instead of one. It exists because ROADMAP Phase D2 measured that
// no arrangement of the GEMV can do this job:
//
//   - B dispatches on ONE matrix cost 0.65B, not ~1, because the weights
//     do not stay cached across them;
//   - encoding them CONCURRENTLY (no serial barrier) recovers only
//     1.07-1.40x, and 8 x the resulting rate lands at the ~355 GiB/s the
//     kernel saturates at, which says the hardware really did move the
//     weight bytes B times.
//
// So the amortization has to happen INSIDE one dispatch: read each packed
// nibble once, multiply it into B accumulators. That is the whole idea,
// and everything else here is `dequant_int4_gemv_simd` unchanged.
//
// Layouts. `x` is [B, N] and `y` is [B, M], both token-major, so each
// token's vectors stay contiguous and a caller can hand one row of either
// to a kernel that still wants a single token. B is capped at
// kMaxBatchRows because the accumulators are a register array; the cap is
// a hard precondition, not a clamp, and the host asserts it.

// Concatenated AFTER dequant_int4.metal (crate Gotcha 4), which supplies
// the includes and kGroupSize. Never dispatch this file's own constant.

constant constexpr uint kMaxBatchRows = 16;
/// Elements one vectorized block covers: 32 lanes x 4 bytes x 2 nibbles.
constant constexpr uint kBlockElems = 256;
constant constexpr uint kThreadsPerGroup = 256;

kernel void dequant_int4_gemm_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        B      [[buffer(7)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]],
    uint                  tid    [[thread_index_in_threadgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    threadgroup half x_tile[kMaxBatchRows * kBlockElems];
    // NOT an early return: every thread must reach the barriers that fill
    // `x_tile`, and a threadgroup whose rows run past M still has threads
    // that have to participate. Guard the WRITE instead.
    const bool active = row < M;

    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    const uint safe_row = active ? row : 0u;
    device const uint8_t* W_row = W      + uint(safe_row) * row_bytes;
    device const bfloat*  s_row = scales + uint(safe_row) * n_groups;
    device const bfloat*  b_row = biases + uint(safe_row) * n_groups;

    float acc[kMaxBatchRows];
    for (uint i = 0; i < kMaxBatchRows; ++i) {
        acc[i] = 0.0f;
    }

    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        // Stage this block's slice of every token's activations in
        // threadgroup memory. Without this the inner loop re-reads `x`
        // from device memory once per batch element AND once per SIMD
        // group, which at B=16 moves 256 bytes of activations for every 4
        // bytes of weight and makes the batched kernel activation-bound:
        // measured c(16) = 0.64-0.77x, against 0.10x if the weight read
        // were the cost. One block is 256 elements, so the tile is
        // B * 256 halfs, 8 KiB at the cap, shared by all 8 rows.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < B * kBlockElems; i += kThreadsPerGroup) {
            const uint bi = i / kBlockElems;
            const uint e = i % kBlockElems;
            x_tile[i] = x[bi * N + blk * kBlockElems + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint byte_base = blk * 128u + lane * 4u;
        // Same 2-byte-aligned pair load as the GEMV: the resident tensors
        // are 2-aligned but not 4-aligned, so a `uint*` load is undefined.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;

        const float q0 = float(  w4        & 0x0Fu);
        const float q1 = float(( w4        & 0xFFu) >> 4);
        const float q2 = float((w4 >> 8)   & 0x0Fu);
        const float q3 = float(((w4 >> 8)  & 0xFFu) >> 4);
        const float q4 = float((w4 >> 16)  & 0x0Fu);
        const float q5 = float(((w4 >> 16) & 0xFFu) >> 4);
        const float q6 = float((w4 >> 24)  & 0x0Fu);
        const float q7 = float(((w4 >> 24) & 0xFFu) >> 4);

        const uint tile_base = lane * 8u;
        for (uint bi = 0; bi < B; ++bi) {
            threadgroup const half* x_b = x_tile + bi * kBlockElems;
            const half4 xa = *((threadgroup const half4*)(x_b + tile_base));
            const half4 xb = *((threadgroup const half4*)(x_b + tile_base + 4u));
            const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
            const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
            float dot = 0.0f;
            dot = fma(q0, e0, dot); dot = fma(q1, e1, dot);
            dot = fma(q2, e2, dot); dot = fma(q3, e3, dot);
            dot = fma(q4, e4, dot); dot = fma(q5, e5, dot);
            dot = fma(q6, e6, dot); dot = fma(q7, e7, dot);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            acc[bi] = fma(s, dot, acc[bi]);
            acc[bi] = fma(b, sum, acc[bi]);
        }
    }

    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float lo = float(uint(byte & 0x0Fu));
        const float hi = float(uint(byte >> 4));
        for (uint bi = 0; bi < B; ++bi) {
            device const half* x_b = x + bi * N;
            const float x0 = float(x_b[g * kGroupSize + lane * 2u]);
            const float x1 = float(x_b[g * kGroupSize + lane * 2u + 1u]);
            float dot = fma(lo, x0, 0.0f);
            dot = fma(hi, x1, dot);
            const float sum = x0 + x1;
            acc[bi] = fma(s, dot, acc[bi]);
            acc[bi] = fma(b, sum, acc[bi]);
        }
    }

    for (uint bi = 0; bi < B; ++bi) {
        const float total = simd_sum(acc[bi]);
        if (lane == 0 && active) {
            y[bi * M + row] = half(total);
        }
    }
}
