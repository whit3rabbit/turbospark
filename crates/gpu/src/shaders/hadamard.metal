#include <metal_stdlib>
using namespace metal;

// ============================================================================
// hadamard -- row-wise signed block-Fast-Walsh-Hadamard transform, the
// activation-side half of the prism Hadamard-folded weight contract (the
// Bonsai-2 line; `docs/BONSAI2.md`).
//
// The CONTRACT. A folded checkpoint stores every quantized matrix in the
// rotated basis W' = W * diag(signs) * H_b / sqrt(block) (H_b: independent
// Hadamard butterflies over `block`-sized segments of the INPUT axis), and
// stores the embedding rows as e' = e * diag(signs) * H_b / sqrt(block). No
// weight ever carries the transform at run time; the caller instead runs
//
//   forward  y = H_b (signs * x) / sqrt(block)   on every folded entry's INPUT
//   inverse  y = signs * (H_b x) / sqrt(block)   on the embedding's OUTPUT
//
// so that W' * forward(x) = W x and inverse(e') = e. This is the same
// decomposition prism's own runtime implements (`fwht` in its bundled
// `runtime.py`): signs first for the forward, signs last for the inverse,
// 1/sqrt(block) scale on every path. The sign vector spans the FULL row
// width and is sliced per segment; one vector is shared by every folded
// entry of the same input width (verified equal across all 402 packed
// modules of the real checkpoint at build time).
//
// The butterfly is the natural (Sylvester) order, copied from
// `attention_tq.metal`'s `tq_rht` -- the Walsh-Hadamard matrix is symmetric,
// so the recursive halving topology computes the standard Hadamard map the
// same as any other evaluation order. Math is FP32 end to end regardless of
// the FP16 storage of src/dst.
//
// One threadgroup per (row, segment): x = row index, y = segment index.
// `threadgroup float buf[4096]` sizes the LARGEST compiled block width;
// the host refuses anything over it, so a wider contract needs the array
// and the guard raised together.
// ============================================================================

constant constexpr uint kHadamardMaxBlock = 4096;

static inline void hadamard_butterfly(threadgroup float* buf,
                                      device const float* signs,
                                      uint dim,
                                      bool forward,
                                      uint lid,
                                      uint lsize) {
    if (forward) {
        for (uint i = lid; i < dim; i += lsize) { buf[i] *= signs[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint h = 1; h < dim; h <<= 1) {
        for (uint i = lid; i < dim / 2; i += lsize) {
            const uint block = i / h;
            const uint offset = i % h;
            const uint idx0 = block * 2u * h + offset;
            const uint idx1 = idx0 + h;
            const float a = buf[idx0];
            const float b = buf[idx1];
            buf[idx0] = a + b;
            buf[idx1] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(float(dim));
    for (uint i = lid; i < dim; i += lsize) { buf[i] *= scale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!forward) {
        for (uint i = lid; i < dim; i += lsize) { buf[i] *= signs[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void hadamard_fwht_rows(device const half* src [[buffer(0)]],
                               device half* dst [[buffer(1)]],
                               device const float* signs [[buffer(2)]],
                               constant uint& width [[buffer(3)]],
                               constant uint& block [[buffer(4)]],
                               constant uint& forward [[buffer(5)]],
                               // ALL THREE POSITION ATTRIBUTES ARE `uint2`: a kernel's
                               // position inputs must be all scalar or all vectors of the
                               // SAME width (the rule `vision.metal`'s GEMM documents), and
                               // this kernel's (row, segment) grid wants the vector.
                               uint2 gid [[threadgroup_position_in_grid]],
                               uint2 lid2 [[thread_position_in_threadgroup]],
                               uint2 lsize2 [[threads_per_threadgroup]]) {
    threadgroup float buf[kHadamardMaxBlock];
    const uint lid = lid2.x;
    const uint lsize = lsize2.x;
    // The host dispatches ONE threadgroup per (row, segment) as a FLAT 1D
    // grid, so decode the pair here: row-major, segments fastest.
    const uint segments = width / block;
    const uint row = gid.x / segments;
    const uint base = (gid.x % segments) * block;
    for (uint i = lid; i < block; i += lsize) {
        buf[i] = float(src[row * width + base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hadamard_butterfly(buf, signs + base, block, forward != 0u, lid, lsize);
    for (uint i = lid; i < block; i += lsize) {
        dst[row * width + base + i] = half(buf[i]);
    }
}

// ============================================================================
// hadamard_fwht_block -- the hybrid shape for the block widths real
// checkpoints ship. Same contract, same (row, segment) grid, same buffer ABI
// as the generic kernel above (`block` is bound but unused: the width is a
// template parameter), at kHadamardHybridThreads threads per threadgroup with
// N/NT elements per thread in registers. Butterflies below the simdgroup
// width exchange through simd_shuffle_xor, the middle stages through
// threadgroup memory, and the rest pair registers of one thread, so the
// stage barriers the generic kernel pays at every step shrink to the middle
// stages only.
//
// BIT-IDENTITY with the generic kernel: every butterfly pairs element e with
// e ^ h, and both kernels compute a + b on the clear-bit partner and a - b on
// the set-bit one over the same h = 1..N/2 stage order, with the 1/sqrt(N)
// scale AFTER the butterflies and the signs exact +/-1 multiplies at the same
// two points. Only the data movement differs, so the per-element arithmetic
// sequence -- and therefore the output bits -- is unchanged.
// ============================================================================

constant constexpr uint kHadamardHybridThreads = 256;

template<int N, int NT>
kernel void hadamard_fwht_block(device const half* src [[buffer(0)]],
                                device half* dst [[buffer(1)]],
                                device const float* signs [[buffer(2)]],
                                constant uint& width [[buffer(3)]],
                                constant uint& block [[buffer(4)]],
                                constant uint& forward [[buffer(5)]],
                                uint gid [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float buf[N];
    constexpr uint NE = N / NT;
    float reg[NE];
    const uint segments = width / N;
    const uint row = gid / segments;
    const uint base = (gid % segments) * N;
    src += row * width + base;
    dst += row * width + base;
    device const float* seg_signs = signs + base;
    for (uint j = 0u; j < NE; ++j) {
        reg[j] = float(src[j * NT + lid]);
    }
    if (forward != 0u) {
        for (uint j = 0u; j < NE; ++j) {
            reg[j] *= seg_signs[j * NT + lid];
        }
    }
    // Stages below the simdgroup width: the partner differs in lane bits.
    for (uint h = 1u; h < 32u; h <<= 1) {
        for (uint j = 0u; j < NE; ++j) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, h);
            reg[j] = (lid & h) == 0u ? val + val2 : val2 - val;
        }
    }
    // Stages up to the threadgroup width: the partner is another thread's
    // element, exchanged through threadgroup memory.
    for (uint h = 32u; h < NT; h <<= 1) {
        for (uint j = 0u; j < NE; ++j) {
            buf[j * NT + lid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < NE; ++j) {
            const float val = reg[j];
            const float val2 = buf[j * NT + (lid ^ h)];
            reg[j] = (lid & h) == 0u ? val + val2 : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Stages above the threadgroup width: the partner is another register of
    // the same thread.
    for (uint h = NT; h < N; h <<= 1) {
        const uint step = h / NT;
        for (uint j = 0u; j < NE; j += 2u * step) {
            for (uint k = 0u; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
    const float scale = rsqrt(float(N));
    for (uint j = 0u; j < NE; ++j) {
        reg[j] *= scale;
    }
    if (forward == 0u) {
        for (uint j = 0u; j < NE; ++j) {
            reg[j] *= seg_signs[j * NT + lid];
        }
    }
    for (uint j = 0u; j < NE; ++j) {
        dst[j * NT + lid] = half(reg[j]);
    }
}

typedef decltype(hadamard_fwht_block<256, kHadamardHybridThreads>) hadamard_fwht_block_fn;
template [[host_name("hadamard_fwht_tg_256")]] kernel hadamard_fwht_block_fn hadamard_fwht_block<256, kHadamardHybridThreads>;
template [[host_name("hadamard_fwht_tg_512")]] kernel hadamard_fwht_block_fn hadamard_fwht_block<512, kHadamardHybridThreads>;
template [[host_name("hadamard_fwht_tg_1024")]] kernel hadamard_fwht_block_fn hadamard_fwht_block<1024, kHadamardHybridThreads>;
template [[host_name("hadamard_fwht_tg_2048")]] kernel hadamard_fwht_block_fn hadamard_fwht_block<2048, kHadamardHybridThreads>;
template [[host_name("hadamard_fwht_tg_4096")]] kernel hadamard_fwht_block_fn hadamard_fwht_block<4096, kHadamardHybridThreads>;
