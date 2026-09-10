#include <metal_stdlib>
using namespace metal;

// One SIMD group per expert, accumulating the original FP32 matrix.
kernel void minimax_router(
    device const float *w [[buffer(0)]], device const half *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    for (uint col = lane; col < hidden; col += 32) {
        sum += w[row * hidden + col] * float(x[col]);
    }
    sum = simd_sum(sum);
    if (lane == 0) out[row] = sum;
}
