// Mamba1 selective scan, one thread per (sequence, channel), the state in
// registers. The recurrence and readout follow the inworld voice-isolator
// reference (`scan.py`, Apache-2.0), whose MPS kernel this mirrors:
//   h_t = exp(delta_t * a) * h_{t-1} + delta_t * u_t * b_t
//   y_t = (c_t . h_t + d * u_t) * silu(z_t)
// b, c and z are read in place from the fused projections through strides.
#include <metal_stdlib>
using namespace metal;

constant uint MAX_STATES = 16;

struct ScanParams {
    uint seqs;
    uint len;
    uint channels;
    uint states;
    uint bc_stride;   // floats between rows of the b/c projection
    uint b_offset;    // first b column in a row
    uint c_offset;    // first c column in a row
    uint z_stride;    // floats between rows of z
    uint z_offset;    // first z column in a row
    uint carrying;    // 1: start from `carried`, else zeros
};

kernel void selective_scan_f32(
    device float* y [[buffer(0)]],
    device float* last [[buffer(1)]],
    device const float* u [[buffer(2)]],
    device const float* delta [[buffer(3)]],
    device const float* a [[buffer(4)]],
    device const float* bc [[buffer(5)]],
    device const float* d [[buffer(6)]],
    device const float* z [[buffer(7)]],
    device const float* carried [[buffer(8)]],
    constant ScanParams& p [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.seqs * p.channels) {
        return;
    }
    const uint seq = index / p.channels;
    const uint channel = index % p.channels;
    float h[MAX_STATES];
    float rate[MAX_STATES];
    for (uint n = 0; n < p.states; ++n) {
        rate[n] = a[channel * p.states + n];
        h[n] = p.carrying ? carried[index * p.states + n] : 0.0f;
    }
    const float skip = d[channel];
    for (uint t = 0; t < p.len; ++t) {
        const uint row = seq * p.len + t;
        const uint at = row * p.channels + channel;
        const uint bt = row * p.bc_stride;
        const float step = delta[at];
        const float x = u[at];
        const float drive = step * x;
        float out = 0.0f;
        for (uint n = 0; n < p.states; ++n) {
            h[n] = precise::exp(step * rate[n]) * h[n] + drive * bc[bt + p.b_offset + n];
            out += h[n] * bc[bt + p.c_offset + n];
        }
        const float g = z[row * p.z_stride + p.z_offset + channel];
        y[at] = (out + x * skip) * precise::divide(g, 1.0f + precise::exp(-g));
    }
    for (uint n = 0; n < p.states; ++n) {
        last[index * p.states + n] = h[n];
    }
}
