// Music 3 sequence kernels: device forms of the host steps between
// projections in a depth decoder layer, so a whole step runs in one command
// buffer. Each reproduces the host arithmetic in the audio crate's
// `precision.rs` for BF16 activations (dtype 2 of `music_round`).
//
// The device may flush denormal operands and results to zero, and the host
// does not. Any operand that is subnormal, and any result whose exponent
// field is zero unless the exact result is zero, raises `inexact`; the
// caller then recomputes the step on the per-operation path. Compiled with
// precise math, so the single add or multiply is correctly rounded.

inline bool music_subnormal(uint u) {
    return (u & 0x7f800000u) == 0u && (u & 0x007fffffu) != 0u;
}

// y = Bf16.round(a + b): the host residual add.
kernel void music3_add_round_bf16(device const float *a [[buffer(0)]],
 device const float *b [[buffer(1)]], device float *y [[buffer(2)]],
 device atomic_uint *inexact [[buffer(3)]], constant uint *p [[buffer(4)]],
 uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    float x = a[gid], z = b[gid], s = x + z;
    uint ux = as_type<uint>(x), uz = as_type<uint>(z), us = as_type<uint>(s);
    // A zero sum is exact only for two zeros or exact cancellation.
    bool exact_zero = ((ux | uz) & 0x7fffffffu) == 0u || (ux ^ uz) == 0x80000000u;
    if (music_subnormal(ux) || music_subnormal(uz)
        || ((us & 0x7f800000u) == 0u && !exact_zero)) {
        atomic_store_explicit(inexact, 1u, memory_order_relaxed);
    }
    y[gid] = music_round(s, 2);
}

// y = Bf16.round(Bf16.silu(g) * u): the host SwiGLU. silu[i] is the host
// activation of the bf16 value with bits i << 16, so the activation matches
// by construction. g must already be a bf16 value (the gate projection
// rounds to bf16); anything else is flagged rather than truncated.
kernel void music3_swiglu_bf16(device const float *g [[buffer(0)]],
 device const float *u [[buffer(1)]], device const float *silu [[buffer(2)]],
 device float *y [[buffer(3)]], device atomic_uint *inexact [[buffer(4)]],
 constant uint *p [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    uint ug = as_type<uint>(g[gid]);
    float s = silu[ug >> 16], v = u[gid], m = s * v;
    uint us = as_type<uint>(s), uv = as_type<uint>(v), um = as_type<uint>(m);
    bool exact_zero = (us & 0x7fffffffu) == 0u || (uv & 0x7fffffffu) == 0u;
    if ((ug & 0xffffu) != 0u || music_subnormal(us) || music_subnormal(uv)
        || ((um & 0x7f800000u) == 0u && !exact_zero)) {
        atomic_store_explicit(inexact, 1u, memory_order_relaxed);
    }
    y[gid] = music_round(m, 2);
}

// Sets flag when any value is infinite or NaN. Run over every operation
// output before readback, matching the per-operation non-finite check.
kernel void music3_nonfinite(device const uint *x [[buffer(0)]],
 device atomic_uint *flag [[buffer(1)]], constant uint *p [[buffer(2)]],
 uint gid [[thread_position_in_grid]]) {
    if (gid < p[0] && (x[gid] & 0x7f800000u) == 0x7f800000u) {
        atomic_store_explicit(flag, 1u, memory_order_relaxed);
    }
}

// y = x as raw bits (no float load or store), so the copy is exact for every
// pattern. Writes AR KV rows into the persistent cache.
kernel void music3_copy(device const uint *x [[buffer(0)]],
 device uint *y [[buffer(1)]], constant uint *p [[buffer(2)]],
 uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) y[gid] = x[gid];
}

// Nonzero result of an op whose exact result is nonzero, but whose float
// may have been flushed: exponent field zero.
inline bool music_flushed(uint result, bool exact_zero) {
    return (result & 0x7f800000u) == 0u && !exact_zero;
}
inline bool music_zero(uint u) { return (u & 0x7fffffffu) == 0u; }

// The host partial NeoX rotary at BF16 (`partial_rope_typed` in the audio
// crate's dit.rs) over `[len, heads * head_dim]`: for d < rotary/2,
//   x[d]        = round(round(a * c) - round(b * s))
//   x[d + hw]   = round(round(b * c) + round(a * s))
// with a = x[d], b = x[d + hw], c = round(cos[t, d]), s = round(sin[t, d]).
// Features past `rotary` pass through unchanged. p = {count, inner,
// head_dim, hw = rotary / 2}.
kernel void music3_partial_rope_bf16(device const float *x [[buffer(0)]],
 device const float *cos_t [[buffer(1)]], device const float *sin_t [[buffer(2)]],
 device float *y [[buffer(3)]], device atomic_uint *inexact [[buffer(4)]],
 constant uint *p [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    uint inner = p[1], head_dim = p[2], hw = p[3];
    uint t = gid / inner, j = (gid % inner) % head_dim;
    if (j >= 2 * hw) {
        y[gid] = x[gid];
        return;
    }
    uint d = j < hw ? j : j - hw;
    uint base = gid - j;
    float a = x[base + d], b = x[base + hw + d];
    float c = music_round(cos_t[t * hw + d], 2), s = music_round(sin_t[t * hw + d], 2);
    // First half: round(a*c) - round(b*s). Second half: round(b*c) + round(a*s).
    float m0 = j < hw ? a * c : b * c;
    float m1 = j < hw ? b * s : a * s;
    float r0 = music_round(m0, 2), r1 = music_round(m1, 2);
    float sum = j < hw ? r0 - r1 : r0 + r1;
    uint ua = as_type<uint>(a), ub = as_type<uint>(b), uc = as_type<uint>(c), us = as_type<uint>(s);
    uint u0 = as_type<uint>(r0), u1 = as_type<uint>(r1);
    bool bad = music_subnormal(ua) || music_subnormal(ub) || music_subnormal(uc) || music_subnormal(us)
        || music_flushed(as_type<uint>(m0), music_zero(j < hw ? ua : ub) || music_zero(uc))
        || music_flushed(as_type<uint>(m1), music_zero(j < hw ? ub : ua) || music_zero(us))
        // r0 - r1 is exactly zero only when r0 == r1; r0 + r1 only when r0 == -r1.
        || music_flushed(as_type<uint>(sum), (music_zero(u0) && music_zero(u1))
            || (j < hw ? u0 == u1 : (u0 ^ u1) == 0x80000000u));
    if (bad) atomic_store_explicit(inexact, 1u, memory_order_relaxed);
    y[gid] = music_round(sum, 2);
}

// The DiT gated activation at BF16: rows of `[state | gate]` (each `hw`
// wide) give round(state * silu(gate)), with silu from the host table as in
// music3_swiglu_bf16. p = {count, hw}.
kernel void music3_gated_silu_bf16(device const float *x [[buffer(0)]],
 device const float *silu [[buffer(1)]], device float *y [[buffer(2)]],
 device atomic_uint *inexact [[buffer(3)]], constant uint *p [[buffer(4)]],
 uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    uint hw = p[1], row = gid / hw, f = gid % hw;
    float state = x[row * 2 * hw + f];
    uint ug = as_type<uint>(x[row * 2 * hw + hw + f]);
    float s = silu[ug >> 16], m = state * s;
    uint us = as_type<uint>(s), ux = as_type<uint>(state);
    if ((ug & 0xffffu) != 0u || music_subnormal(us) || music_subnormal(ux)
        || music_flushed(as_type<uint>(m), music_zero(us) || music_zero(ux))) {
        atomic_store_explicit(inexact, 1u, memory_order_relaxed);
    }
    y[gid] = music_round(m, 2);
}

// F32 forms of the three host steps above, for a DiT whose stages promote
// to F32 (Music 3 Turbo). Add and rotary are single correctly rounded
// operations like the host's. The SiLU uses the device `exp`, which can
// differ from the host's in the last bit, so a caller uses this one only
// where the per-operation path is not a reference it must match.

// y = a + b.
kernel void music3_add_f32(device const float *a [[buffer(0)]],
 device const float *b [[buffer(1)]], device float *y [[buffer(2)]],
 device atomic_uint *inexact [[buffer(3)]], constant uint *p [[buffer(4)]],
 uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    float x = a[gid], z = b[gid], s = x + z;
    uint ux = as_type<uint>(x), uz = as_type<uint>(z);
    bool exact_zero = ((ux | uz) & 0x7fffffffu) == 0u || (ux ^ uz) == 0x80000000u;
    if (music_subnormal(ux) || music_subnormal(uz)
        || music_flushed(as_type<uint>(s), exact_zero)) {
        atomic_store_explicit(inexact, 1u, memory_order_relaxed);
    }
    y[gid] = s;
}

// The host F32 partial NeoX rotary (`partial_rope` in dit.rs):
//   x[d] = a * c - b * s,   x[d + hw] = b * c + a * s
// with the same layout and parameters as music3_partial_rope_bf16.
kernel void music3_partial_rope_f32(device const float *x [[buffer(0)]],
 device const float *cos_t [[buffer(1)]], device const float *sin_t [[buffer(2)]],
 device float *y [[buffer(3)]], device atomic_uint *inexact [[buffer(4)]],
 constant uint *p [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    uint inner = p[1], head_dim = p[2], hw = p[3];
    uint t = gid / inner, j = (gid % inner) % head_dim;
    if (j >= 2 * hw) {
        y[gid] = x[gid];
        return;
    }
    uint d = j < hw ? j : j - hw;
    uint base = gid - j;
    float a = x[base + d], b = x[base + hw + d];
    float c = cos_t[t * hw + d], s = sin_t[t * hw + d];
    float m0 = j < hw ? a * c : b * c;
    float m1 = j < hw ? b * s : a * s;
    float sum = j < hw ? m0 - m1 : m0 + m1;
    uint ua = as_type<uint>(a), ub = as_type<uint>(b), uc = as_type<uint>(c), us = as_type<uint>(s);
    uint u0 = as_type<uint>(m0), u1 = as_type<uint>(m1);
    bool bad = music_subnormal(ua) || music_subnormal(ub) || music_subnormal(uc) || music_subnormal(us)
        || music_flushed(u0, music_zero(j < hw ? ua : ub) || music_zero(uc))
        || music_flushed(u1, music_zero(j < hw ? ub : ua) || music_zero(us))
        || music_flushed(as_type<uint>(sum), (music_zero(u0) && music_zero(u1))
            || (j < hw ? u0 == u1 : (u0 ^ u1) == 0x80000000u));
    if (bad) atomic_store_explicit(inexact, 1u, memory_order_relaxed);
    y[gid] = sum;
}

// round-free DiT gated activation: rows of `[state | gate]` give
// state * gate / (1 + exp(-gate)), the host F32 `silu`. p = {count, hw}.
kernel void music3_gated_silu_f32(device const float *x [[buffer(0)]],
 device float *y [[buffer(1)]], device atomic_uint *inexact [[buffer(2)]],
 constant uint *p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p[0]) return;
    uint hw = p[1], row = gid / hw, f = gid % hw;
    float state = x[row * 2 * hw + f], gate = x[row * 2 * hw + hw + f];
    float silu = gate / (1.0f + exp(-gate)), m = state * silu;
    uint ux = as_type<uint>(state), ug = as_type<uint>(gate), us = as_type<uint>(silu);
    if (music_subnormal(ux) || music_subnormal(ug) || music_subnormal(us)
        || music_flushed(as_type<uint>(m), music_zero(us) || music_zero(ux))) {
        atomic_store_explicit(inexact, 1u, memory_order_relaxed);
    }
    y[gid] = m;
}
