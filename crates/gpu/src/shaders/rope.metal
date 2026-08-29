#include <metal_stdlib>
using namespace metal;

constant uint FC_ROPE_HEAD_DIM [[function_constant(50)]];
constant uint FC_ROPE_NUM_HEADS [[function_constant(51)]];
constant uint FC_ROPE_ROTATED_PAIRS [[function_constant(52)]];
constant bool FC_ROPE_USE_FC [[function_constant(53)]];

static inline uint rope_head_dim(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_HEAD_DIM))
        ? FC_ROPE_HEAD_DIM
        : runtime_value;
}
static inline uint rope_num_heads(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_NUM_HEADS))
        ? FC_ROPE_NUM_HEADS
        : runtime_value;
}

static inline uint rope_rotated_pairs(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_ROTATED_PAIRS))
        ? FC_ROPE_ROTATED_PAIRS
        : runtime_value;
}

static inline void apply_neox_pair(
    device half* head,
    uint pair,
    uint half_dim,
    uint frequency_divisor,
    float position,
    float theta
) {
    const float exponent = -float(2u * pair) / float(frequency_divisor);
    const float angle = position * pow(theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const uint lower = pair;
    const uint upper = half_dim + pair;
    const float x0 = float(head[lower]);
    const float x1 = float(head[upper]);
    head[lower] = half(x0 * cosine - x1 * sine);
    head[upper] = half(x0 * sine + x1 * cosine);
}

kernel void rope_default_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_dimension = dimension / 2u;
    if (pair >= half_dimension || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}

// Qwen-style partial RoPE: rotation confined to the first `rotary_dim`
// elements of each head, NeoX pairing (i, rotary_dim/2 + i) inside that
// window, frequency divisor = rotary_dim. Elements >= rotary_dim pass
// through untouched. (Gemma's proportional variant instead pairs across the
// full head and divides frequencies by head_dim — a different element set.)
kernel void rope_neox_subdim(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_rotary = rotary_dim / 2u;
    if (pair >= half_rotary || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_rotary, rotary_dim,
                    float(position), theta);
}

// Qwen 3.8's INTERLEAVED mRoPE (ROADMAP M-V5, `docs/VISION_PHASE0.md` item
// 2). `rope_neox_subdim` with the scalar position replaced by a per-pair
// choice among (t, h, w).
//
// IT CALLS `apply_neox_pair`, WHICH IS THE POINT. A text token gets
// t == h == w from the reference's own `get_rope_index`, so at that input
// every pair selects the same number and this kernel executes the identical
// float sequence `rope_neox_subdim` executes -- BIT-identical, not merely
// close. The trunk therefore stays byte-exact for every text token of a
// mixed prompt and only an image's own tokens take a different angle. Any
// rewrite that precomputes cos/sin on the host, or reassociates the angle,
// gives that up and turns a structural invariant into an FP coincidence.
//
// The selector is `_interleaved_position_selector`
// (`mlx_vlm/models/rope_utils.py:350-355`): default t, then h claims residue
// 1 and w residue 2, each stopping at `min(section * 3, half_rotary)`.
// THE CLAMPS ARE IMPLEMENTED RATHER THAN THE `i % 3` COLLAPSE: that collapse
// holds only because this family's [11, 11, 10] tiles freq_dim 32 exactly.
//
// `section_t` is not a parameter. t is what a pair gets when neither of the
// other two claims it, so passing its width would be a second copy of a
// number this already derives.
kernel void rope_mrope_interleaved(
    device half* data [[buffer(0)]],
    constant uint& position_t [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    constant uint& position_h [[buffer(6)]],
    constant uint& position_w [[buffer(7)]],
    constant uint& section_h [[buffer(8)]],
    constant uint& section_w [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_rotary = rotary_dim / 2u;
    if (pair >= half_rotary || head_index >= heads) return;

    uint selected = position_t;
    if (pair % 3u == 1u && pair < min(section_h * 3u, half_rotary)) {
        selected = position_h;
    } else if (pair % 3u == 2u && pair < min(section_w * 3u, half_rotary)) {
        selected = position_w;
    }

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_rotary, rotary_dim,
                    float(selected), theta);
}

kernel void rope_proportional_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotated_pairs [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint active_pairs = rope_rotated_pairs(rotated_pairs);
    if (pair >= active_pairs || head_index >= heads) return;

    const uint half_dimension = dimension / 2u;
    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}

// Port-local (ROADMAP M5): NeoX rope from a PRECOMPUTED per-pair frequency
// table, with a magnitude scale on cos and sin.
//
// The three kernels above derive each pair's frequency from a scalar theta
// (`pow(theta, -2i/D)`), which is every rope this port had until `gpt-oss`.
// YaRN is not expressible that way: it interpolates per dimension between
// extrapolating and interpolating the trained frequency along a ramp, so the
// per-pair frequencies are no longer a function of one number.
//
// The table is position-INDEPENDENT (see `turbospark_compute::yarn_spec`), so
// it is built once at open and bound, rather than recomputed per token.
//
// `mscale` is not decoration and does not cancel: YaRN scales q and k, and
// therefore `q.k` by its square. See the reference's doc comment for why
// llama.cpp's own source makes it look like a no-op.
//
// This also happens to be the shape a LEARNED frequency table needs, which
// is what `rope_freqs.weight` is -- refused by name since M4. Nothing here
// wires that up; it is only worth noting that the kernel would not be the
// obstacle.
kernel void rope_neox_freqs(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    device const float* frequencies [[buffer(4)]],
    constant uint& rotated_pairs [[buffer(5)]],
    constant float& mscale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint active_pairs = rope_rotated_pairs(rotated_pairs);
    if (pair >= active_pairs || head_index >= heads) return;

    const uint half_dimension = dimension / 2u;
    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;

    const float angle = float(position) * frequencies[pair];
    const float cosine = cos(angle) * mscale;
    const float sine = sin(angle) * mscale;
    const uint lower = pair;
    const uint upper = half_dimension + pair;
    const float x0 = float(head[lower]);
    const float x1 = float(head[upper]);
    head[lower] = half(x0 * cosine - x1 * sine);
    head[upper] = half(x0 * sine + x1 * cosine);
}
