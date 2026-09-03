//! Parity tests for `gdn_conv_mix_decode`/`gdn_conv_mix_prefill`/
//! `gdn_conv_tail_update`'s `dilation` parameter (`qwen4_exp`'s PLE conv,
//! `docs/QWEN4_PHASE0.md` section 4: `kernel_size=4`, `dilation=3`) against
//! `turbospark_compute::dilated_conv_step`.
//!
//! **`gdn_parity.rs`'s existing seven-test suite is the OTHER half of this
//! change's verification, and it is deliberately left untouched.** Every
//! call site there now passes `dilation=1` explicitly, and the whole
//! `gdn_conv_mix_*` family algebraically reduces to its pre-dilation form at
//! `dilation=1` (see the shader's own doc comment) -- so that suite staying
//! green with no edits beyond the new argument is the "provably unmoved"
//! evidence for the two production families that already dispatch these
//! kernels, matching how the sigmoid `gdn_gated_norm` variant was verified.

#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::{bf16_to_f32, dilated_conv_step, f32_to_bf16};
use turbospark_gpu::{
    encode_gdn_conv_decode, encode_gdn_conv_prefill, encode_gdn_conv_tail_update, read_buffer_f16,
    GdnShape, MetalContext,
};

// `encode_gdn_conv_*` derives its dispatch `channels` from
// `shape.qkv_dim() == 2*num_k_heads*key_head_dim + num_v_heads*value_head_dim`
// (`gdn_shape.rs`), never from a value this fixture picks directly -- and
// `validate()` floors `key_head_dim` at 32. So CHANNELS has to be exactly
// this shape's `qkv_dim()` (2*32 + 4 = 68), not an arbitrary small number.
const CHANNELS: usize = 68;
const K: usize = 4;
const DILATION: usize = 3;
const HISTORY: usize = (K - 1) * DILATION; // 9

fn shape() -> GdnShape {
    let s = GdnShape {
        num_k_heads: 1,
        num_v_heads: 1,
        key_head_dim: 32,
        value_head_dim: 4,
        conv_kernel_size: K as u32,
    };
    assert_eq!(
        s.qkv_dim() as usize,
        CHANNELS,
        "CHANNELS must equal qkv_dim"
    );
    s
}

fn deterministic(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state % 2000) as f32 / 1000.0) - 1.0) * scale
        })
        .collect()
}

fn f16_row(seed: u64, n: usize, scale: f32) -> (Vec<f16>, Vec<f32>) {
    let f32s = deterministic(seed, n, scale);
    let f16s: Vec<f16> = f32s.iter().map(|&v| f16::from_f32(v)).collect();
    let rounded: Vec<f32> = f16s.iter().map(|v| v.to_f32()).collect();
    (f16s, rounded)
}

fn bf16_conv_w(seed: u64, channels: usize, k: usize) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = deterministic(seed, channels * k, 0.6)
        .iter()
        .map(|&v| f32_to_bf16(v))
        .collect();
    let values = bits.iter().map(|&b| bf16_to_f32(b)).collect();
    (bits, values)
}

fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Runs `steps` decode calls of `gdn_conv_mix_decode` at `dilation=DILATION`
/// through a persistent GPU tail buffer, comparing each step's output (and
/// the shifted tail's final contents) against `dilated_conv_step`'s own
/// state carried the same way on the CPU side.
///
/// **The fixture's rows are all DISTINCT** (seeded per step), which is what
/// makes a wrong tap stride (reading `tail[j]` instead of `tail[j*D]`, the
/// exact bug this generalization could introduce) visible: with `D=3` taps
/// 0 and 1 land on tail rows 0 and 3, not 0 and 1, so a stride-1 bug reads
/// the WRONG neighboring row rather than a plausible one.
#[test]
fn dilated_decode_matches_cpu_reference_across_several_steps() {
    let mut context = MetalContext::new().expect("Metal device");
    let (conv_w_bits, conv_w32) = bf16_conv_w(7, CHANNELS, K);

    let tail_buf = context.new_output_buffer((HISTORY * CHANNELS * 2) as u64);
    turbospark_gpu::write_buffer_bytes(&tail_buf, 0, &vec![0u8; HISTORY * CHANNELS * 2]);
    let conv_w_buf = context.new_buffer_with_data(&u16_bytes(&conv_w_bits));

    let mut cpu_tail: Vec<Vec<f32>> = vec![vec![0.0f32; CHANNELS]; HISTORY];

    for step in 0..12u64 {
        let (raw16, raw32) = f16_row(1000 + step, CHANNELS, 0.5);
        let raw_buf = context.new_buffer_with_data(&half_bytes(&raw16));
        let out_buf = context.new_output_buffer((CHANNELS * 2) as u64);

        let pass = context.begin_pass();
        encode_gdn_conv_decode(
            &mut context,
            &pass,
            shape(),
            (&tail_buf, 0),
            (&raw_buf, 0),
            (&conv_w_buf, 0),
            (&out_buf, 0),
            DILATION as u32,
        )
        .expect("encode");
        pass.commit_and_wait();

        let got: Vec<f32> = read_buffer_f16(&out_buf, 0, CHANNELS)
            .iter()
            .map(|h| h.to_f32())
            .collect();
        let expected = dilated_conv_step(&mut cpu_tail, &raw32, &conv_w32, CHANNELS, K, DILATION);

        for ch in 0..CHANNELS {
            let diff = (got[ch] - expected[ch]).abs();
            assert!(
                diff <= 2e-3_f32.max(expected[ch].abs() * 1e-2),
                "step={step} ch={ch}: got {} want {}",
                got[ch],
                expected[ch]
            );
        }
    }

    // The GPU tail's final contents must also agree with the CPU
    // reference's, not just the last output -- a kernel that computed the
    // right output but shifted the tail incorrectly would diverge on the
    // NEXT call, which this catches one step early.
    let gpu_tail: Vec<f32> = read_buffer_f16(&tail_buf, 0, HISTORY * CHANNELS)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    for j in 0..HISTORY {
        for ch in 0..CHANNELS {
            let diff = (gpu_tail[j * CHANNELS + ch] - cpu_tail[j][ch]).abs();
            assert!(
                diff <= 2e-3,
                "tail row {j} ch {ch}: got {} want {}",
                gpu_tail[j * CHANNELS + ch],
                cpu_tail[j][ch]
            );
        }
    }
}

/// **PREFILL + TAIL_UPDATE MUST MATCH SEQUENTIAL DECODE, at `dilation=3`
/// too.** Runs the SAME row sequence two ways: one row at a time through
/// `gdn_conv_mix_decode`, and all rows at once through
/// `gdn_conv_mix_prefill` + `gdn_conv_tail_update` -- both starting from an
/// identical zeroed tail. A dilation bug that only manifested in the
/// PREFILL kernel's virtual-row addressing (`t + j*D - history`, as opposed
/// to decode's tail-stride addressing) would diverge here even if the
/// decode-only case above happened to pass.
#[test]
fn dilated_prefill_matches_sequential_decode() {
    let mut context = MetalContext::new().expect("Metal device");
    let (conv_w_bits, _conv_w32) = bf16_conv_w(11, CHANNELS, K);
    let conv_w_buf = context.new_buffer_with_data(&u16_bytes(&conv_w_bits));

    let rows = 5usize;
    let mut all_rows_f16: Vec<f16> = Vec::with_capacity(rows * CHANNELS);
    let mut all_rows_f32: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for r in 0..rows as u64 {
        let (r16, r32) = f16_row(2000 + r, CHANNELS, 0.5);
        all_rows_f16.extend_from_slice(&r16);
        all_rows_f32.push(r32);
    }

    // Sequential decode.
    let decode_tail = context.new_output_buffer((HISTORY * CHANNELS * 2) as u64);
    turbospark_gpu::write_buffer_bytes(&decode_tail, 0, &vec![0u8; HISTORY * CHANNELS * 2]);
    let mut decode_last = vec![0.0f32; CHANNELS];
    for row32 in &all_rows_f32 {
        let row_f16: Vec<f16> = row32.iter().map(|&v| f16::from_f32(v)).collect();
        let raw_buf = context.new_buffer_with_data(&half_bytes(&row_f16));
        let out_buf = context.new_output_buffer((CHANNELS * 2) as u64);
        let pass = context.begin_pass();
        encode_gdn_conv_decode(
            &mut context,
            &pass,
            shape(),
            (&decode_tail, 0),
            (&raw_buf, 0),
            (&conv_w_buf, 0),
            (&out_buf, 0),
            DILATION as u32,
        )
        .expect("encode decode");
        pass.commit_and_wait();
        decode_last = read_buffer_f16(&out_buf, 0, CHANNELS)
            .iter()
            .map(|h| h.to_f32())
            .collect();
    }

    // One prefill dispatch + tail_update over the same rows.
    let prefill_tail = context.new_output_buffer((HISTORY * CHANNELS * 2) as u64);
    turbospark_gpu::write_buffer_bytes(&prefill_tail, 0, &vec![0u8; HISTORY * CHANNELS * 2]);
    let all_rows_buf = context.new_buffer_with_data(&half_bytes(&all_rows_f16));
    let prefill_out = context.new_output_buffer((rows * CHANNELS * 2) as u64);
    let pass = context.begin_pass();
    encode_gdn_conv_prefill(
        &mut context,
        &pass,
        shape(),
        (&prefill_tail, 0),
        (&all_rows_buf, 0),
        (&conv_w_buf, 0),
        (&prefill_out, 0),
        rows as u32,
        DILATION as u32,
    )
    .expect("encode prefill");
    encode_gdn_conv_tail_update(
        &mut context,
        &pass,
        shape(),
        (&prefill_tail, 0),
        (&all_rows_buf, 0),
        rows as u32,
        DILATION as u32,
    )
    .expect("encode tail_update");
    pass.commit_and_wait();

    let prefill_all: Vec<f32> = read_buffer_f16(&prefill_out, 0, rows * CHANNELS)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    let prefill_last = &prefill_all[(rows - 1) * CHANNELS..rows * CHANNELS];

    for ch in 0..CHANNELS {
        let diff = (decode_last[ch] - prefill_last[ch]).abs();
        assert!(
            diff <= 2e-3_f32.max(decode_last[ch].abs() * 1e-2),
            "ch={ch}: sequential decode {} vs prefill {}",
            decode_last[ch],
            prefill_last[ch]
        );
    }

    let decode_tail_final: Vec<f32> = read_buffer_f16(&decode_tail, 0, HISTORY * CHANNELS)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    let prefill_tail_final: Vec<f32> = read_buffer_f16(&prefill_tail, 0, HISTORY * CHANNELS)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    for i in 0..HISTORY * CHANNELS {
        let diff = (decode_tail_final[i] - prefill_tail_final[i]).abs();
        assert!(
            diff <= 2e-3,
            "tail element {i}: decode {} vs prefill {}",
            decode_tail_final[i],
            prefill_tail_final[i]
        );
    }
}
