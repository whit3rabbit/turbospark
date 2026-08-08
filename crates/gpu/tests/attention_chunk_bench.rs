//! Isolated measurement of split-KV (`num_chunks > 1`) on
//! `attention_decode_partial`, at the real Gemma 4 26B attention shapes.
//!
//! Why this exists as its own bench rather than an end-to-end A/B: the
//! production dispatch hardcodes `num_chunks = 1`, which launches exactly
//! `num_q_heads` threadgroups. Gemma 4 26B has 16 Q heads, so one decode
//! attention dispatch occupies 16 threadgroups of 256 threads on a GPU
//! with tens of cores, and each threadgroup then walks its whole KV range
//! serially with two threadgroup-wide reductions per position. Whether
//! that is bandwidth-bound (chunking cannot help) or occupancy-bound
//! (chunking is a large win) is a property of the kernel alone, so
//! measuring it in isolation removes every confound an end-to-end A/B
//! carries: expert cache hit rate, host encode time, and the phase
//! counters' averaging of prefill into decode.
//!
//! Ratios here are clock-invariant, so they stay meaningful on a
//! throttled (for example battery-powered) machine where absolute
//! ms/token numbers do not.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test attention_chunk_bench --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use half::f16;
use metal::{FunctionConstantValues, MTLDataType};
use turbospark_gpu::MetalContext;

const SOURCE: &str = include_str!("../src/shaders/attention.metal");
const THREADS_PER_GROUP: u64 = 256;

/// The two attention shapes in Gemma 4 26B-A4B: 25 sliding-window layers
/// and 5 full-attention ones. `(label, head_dim, num_q_heads, num_kv_heads)`.
const SHAPES: [(&str, u32, u32, u32); 2] = [("swa ", 256, 16, 8), ("full", 512, 16, 2)];

fn u32_bytes(v: &u32) -> &[u8] {
    unsafe { std::slice::from_raw_parts((v as *const u32).cast(), 4) }
}

fn f32_bytes(v: &f32) -> &[u8] {
    unsafe { std::slice::from_raw_parts((v as *const f32).cast(), 4) }
}

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// Mirrors the private `attention_function_constants`, but with
/// `FC_ATTN_NUM_CHUNKS` (65) set to this dispatch's real chunk count
/// instead of a hardcoded 1. The shader checks 64 and 65
/// unconditionally, so a wrong value here is used in place of the buffer
/// argument rather than ignored.
fn constants(scale: f32, num_chunks: u32) -> FunctionConstantValues {
    let values = FunctionConstantValues::new();
    let zero: u32 = 0;
    let use_fc = false;
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 60);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 61);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 62);
    values.set_constant_value_at_index((&use_fc as *const bool).cast(), MTLDataType::Bool, 63);
    values.set_constant_value_at_index((&scale as *const f32).cast(), MTLDataType::Float, 64);
    values.set_constant_value_at_index((&num_chunks as *const u32).cast(), MTLDataType::UInt, 65);
    values.set_constant_value_at_index((&zero as *const u32).cast(), MTLDataType::UInt, 69);
    values
}

struct Case {
    q: metal::Buffer,
    k: metal::Buffer,
    v: metal::Buffer,
    out: metal::Buffer,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    scale: f32,
}

impl Case {
    fn new(
        context: &MetalContext,
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        seq_len: u32,
    ) -> Self {
        let q: Vec<f16> = (0..(num_q_heads * head_dim) as usize)
            .map(|i| f16::from_f32(((i as f32) * 0.019).sin() * 0.5))
            .collect();
        let kv_len = (seq_len * num_kv_heads * head_dim) as usize;
        let k: Vec<f16> = (0..kv_len)
            .map(|i| f16::from_f32(((i as f32) * 0.0007).cos() * 0.3))
            .collect();
        // Offset away from zero on purpose: a zero-mean V makes the
        // chunks-agree assertion vacuous, since a dispatch that writes
        // nothing also averages to zero (see attention_decode_parity.rs).
        let v: Vec<f16> = (0..kv_len)
            .map(|i| f16::from_f32(0.7 + ((i as f32) * 0.0011).sin() * 0.3))
            .collect();
        Self {
            q: context.new_buffer_with_data(&to_le(&q)),
            k: context.new_buffer_with_data(&to_le(&k)),
            v: context.new_buffer_with_data(&to_le(&v)),
            out: context.new_output_buffer((num_q_heads * head_dim) as u64 * 2),
            head_dim,
            num_q_heads,
            num_kv_heads,
            seq_len,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

/// Encodes `repeats` back-to-back partial+combine pairs at `num_chunks`
/// and returns the whole buffer's GPU busy seconds. Repeating inside one
/// command buffer amortizes submission overhead, which is what a decode
/// token does anyway (30 layers of attention per buffer).
fn time_chunks(context: &mut MetalContext, case: &Case, num_chunks: u32, repeats: u32) -> f64 {
    let range = case.seq_len;
    let chunk_len = range.div_ceil(num_chunks);
    let key = {
        let mut k = [0u8; 8];
        k[..4].copy_from_slice(&case.scale.to_le_bytes());
        k[4..].copy_from_slice(&num_chunks.to_le_bytes());
        k
    };
    let partial = context
        .pipeline(
            SOURCE,
            "attention_decode_partial",
            &constants(case.scale, num_chunks),
            &key,
        )
        .expect("partial pipeline");
    let combine = context
        .pipeline(
            SOURCE,
            "attention_decode_combine",
            &constants(case.scale, num_chunks),
            &key,
        )
        .expect("combine pipeline");

    let slots = (case.num_q_heads * num_chunks) as u64;
    let m = context.new_output_buffer(slots * 4);
    let d = context.new_output_buffer(slots * 4);
    let o = context.new_output_buffer(slots * case.head_dim as u64 * 4);
    let kv_start: u32 = 0;

    let pass = context.begin_pass();
    for _ in 0..repeats {
        pass.encode_threadgroups(
            &partial,
            &[
                (&case.q, 0, 0),
                (&case.k, 1, 0),
                (&case.v, 2, 0),
                (&m, 3, 0),
                (&d, 4, 0),
                (&o, 5, 0),
            ],
            &[
                (u32_bytes(&case.head_dim), 6),
                (u32_bytes(&case.num_q_heads), 7),
                (u32_bytes(&case.num_kv_heads), 8),
                (u32_bytes(&case.seq_len), 9),
                (u32_bytes(&kv_start), 10),
                (u32_bytes(&chunk_len), 11),
                (u32_bytes(&num_chunks), 12),
                (f32_bytes(&case.scale), 13),
            ],
            slots,
            THREADS_PER_GROUP,
        );
        pass.encode_threadgroups(
            &combine,
            &[(&m, 0, 0), (&d, 1, 0), (&o, 2, 0), (&case.out, 3, 0)],
            &[(u32_bytes(&case.head_dim), 4), (u32_bytes(&num_chunks), 5)],
            case.num_q_heads as u64,
            THREADS_PER_GROUP,
        );
    }
    pass.commit_and_wait_with_gpu_time()
}

fn read_out(case: &Case) -> Vec<f32> {
    let n = (case.num_q_heads * case.head_dim) as usize;
    let ptr = case.out.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// The measurement. Prints a table; asserts only that every chunk count
/// computes the same attention, so a timing shift can never be a silently
/// wrong kernel.
#[test]
#[ignore = "performance measurement on a real Metal device; run explicitly"]
fn split_kv_speedup_at_gemma4_shapes() {
    let mut context = MetalContext::new().expect("Metal device");
    let chunk_counts = [1u32, 2, 4, 8, 16, 32];
    let repeats = 30; // one decode token's worth of layers.

    println!("split-KV, {repeats} dispatch pairs per measurement, us per pair");
    println!("shape  seq   chunks=1   2      4      8      16     32     best speedup");
    for (label, head_dim, num_q_heads, num_kv_heads) in SHAPES {
        for seq_len in [256u32, 1024, 4096] {
            let case = Case::new(&context, head_dim, num_q_heads, num_kv_heads, seq_len);
            // Warm every pipeline and the GPU's clocks before timing: a
            // cold first dispatch reads high for reasons unrelated to the
            // kernel (AGENTS.md Gotcha 20).
            let mut reference: Option<Vec<f32>> = None;
            for &c in &chunk_counts {
                time_chunks(&mut context, &case, c, 2);
                let got = read_out(&case);
                match &reference {
                    None => reference = Some(got),
                    Some(want) => {
                        let err = got
                            .iter()
                            .zip(want)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        assert!(err < 2e-3, "{label} seq={seq_len} chunks={c}: err={err}");
                    }
                }
            }

            let mut us = Vec::new();
            for &c in &chunk_counts {
                // Best of three: the minimum is the least contaminated by
                // whatever else is using the GPU.
                let best = (0..3)
                    .map(|_| time_chunks(&mut context, &case, c, repeats))
                    .fold(f64::INFINITY, f64::min);
                us.push(best * 1e6 / repeats as f64);
            }
            let baseline = us[0];
            let best = us.iter().cloned().fold(f64::INFINITY, f64::min);
            print!("{label} {seq_len:>5}");
            for t in &us {
                print!(" {t:>6.1}");
            }
            println!("   {:.2}x", baseline / best);
        }
    }
}
