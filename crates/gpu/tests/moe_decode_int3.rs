#![cfg(target_os = "macos")]
//! Parity for the MLX affine INT3 routed pair (`moe_phase1_gate_up_act_int3`
//! + `moe_phase2_down_reduce_int3`) against a CPU dequant reference, read
//! through a real `RoutedBlobs` argument buffer the way the runtime binds it.
//!
//! The packing is MLX's (`mx.quantize(..., bits=3)`): a little-endian
//! bitstream with element `j` at bits `[3j, 3j+3)`. `mlx_pack_matches_mlx`
//! pins the packer to bytes MLX 0.32.4 actually wrote, so the reference is not
//! just agreeing with the kernel's own reading of the format.

use half::{bf16, f16};
use turbospark_gpu::{MetalContext, MoeExpertOffsets, RoutedBlobsBuffer};

const GROUP: usize = 64;

struct Rows {
    codes: Vec<u8>,
    scales: Vec<bf16>,
    biases: Vec<bf16>,
    rows: usize,
    cols: usize,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 20_001) as f32 / 10_000.0 - 1.0
    }
}

/// Random codes with per-group scales and biases in the range the real
/// checkpoint's experts carry (scales ~1e-2, biases of either sign).
fn random_rows(rows: usize, cols: usize, seed: u64) -> Rows {
    assert_eq!(cols % GROUP, 0);
    let mut rng = Rng(seed | 1);
    let codes = (0..rows * cols).map(|_| (rng.next() % 8) as u8).collect();
    let groups = rows * cols / GROUP;
    let scales = (0..groups)
        .map(|_| bf16::from_f32(0.004 + 0.02 * rng.unit().abs()))
        .collect();
    let biases = (0..groups)
        .map(|_| bf16::from_f32(0.06 * rng.unit()))
        .collect();
    Rows {
        codes,
        scales,
        biases,
        rows,
        cols,
    }
}

/// MLX's 3-bit row packing: 8 codes per 3 bytes, element `j` at bit `3j`.
fn mlx_pack3(codes: &[u8]) -> Vec<u8> {
    assert_eq!(codes.len() % 8, 0);
    let mut out = Vec::with_capacity(codes.len() * 3 / 8);
    for pack in codes.chunks(8) {
        let mut v = 0u32;
        for (j, &c) in pack.iter().enumerate() {
            assert!(c < 8);
            v |= (c as u32) << (3 * j);
        }
        out.extend_from_slice(&v.to_le_bytes()[..3]);
    }
    out
}

impl Rows {
    fn dequant(&self, r: usize, c: usize) -> f32 {
        let g = (r * self.cols + c) / GROUP;
        self.codes[r * self.cols + c] as f32 * self.scales[g].to_f32() + self.biases[g].to_f32()
    }

    /// `W x` in f64, the reference the kernel's f32 accumulation is held to.
    fn gemv(&self, x: &[f32]) -> Vec<f64> {
        (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .map(|c| self.dequant(r, c) as f64 * x[c] as f64)
                    .sum()
            })
            .collect()
    }

    /// `sum |w * x|`, the scale a summation-order difference is relative to.
    fn abs_gemv(&self, x: &[f32]) -> Vec<f64> {
        (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .map(|c| (self.dequant(r, c) as f64 * x[c] as f64).abs())
                    .sum()
            })
            .collect()
    }
}

fn le16(v: impl IntoIterator<Item = u16>) -> Vec<u8> {
    v.into_iter().flat_map(|b| b.to_le_bytes()).collect()
}

/// One expert blob, `gate, gate_s, gate_b, up, ..., down, ...` back to back
/// as the repack writer lays it out. `pad` shifts every weight run off a
/// word boundary, which a real 3-bit blob can also be (the kernels read
/// weights a byte at a time). It must be even: the BF16 companions keep the
/// 2-byte alignment every routed kernel requires, and a real 3-bit run is a
/// multiple of 24 bytes per row, so it never breaks that.
fn build_blob(gate: &Rows, up: &Rows, down: &Rows, pad: usize) -> (Vec<u8>, MoeExpertOffsets) {
    assert_eq!(pad % 2, 0, "companions must stay 2-byte aligned");
    let mut blob = vec![0xA5u8; pad];
    let mut off = [0u32; 9];
    for (i, rows) in [gate, up, down].into_iter().enumerate() {
        off[3 * i] = blob.len() as u32;
        blob.extend(mlx_pack3(&rows.codes));
        off[3 * i + 1] = blob.len() as u32;
        blob.extend(le16(rows.scales.iter().map(|s| s.to_bits())));
        off[3 * i + 2] = blob.len() as u32;
        blob.extend(le16(rows.biases.iter().map(|b| b.to_bits())));
        blob.extend(std::iter::repeat_n(0x5Au8, pad));
    }
    (
        blob,
        MoeExpertOffsets {
            gate_w: off[0],
            gate_s: off[1],
            gate_b: off[2],
            up_w: off[3],
            up_s: off[4],
            up_b: off[5],
            down_w: off[6],
            down_s: off[7],
            down_b: off[8],
        },
    )
}

fn read_f16(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const u16;
    // SAFETY: the buffer was allocated with at least `n` halfs and the pass
    // that wrote it has completed.
    let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Runs phase 1 and phase 2 over `top_k` random experts at `(d, f)` and
/// checks both outputs against the CPU reference.
fn check_pair(d: usize, f: usize, top_k: usize, seed: u64, pad: usize) {
    let mut context = MetalContext::new().expect("Metal device");
    let experts: Vec<(Rows, Rows, Rows)> = (0..top_k as u64)
        .map(|e| {
            let s = seed + 1000 * e;
            (
                random_rows(f, d, s + 1),
                random_rows(f, d, s + 2),
                random_rows(d, f, s + 3),
            )
        })
        .collect();

    let mut rng = Rng(seed ^ 0xDEAD_BEEF);
    let x16: Vec<f16> = (0..d).map(|_| f16::from_f32(rng.unit())).collect();
    let x: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let residual16: Vec<f16> = (0..d).map(|_| f16::from_f32(0.5 * rng.unit())).collect();
    let weights16: Vec<f16> = (0..top_k)
        .map(|k| f16::from_f32(1.0 / (k as f32 + 1.5)))
        .collect();

    // Phase 1 reference, and the f16 acts phase 2 reads (taken from the GPU
    // below, so phase 2 is checked on its own and not on phase 1's rounding).
    let mut want_acts = Vec::new();
    let mut act_tol = Vec::new();
    for (gate, up, _) in &experts {
        let (g, u) = (gate.gemv(&x), up.gemv(&x));
        let (ga, ua) = (gate.abs_gemv(&x), up.abs_gemv(&x));
        for i in 0..f {
            let a = silu(g[i]) * u[i];
            want_acts.push(a);
            // f32 summation-order error on g and u (relative to the absolute
            // sums), propagated through silu(g) * u, plus the f16 store.
            let dg = 1e-5 * ga[i];
            let du = 1e-5 * ua[i];
            act_tol.push(dg * u[i].abs() * 1.2 + du * silu(g[i]).abs() + a.abs() * 1e-3 + 1e-6);
        }
    }

    let blobs: Vec<(Vec<u8>, MoeExpertOffsets)> = experts
        .iter()
        .map(|(g, u, dn)| build_blob(g, u, dn, pad))
        .collect();
    let offsets = blobs[0].1;
    let blob_bufs: Vec<metal::Buffer> = blobs
        .iter()
        .map(|(b, _)| context.new_buffer_with_data(b))
        .collect();
    let x_buf = context.new_buffer_with_data(&le16(x16.iter().map(|v| v.to_bits())));
    let res_buf = context.new_buffer_with_data(&le16(residual16.iter().map(|v| v.to_bits())));
    let mut routing = vec![f16::ZERO; turbospark_gpu::MAX_STREAMED_EXPERTS];
    routing[..top_k].copy_from_slice(&weights16);
    let rw_buf = context.new_buffer_with_data(&le16(routing.iter().map(|v| v.to_bits())));
    let acts_buf = context.new_output_buffer((top_k * f * 2) as u64);
    let y_buf = context.new_output_buffer((d * 2) as u64);

    // Bound through the INT4 kernel's reflection, as the runtime does
    // (`RoutedBlobLayout::source_function`): same library, same struct.
    let routed = RoutedBlobsBuffer::new(&mut context, true).expect("arg buffer");
    let refs: Vec<(&metal::Buffer, u64)> = blob_bufs.iter().map(|b| (b, 0u64)).collect();
    routed.bind(&mut context, true, &refs).expect("bind");

    let pass = context.begin_pass();
    for b in &blob_bufs {
        pass.use_read_buffer(b);
    }
    turbospark_gpu::encode_moe_phase1_int3(
        &mut context,
        &pass,
        &routed,
        &offsets,
        (&x_buf, 0),
        (&acts_buf, 0),
        d as u32,
        f as u32,
        top_k as u32,
        true,
    )
    .expect("phase1");
    turbospark_gpu::encode_moe_phase2_int3(
        &mut context,
        &pass,
        &routed,
        &offsets,
        (&acts_buf, 0),
        (&rw_buf, 0),
        (&res_buf, 0),
        (&y_buf, 0),
        d as u32,
        f as u32,
        top_k as u32,
        true,
    )
    .expect("phase2");
    pass.commit_and_wait();

    let acts = read_f16(&acts_buf, top_k * f);
    let mut worst = 0.0f64;
    for i in 0..top_k * f {
        let err = (acts[i] as f64 - want_acts[i]).abs();
        worst = worst.max(err / act_tol[i]);
        assert!(
            err <= act_tol[i],
            "phase 1 slot {} row {}: got {} want {} (tol {})",
            i / f,
            i % f,
            acts[i],
            want_acts[i],
            act_tol[i]
        );
    }

    let y = read_f16(&y_buf, d);
    let downs: Vec<(f64, Vec<f64>, Vec<f64>)> = experts
        .iter()
        .enumerate()
        .map(|(k, (_, _, down))| {
            let a = &acts[k * f..(k + 1) * f];
            (weights16[k].to_f32() as f64, down.gemv(a), down.abs_gemv(a))
        })
        .collect();
    for row in 0..d {
        let mut want = residual16[row].to_f32() as f64;
        let mut scale = want.abs();
        for (w, value, abs) in &downs {
            want += w * value[row];
            scale += w * abs[row];
        }
        let tol = 1e-5 * scale + want.abs() * 1e-3 + 1e-6;
        let err = (y[row] as f64 - want).abs();
        worst = worst.max(err / tol);
        assert!(
            err <= tol,
            "phase 2 row {row}: got {} want {want} (tol {tol})",
            y[row]
        );
    }
    eprintln!("d={d} f={f} top_k={top_k}: worst error / tolerance {worst:.3}");
}

#[test]
fn mlx_pack_matches_mlx() {
    // `mx.quantize` of a row whose codes run 7, 6, ..., 0 twice (MLX 0.32.4,
    // group 64, bits 3) wrote these first six bytes.
    let codes: Vec<u8> = (0..16).map(|i| 7 - (i % 8) as u8).collect();
    assert_eq!(mlx_pack3(&codes), [119, 57, 5, 119, 57, 5]);
}

/// Flint's routed shape: hidden 2048 (32 groups, the 4-group block path
/// only), expert width 512 (8 groups), top-6.
#[test]
fn int3_pair_matches_cpu_at_flints_shape() {
    check_pair(2048, 512, 6, 0x3B17, 6);
}

/// Group counts that are not a multiple of 4 run the per-group tail loop:
/// d = 320 is one block plus one tail group, f = 192 is tail only. The pad
/// keeps every weight run off a 4-byte boundary.
#[test]
fn int3_pair_matches_cpu_on_the_tail_path() {
    check_pair(320, 192, 3, 0x7A11, 2);
}

/// The runtime binds INT3 layers through the INT4 phase-1 kernel's
/// argument-buffer reflection, which is sound only while both declare the
/// same `RoutedBlobs` at index 0.
#[test]
fn int3_and_int4_agree_on_the_routed_blobs_argument_buffer() {
    let mut context = MetalContext::new().expect("Metal device");
    let int4 = RoutedBlobsBuffer::new(&mut context, true).expect("int4");
    let int3 = RoutedBlobsBuffer::new_for(
        &mut context,
        turbospark_gpu::moe_decode_source(),
        "moe_phase1_gate_up_act_int3",
        true,
    )
    .expect("int3");
    assert_eq!(int4.buffer().length(), int3.buffer().length());
}
