#![cfg(target_os = "macos")]
//! Proves the resident-weights Metal wrap is genuinely zero-copy: the
//! `MTLBuffer` created by `ResidentGpuWeights::wrap` aliases the very same
//! virtual pages the mmap of `model_weights.bin` faulted in, rather than
//! holding a copy, and the `.gturbo` writer's 16 KiB index alignment makes
//! the resident region start page-aligned (zero slice shift).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use mrefrust_gpu::{MetalContext, ResidentGpuWeights};
use mrefrust_repack::build_synthetic_gemma4_install;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "mrefrust-resident-metal-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn wrap_is_zero_copy_and_page_aligned() {
    let dir = temp_dir();
    build_synthetic_gemma4_install(&dir, 256, 1, "resident-metal-test")
        .expect("synthetic install should write");

    let weights_path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&weights_path).expect("index should load");
    let resident = model_io::ResidentBuffer::map(
        &weights_path,
        index.header.index_size,
        index.header.resident_size,
    )
    .expect("resident region should map");

    // The writer 16 KiB-aligns the index region, so on Apple Silicon
    // (16 KiB pages) the resident region starts exactly on a page
    // boundary and the mapping needs no shift.
    assert_eq!(index.header.index_size % 16_384, 0);
    assert_eq!(resident.slice_shift(), 0);

    let mapped_base = resident.mapped_bytes().as_ptr();
    let mapped_len = resident.mapped_bytes().len();
    let data_snapshot = resident.data().to_vec();

    let context = MetalContext::new().expect("Metal device should exist");
    let weights =
        ResidentGpuWeights::wrap(context.device(), resident).expect("wrap should succeed");

    // No copy: the Metal buffer's contents pointer IS the mapping base.
    assert_eq!(weights.buffer().contents() as *const u8, mapped_base);
    assert_eq!(weights.buffer().length(), mapped_len as u64);

    // Same bytes through the buffer as through the mmap slice, at the
    // gpu_offset the tensors would be bound with.
    let via_buffer = unsafe {
        std::slice::from_raw_parts(
            (weights.buffer().contents() as *const u8).add(weights.gpu_offset(0) as usize),
            data_snapshot.len(),
        )
    };
    assert_eq!(via_buffer, data_snapshot.as_slice());
}

/// The offset-bound resident GEMV must produce bit-identical output to
/// the staged-copy GEMV over the same tensor: same kernel, same math,
/// only the weight binding differs.
#[test]
fn resident_bound_gemv_matches_staged_gemv() {
    let dir = temp_dir();
    build_synthetic_gemma4_install(&dir, 256, 1, "resident-gemv-test")
        .expect("synthetic install should write");

    let weights_path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&weights_path).expect("index should load");
    let resident = model_io::ResidentBuffer::map(
        &weights_path,
        index.header.index_size,
        index.header.resident_size,
    )
    .expect("resident region should map");

    let entry = index.entries.get("layer0.q_proj").expect("tensor exists");
    let rows = entry.shape.0 as usize;
    let cols = entry.shape.1 as usize;
    let index_size = index.header.index_size;

    let mut context = MetalContext::new().expect("Metal device should exist");

    // Staged path: slice rows out of the mapped data the way the
    // pre-zero-copy runner did.
    let data = resident.data().to_vec();
    let packed_local = (entry.file_offset - index_size) as usize;
    let scale_local = (entry.scale_offset - index_size) as usize;
    let bias_local = (entry.bias_offset - index_size) as usize;
    let packed = &data[packed_local..packed_local + entry.size_bytes as usize];
    let to_u16 = |b: &[u8]| -> Vec<u16> {
        b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let scales = to_u16(&data[scale_local..scale_local + entry.scale_size as usize]);
    let biases = to_u16(&data[bias_local..bias_local + entry.bias_size as usize]);
    let row_bytes = cols / 2;
    let groups = cols / 64;
    let staged_rows: Vec<mrefrust_gpu::Int4AffineRowGpu> = (0..rows)
        .map(|r| mrefrust_gpu::Int4AffineRowGpu {
            packed: &packed[r * row_bytes..(r + 1) * row_bytes],
            scales: &scales[r * groups..(r + 1) * groups],
            biases: &biases[r * groups..(r + 1) * groups],
        })
        .collect();

    let x: Vec<half::f16> = (0..cols)
        .map(|i| half::f16::from_f32((i as f32 * 0.37).sin()))
        .collect();

    let staged =
        mrefrust_gpu::dequant_int4_gemv(&mut context, &staged_rows, &x, cols).expect("staged gemv");

    let weights =
        ResidentGpuWeights::wrap(context.device(), resident).expect("wrap should succeed");
    let matrix = mrefrust_gpu::Int4ResidentMatrix {
        buffer: weights.buffer(),
        weights_offset: weights.gpu_offset(entry.file_offset - index_size),
        scales_offset: weights.gpu_offset(entry.scale_offset - index_size),
        biases_offset: weights.gpu_offset(entry.bias_offset - index_size),
        rows,
        cols,
    };
    let bound =
        mrefrust_gpu::dequant_int4_gemv_resident(&mut context, &matrix, &x).expect("resident gemv");

    assert_eq!(staged.len(), bound.len());
    for (a, b) in staged.iter().zip(bound.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "outputs must be bit-identical");
    }
}
