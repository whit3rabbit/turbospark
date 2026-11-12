#![cfg(target_os = "macos")]
//! **Does a mapped expert table stay GPU-resident when a decode cycles
//! through more of it?**
//!
//! Mapped residency (`docs/EXPERT_RESIDENCY.md`) wraps each layer's
//! `packed_experts/layer_NN.bin` mapping in one no-copy Metal buffer, and
//! every routed pass binds its layer's buffer. Gemma 4 (30 layers, 12.3 GB)
//! decodes faster that way than streamed. The `QwenGdnMoe` trunks (Fish Audio
//! Drama 3, MiniCPM-V 4.7: 40 layers, 16-17 GB) decode 70 to 180 times
//! slower, with the Metal queue thread blocked inside command-buffer
//! submission.
//!
//! The suspicion is cyclic eviction: if the kernel keeps only so many bytes of
//! no-copy mappings GPU-resident, a decode that walks every layer in order
//! re-admits each layer just after it was dropped, and every submission pays
//! for a whole layer. This probe walks the first `n` layers in order, one
//! small INT4 GEMV per layer per submission, for growing `n`, and reports the
//! median submission time. Flat times mean no budget is hit. A step at some
//! `n` names the budget in layers, and so in bytes.
//!
//! `TURBOSPARK_PROBE_MAP_EXPERTS=k` maps only each layer's first `k` experts
//! (the probe reads experts 0 to 3), so the same layer COUNT can be walked
//! with fewer BYTES: that separates a limit on mapped bytes from a limit on
//! mapped buffers.
//!
//! `TURBOSPARK_PROBE_WITH_RESIDENT=1` also maps the install's
//! `model_weights.bin` the way the runtime does (one no-copy buffer) and
//! reads from it in every submission, as a decode reads its resident
//! weights, to see whether those bytes count against the same budget. That
//! read is a small GEMV over the region's first bytes; its output is
//! discarded (the bytes are not a matrix).
//!
//! `TURBOSPARK_PROBE_PRESSURE_GIB=g` holds `g` GiB of dirty anonymous memory
//! for the whole walk, the controlled stand-in for other processes (parallel
//! compiles, another model) squeezing the page cache.
//!
//! Read-only on the mapping: the GEMV reads the matrix and writes `y`.
//! `#[ignore]`d and env-gated like the other real-install probes here.

use std::path::PathBuf;
use std::time::Instant;

use gpu::{encode_dequant_int4_gemv_resident, wrap_page_aligned_no_copy, Int4ResidentMatrix};

const INSTALL_VAR: &str = "TURBOSPARK_PROBE_INSTALL_DIR";
const LAYOUT_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// Full walks timed per `n`, after one untimed walk.
const ROUNDS: usize = 3;
const MAP_EXPERTS_VAR: &str = "TURBOSPARK_PROBE_MAP_EXPERTS";

#[test]
#[ignore = "needs a real MoE install; set TURBOSPARK_PROBE_INSTALL_DIR"]
fn cycling_more_mapped_layers_is_timed_per_submission() {
    let Ok(dir) = std::env::var(INSTALL_VAR) else {
        eprintln!("{INSTALL_VAR} unset; skipping");
        return;
    };
    let dir = PathBuf::from(dir);
    let experts_dir = dir.join("packed_experts");
    let layout = model_io::load_packed_experts_layout(&dir, LAYOUT_MAX_BYTES)
        .expect("packed_experts/layout.json should load");
    let arch = repack::peek_manifest_arch(&dir).expect("manifest should peek");
    let (rows, cols) = (
        arch.moe_intermediate_size as usize,
        arch.hidden_size as usize,
    );

    let map_experts: Option<usize> = std::env::var(MAP_EXPERTS_VAR)
        .ok()
        .map(|v| v.parse().expect("an expert count"));
    if let Some(k) = map_experts {
        assert!(
            (4..=layout.experts_per_layer).contains(&k),
            "map at least the 4 experts read"
        );
    }
    // The mappings outlive the buffers that alias them.
    let mut mappings = Vec::with_capacity(layout.num_layers);
    let mut layer_bytes = Vec::with_capacity(layout.num_layers);
    for (index, layer) in layout.layers.iter().enumerate() {
        let path = experts_dir.join(&layer.file);
        let file_len = std::fs::metadata(&path).expect("layer file stats").len();
        // Up to the end of expert k - 1, rounded up to a 16 KiB page.
        let len = match map_experts {
            Some(k) => {
                let last = layout.expert(index, k - 1);
                (last.offset + layer.expert_stride).div_ceil(16384) * 16384
            }
            None => file_len,
        }
        .min(file_len);
        mappings.push(model_io::ResidentBuffer::map(&path, 0, len).expect("layer file maps"));
        layer_bytes.push(len);
    }
    // Pseudo-random bytes in every word: the memory compressor absorbs a
    // mostly-zero allocation, which would hold almost nothing.
    let pressure_gib: usize = std::env::var("TURBOSPARK_PROBE_PRESSURE_GIB")
        .ok()
        .map_or(0, |v| v.parse().expect("a GiB count"));
    let mut pressure = vec![0u64; (pressure_gib << 30) / 8];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for word in pressure.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *word = state | 1;
    }
    let mut context = gpu::MetalContext::new().expect("Metal device");
    let resident_map = std::env::var("TURBOSPARK_PROBE_WITH_RESIDENT")
        .is_ok_and(|v| v == "1")
        .then(|| {
            let path = dir.join("model_weights.bin");
            let len = std::fs::metadata(&path)
                .expect("model_weights.bin stats")
                .len();
            model_io::ResidentBuffer::map(&path, 0, len).expect("model_weights.bin maps")
        });
    let resident = resident_map.as_ref().map(|m| {
        let bytes = m.mapped_bytes();
        wrap_page_aligned_no_copy(context.device(), bytes.as_ptr(), bytes.len())
            .expect("no-copy wrap of the resident region")
    });
    let y_resident = context.new_output_buffer(64 * 2);
    let resident_gib = resident_map
        .as_ref()
        .map_or(0.0, |m| m.mapped_bytes().len() as f64 / (1u64 << 30) as f64);
    let buffers: Vec<_> = mappings
        .iter()
        .map(|m| {
            let bytes = m.mapped_bytes();
            wrap_page_aligned_no_copy(context.device(), bytes.as_ptr(), bytes.len())
                .expect("no-copy wrap of a page-aligned mapping")
        })
        .collect();

    const FP16_ONE: u16 = 0x3C00;
    let x = context.new_buffer_with_data(&vec![FP16_ONE; cols]);
    let y = context.new_output_buffer((rows * 2) as u64);
    // One GEMV over expert `expert` of `layer`, in its own submission.
    let mut read = |layer: usize, expert: usize| -> f64 {
        let entry = layout.expert(layer, expert);
        let sub = |name: &str| entry.sub_tensors.get(name).expect(name).offset;
        let started = Instant::now();
        gpu::autorelease_pool(|| {
            let pass = context.begin_pass();
            if let Some(resident) = &resident {
                encode_dequant_int4_gemv_resident(
                    &mut context,
                    &pass,
                    &Int4ResidentMatrix {
                        buffer: resident,
                        weights_offset: 0,
                        scales_offset: 1 << 20,
                        biases_offset: 2 << 20,
                        rows: 64,
                        cols,
                    },
                    (&x, 0),
                    (&y_resident, 0),
                )
                .expect("GEMV encodes against the resident region");
            }
            encode_dequant_int4_gemv_resident(
                &mut context,
                &pass,
                &Int4ResidentMatrix {
                    buffer: &buffers[layer],
                    weights_offset: entry.offset + sub("gate"),
                    scales_offset: entry.offset + sub("gate_scales"),
                    biases_offset: entry.offset + sub("gate_biases"),
                    rows,
                    cols,
                },
                (&x, 0),
                (&y, 0),
            )
            .expect("GEMV encodes against a mapped expert");
            pass.commit_and_wait();
        });
        started.elapsed().as_secs_f64() * 1e3
    };

    let total = layout.num_layers;
    let mut ns: Vec<usize> = (5..total).step_by(5).collect();
    ns.push(total);
    println!("\n=== mapped residency, cyclic layer walk ===");
    println!(
        "install {} ({total} layers, resident {resident_gib:.2} GiB read per pass, \
         {pressure_gib} GiB held)",
        dir.display()
    );
    println!(
        "{:>4} {:>10} {:>14} {:>14}",
        "n", "GiB", "median ms", "max ms"
    );
    for &n in &ns {
        let gib = layer_bytes[..n].iter().sum::<u64>() as f64 / (1u64 << 30) as f64;
        // One untimed walk admits whatever this `n` can hold.
        for layer in 0..n {
            read(layer, 0);
        }
        let mut times = Vec::with_capacity(ROUNDS * n);
        for round in 0..ROUNDS {
            for layer in 0..n {
                // A different expert each round, so a page cache hit on the
                // same expert cannot stand in for residency of the layer.
                times.push(read(layer, (round + 1) % 4));
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{n:>4} {gib:>10.2} {:>14.3} {:>14.3}",
            times[times.len() / 2],
            times[times.len() - 1]
        );
    }
    // Keeps the pressure alive (and committed) until the walk is done.
    assert!(pressure.iter().step_by(2048).all(|&w| w & 1 == 1));
    let out = gpu::read_buffer_f16(&y, 0, rows);
    assert!(
        out.iter().all(|v| v.to_f32().is_finite()),
        "a GEMV over mapped bytes produced a non-finite row; the probe read nothing real"
    );
}
