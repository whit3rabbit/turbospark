//! Timing profile for the MiniMax Music 3 port. Reporting only: no
//! wall-clock assertion passes or fails on a duration. Every number
//! printed here is a CPU f32 measurement of this port on the host that
//! ran the test; none of it is a Metal, quality, or real-checkpoint
//! claim. Run in release (a debug build's numbers are meaningless):
//!
//! ```sh
//! cargo test -p turbospark-audio --release minimax_music3::timing \
//!   -- --ignored --nocapture --test-threads=1
//!
//! --test-threads=1 is part of the measurement, not a nicety: these
//! targets share one test binary, and concurrent timings contaminate
//! each other (measured: the 9000-frame AR run read 79.6 s alone and
//! 138.6 s beside the other benches).
//! ```
//!
//! Two profiles need synthetic trees at enlarged dims, generated on
//! demand by `tools/gen_minimax_music3_bench_tree.py` (a full fp32
//! real-dims tree is ~44 GB and does not fit this host's 36 GiB, so
//! the AR profile runs real-width layers with a reduced vocab and
//! layer count, and the flow profile runs the real DiT and vocoder
//! with a tiny backbone). Point `TURBOSPARK_MUSIC3_BENCH_TREES` at
//! their parent directory; the real-dims tests skip with a note when
//! it is unset. Compositions that extrapolate beyond what was measured
//! are printed as PROJECTED and are arithmetic on measured rates, not
//! measurements.

use std::path::{Path, PathBuf};
use std::time::Instant;

use super::*;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/minimax_music3")
        .join(name)
}

fn bench_dir() -> Option<PathBuf> {
    std::env::var_os("TURBOSPARK_MUSIC3_BENCH_TREES").map(PathBuf::from)
}

fn print_header(label: &str) {
    let chip = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown cpu".to_string());
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let profile = if cfg!(debug_assertions) {
        "debug (numbers are meaningless)"
    } else {
        "release"
    };
    println!(
        "music3 timing [{label}] on {chip}, {threads} threads, {profile} build; \
         CPU f32 port, decode GEMVs run single-threaded"
    );
}

/// AR-stage weight bytes read per frame: every linear is streamed once
/// per decode step (the embedding lookup touches two rows and is not a
/// bandwidth term). Analytic from the config, used to turn a measured
/// frame time into an implied bandwidth.
fn ar_weight_bytes(config: &ModelConfig) -> usize {
    let h = config.hidden_size;
    let q_out = config.num_attention_heads * config.head_dim;
    let kv_out = config.num_key_value_heads * config.head_dim;
    let per_layer = h * q_out + h * kv_out * 2 + q_out * h + 3 * h * config.intermediate_size;
    let depth_per_layer =
        h * h * 4 + 3 * h * config.depth_intermediate_size + h * config.audio_vocab_size;
    let lm_head = if config.tie_word_embeddings {
        0
    } else {
        config.vocab_size * h
    };
    let bytes = lm_head
        + config.num_hidden_layers * per_layer
        + config.depth_num_layers * depth_per_layer
        + (config.num_codebooks - 1) * config.audio_vocab_size * h
        + h * h;
    bytes * 4
}

#[test]
#[ignore = "timing report, not a gate"]
fn tiny_end_to_end_real_time_factor() {
    print_header("tiny end-to-end");
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let text_ids: Vec<i32> = vec![1, 5, 6, 2];
    let request = GenerateRequest {
        text_ids: text_ids.clone(),
        frames: 201,
        steps: 2,
        seed: 7,
    };
    // One discarded warmup, then three measured end-to-end runs.
    model.generate(&request).expect("warmup");
    for _ in 0..3 {
        let start = Instant::now();
        let generation = model.generate(&request).expect("generation");
        let wall = start.elapsed();
        let seconds = generation.samples as f64 / f64::from(generation.sample_rate);
        println!(
            "  end-to-end: {:.3?} for {} samples ({:.2} s audio), RTF {:.4} \
             (wall / audio; <1 means faster than real time)",
            wall,
            generation.samples,
            seconds,
            wall.as_secs_f64() / seconds
        );
    }
    // Stage split on one pass.
    let start = Instant::now();
    let (hiddens, codes) = model
        .generate_frame_hiddens(&text_ids, 201, 7)
        .expect("AR stage");
    let ar_wall = start.elapsed();
    let emitted = codes.len();
    let start = Instant::now();
    let stereo = model.run_flow(&hiddens, emitted, 2, 7).expect("flow stage");
    let flow_wall = start.elapsed();
    println!(
        "  stage split: AR {:.3?} ({} frames emitted, end token stopped the run), \
         flow {:.3?} ({} samples)",
        ar_wall,
        emitted,
        flow_wall,
        stereo.len() / 2
    );
}

/// At tiny dims the GEMV work is negligible, so the growth of the
/// per-frame cost with position isolates the KV-cache handling term
/// (append re-lays the whole cache each step; O(t^2) per generation).
#[test]
#[ignore = "timing report, not a gate"]
fn tiny_ar_scaling_profile() {
    print_header("tiny AR scaling");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/minimax_music3_bench/tiny_long");
    if !dir.join("config.json").is_file() {
        panic!(
            "missing {}; generate it with tools/gen_minimax_music3_bench_tree.py --variant tiny_long",
            dir.display()
        );
    }
    let model = Model::load_converted(&dir).expect("tiny_long loads");
    let text_ids: Vec<i32> = vec![1, 5, 6, 2];
    for frames in [1000usize, 2000, 4000, MAX_AUDIO_FRAMES] {
        // Two runs per size; the faster one is the number to quote.
        let mut walls = Vec::new();
        for _ in 0..2 {
            let start = Instant::now();
            let (hiddens, codes) = model
                .generate_frame_hiddens(&text_ids, frames, 7)
                .expect("AR run");
            walls.push(start.elapsed());
            assert_eq!(codes.len(), frames, "end token must not stop the run");
            assert_eq!(hiddens.len(), frames * 8 * model.config().hidden_size);
        }
        walls.sort();
        let best = walls[0];
        println!(
            "  AR tiny {frames:>5} frames: best {:.3?} ({:.3} ms/frame)",
            best,
            best.as_secs_f64() * 1e3 / frames as f64
        );
    }
}

#[test]
#[ignore = "timing report, not a gate"]
fn tiny_flow_multichunk_profile() {
    print_header("tiny flow, multi-chunk");
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let fused = model.config().num_codebooks * model.config().hidden_size;
    for frames in [201usize, 601, 1201] {
        let starts = chunk_starts(frames);
        let mut hiddens = Vec::with_capacity(frames * fused);
        for frame in 0..frames {
            let pattern = (frame % 8) as f32 * 0.01 - 0.03;
            hiddens.extend(std::iter::repeat_n(pattern, fused));
        }
        let mut walls = Vec::new();
        for _ in 0..2 {
            let start = Instant::now();
            let stereo = model.run_flow(&hiddens, frames, 2, 7).expect("flow");
            walls.push(start.elapsed());
            assert!(stereo.iter().all(|v| v.is_finite()));
        }
        walls.sort();
        println!(
            "  flow tiny {frames:>5} frames ({} chunks, 2 steps): best {:.3?} ({:.1} ms/chunk)",
            starts.len(),
            walls[0],
            walls[0].as_secs_f64() * 1e3 / starts.len() as f64
        );
    }
}

/// Real-width AR profile: real hidden size, layer width, head layout,
/// and depth decoder at a reduced layer count and vocab (a full
/// real-dims fp32 tree does not fit this host). The per-layer and
/// real-vocab projections below are arithmetic on the measured frame
/// time and the analytic weight bytes, not measurements.
#[test]
#[ignore = "timing report, needs TURBOSPARK_MUSIC3_BENCH_TREES"]
fn real_width_ar_profile() {
    let Some(root) = bench_dir() else {
        println!("skipped: TURBOSPARK_MUSIC3_BENCH_TREES is not set");
        return;
    };
    print_header("real-width AR");
    let model = Model::load_converted(&root.join("ar_real")).expect("ar_real loads");
    let config = model.config();
    println!(
        "  ar_real config: hidden {} x {} layers, ffn {}, heads {}/kv {}, head_dim {}, \
         vocab {} (real: 4096 x 36, ffn 12288, 32/8, 128, 200k)",
        config.hidden_size,
        config.num_hidden_layers,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.vocab_size
    );
    let text_ids: Vec<i32> = vec![1, 5, 6, 2];
    // Discarded warmup.
    model
        .generate_frame_hiddens(&text_ids, 4, 7)
        .expect("warmup");
    for frames in [50usize, 200] {
        let start = Instant::now();
        let (hiddens, codes) = model
            .generate_frame_hiddens(&text_ids, frames, 7)
            .expect("AR run");
        let wall = start.elapsed();
        assert_eq!(codes.len(), frames);
        assert!(hiddens.iter().all(|v| v.is_finite()));
        let per_frame = wall.as_secs_f64() / frames as f64;
        let bytes = ar_weight_bytes(config) as f64;
        println!(
            "  AR real-width {frames:>3} frames: {wall:.3?} ({:.1} ms/frame); \
             {:.2} GiB weights/frame implies {:.2} GiB/s effective",
            per_frame * 1e3,
            bytes / (1 << 30) as f64,
            bytes / per_frame / (1 << 30) as f64
        );
        if frames == 200 {
            println!("  (see real_width_ar_components for the per-piece decomposition)");
        }
    }
}

/// Component decomposition of one real-width AR frame at a warm
/// ~512-position cache: one LM decode step across all layers, the
/// vocab head, and one depth-decoder expansion (the AR stage runs
/// seven per frame at growing sequence lengths; seq 8 is the largest).
/// These price the real-dims projection: the depth decoder is already
/// at real dims and does not scale with the layer count; the LM step
/// and head do.
#[test]
#[ignore = "timing report, needs TURBOSPARK_MUSIC3_BENCH_TREES"]
fn real_width_ar_components() {
    let Some(root) = bench_dir() else {
        println!("skipped: TURBOSPARK_MUSIC3_BENCH_TREES is not set");
        return;
    };
    print_header("real-width AR components");
    let model = Model::load_converted(&root.join("ar_real")).expect("ar_real loads");
    let config = model.config();
    let warm = 512usize;
    let mut cache: Vec<KvCache> = (0..config.num_hidden_layers)
        .map(|_| KvCache::new())
        .collect();
    let embeddings = vec![0.01f32; 2 * warm * config.hidden_size];
    model
        .lm
        .hidden_forward(&embeddings, 2, warm, &mut cache)
        .expect("prefill");
    let feedback = vec![0.01f32; 2 * config.hidden_size];
    let last_hidden = vec![0.01f32; 2 * config.hidden_size];
    let runs = 10usize;
    let start = Instant::now();
    for _ in 0..runs {
        drop(
            model
                .lm
                .hidden_forward(&feedback, 2, 1, &mut cache)
                .expect("decode step"),
        );
    }
    let lm_step = start.elapsed() / runs as u32;
    let start = Instant::now();
    for _ in 0..runs {
        drop(model.lm.logits(&last_hidden).expect("logits"));
    }
    let head = start.elapsed() / runs as u32;
    // One AR frame expands the depth decoder seven times at sequence
    // lengths 2..=8; time that exact sequence rather than seven seq-8
    // calls, which would overstate the per-frame cost by ~1.6x.
    let start = Instant::now();
    for _ in 0..runs {
        for seq in 2..=8usize {
            let depth_in = vec![0.01f32; 2 * seq * config.hidden_size];
            drop(model.depth.forward(&depth_in, 2, seq).expect("depth"));
        }
    }
    let depth_frame = start.elapsed() / runs as u32;
    println!(
        "  components at t~{warm}: LM decode step ({} layers) {:.3?}, vocab head ({} rows) {:.3?}, \
         depth expansion (seqs 2..=8) {:.3?} per frame",
        config.num_hidden_layers,
        lm_step,
        config.vocab_size,
        head,
        depth_frame
    );
    // PROJECTED real dims: the LM step scales linearly with layers
    // (identical per-layer GEMV shapes), the head with its row count,
    // and the depth decoder is already real. Arithmetic composition,
    // not a measurement; a real checkpoint would run on Metal, not
    // this CPU path.
    let projected_lm = lm_step.mul_f64(36.0 / config.num_hidden_layers as f64);
    let projected_head = head.mul_f64(200_000.0 / config.vocab_size as f64);
    println!(
        "  PROJECTED 36-layer 200k-vocab AR frame: LM {:.0?} + head {:.0?} + depth {:.0?} \
         + attention (grows with t; see minimax_music3::qwen3::bench)",
        projected_lm, projected_head, depth_frame
    );
}

/// Real-DiTs flow profile: one 200-frame chunk through the real
/// condition encoder, DiT, and vocoder sizes. Each Euler step runs two
/// DiT forwards (CFG), so a 30-step chunk is 60 forwards.
#[test]
#[ignore = "timing report, needs TURBOSPARK_MUSIC3_BENCH_TREES"]
fn real_dims_flow_profile() {
    let Some(root) = bench_dir() else {
        println!("skipped: TURBOSPARK_MUSIC3_BENCH_TREES is not set");
        return;
    };
    print_header("real-dims flow, one chunk");
    let model = Model::load_converted(&root.join("flow_real")).expect("flow_real loads");
    let config = model.config();
    let fused = config.num_codebooks * config.hidden_size;
    let frames = 200usize;
    let mut hiddens = Vec::with_capacity(frames * fused);
    for frame in 0..frames {
        let pattern = (frame % 8) as f32 * 0.01 - 0.03;
        hiddens.extend(std::iter::repeat_n(pattern, fused));
    }
    let start = Instant::now();
    let condition = model
        .condition
        .forward(&hiddens, frames)
        .expect("condition");
    let cond_wall = start.elapsed();
    let cond_dim = config.condition_out_dim;
    let target = condition.len() / cond_dim;
    let noise = rng::normal(
        rng::KeySequence::new(7).next(),
        config.dit_in_channels * target,
    );
    let channel_major = condition_row_major_to_channel(&condition, cond_dim, target);
    // Discarded warmup, then two measured two-step denoise runs.
    euler::denoise_chunk(
        &model.transformer,
        &noise,
        &channel_major,
        config.dit_in_channels,
        cond_dim,
        target,
        2,
        DIT_CFG_SCALE,
        None,
        None,
    )
    .expect("warmup denoise");
    let mut denoise_walls = Vec::new();
    for _ in 0..2 {
        let start = Instant::now();
        let (latents, _) = euler::denoise_chunk(
            &model.transformer,
            &noise,
            &channel_major,
            config.dit_in_channels,
            cond_dim,
            target,
            2,
            DIT_CFG_SCALE,
            None,
            None,
        )
        .expect("denoise");
        denoise_walls.push(start.elapsed());
        let start = Instant::now();
        let wave = model.vocoder.forward(&latents, target).expect("vocoder");
        let voc_wall = start.elapsed();
        assert!(wave.iter().all(|v| v.is_finite()), "vocoder output");
        if denoise_walls.len() == 2 {
            let denoise = denoise_walls[1];
            let per_forward = denoise.as_secs_f64() / 4.0;
            let seconds = wave.len() as f64 / 2.0 / f64::from(SAMPLING_RATE);
            println!(
                "  chunk: latent len {target}, condition {:.3?}, \
                 2-step denoise {:.3?} ({:.0} s/DiT forward, 2 CFG forwards per step), \
                 vocoder {:.3?} ({:.0} K samples/s, {:.2} s audio)",
                cond_wall,
                denoise,
                per_forward,
                voc_wall,
                wave.len() as f64 / voc_wall.as_secs_f64() / 1e3,
                seconds
            );
            // PROJECTED 30-step chunk: the schedule is linear in steps.
            println!(
                "  PROJECTED 30-step chunk: {:.1?} (linear in steps; 60 DiT forwards)",
                cond_wall + denoise.mul_f64(15.0) + voc_wall
            );
        }
    }
}

/// Decomposes one tiny AR frame into its pieces at a ~1k-token cache:
/// the LM decode step, the vocab head, the seven depth-decoder
/// expansions, the per-frame samplers, and (via the qwen3 bench) the
/// rope-table / KV-cache terms. This is the attribution behind the
/// scaling profile above.
#[test]
#[ignore = "timing report, not a gate"]
fn tiny_ar_frame_anatomy() {
    print_header("tiny AR frame anatomy");
    let model = Model::load_converted(&testdata("converted_plain")).expect("plain loads");
    let config = model.config();
    let hidden = config.hidden_size;
    let warm = 1024usize;
    // Grow a real cache to `warm` positions with one synthetic prefill.
    let mut cache: Vec<KvCache> = (0..config.num_hidden_layers)
        .map(|_| KvCache::new())
        .collect();
    let embeddings = vec![0.01f32; 2 * warm * hidden];
    model
        .lm
        .hidden_forward(&embeddings, 2, warm, &mut cache)
        .expect("prefill");
    let feedback = vec![0.01f32; 2 * hidden];
    let last_hidden = vec![0.01f32; 2 * hidden];

    let steps = 200usize;
    let start = Instant::now();
    for _ in 0..steps {
        drop(
            model
                .lm
                .hidden_forward(&feedback, 2, 1, &mut cache)
                .expect("decode step"),
        );
    }
    let decode = start.elapsed() / steps as u32;
    let start = Instant::now();
    for _ in 0..steps {
        drop(model.lm.logits(&last_hidden).expect("logits"));
    }
    let head = start.elapsed() / steps as u32;
    let depth_in = vec![0.01f32; 2 * 8 * hidden];
    let start = Instant::now();
    for _ in 0..steps {
        drop(model.depth.forward(&depth_in, 2, 8).expect("depth"));
    }
    let depth_once = start.elapsed() / steps as u32;
    let guided = vec![0.0f32; config.vocab_size];
    let start = Instant::now();
    let mut key = rng::Key::new(7);
    for _ in 0..steps {
        let drawn;
        (drawn, key) = rng::sample_top_k(&guided, key, AR_SAMPLING_TOP_K);
        let _ = drawn;
    }
    let sample = start.elapsed() / steps as u32;
    println!(
        "  per frame at t~{warm}: LM decode step {decode:.3?} + vocab head {head:.3?} + \
         7x depth {:.3?} + 8x sample_top_k {:.3?} = {:.1?} accounted before rope/KV terms",
        depth_once * 7,
        sample * 8,
        decode + head + depth_once * 7 + sample * 8
    );
}
