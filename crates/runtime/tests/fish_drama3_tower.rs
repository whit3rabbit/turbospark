#![cfg(target_os = "macos")]
//! The Metal Drama 3 tower against the pinned mlx-audio goldens and the f32
//! CPU tower, on the tiny 8-bit fixture (`crates/audio/testdata/fish_drama3`).

use audio::tts::fish_drama3::tower::{CpuTower, Tower, TowerConfig, TowerModel};
use audio::tts::sampling::argmax;
use model_io::safetensors::SafetensorsFile;
use std::path::PathBuf;
use turbospark_runtime::fish_drama3_tower::MetalTower;

const CAPACITY: usize = 64;

struct Golden {
    value: serde_json::Value,
    file: SafetensorsFile,
}

fn golden() -> Golden {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../audio/testdata/fish_drama3");
    let text = std::fs::read_to_string(dir.join("tower.json")).expect("tower fixture");
    let value: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    assert_eq!(
        value["commit"],
        audio::tts::fish_drama3::REFERENCE_COMMIT,
        "fixture from another reference commit"
    );
    let file = SafetensorsFile::open(&dir.join("tower_side.safetensors")).expect("weights");
    Golden { value, file }
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

fn ints(v: &serde_json::Value) -> Vec<i32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("int") as i32)
        .collect()
}

impl Golden {
    fn config(&self) -> TowerConfig {
        TowerConfig::from_json(&self.value["tower"]).expect("config")
    }

    fn band(&self) -> std::ops::RangeInclusive<i32> {
        self.value["semantic_begin"].as_i64().unwrap() as i32
            ..=self.value["semantic_end"].as_i64().unwrap() as i32
    }

    fn metal(&self) -> MetalTower {
        MetalTower::load(&self.file, self.config(), self.band(), CAPACITY).expect("metal tower")
    }

    fn cpu(&self) -> CpuTower {
        CpuTower::new(Tower::load(&self.file, self.config(), self.band()).expect("cpu tower"))
    }

    fn prompt(&self) -> Vec<Vec<i32>> {
        self.value["prompt"]
            .as_array()
            .unwrap()
            .iter()
            .map(ints)
            .collect()
    }

    fn prompt_hidden(&self) -> Vec<f32> {
        self.value["prompt_hidden"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(floats)
            .collect()
    }

    fn steps(&self) -> &Vec<serde_json::Value> {
        self.value["steps"].as_array().unwrap()
    }

    /// Teacher-forced prefill of every prompt row but the last.
    fn prefill(&self, tower: &mut dyn TowerModel) {
        let prompt = self.prompt();
        let hidden = self.prompt_hidden();
        let dim = tower.config().dim;
        let last = prompt.len() - 1;
        let frames: Vec<&[i32]> = prompt[..last].iter().map(Vec::as_slice).collect();
        let labels: Vec<&[i32]> = prompt[1..].iter().map(Vec::as_slice).collect();
        tower
            .prefill(&hidden[..last * dim], &frames, &labels)
            .expect("prefill");
        assert_eq!(tower.position(), last);
    }
}

/// Worst absolute error relative to the reference's largest magnitude.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

/// One greedy decode step: the frame and every head's logits.
fn step(
    tower: &mut dyn TowerModel,
    hidden: &[f32],
    main: i32,
    frame: &[i32],
) -> (Vec<i32>, Vec<Vec<f32>>) {
    let mut logits = Vec::new();
    let codes = tower
        .residual_codes(hidden, main, frame, &mut |_, l| {
            logits.push(l.to_vec());
            argmax(l) as i32
        })
        .expect("decode step");
    let mut out = vec![main];
    out.extend(codes);
    (out, logits)
}

#[test]
fn metal_tower_matches_the_reference_and_the_cpu_tower() {
    let g = golden();
    let mut metal = g.metal();
    let mut cpu = g.cpu();
    g.prefill(&mut metal);
    g.prefill(&mut cpu);

    let mut frame = g.prompt().last().unwrap().clone();
    let (mut worst_ref, mut worst_cpu) = (0.0f32, 0.0f32);
    for (s, golden_step) in g.steps().iter().enumerate() {
        let hidden = floats(&golden_step["hidden"]);
        let main = golden_step["main"].as_i64().unwrap() as i32;
        let semantic = rel_err(
            &metal.semantic_logits(&hidden).unwrap(),
            &cpu.semantic_logits(&hidden).unwrap(),
        );
        eprintln!("step {s} semantic head vs cpu: {semantic:.2e}");
        worst_cpu = worst_cpu.max(semantic);

        let (got_frame, got) = step(&mut metal, &hidden, main, &frame);
        let (cpu_frame, cpu_logits) = step(&mut cpu, &hidden, main, &frame);
        let want: Vec<Vec<f32>> = golden_step["head_logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        assert_eq!(got.len(), want.len());
        for (head, ((m, c), w)) in got.iter().zip(&cpu_logits).zip(&want).enumerate() {
            let (er, ec) = (rel_err(m, w), rel_err(m, c));
            eprintln!("step {s} head {head}: vs reference {er:.2e}, vs cpu {ec:.2e}");
            worst_ref = worst_ref.max(er);
            worst_cpu = worst_cpu.max(ec);
        }
        assert_eq!(
            got_frame,
            ints(&golden_step["frame"]),
            "step {s}: greedy codes"
        );
        assert_eq!(got_frame, cpu_frame, "step {s}: greedy codes against cpu");
        frame = got_frame;
    }
    assert_eq!(metal.position(), cpu.position());
    // f16 activations here, bf16 in the reference, f32 on the cpu. The cpu
    // bound sits 4x over a measured 2.5e-3; an unpermuted q_norm (a rope
    // reuse bug) reads 4.7e-2.
    assert!(
        worst_ref < 3e-2,
        "worst relative error vs reference {worst_ref:.2e}"
    );
    assert!(
        worst_cpu < 1e-2,
        "worst relative error vs cpu {worst_cpu:.2e}"
    );
}

/// Rows split across prefill calls see the same cache as one call, and a
/// truncated tower replays a step bit for bit.
#[test]
fn split_prefill_and_truncate_replay_bit_for_bit() {
    let g = golden();
    let prompt = g.prompt();
    let hidden = g.prompt_hidden();
    let first = &g.steps()[0];
    let (h0, main0) = (
        floats(&first["hidden"]),
        first["main"].as_i64().unwrap() as i32,
    );
    let last = prompt.last().unwrap();

    let mut whole = g.metal();
    g.prefill(&mut whole);
    let (frame_whole, logits_whole) = step(&mut whole, &h0, main0, last);

    let mut split = g.metal();
    let dim = split.config().dim;
    for r in 0..prompt.len() - 1 {
        split
            .prefill(
                &hidden[r * dim..][..dim],
                &[prompt[r].as_slice()],
                &[prompt[r + 1].as_slice()],
            )
            .unwrap();
    }
    let (frame_split, logits_split) = step(&mut split, &h0, main0, last);
    assert_eq!(frame_split, frame_whole);
    assert_eq!(
        logits_split, logits_whole,
        "split prefill changed the cache"
    );

    let before = split.position() - 1;
    split.truncate(before).unwrap();
    let (frame_again, logits_again) = step(&mut split, &h0, main0, last);
    assert_eq!(frame_again, frame_whole);
    assert_eq!(logits_again, logits_whole, "truncate did not replay");
    assert!(split.truncate(split.position() + 1).is_err());
}

#[test]
fn a_full_tower_refuses_more_rows() {
    let g = golden();
    let mut tower = MetalTower::load(&g.file, g.config(), g.band(), 2).expect("metal tower");
    let prompt = g.prompt();
    let hidden = g.prompt_hidden();
    let dim = tower.config().dim;
    let frames: Vec<&[i32]> = prompt[..3].iter().map(Vec::as_slice).collect();
    let labels: Vec<&[i32]> = prompt[1..4].iter().map(Vec::as_slice).collect();
    let err = tower
        .prefill(&hidden[..3 * dim], &frames, &labels)
        .expect_err("three rows into a two-row tower");
    assert!(err.to_string().contains("full"), "{err}");
    assert_eq!(tower.position(), 0);
}
