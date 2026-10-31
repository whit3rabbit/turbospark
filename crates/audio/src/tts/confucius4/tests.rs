//! Parity tests for the Confucius4 family against the fixture recorded
//! by `tools/gen_confucius4_fixtures.py` (tiny seeded w2v-bert, CAMPPlus,
//! prefix, T2S, S2A, and BigVGAN stacks; the T2S run uses the greedy
//! stand-in so the trajectory is deterministic, with the sampled
//! distribution math pinned separately).

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{Model, ModelConfig};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_confucius4")
        .join(name)
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("fixture json");
    serde_json::from_str(&text).expect("fixture json")
}

fn f32_vec(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("values")
        .iter()
        .map(|v| v.as_f64().expect("value") as f32)
        .collect()
}

fn i32_vec(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_i64().expect("id") as i32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    diff
}

fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let max = b.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    max_abs_diff(a, b) / max
}

/// Measured gates frozen with the fixtures (measured values in the
/// comments; see the family README): the DSP/conformer stacks sit at
/// the conv/attention reduction level, the euler mel covers the solver
/// accumulation, and the waveform gate is relative over the vocoder
/// conv stacks.
const FEATS_TOLERANCE: f32 = 2e-4;
/// The reference mel chain runs numpy float64 end to end (rfft
/// included) while the port is f32; measured 2.4e-4 at log-mel
/// magnitudes near 11.5.
const REF_MEL_TOLERANCE: f32 = 4e-4;
const STATE_TOLERANCE: f32 = 2e-5;
const MEL_TOLERANCE: f32 = 1e-4;
const WAVEFORM_REL_TOLERANCE: f32 = 1e-5;

fn open_case() -> Model {
    let config = read_json(&testdata("pipeline/config.json"));
    Model::open_with_config(&testdata("pipeline"), config).expect("model loads")
}

#[test]
fn dsp_and_encoder_stages_match_the_reference() {
    let traces = read_json(&testdata("pipeline/traces.json"));
    let model = open_case();
    let audio = f32_vec(&traces["audio"]);

    // The w2v fbank features.
    let feats =
        super::dsp::fbank_160(&audio, &model.fbank_mel, &model.fbank_window).expect("fbank");
    let diff = max_abs_diff(&feats, &f32_vec(&traces["feats"]));
    assert!(diff <= FEATS_TOLERANCE, "fbank diff {diff}");

    // hidden17 and the normalized condition vector.
    let frames = feats.len() / 160;
    let h17 = model.w2vbert.hidden17(&feats, frames).expect("hidden17");
    let diff = max_abs_diff(&h17, &f32_vec(&traces["h17"]));
    // The two-layer conformer accumulates conv/attention order noise;
    // measured 5.7e-4 at state magnitudes near 3 (see the README).
    // Conformer conv/attention reduction order noise; measured 5.7e-4.
    assert!(diff <= 1e-3, "h17 diff {diff}");
    let cond_dim = model.config.prefix.cond_dim;
    let mut cond_vec = vec![0.0f32; h17.len()];
    for (index, value) in h17.iter().enumerate() {
        let col = index % cond_dim;
        cond_vec[index] = (value - model.stats_mean[col]) / model.stats_std[col];
    }
    let diff = max_abs_diff(&cond_vec, &f32_vec(&traces["cond_vec"]));
    // Downstream of h17; measured 7.1e-4.
    assert!(diff <= 1e-3, "cond vec diff {diff}");

    // The CAMPPlus kaldi fbank and the style embedding.
    let camp_fbank = super::dsp::kaldi_fbank(&audio, 16000, 80).expect("camp fbank");
    let normalized = super::mean_normalize(&camp_fbank);
    let diff = max_abs_diff(&normalized, &f32_vec(&traces["camp_fbank"]));
    assert!(diff <= FEATS_TOLERANCE, "camp fbank diff {diff}");
    let style = model.campplus.inference(&normalized, camp_fbank.len() / 80);
    let diff = max_abs_diff(&style, &f32_vec(&traces["style"]));
    assert!(diff <= STATE_TOLERANCE, "style diff {diff}");

    // The reference mel: `_ref_mel` returns (1, nfr, 80), so the trace
    // is [frames, mels] row-major, exactly the port's layout.
    let ref_mel = super::dsp::ref_mel(&audio).expect("ref mel");
    let want = f32_vec(&traces["ref_mel"]);
    assert_eq!(want.len(), ref_mel.len(), "ref mel length");
    let diff = max_abs_diff(&ref_mel, &want);
    assert!(diff <= REF_MEL_TOLERANCE, "ref mel diff {diff}");
}

#[test]
fn prefix_and_t2s_stages_match_the_reference() {
    let traces = read_json(&testdata("pipeline/traces.json"));
    let model = open_case();
    let cond_vec = f32_vec(&traces["cond_vec"]);
    let text_ids = i32_vec(&traces["text_ids"]);

    let (cond_emb, text_emb) = model.prefix_embeddings(&cond_vec, &text_ids).unwrap();
    let diff = max_abs_diff(&cond_emb, &f32_vec(&traces["cond_emb"]));
    // Measured 7.2e-7.
    assert!(diff <= 1e-5, "cond emb diff {diff}");
    let diff = max_abs_diff(&text_emb, &f32_vec(&traces["text_emb"]));
    // Measured 1.2e-7.
    assert!(diff <= 1e-5, "text emb diff {diff}");

    // Greedy T2S: codes, latent, and every step's raw logits.
    let (codes, latent) = model
        .t2s_generate(&cond_emb, &text_emb, 8, 0.8, 30, 0.8, 10.0, true, 0)
        .expect("t2s");
    assert_eq!(codes, i32_vec(&traces["codes"]), "codes");
    let latent_diff = max_abs_diff(&latent, &f32_vec(&traces["latent"]));
    // Measured 2.4e-6.
    assert!(latent_diff <= 1e-5, "latent diff {latent_diff}");
    for (step, want) in traces["t2s_logits"]
        .as_array()
        .expect("logits")
        .iter()
        .enumerate()
    {
        // Re-run the prefill/step logits through the public path by
        // regenerating stepwise: the greedy run above already pins the
        // tokens, so compare the recorded logits against a fresh run's
        // head outputs via the internal hooks.
        let _ = (step, want);
    }
}

#[test]
fn t2s_step_logits_match_the_reference() {
    let traces = read_json(&testdata("pipeline/traces.json"));
    let model = open_case();
    let cond_emb = f32_vec(&traces["cond_emb"]);
    let text_emb = f32_vec(&traces["text_emb"]);
    let text_len = text_emb.len() / model.config.t2s.d_model;
    let d = model.config.t2s.d_model;
    // Step logits are pinned through a public stepwise helper.
    let logits = model
        .t2s_step_logits(&cond_emb, &text_emb, &i32_vec(&traces["codes"]))
        .expect("step logits");
    let want_steps = traces["t2s_logits"].as_array().expect("logits");
    assert_eq!(logits.len(), want_steps.len(), "step count");
    for (step, (got, want)) in logits.iter().zip(want_steps).enumerate() {
        let diff = max_abs_diff(got, &f32_vec(want));
        // Measured 1.7e-6 at the widest step.
        assert!(diff <= 1e-5, "step {step} logits diff {diff}");
    }
    let _ = (text_len, d);
}

#[test]
fn sampled_distribution_matches_the_reference() {
    let traces = read_json(&testdata("pipeline/traces.json"));
    let logits = f32_vec(traces["t2s_logits"].as_array().unwrap().last().unwrap());
    let want = f32_vec(&traces["sampled_probs"].as_array().unwrap()[0]);
    let got = super::t2s::sample_distribution(&logits, &[super::t2s::BOS], 0.8, 30, 0.8, 10.0);
    // 8194-way probability vector; the f32 chain against the f64
    // reference sits at the mass level.
    let diff = max_abs_diff(&got, &want);
    // 8194-way probability vector; the f32 chain against the f64
    // reference sits at the mass level (measured 7.5e-9).
    assert!(diff <= 2e-6, "sampled distribution diff {diff}");
}

#[test]
fn s2a_and_vocoder_stages_match_the_reference() {
    let traces = read_json(&testdata("pipeline/traces.json"));
    let model = open_case();
    let codes = i32_vec(&traces["codes"]);
    let latent = f32_vec(&traces["latent"]);
    let style = f32_vec(&traces["style"]);
    let t_ref = traces["t_ref"].as_u64().unwrap() as usize;
    // The reference ref mel, channel-first as recorded.
    let want_ref_mel = f32_vec(&traces["ref_mel"]);
    let total_frames = traces["mel_frames"].as_u64().unwrap() as usize + t_ref;
    let z = f32_vec(&traces["z"]);

    let mu = model.s2a.build_mu(&codes, &latent, t_ref).unwrap();
    let diff = max_abs_diff(&mu, &f32_vec(&traces["mu"]));
    assert!(diff <= STATE_TOLERANCE, "mu diff {diff}");
    assert_eq!(
        mu.len() / model.config.s2a.hidden,
        total_frames,
        "mu frames"
    );

    let mel = model
        .s2a_solve(&codes, &latent, &style, &want_ref_mel, t_ref, &z)
        .unwrap();
    // The recorded mel is channel-first [80, target frames].
    let target = total_frames - t_ref;
    let want_mel = f32_vec(&traces["mel"]);
    let mut got_cm = vec![0.0f32; target * 80];
    for f in 0..target {
        for c in 0..80 {
            got_cm[c * target + f] = mel[(t_ref + f) * 80 + c];
        }
    }
    let diff = max_abs_diff(&got_cm, &want_mel);
    assert!(diff <= MEL_TOLERANCE, "mel diff {diff}");

    let samples = model.vocode(&mel, total_frames, t_ref).unwrap();
    let diff = rel_diff(&samples, &f32_vec(&traces["waveform"]));
    assert!(diff <= WAVEFORM_REL_TOLERANCE, "waveform rel diff {diff}");
}

#[test]
fn config_defaults_match_the_reference() {
    let config = ModelConfig::default();
    assert_eq!(config.model_type, "confucius4");
    assert_eq!(config.sample_rate, 22050);
    assert_eq!(config.w2vbert.layers, 17);
    assert_eq!(config.t2s.d_model, 1280);
    assert_eq!(config.t2s.semantic_vocab, 8194);
    assert_eq!(config.s2a.depth, 13);
    assert_eq!(config.s2a.hidden, 512);
    assert_eq!(config.campplus.embedding_size, 192);
    let value = read_json(&testdata("pipeline/config.json"));
    let parsed = ModelConfig::from_json(&value);
    assert_eq!(parsed.t2s.semantic_vocab, 8194);
    assert_eq!(parsed.s2a.depth, 5);
    assert_eq!(parsed.campplus.m_channels, 8);
}

#[test]
fn short_w2v_stats_are_refused_at_open() {
    // A stats tensor shorter than cond_dim used to panic in
    // compute_condition; it is now a BadConfig at open.
    assert!(super::check_stats_len(1024, 1024, 1024).is_ok());
    assert!(super::check_stats_len(1025, 1024, 1024).is_ok());
    let err = super::check_stats_len(512, 1024, 1024).expect_err("short mean");
    assert!(
        matches!(&err, crate::SpeechError::BadConfig { field, .. } if field == "w2v_stats"),
        "unexpected error: {err}"
    );
    assert!(super::check_stats_len(1024, 100, 1024).is_err());
}

#[test]
fn non_positive_temperature_is_refused_before_sampling() {
    use super::t2s::validate_temperature;
    assert!(validate_temperature(0.8).is_ok());
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(validate_temperature(bad).is_err(), "accepted {bad}");
    }
}
