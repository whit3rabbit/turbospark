#![cfg(target_os = "macos")]
//! Does `--kv-bits` do what it claims on a REAL trained checkpoint: does it
//! actually shrink the counted footprint, and how much does the reference
//! answer's perplexity move? (`docs/TRUBOQUANT.md`.)
//!
//! Phase 3's per-family tests (`crates/runtime/tests/real_forward_*_kv_quant.rs`)
//! already prove the write path and the attention fork engage, on synthetic
//! UNTRAINED weights where the only questions answerable are "does it
//! decode" and "does the output move". Neither answers whether the
//! quantization is a small, tolerable perturbation on a real distribution,
//! or how many MiB it actually buys -- both need real weights, which this
//! sandboxed development session had none of. This probe is what the next
//! session with a real install runs to answer them.
//!
//! # What it measures, and what it does not
//!
//! For `KvQuant::Off` and each TurboQuant width, one open of the SAME
//! install, at the frozen protocol's own per-family window
//! (`real_model::protocol_parameters`, the same one every memory oracle
//! opens at -- see `crates/bench/CLAUDE.md` Gotcha 16):
//!
//! 1. **Peak `phys_footprint`** over a greedy decode of the frozen
//!    `short-explanation` case, via the same mach sampler the memory
//!    oracles use.
//! 2. **Reference-answer perplexity**, via `quality_common`'s own
//!    `teacher_forced_nll` primitive, teacher-forcing the reference answer
//!    into the assistant slot after the frozen protocol's first prompt.
//! 3. **Determinism**: the SAME quantized generation run twice must produce
//!    one greedy digest, the real-checkpoint counterpart of the synthetic
//!    determinism cases Phase 3 already has.
//!
//! **THE FOOTPRINT DELTA IS SMALL AT THIS PROTOCOL'S WINDOW ON MOST
//! FAMILIES, AND THAT IS EXPECTED RATHER THAN A BUG.** `AGENTS.md` Gotcha 36
//! and `crates/bench/CLAUDE.md` Gotcha 1 both establish that the expert
//! slot cache (`slots x layers x expert_stride`) dominates a streamed MoE
//! install's counted footprint, not KV -- so quantizing KV alone moves a
//! small fraction of the total on those families. On a DENSE family
//! (Mistral, the dense `llama` half, `muse_glimmer`) KV is most of the
//! counted footprint (AGENTS.md Gotcha 40), so that is where this probe's
//! percentage column is the most informative, and where the memory
//! decision this feature was built to inform actually lives (context
//! length is the axis KV scales with, not model size). A short prompt at a
//! short protocol window additionally under-states TurboQuant's real
//! payoff, which grows with context length; re-run at a longer
//! `--max-context` by hand (this probe does not sweep it) before quoting a
//! percentage as the feature's ceiling.
//!
//! **THE PERPLEXITY NUMBER IS A REGRESSION SENTINEL, NOT A PASS/FAIL
//! GATE.** Unlike `quality_gate.rs`'s frozen row, there is no reference
//! value to compare against here -- TurboQuant is EXPECTED to move the
//! bytes and therefore the perplexity, by construction. What is asserted
//! is only that it stays FINITE; how much it may reasonably move is a
//! judgement call this probe reports the inputs for rather than makes.
//!
//! ```sh
//! TURBOSPARK_KV_QUANT_INSTALL_DIR=~/models/gemma4.gturbo \
//!   cargo test -p turbospark-bench --test kv_quant_probe --release -- --ignored --nocapture
//! ```
//!
//! Needs a family and a `head_dim` TurboQuant supports (`model_io::rht_supported`:
//! a power of two in 32..=512); an unsupported install is reported by name
//! per width rather than failing the whole probe, since not every install
//! on hand will qualify.

mod quality_common;

use foundation::LogitValue;
use model_io::KvQuant;
use quality_common::{teacher_forced_nll, user_turn_ids, REFERENCE_ANSWER};
use runtime::{DraftPolicies, LogitProducer, RealForwardRunner};
use tokenizer::MfTokenizer;
use turbospark_bench::memory::AppMemorySampler;
use turbospark_bench::real_model::open_model_runner_for_protocol_speculative_kv_quant;

/// Enough to warm the expert cache and show real decode-time KV growth
/// without spending minutes per width; the protocol's own oracles decode
/// hundreds of tokens per case, but this probe cares about the DELTA
/// between widths on one short case, not an absolute ceiling.
const GREEDY_TOKENS: usize = 48;

fn bits(logits: &[LogitValue]) -> Vec<u16> {
    logits.iter().map(|v| v.to_bits()).collect()
}

fn argmax(v: &[LogitValue]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Greedy decode from a fresh reset, sampling the peak footprint across it.
/// Returns the digest (bit patterns of every step's logits, which is what
/// `oracle_common`'s own determinism checks compare), whether every logit
/// seen was finite, and the peak bytes [`AppMemorySampler`] saw across the
/// whole walk.
///
/// **EVERY ROW IS CHECKED FINITE AS IT ARRIVES, not reconstructed from the
/// digest afterward**: a non-finite value survives the `to_bits()` round
/// trip that builds the digest (it is still a valid `u16` bit pattern), so
/// checking after the fact would silently pass a NaN or infinity through
/// undetected -- AGENTS.md Gotcha 59's failure mode, one probe over.
fn decode_and_sample(
    runner: &mut RealForwardRunner,
    tokenizer: &MfTokenizer,
    sampler: &mut AppMemorySampler,
) -> (Vec<u16>, bool, u64) {
    let prompt = user_turn_ids(tokenizer);
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); runner.vocab_size()];
    let mut digest = Vec::new();
    let mut finite = true;
    for (position, &token) in prompt.iter().enumerate() {
        runner
            .produce(token, position, &mut logits)
            .expect("prefill");
    }
    finite &= logits.iter().all(|v| v.to_f32().is_finite());
    digest.extend(bits(&logits));
    sampler.sample();
    let mut token = argmax(&logits);
    for step in 0..GREEDY_TOKENS {
        runner
            .produce(token, prompt.len() + step, &mut logits)
            .expect("decode");
        finite &= logits.iter().all(|v| v.to_f32().is_finite());
        digest.extend(bits(&logits));
        sampler.sample();
        token = argmax(&logits);
    }
    (digest, finite, sampler.peak_bytes().unwrap_or(0))
}

/// Reference-answer perplexity under `runner`, at whatever state it is
/// currently opened at. `assistant_prefix` is deliberately empty, matching
/// `quality_common::measure_perplexity`'s own scope: correct for Gemma,
/// ChatML, Mistral and Qwen, and understating badly for Harmony (`gpt-oss`)
/// families, whose generation prompt ends mid-frame (AGENTS.md Gotcha 46's
/// `crates/bench/CLAUDE.md` Gotcha 13 sibling). Point this probe at a
/// non-Harmony install, or read a huge absolute number as that artifact
/// rather than as TurboQuant damage.
fn reference_ppl(runner: &mut RealForwardRunner, tokenizer: &MfTokenizer) -> f64 {
    let prompt = user_turn_ids(tokenizer);
    let answer = tokenizer.encode(REFERENCE_ANSWER, false);
    assert!(!answer.is_empty(), "the reference answer must tokenize");
    teacher_forced_nll(runner, &prompt, &answer).exp()
}

struct Row {
    label: &'static str,
    kv_quant: KvQuant,
}

#[test]
#[ignore = "needs a real install via TURBOSPARK_KV_QUANT_INSTALL_DIR"]
fn kv_bits_moves_footprint_and_stays_finite_on_a_real_install() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_KV_QUANT_INSTALL_DIR")
            .expect("TURBOSPARK_KV_QUANT_INSTALL_DIR"),
    );
    let arch = repack::peek_manifest_arch(&dir)
        .unwrap_or_else(|e| panic!("{}: failed to read manifest arch: {e}", dir.display()));
    println!(
        "kv_quant_probe: install {}, family {:?}",
        dir.display(),
        arch.family
    );

    let rows = [
        Row {
            label: "off",
            kv_quant: KvQuant::Off,
        },
        Row {
            label: "2",
            kv_quant: KvQuant::TurboQuant {
                k_bits: 2,
                v_bits: 2,
            },
        },
        Row {
            label: "3",
            kv_quant: KvQuant::TurboQuant {
                k_bits: 3,
                v_bits: 3,
            },
        },
        Row {
            label: "3.5",
            kv_quant: KvQuant::TurboQuant {
                k_bits: 3,
                v_bits: 4,
            },
        },
        Row {
            label: "4",
            kv_quant: KvQuant::TurboQuant {
                k_bits: 4,
                v_bits: 4,
            },
        },
    ];

    // `slots` matches `PROTOCOL_EXPERT_CACHE_SLOTS`, the same pin every
    // memory oracle uses, so a footprint delta between widths is not
    // confounded with a slot-count difference (crate Gotcha 5).
    let slots = turbospark_bench::protocol::PROTOCOL_EXPERT_CACHE_SLOTS;

    let mut off_peak: Option<u64> = None;
    let mut off_ppl: Option<f64> = None;
    println!(
        "\n{:<6} {:>12} {:>10} {:>14} {:>10}",
        "width", "peak MiB", "delta", "ref ppl", "ppl delta"
    );
    for row in &rows {
        let opened = open_model_runner_for_protocol_speculative_kv_quant(
            &dir,
            slots,
            DraftPolicies::off(),
            row.kv_quant,
        );
        let (mut runner, tokenizer, _params) = match opened {
            Ok(triple) => triple,
            Err(e) => {
                println!("{:<6} REFUSED: {e}", row.label,);
                if row.label == "off" {
                    panic!(
                        "the FP16 baseline (kv-bits off) must always open; this refusal means \
                         the install itself is unreadable, not that it lacks TurboQuant support"
                    );
                }
                continue;
            }
        };

        let mut sampler = AppMemorySampler::new();
        let (digest_a, finite, peak) = decode_and_sample(&mut runner, &tokenizer, &mut sampler);
        assert!(
            finite,
            "{}: a non-finite logit appeared during decode",
            row.label
        );

        let ppl = reference_ppl(&mut runner, &tokenizer);
        assert!(
            ppl.is_finite(),
            "{}: reference-answer perplexity is not finite ({ppl})",
            row.label
        );

        if row.kv_quant.is_on() {
            // Determinism on the REAL checkpoint: the same quantized
            // generation run twice, from a fresh reset, must agree exactly.
            // Phase 3 already proves this on untrained synthetic weights
            // per family; this is the same check on real ones, which is
            // cheap and closes the one thing untrained weights cannot
            // exercise (a real trained distribution feeding the codec's
            // Lloyd-Max quantizer real, non-adversarial values).
            let (digest_b, _, _) = decode_and_sample(&mut runner, &tokenizer, &mut sampler);
            assert_eq!(
                digest_a, digest_b,
                "{}: two greedy runs from a fresh reset produced different digests; \
                 TurboQuant introduced nondeterminism on this real checkpoint",
                row.label
            );
        }

        let peak_mib = peak as f64 / 1_048_576.0;
        let (delta_str, ppl_delta_str) = match (off_peak, off_ppl) {
            (Some(base_peak), Some(base_ppl)) => {
                let d_mib = (peak as f64 - base_peak as f64) / 1_048_576.0;
                let d_pct = 100.0 * d_mib / (base_peak as f64 / 1_048_576.0);
                let d_ppl = ppl - base_ppl;
                (
                    format!("{d_mib:+.1} MiB ({d_pct:+.1}%)"),
                    format!("{d_ppl:+.4}"),
                )
            }
            _ => ("(baseline)".to_string(), "(baseline)".to_string()),
        };
        println!(
            "{:<6} {peak_mib:>12.1} {delta_str:>10} {ppl:>14.4} {ppl_delta_str:>10}",
            row.label
        );

        if row.label == "off" {
            off_peak = Some(peak);
            off_ppl = Some(ppl);
        }
    }

    println!(
        "\nNOTE: read the module doc before quoting the MiB delta as a ceiling -- it is \
         measured at this protocol's own window and grows with context length on a family \
         where KV dominates the counted footprint (AGENTS.md Gotcha 40), which is not every \
         family here (Gotcha 36 and crate Gotcha 1)."
    );
}
