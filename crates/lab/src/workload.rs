//! What one worker run executes, and under which arm.
//!
//! These types cross the orchestrator/worker boundary as JSON, so they are
//! portable plain data. An arm is ENVIRONMENT plus open parameters only:
//! every runtime knob is read from the environment at open or through a
//! process-wide `OnceLock`, and a fresh process per run makes both kinds
//! apply the same way.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One prompt the workload prefills.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSource {
    /// A frozen protocol case id from `crates/bench/prompts/real-generation-v1`
    /// (`short-explanation`, `medium-review`, `long-synthesis`), chat-templated
    /// as the CLI templates it.
    Protocol(String),
    /// A `--messages-file`-shaped JSON conversation, chat-templated.
    MessagesFile(PathBuf),
    /// A raw text file tokenized without a template.
    TextFile(PathBuf),
    /// A slice of the distribution-probe corpus
    /// (`turbospark_bench::quality_corpus`) of this many tokens. The k-th
    /// corpus prompt in a workload takes the probes' k-th offset, so a
    /// workload of the probe lengths prefills the probes' exact token ids.
    Corpus(usize),
    /// A multi-turn conversation (`worker::ChatFile`): the opening turn, then
    /// each follow-up, with the model's replies carried back WITHOUT their
    /// reasoning, as most OpenAI-style clients send them. Prefix reuse is on
    /// for the whole chat, so a follow-up's time to first token measures what
    /// a client waits for. Judged on generated tokens, like a speculative arm.
    Chat(PathBuf),
}

impl PromptSource {
    pub fn is_chat(&self) -> bool {
        matches!(self, PromptSource::Chat(_))
    }
}

/// Expert cache slot policy for the open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Slots {
    #[default]
    Auto,
    Fixed(usize),
}

/// Expert residency for the open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    #[default]
    Auto,
    Streamed,
    Mapped,
}

/// Parameters that are part of the OPEN rather than the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OpenParams {
    #[serde(default)]
    pub slots: Slots,
    #[serde(default)]
    pub residency: Residency,
}

/// One arm: a name, the `TURBOSPARK_*` variables it sets, and its open
/// parameters. Variables not named here are cleared in the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arm {
    pub name: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub open: OpenParams,
    /// Decode through the speculative loop with this drafter and depth. An
    /// arm's setting wins over the workload's, so one session can compare
    /// plain decode against MTP or DFlash2 at several depths.
    #[serde(default)]
    pub speculation: Option<Speculation>,
    /// Engagement lines (`[engaged] NAME`) a run of this arm must print: how
    /// a spec proves its new switch's lane ran (evaluator item L3). Lines
    /// for registered knobs come from `knobs.rs` and need not be listed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expect_engaged: Vec<String>,
    /// Controlled host contention (`~hog:N`): the orchestrator runs N busy
    /// threads beside the worker for the whole run. The run is never
    /// quotable; it exists to read handoff tails under load
    /// (`docs/SPEEDUP_PLAYBOOK.md`, "Not pursued").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contention: Option<Contention>,
}

/// A deliberate CPU load beside the measured worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contention {
    pub hog_threads: usize,
}

impl Arm {
    /// The speculation this arm runs under a given workload.
    pub fn effective_speculation(&self, w: &Workload) -> Option<Speculation> {
        self.speculation.clone().or_else(|| w.speculation.clone())
    }
}

/// What a worker runs once the model is open.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    pub prompts: Vec<PromptSource>,
    /// Greedy decode steps after each prefill. A FIXED count that ignores
    /// end-of-turn, so both arms of an A/B do the same decode work even when
    /// a numerics change moves the stop point. 0 measures prefill only.
    #[serde(default)]
    pub decode_tokens: usize,
    /// Prefill chunk size, or `None` for the sequential per-token loop.
    #[serde(default = "default_chunk")]
    pub prefill_chunk: Option<usize>,
    /// KV window at open. Must cover the longest prompt plus `decode_tokens`.
    #[serde(default = "default_context")]
    pub max_context: u32,
    /// Decode through the speculative loop instead of plain greedy steps.
    /// The runtime then owns prefill as well (the speculative loop has its
    /// own), so a speculative record has one combined phase report and no
    /// prefill/decode split of the counters.
    #[serde(default)]
    pub speculation: Option<Speculation>,
    /// An audio install's job. When set, `prompts` and the decode fields do
    /// not apply: the worker opens the audio pipeline and runs this instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioJob>,
}

/// One audio request, run once warm and then measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "task")]
pub enum AudioJob {
    /// Caption and lyrics to a stereo waveform (MiniMax Music 3).
    Music {
        caption: String,
        #[serde(default)]
        lyrics: String,
        duration_s: f64,
        #[serde(default)]
        steps: Option<usize>,
        #[serde(default)]
        seed: u64,
    },
    /// A WAV file to a transcript (Whisper, Moonshine, Qwen3-ASR).
    Transcribe {
        wav: PathBuf,
        #[serde(default)]
        language: Option<String>,
    },
    /// Text to a waveform (Kokoro, MegaTTS3). A MegaTTS3 job clones the
    /// voice in `reference_wav` and needs `consent: true` for that exact
    /// reference, the workspace's voice-cloning rule; without it the arm is
    /// a recorded refusal.
    Speak {
        text: String,
        #[serde(default)]
        voice: Option<String>,
        /// Diffusion steps (MegaTTS3; ignored by Kokoro).
        #[serde(default)]
        steps: Option<usize>,
        #[serde(default)]
        reference_wav: Option<PathBuf>,
        #[serde(default)]
        consent: bool,
        #[serde(default)]
        seed: u64,
    },
}

impl AudioJob {
    pub fn task(&self) -> &'static str {
        match self {
            AudioJob::Music { .. } => "music",
            AudioJob::Transcribe { .. } => "transcribe",
            AudioJob::Speak { .. } => "speak",
        }
    }
}

/// Which drafter, and how deep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Speculation {
    #[serde(default)]
    pub drafter: Drafter,
    /// Tokens drafted per round, or the cap when `adaptive`.
    pub block: usize,
    /// Let the runtime's `AdaptiveDepth` controller move the depth each
    /// round between its floor and `block` (step drafters only).
    #[serde(default)]
    pub adaptive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Drafter {
    /// Whatever drafter the install carries.
    #[default]
    Auto,
    Mtp,
    Dflash,
}

fn default_chunk() -> Option<usize> {
    Some(128)
}

fn default_context() -> u32 {
    4096
}

/// Which part of a run a metric reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Prompt tokens per prefill wall second.
    PrefillTokS,
    /// Decode tokens per decode wall second.
    DecodeTokS,
    /// Prefill wall seconds.
    PrefillS,
    /// Peak `phys_footprint`, bytes.
    PeakBytes,
    /// Mean time to first token over every chat turn after the first, in
    /// seconds: from the generation call to the first token, so it includes
    /// whatever reuse or restore the turn did. Chat workloads only.
    FollowupTtftS,
    /// Wall seconds of the measured audio job.
    AudioRunS,
    /// Audio job wall time over the audio's duration (real-time factor).
    Rtf,
    /// Modeled joules over the measured prefills (IOReport, whole machine;
    /// `energy::ENERGY_MODEL`). Comparable within one experiment only.
    PrefillJ,
    /// Modeled joules over the measured decode steps.
    DecodeJ,
}

impl Metric {
    pub fn direction(self) -> crate::stats::Direction {
        use crate::stats::Direction::*;
        match self {
            Metric::PrefillTokS | Metric::DecodeTokS => HigherIsBetter,
            Metric::PrefillS
            | Metric::PeakBytes
            | Metric::FollowupTtftS
            | Metric::AudioRunS
            | Metric::Rtf
            | Metric::PrefillJ
            | Metric::DecodeJ => LowerIsBetter,
        }
    }

    /// Read this metric from a worker record (`worker::RunRecord` as JSON).
    pub fn read(self, record: &serde_json::Value) -> Option<f64> {
        let totals = &record["totals"];
        match self {
            Metric::PrefillTokS => totals["prefill_tok_s"].as_f64(),
            Metric::DecodeTokS => totals["decode_tok_s"].as_f64(),
            Metric::PrefillS => totals["prefill_s"].as_f64(),
            Metric::PeakBytes => record["peak_footprint_bytes"].as_f64(),
            Metric::FollowupTtftS => totals["followup_ttft_s"].as_f64(),
            Metric::AudioRunS => record["audio"]["run_s"].as_f64(),
            Metric::Rtf => record["audio"]["rtf"].as_f64(),
            Metric::PrefillJ => totals["prefill_j"].as_f64(),
            Metric::DecodeJ => totals["decode_j"].as_f64(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workload_round_trips_with_defaults() {
        let w: Workload = serde_json::from_str(
            r#"{"prompts":[{"protocol":"long-synthesis"},{"text_file":"/x.txt"}],
                "speculation":{"drafter":"mtp","block":2}}"#,
        )
        .unwrap();
        assert_eq!(w.speculation.as_ref().unwrap().drafter, Drafter::Mtp);
        assert_eq!(w.prefill_chunk, Some(128));
        assert_eq!(w.max_context, 4096);
        assert_eq!(w.decode_tokens, 0);
        let back: Workload = serde_json::from_str(&serde_json::to_string(&w).unwrap()).unwrap();
        assert_eq!(back, w);
    }

    #[test]
    fn audio_jobs_parse_by_task() {
        let w: Workload = serde_json::from_str(
            r#"{"prompts":[],"audio":{"task":"music","caption":"lofi","duration_s":2.0,"steps":4}}"#,
        )
        .unwrap();
        assert!(matches!(
            w.audio,
            Some(AudioJob::Music {
                steps: Some(4),
                seed: 0,
                ..
            })
        ));
        let t: AudioJob = serde_json::from_str(r#"{"task":"transcribe","wav":"/a.wav"}"#).unwrap();
        assert!(matches!(t, AudioJob::Transcribe { language: None, .. }));
        assert_eq!(t.task(), "transcribe");
        let s: AudioJob = serde_json::from_str(r#"{"task":"speak","text":"hi"}"#).unwrap();
        assert!(matches!(
            s,
            AudioJob::Speak {
                consent: false,
                steps: None,
                seed: 0,
                ..
            }
        ));
        assert_eq!(s.task(), "speak");
        let r = serde_json::json!({"audio": {"run_s": 2.0, "rtf": 0.5}});
        assert_eq!(Metric::Rtf.read(&r), Some(0.5));
        assert_eq!(Metric::AudioRunS.read(&r), Some(2.0));
    }

    #[test]
    fn arm_open_params_parse() {
        let a: Arm = serde_json::from_str(
            r#"{"name":"s32","open":{"slots":{"fixed":32},"residency":"streamed"},"env":{"TURBOSPARK_X":"1"}}"#,
        )
        .unwrap();
        assert_eq!(a.open.slots, Slots::Fixed(32));
        assert_eq!(a.open.residency, Residency::Streamed);
        assert_eq!(a.env["TURBOSPARK_X"], "1");
    }
}
