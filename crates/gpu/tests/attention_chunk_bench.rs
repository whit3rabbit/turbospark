//! One-arm matched synthetic GPU attention measurement for batched decode.
//!
//! Each invocation selects exactly one arm with
//! `TURBOSPARK_ATTENTION_BENCH_ARM=batched|serial`. The serial arm encodes M
//! calls to the existing one-row encoder; the batched arm encodes the public
//! multi-row API. Both use the same row-major Q tensor, linear K/V tensors,
//! contiguous query positions, and head shape. One selected-arm warmup is
//! discarded. Each measured repetition is one command buffer, and the test
//! prints one JSON record containing raw per-repetition GPU times and their
//! summary. Run each arm in a separate fresh process, for example:
//!
//! ```sh
//! TURBOSPARK_ATTENTION_BENCH_ARM=batched cargo test -p turbospark-gpu --test attention_chunk_bench --release one_arm_attention_benchmark -- --ignored --nocapture
//! TURBOSPARK_ATTENTION_BENCH_ARM=serial cargo test -p turbospark-gpu --test attention_chunk_bench --release one_arm_attention_benchmark -- --ignored --nocapture
//! ```
//!
//! This synthetic kernel timing is diagnostic only. It does not close the
//! real-model or end-to-end prefill performance gates.

#![cfg(target_os = "macos")]

use std::fmt::Write as _;
use std::process::Command;

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_attention_decode, encode_attention_decode_batch, AttentionScratch,
    BatchAttentionInputContract, BatchAttentionKvFormat, BatchAttentionKvLayout,
    BatchAttentionScratch, BatchAttentionScratchLayout, MetalContext, MAX_BATCH_ROWS,
};

const HEAD_DIM: u32 = 128;
const NUM_Q_HEADS: u32 = 32;
const NUM_KV_HEADS: u32 = 8;
const FIRST_QUERY_POSITION: u32 = 240;
const DEFAULT_LIVE_ROWS: usize = 16;
const DEFAULT_REPETITIONS: usize = 7;
const MAX_REPETITIONS: usize = 100;
const ARM_ENV: &str = "TURBOSPARK_ATTENTION_BENCH_ARM";
const ROWS_ENV: &str = "TURBOSPARK_ATTENTION_BENCH_M";
const REPETITIONS_ENV: &str = "TURBOSPARK_ATTENTION_BENCH_REPETITIONS";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    Batched,
    Serial,
}

impl Arm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "batched" => Ok(Self::Batched),
            "serial" => Ok(Self::Serial),
            _ => Err(format!("{ARM_ENV} must be exactly 'batched' or 'serial'")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Batched => "batched",
            Self::Serial => "serial",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Config {
    arm: Arm,
    live_rows: usize,
    repetitions: usize,
}

impl Config {
    fn parse(arm: &str, rows: Option<&str>, repetitions: Option<&str>) -> Result<Self, String> {
        let arm = Arm::parse(arm)?;
        let live_rows = match rows {
            Some(value) => value
                .parse::<usize>()
                .map_err(|_| format!("{ROWS_ENV} must be an integer in 1..={MAX_BATCH_ROWS}"))?,
            None => DEFAULT_LIVE_ROWS,
        };
        if !(1..=MAX_BATCH_ROWS).contains(&live_rows) {
            return Err(format!(
                "{ROWS_ENV} must be an integer in 1..={MAX_BATCH_ROWS}"
            ));
        }

        let repetitions = match repetitions {
            Some(value) => value.parse::<usize>().map_err(|_| {
                format!("{REPETITIONS_ENV} must be an integer in 2..={MAX_REPETITIONS}")
            })?,
            None => DEFAULT_REPETITIONS,
        };
        if !(2..=MAX_REPETITIONS).contains(&repetitions) {
            return Err(format!(
                "{REPETITIONS_ENV} must be an integer in 2..={MAX_REPETITIONS}"
            ));
        }

        Ok(Self {
            arm,
            live_rows,
            repetitions,
        })
    }

    fn from_env() -> Result<Self, String> {
        let arm = std::env::var(ARM_ENV).map_err(|_| format!("set {ARM_ENV}=batched|serial"))?;
        let rows = std::env::var(ROWS_ENV).ok();
        let repetitions = std::env::var(REPETITIONS_ENV).ok();
        Self::parse(&arm, rows.as_deref(), repetitions.as_deref())
    }
}

struct Case {
    q: metal::Buffer,
    k: metal::Buffer,
    v: metal::Buffer,
    output: metal::Buffer,
    scratch: ArmScratch,
    row_positions: Vec<u32>,
    q_row_offset_bytes: u64,
    output_row_offset_bytes: u64,
    scale: f32,
}

enum ArmScratch {
    Batched(BatchAttentionScratch),
    Serial(Vec<AttentionScratch>),
}

impl Case {
    fn new(context: &MetalContext, config: Config) -> Self {
        let layout = BatchAttentionScratchLayout::new(NUM_Q_HEADS as usize, HEAD_DIM as usize)
            .expect("fixed benchmark shape has a valid batch scratch layout");
        let capacity = layout.capacity();
        assert_eq!(capacity, MAX_BATCH_ROWS);
        let q_width = NUM_Q_HEADS as usize * HEAD_DIM as usize;
        let kv_width = NUM_KV_HEADS as usize * HEAD_DIM as usize;
        let max_seq_len = FIRST_QUERY_POSITION as usize + config.live_rows;

        // Include every Q row in the fixed-capacity allocation. Only the
        // selected live prefix is read by either arm.
        let q: Vec<f16> = (0..capacity * q_width)
            .map(|index| f16::from_f32(((index as f32) * 0.019).sin() * 0.5))
            .collect();
        let k: Vec<f16> = (0..max_seq_len * kv_width)
            .map(|index| f16::from_f32(((index as f32) * 0.0007).cos() * 0.3))
            .collect();
        let v: Vec<f16> = (0..max_seq_len * kv_width)
            .map(|index| f16::from_f32(0.7 + ((index as f32) * 0.0011).sin() * 0.3))
            .collect();

        let row_positions = (0..config.live_rows)
            .map(|row| FIRST_QUERY_POSITION + row as u32)
            .collect();
        let q_row_offset_bytes = (q_width * std::mem::size_of::<f16>()) as u64;
        let output_row_offset_bytes = q_row_offset_bytes;
        let scratch = match config.arm {
            Arm::Batched => ArmScratch::Batched(
                BatchAttentionScratch::new(context, layout)
                    .expect("fixed benchmark batch scratch allocation"),
            ),
            Arm::Serial => ArmScratch::Serial(
                (0..config.live_rows)
                    .map(|_| AttentionScratch::new(context, NUM_Q_HEADS, HEAD_DIM))
                    .collect(),
            ),
        };

        Self {
            q: context.new_buffer_with_data(&to_le_bytes(&q)),
            k: context.new_buffer_with_data(&to_le_bytes(&k)),
            v: context.new_buffer_with_data(&to_le_bytes(&v)),
            output: context.new_output_buffer(layout.output_buffer_bytes() as u64),
            scratch,
            row_positions,
            q_row_offset_bytes,
            output_row_offset_bytes,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        }
    }
}

fn to_le_bytes(values: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn linear_fp16_contract() -> BatchAttentionInputContract {
    BatchAttentionInputContract {
        kv_layout: BatchAttentionKvLayout::Linear,
        kv_format: BatchAttentionKvFormat::Fp16,
        kv_start: 0,
        ring_capacity: 0,
        sink_count: 0,
    }
}

fn run_selected_arm(context: &mut MetalContext, case: &Case, config: Config) -> f64 {
    let pass = context.begin_pass();
    match config.arm {
        Arm::Batched => {
            let ArmScratch::Batched(scratch) = &case.scratch else {
                panic!("selected batched arm has batched scratch");
            };
            encode_attention_decode_batch(
                context,
                &pass,
                (&case.q, 0),
                &case.k,
                &case.v,
                scratch,
                (&case.output, 0),
                linear_fp16_contract(),
                FIRST_QUERY_POSITION,
                config.live_rows,
                HEAD_DIM,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                case.scale,
            )
            .expect("batched attention encoder accepts the matched linear FP16 input");
        }
        Arm::Serial => {
            let ArmScratch::Serial(scratch) = &case.scratch else {
                panic!("selected serial arm has one-row scratch");
            };
            for (row, &position) in case.row_positions.iter().enumerate() {
                let seq_len = position + 1;
                encode_attention_decode(
                    context,
                    &pass,
                    (&case.q, row as u64 * case.q_row_offset_bytes),
                    &case.k,
                    &case.v,
                    &scratch[row],
                    (&case.output, row as u64 * case.output_row_offset_bytes),
                    HEAD_DIM,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    seq_len,
                    0,
                    0,
                    case.scale,
                    None,
                )
                .expect("existing one-row attention encoder accepts the matched input");
            }
        }
    }
    pass.commit_and_wait_with_gpu_time()
}

#[derive(Debug)]
struct TimingRecord {
    arm: Arm,
    live_rows: usize,
    row_positions: Vec<u32>,
    source_revision: String,
    benchmark_source_dirty: bool,
    release_mode: &'static str,
    metal_device: String,
    samples_seconds: Vec<f64>,
}

impl TimingRecord {
    fn json_line(&self) -> String {
        let mut out = String::new();
        let total_seconds: f64 = self.samples_seconds.iter().sum();
        let mut sorted = self.samples_seconds.clone();
        sorted.sort_by(f64::total_cmp);
        let median_seconds = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        let min_seconds = sorted.first().copied().unwrap_or(0.0);
        let max_seconds = sorted.last().copied().unwrap_or(0.0);
        let _ = write!(
            out,
            "{{\"arm\":\"{}\",\"M\":{},\"row_positions\":[",
            self.arm.as_str(),
            self.live_rows
        );
        for (index, position) in self.row_positions.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let _ = write!(out, "{position}");
        }
        let _ = write!(
            out,
            "],\"head_shape\":{{\"head_dim\":{},\"num_q_heads\":{},\"num_kv_heads\":{}}},\"source_revision\":\"{}\",\"benchmark_source_dirty\":{},\"release_mode\":\"{}\",\"metal_device\":\"{}\",\"repetitions\":{},\"sample_count\":{},\"elapsed_seconds_total\":{:.9},\"elapsed_seconds_per_repetition_median\":{:.9},\"elapsed_seconds_per_repetition_range\":[{:.9},{:.9}],\"elapsed_seconds_per_repetition_samples\":[",
            HEAD_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            json_escape(&self.source_revision),
            self.benchmark_source_dirty,
            self.release_mode,
            json_escape(&self.metal_device),
            self.samples_seconds.len(),
            self.samples_seconds.len(),
            total_seconds,
            median_seconds,
            min_seconds,
            max_seconds,
        );
        for (index, sample) in self.samples_seconds.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let _ = write!(out, "{sample:.9}");
        }
        out.push_str("]}");
        out
    }
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn git_output(args: &[&str]) -> String {
    let output = Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR")])
        .args(args)
        .output()
        .expect("git is available to identify the benchmark source");
    assert!(output.status.success(), "git command failed: {args:?}");
    String::from_utf8(output.stdout)
        .expect("git metadata is UTF-8")
        .trim()
        .to_owned()
}

fn record_metadata(context: &MetalContext) -> (String, bool, String) {
    let revision = git_output(&["rev-parse", "HEAD"]);
    let benchmark_source_dirty = !git_output(&[
        "status",
        "--porcelain",
        "--",
        "tests/attention_chunk_bench.rs",
    ])
    .is_empty();
    let device = context.device().name().to_owned();
    (revision, benchmark_source_dirty, device)
}

/// Each process measures only the arm selected by `TURBOSPARK_ATTENTION_BENCH_ARM`.
#[test]
#[ignore = "one-arm performance measurement on a real Metal device; run explicitly"]
fn one_arm_attention_benchmark() {
    let config = Config::from_env().unwrap_or_else(|error| panic!("{error}"));
    let mut context = MetalContext::new().expect("Metal device");
    let case = Case::new(&context, config);

    // Compile/warm only the selected arm, and discard this GPU timing sample.
    autorelease_pool(|| {
        let _discarded_warmup_seconds = run_selected_arm(&mut context, &case, config);
    });

    let mut samples_seconds = Vec::with_capacity(config.repetitions);
    for _ in 0..config.repetitions {
        let elapsed = autorelease_pool(|| run_selected_arm(&mut context, &case, config));
        samples_seconds.push(elapsed);
    }
    let (source_revision, benchmark_source_dirty, metal_device) = record_metadata(&context);
    let record = TimingRecord {
        arm: config.arm,
        live_rows: config.live_rows,
        row_positions: case.row_positions,
        source_revision,
        benchmark_source_dirty,
        release_mode: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        metal_device,
        samples_seconds,
    };
    println!("{}", record.json_line());
}

#[cfg(test)]
mod tests {
    use super::{Arm, Config, TimingRecord};

    #[test]
    fn arm_selection_accepts_one_arm_only() {
        assert_eq!(Arm::parse("batched"), Ok(Arm::Batched));
        assert_eq!(Arm::parse("serial"), Ok(Arm::Serial));
        assert!(Arm::parse("both").is_err());
        assert!(Arm::parse("").is_err());
    }

    #[test]
    fn configuration_defaults_and_bounds_batch_size_and_repetitions() {
        let defaults = Config::parse("batched", None, None).expect("default config");
        assert_eq!(defaults.live_rows, 16);
        assert_eq!(defaults.repetitions, 7);

        let configured = Config::parse("serial", Some("3"), Some("5")).expect("custom config");
        assert_eq!(configured.live_rows, 3);
        assert_eq!(configured.repetitions, 5);

        for rows in ["0", "17", "nope"] {
            assert!(Config::parse("batched", Some(rows), None).is_err());
        }
        for repetitions in ["0", "1", "101", "nope"] {
            assert!(Config::parse("batched", None, Some(repetitions)).is_err());
        }
    }

    #[test]
    fn timing_record_is_one_json_line_with_raw_samples_and_escaped_strings() {
        let record = TimingRecord {
            arm: Arm::Batched,
            live_rows: 2,
            row_positions: vec![240, 241],
            source_revision: "rev\"x".to_owned(),
            benchmark_source_dirty: true,
            release_mode: "release",
            metal_device: "GPU\\name".to_owned(),
            samples_seconds: vec![0.004, 0.002, 0.006],
        };
        let json = record.json_line();
        assert!(!json.contains('\n'));
        assert!(json.contains("\"arm\":\"batched\""));
        assert!(json.contains("\"M\":2"));
        assert!(json.contains("\"row_positions\":[240,241]"));
        assert!(json.contains("\"source_revision\":\"rev\\\"x\""));
        assert!(json.contains("\"metal_device\":\"GPU\\\\name\""));
        assert!(json.contains("\"repetitions\":3,\"sample_count\":3"));
        assert!(json.contains("\"elapsed_seconds_per_repetition_median\":0.004000000"));
        assert!(json.contains("\"elapsed_seconds_per_repetition_range\":[0.002000000,0.006000000]"));
        assert!(json.contains(
            "\"elapsed_seconds_per_repetition_samples\":[0.004000000,0.002000000,0.006000000]"
        ));
        assert!(json.ends_with('}'));
    }
}
