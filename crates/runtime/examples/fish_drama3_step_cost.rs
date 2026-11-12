//! Why does a Drama 3 frame cost more backbone time than a text token?
//!
//! Steps one headless trunk two ways and prints the per-step phase breakdown
//! (`TURBOSPARK_PHASES`'s counters) of each:
//!
//! - **text**: a paragraph's own token rows, one `step` each, as a text
//!   decode feeds them (teacher-forced);
//! - **frames**: the backbone rows of a real Drama generation (Metal tower,
//!   seed 0), measured on the decode steps only.
//!
//! Same trunk, slots, residency and code path, so a difference is the rows.
//! The two modes alternate for `rounds` rounds, with a reset between them,
//! and the first `warmup` steps of each are not counted.
//!
//! ```sh
//! cargo run --release -p turbospark-runtime --example fish_drama3_step_cost -- \
//!     <install-dir> [expert-slots=16] [steps=160] [rounds=2]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use audio::tts::fish_drama3::generate::{Backbone, BackboneRow, Flow, GenerateOptions, TRUNK_DIR};
use turbospark_runtime::fish_drama3::{open_drama, TrunkBackbone};
use turbospark_runtime::{ExpertCacheSlots, ExpertResidency, HiddenTrunk, PhaseCounters};

const WARMUP: usize = 16;

const PARAGRAPH: &str = "The harbor was quiet in the early morning, and the fishing boats \
    rocked gently against the wooden piers while gulls circled overhead. An old man sat on \
    a crate near the water, mending a net with slow and careful hands, humming a tune that \
    nobody else remembered. Across the bay, the lighthouse blinked one last time before \
    the sun rose high enough to make it pointless. Children ran along the sea wall, \
    shouting about the crabs they had seen under the rocks, and their mother called after \
    them to stay away from the edge. By noon the market would be loud with buyers and \
    sellers arguing over the price of mackerel, but for now the town held its breath, as \
    if it were listening to the tide come in. A delivery truck rattled down the hill, its \
    driver waving to the baker who was already stacking warm loaves in the window.";

const SENTENCE: &str = "The quick brown fox jumps over the lazy dog, and then it takes a \
    short nap in the warm afternoon sun, dreaming of fields and rivers far away.";

/// Per-step sums over the counted steps.
#[derive(Default)]
struct Cost {
    steps: usize,
    wall_ms: f64,
    phases: Vec<PhaseCounters>,
}

impl Cost {
    fn record(&mut self, before: PhaseCounters, after: PhaseCounters, wall_ms: f64) {
        self.steps += 1;
        self.wall_ms += wall_ms;
        self.phases.push(delta(before, after));
    }

    fn report(&self, label: &str) {
        let n = self.steps.max(1) as f64;
        let ms = |f: fn(&PhaseCounters) -> u64| {
            self.phases.iter().map(|p| f(p) as f64).sum::<f64>() / 1e6 / n
        };
        let requests: u64 = self.phases.iter().map(|p| p.expert_requests).sum();
        let hits: u64 = self.phases.iter().map(|p| p.expert_hits).sum();
        let bytes: u64 = self
            .phases
            .iter()
            .map(|p| p.expert_io_bytes_requested)
            .sum();
        // Physical bytes are measured only under TURBOSPARK_EXPERT_DISK_IO=1.
        let physical: u64 = self.phases.iter().map(|p| p.expert_io_bytes_physical).sum();
        let samples: u64 = self.phases.iter().map(|p| p.expert_io_samples).sum();
        let disk = if samples > 0 {
            format!(
                " | disk {:5.1}% of reads",
                100.0 * physical as f64 / bytes.max(1) as f64
            )
        } else {
            String::new()
        };
        println!(
            "{label:<8} {:>4} steps | wall {:6.2} ms | gpu_wait {:6.2} | final_wait {:5.2} | \
             router {:5.2} | expert_io {:6.2} | bind {:5.2} | pipe_wait {:5.2} | \
             cb1_gpu {:6.2} | routed_gpu {:6.2} | hit {:5.1}% | read {:6.1} MiB/step{disk}",
            self.steps,
            self.wall_ms / n,
            ms(|p| p.gpu_wait_nanos),
            ms(|p| p.final_wait_nanos),
            ms(|p| p.router_nanos),
            ms(|p| p.expert_io_nanos),
            ms(|p| p.bind_nanos),
            ms(|p| p.pipeline_wait_nanos),
            ms(|p| p.cb1_gpu_nanos),
            ms(|p| p.routed_cb_gpu_nanos),
            100.0 * hits as f64 / requests.max(1) as f64,
            bytes as f64 / (1u64 << 20) as f64 / n,
        );
    }
}

fn delta(a: PhaseCounters, b: PhaseCounters) -> PhaseCounters {
    PhaseCounters {
        calls: b.calls - a.calls,
        total_nanos: b.total_nanos - a.total_nanos,
        gpu_wait_nanos: b.gpu_wait_nanos - a.gpu_wait_nanos,
        final_wait_nanos: b.final_wait_nanos - a.final_wait_nanos,
        router_nanos: b.router_nanos - a.router_nanos,
        expert_io_nanos: b.expert_io_nanos - a.expert_io_nanos,
        bind_nanos: b.bind_nanos - a.bind_nanos,
        pipeline_wait_nanos: b.pipeline_wait_nanos - a.pipeline_wait_nanos,
        cb1_gpu_nanos: b.cb1_gpu_nanos - a.cb1_gpu_nanos,
        routed_cb_gpu_nanos: b.routed_cb_gpu_nanos - a.routed_cb_gpu_nanos,
        final_cb_gpu_nanos: b.final_cb_gpu_nanos - a.final_cb_gpu_nanos,
        expert_requests: b.expert_requests - a.expert_requests,
        expert_hits: b.expert_hits - a.expert_hits,
        expert_io_bytes_requested: b.expert_io_bytes_requested - a.expert_io_bytes_requested,
        expert_io_bytes_physical: b.expert_io_bytes_physical - a.expert_io_bytes_physical,
        expert_io_samples: b.expert_io_samples - a.expert_io_samples,
        ..b
    }
}

/// Counts decode steps only, after the first `WARMUP` of them.
struct Measured {
    inner: TrunkBackbone,
    seen: usize,
    cost: Cost,
}

impl Backbone for Measured {
    type Snapshot = <TrunkBackbone as Backbone>::Snapshot;

    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    fn embedding_row(&mut self, token: i32, out: &mut [f32]) -> audio::Result<()> {
        self.inner.embedding_row(token, out)
    }

    fn prefill(&mut self, rows: &[BackboneRow], hidden_out: &mut [f32]) -> audio::Result<()> {
        self.inner.prefill(rows, hidden_out)
    }

    fn step(&mut self, row: &BackboneRow, hidden_out: &mut [f32]) -> audio::Result<()> {
        let before = self.inner.trunk().phase_counters();
        let t = Instant::now();
        let r = self.inner.step(row, hidden_out);
        let wall = t.elapsed().as_secs_f64() * 1e3;
        self.seen += 1;
        if self.seen > WARMUP {
            self.cost
                .record(before, self.inner.trunk().phase_counters(), wall);
        }
        r
    }

    fn snapshot(&mut self) -> audio::Result<Self::Snapshot> {
        self.inner.snapshot()
    }

    fn restore(&mut self, snapshot: &Self::Snapshot) -> audio::Result<()> {
        self.inner.restore(snapshot)
    }

    fn reset(&mut self) {
        self.inner.reset()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let install = PathBuf::from(args.next().ok_or("install dir")?);
    let slots: usize = args.next().map_or(Ok(16), |s| s.parse())?;
    let steps: usize = args.next().map_or(Ok(160), |s| s.parse())?;
    let rounds: usize = args.next().map_or(Ok(2), |s| s.parse())?;

    let context = 4096;
    let mut drama = open_drama(&install, context)?;
    let trunk_dir = install.join(TRUNK_DIR);
    let arch = turbospark_repack::peek_manifest_arch(&trunk_dir)?;
    let trunk = HiddenTrunk::open(
        &trunk_dir,
        arch,
        context,
        ExpertCacheSlots::Fixed(slots),
        ExpertResidency::Streamed,
    )?;
    let mut backbone = Measured {
        inner: TrunkBackbone::new(trunk),
        seen: 0,
        cost: Cost::default(),
    };

    let ids: Vec<i32> = drama
        .tokenizer
        .encode(PARAGRAPH, false)
        .map_err(|e| format!("tokenize: {e}"))?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    if ids.len() < steps + WARMUP {
        return Err(format!(
            "the paragraph is {} tokens, fewer than {} steps plus {WARMUP} warmup",
            ids.len(),
            steps
        )
        .into());
    }
    println!("{slots} expert slots, streamed; {steps} counted steps per mode, {WARMUP} warmup");
    let hidden = backbone.hidden_size();
    let mut out = vec![0.0f32; hidden];
    for round in 0..rounds {
        // Text: the paragraph's own tokens, one step each.
        backbone.reset();
        backbone.seen = 0;
        backbone.cost = Cost::default();
        for &id in &ids[..steps + WARMUP] {
            backbone.step(&BackboneRow::Token(id), &mut out)?;
        }
        backbone.cost.report(&format!("text {round}"));

        // Frames: a real generation's decode steps.
        backbone.reset();
        backbone.seen = 0;
        backbone.cost = Cost::default();
        let options = GenerateOptions {
            seed: 0,
            max_frames: steps + WARMUP,
            ..GenerateOptions::default()
        };
        drama.generate(&mut backbone, &[], SENTENCE, options, |_| Flow::Continue)?;
        backbone.cost.report(&format!("frames {round}"));
    }
    Ok(())
}
