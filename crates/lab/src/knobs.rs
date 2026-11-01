//! The runtime switches tslab knows about, in one typed table.
//!
//! There is no code-level registry in the runtime: each knob is an env read
//! at its use site, captured at open or cached in a `OnceLock`. This table
//! does not replace those reads. It records what a measurement needs to know
//! about each one: its values, how it is read (which decides whether an A/B
//! needs a fresh process, and tslab always uses one), what exactness a change
//! of value promises, and whether a run with it set may be quoted.
//!
//! `tests/knob_registry.rs` holds the table to the tree: every env literal
//! must still appear in `crates/*/src`, and the doc each entry names must
//! mention it. A knob the runtime renamed or removed fails there instead of
//! becoming an A/B arm that silently measures the default twice.
//!
//! A NEW switch under test (the autoresearch loop's `knob_env`) does not need
//! a row: spec arms may set any `TURBOSPARK_*` variable. Rows are for
//! switches that outlive one experiment.

use serde::Serialize;

/// The slot counts the runtime accepts (`--expert-cache-slots`).
pub use foundation::runtime_config::ALLOWED_CACHE_SLOTS as ALLOWED_SLOTS;

/// How the runtime reads the knob. tslab gives every run a fresh process, so
/// this is information for the reader, not a branch in the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadAt {
    /// Captured into the runner at open (a pub setter usually exists too).
    Open,
    /// A process-wide `OnceLock`: changing it needs a new process.
    ProcessCached,
    /// Read again on every call (diagnostics).
    PerCall,
}

/// What switching the knob may do to output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Exactness {
    /// Output bits are identical across values; the gate checks digests.
    BitExact,
    /// Output moves within a distribution bound; the gate runs the KL probe.
    KlBounded,
    /// Changes what is computed or measured (residency, slot policy that
    /// reorders reduction, diagnostics); compare within one value only.
    NotComparable,
}

#[derive(Debug, Clone, Serialize)]
pub struct Knob {
    /// Short name used on the `tslab` command line.
    pub name: &'static str,
    pub env: &'static str,
    /// Accepted values, with the default first. `"<unset>"` means the
    /// variable is absent.
    pub values: &'static [&'static str],
    pub read_at: ReadAt,
    pub exactness: Exactness,
    /// Families the knob changes anything for (`ModelFamily::as_str`). Empty
    /// means every family. A hint only: applicability is decided by trying
    /// the arm and recording a refusal.
    pub families: &'static [&'static str],
    /// False when setting the knob perturbs timing (profilers, extra
    /// syscalls): such runs are kept but never quoted.
    pub quotable: bool,
    /// A repository doc that describes the knob (checked by the registry test).
    pub doc: &'static str,
    /// What the run's stderr must (or must not) say when the arm sets the
    /// knob, under `TURBOSPARK_ENGAGEMENT=1` (`foundation::engaged!`). Empty:
    /// the lane prints nothing the lab checks.
    pub engagement: &'static [Engagement],
}

/// One engagement expectation: when an arm sets the knob to `value` (`*`
/// for any value but unset), the `[engaged] line` must appear (`present`)
/// or must not (`!present`, for switches that turn a default lane off).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Engagement {
    pub value: &'static str,
    pub line: &'static str,
    pub present: bool,
}

const fn on(value: &'static str, line: &'static str) -> Engagement {
    Engagement {
        value,
        line,
        present: true,
    }
}

const fn off(value: &'static str, line: &'static str) -> Engagement {
    Engagement {
        value,
        line,
        present: false,
    }
}

pub const KNOBS: &[Knob] = &[
    Knob {
        name: "batched_gemv",
        env: "TURBOSPARK_BATCHED_GEMV",
        values: &["<unset>", "1", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        // Read by every flow's batched prefill (`real_forward_open.rs`), so
        // the list is every family with a prefill.rs that takes the arm.
        families: &[
            "qwen35",
            "qwen36",
            "gemma4",
            "llama",
            "qwen3moe",
            "qwen3",
            "qwen2",
            "minimax_m2",
            "qwen3_vl",
            "flint",
            "gptOss",
            "museGlimmer",
            "qwen4exp",
            "spark2_5",
        ],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "prefill_steel",
        env: "TURBOSPARK_PREFILL_STEEL",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["qwen35", "qwen36", "lfm3"],
        quotable: true,
        doc: "docs/BATCHED_PREFILL.md",
        engagement: &[off("0", "prefill_steel")],
    },
    Knob {
        name: "prefill_small_n",
        env: "TURBOSPARK_PREFILL_SMALL_N",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/BATCHED_PREFILL.md",
        engagement: &[off("0", "prefill_small_n")],
    },
    Knob {
        name: "prefill_rows_attn",
        env: "TURBOSPARK_PREFILL_ROWS_ATTN",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/SPEEDUP_PLAYBOOK.md",
        engagement: &[off("0", "prefill_rows_attn")],
    },
    Knob {
        name: "qwen_batched_row_ops",
        env: "TURBOSPARK_QWEN_BATCHED_ROW_OPS",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35", "qwen36", "lfm3"],
        quotable: true,
        doc: "docs/SPEEDUP_PLAYBOOK.md",
        engagement: &[],
    },
    Knob {
        name: "qwen_fused_ffn",
        env: "TURBOSPARK_QWEN_FUSED_FFN",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35", "qwen36"],
        quotable: true,
        doc: "docs/KAPITSA_KERNELS.md",
        engagement: &[off("0", "qwen_fused_ffn")],
    },
    Knob {
        name: "gdn_lazy_tape",
        env: "TURBOSPARK_GDN_LAZY_TAPE",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35", "qwen36"],
        quotable: true,
        doc: "docs/KAPITSA_KERNELS.md",
        engagement: &[off("0", "gdn_lazy_tape")],
    },
    Knob {
        // Off is the token-at-a-time walk; on takes the chunked driver's
        // default arm (Steel), so the prompt's numerics move with it.
        name: "dflash_chunked_prefill",
        env: "TURBOSPARK_DFLASH_CHUNKED_PREFILL",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/DFLASH2.md",
        engagement: &[off("0", "dflash_chunked_prefill")],
    },
    Knob {
        // Lossless by construction (every committed token is the target's
        // greedy pick); the stream equals the chain's whenever both equal
        // sequential decode.
        name: "dflash_tree",
        env: "TURBOSPARK_DFLASH_TREE",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/KAPITSA_KERNELS.md",
        engagement: &[off("0", "dflash_tree")],
    },
    Knob {
        // `1` runs decode and verify on the 8-row tile (bit-exact against
        // each other, not against the GEMV), `verify` only the batched calls.
        // It reaches every family through the generic INT4 dispatchers.
        name: "int4_mma8",
        env: "TURBOSPARK_INT4_MMA8",
        values: &["<unset>", "1", "verify"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &[],
        quotable: true,
        doc: "docs/INT4_SMALL_M_MMA.md",
        engagement: &[on("*", "int4_mma8")],
    },
    Knob {
        // Which shapes the tile serves under `int4_mma8`; meaningless without it.
        name: "int4_mma8_min_rows",
        env: "TURBOSPARK_INT4_MMA8_MIN_ROWS",
        values: &["<unset>", "1", "1024"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &[],
        quotable: true,
        doc: "docs/INT4_SMALL_M_MMA.md",
        engagement: &[],
    },
    Knob {
        // The per-family default lives at the runner's scratch
        // (`real_forward_types.rs`): the dense qwen GDN family and, since
        // the 2026-10-29 flip (mistral7b +5.7%, qwen2 +3.9%, qwen3vl
        // +19.8%, spark25 +3.9%), dense Llama, Qwen2Dense, Qwen3Dense,
        // Qwen3Vl and Spark25 default to gqa. Qwen3Moe (+3.4%, but its gqa
        // arm decoded different tokens), Mixtral and the rest default to
        // barrier, and the encoder falls back to barrier for any shape
        // `attention_decode_gqa_supported` rejects. This env overrides
        // process-wide either way.
        name: "attn_decode_kernel",
        env: "TURBOSPARK_ATTN_DECODE_KERNEL",
        values: &["<unset>", "barrier", "gqa"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &[],
        quotable: true,
        doc: "docs/SPEEDUP_PLAYBOOK.md",
        engagement: &[
            on("barrier", "attn_decode_barrier"),
            off("barrier", "attn_decode_gqa"),
            on("gqa", "attn_decode_gqa"),
        ],
    },
    Knob {
        // Default ON: `=0` keeps the delta kernels on the runtime-Dk loop, and
        // the baked build's line must then be absent.
        name: "gdn_dk128",
        env: "TURBOSPARK_GDN_DK128",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35", "qwen36"],
        quotable: true,
        doc: "docs/KAPITSA_KERNELS.md",
        engagement: &[off("0", "gdn_dk128")],
    },
    Knob {
        // Default ON: `=0` puts every vision tower projection back on the
        // per-output `vision_matmul_fp16`, and the GEMM's line must then be
        // absent. Different summation order, so KL-bounded, not bit-exact.
        name: "vision_mma",
        env: "TURBOSPARK_VISION_MMA",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &["qwen35", "qwen36", "qwen3_vl"],
        quotable: true,
        doc: "docs/VISION.md",
        engagement: &[off("0", "vision_mma")],
    },
    Knob {
        name: "routed_batch",
        env: "TURBOSPARK_ROUTED_BATCH",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        // Every flow with a routed half reads it; Flint refuses it by name.
        families: &[
            "gemma4",
            "gptOss",
            "qwen3moe",
            "llama",
            "minimax_m2",
            "qwen36",
            "qwen4exp",
            "flint",
        ],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[on("1", "routed_batch")],
    },
    Knob {
        // The routed batch's read-ahead: the next sub-batch's pread under
        // the GPU running the previous one. Slot choice only, so exact. The
        // line prints only when a sub-batch actually read ahead, so an arm
        // whose slots never held two unions (32 and below) reads as dirty.
        name: "routed_batch_overlap",
        env: "TURBOSPARK_ROUTED_BATCH_OVERLAP",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen36"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[on("1", "routed_batch_overlap")],
    },
    Knob {
        name: "routed_pipeline",
        env: "TURBOSPARK_ROUTED_PIPELINE",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "shared_cb",
        env: "TURBOSPARK_SHARED_CB",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[
            "gemma4",
            "gptOss",
            "qwen4exp",
            "flint",
            "llama",
            "qwen3moe",
            "minimax_m2",
        ],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "mtp_batched_prime",
        env: "TURBOSPARK_MTP_BATCHED_PRIME",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/MTP.md",
        engagement: &[],
    },
    Knob {
        name: "qwen_moe_chunked",
        env: "TURBOSPARK_QWEN_MOE_CHUNKED",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen36"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "expert_cache_decay",
        env: "TURBOSPARK_EXPERT_CACHE_DECAY",
        values: &["<unset>", "16"],
        read_at: ReadAt::Open,
        exactness: Exactness::NotComparable,
        families: &[],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "ane_prefill",
        env: "TURBOSPARK_ANE_PREFILL",
        values: &["<unset>", "split", "break", "auto"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/SPEEDUP_PLAYBOOK.md",
        // Missing programs turn the split off with one warning and prefill
        // runs on Steel; without these lines that run reads as the ANE arm.
        // `auto` is left out: it legitimately stays on Steel below M4.
        engagement: &[on("split", "ane_prefill"), on("break", "ane_prefill_break")],
    },
    Knob {
        // Where the split's Core ML programs live. A path, not a lane: it
        // only matters beside `ane_prefill`.
        name: "ane_prefill_dir",
        env: "TURBOSPARK_ANE_PREFILL_DIR",
        values: &["<unset>"],
        read_at: ReadAt::Open,
        exactness: Exactness::NotComparable,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Lets the split run with a DFlash2 drafter open. Set beside
        // `ane_prefill`, so the line it holds is the split's own: a drafter
        // arm whose prefill stayed on Steel is dirty.
        name: "ane_prefill_dflash",
        env: "TURBOSPARK_ANE_PREFILL_DFLASH",
        values: &["<unset>", "1"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[on("1", "ane_prefill")],
    },
    Knob {
        name: "metal_kernel_warmup",
        env: "TURBOSPARK_METAL_KERNEL_WARMUP",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        // Read at every open (`real_forward_open.rs`), not only the MoE one.
        families: &[],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Counters and host wall timers around phases the loop already
        // waits on; it adds no GPU sync, so runs with it stay quotable.
        name: "spec_stats",
        env: "TURBOSPARK_SPEC_STATS",
        values: &["<unset>", "1"],
        read_at: ReadAt::PerCall,
        exactness: Exactness::BitExact,
        families: &["qwen35", "qwen36"],
        quotable: true,
        doc: "docs/CLI.md",
        engagement: &[],
    },
    Knob {
        // Only changes which depth each round proposes; the speculative loop
        // stays lossless, so output is identical at any setting.
        name: "adaptive_depth",
        env: "TURBOSPARK_ADAPTIVE_DEPTH",
        values: &["<unset>", "8,6,4,2"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Only changes how much of a chat follow-up the runtime continues
        // from instead of prefilling; the turn snapshot gate holds tokens
        // identical to a full prefill. Measure it with a chat: workload and
        // the followup_ttft_s metric; on single prompts it never fires.
        name: "gdn_turn_snapshot",
        env: "TURBOSPARK_GDN_TURN_SNAPSHOT",
        values: &["<unset>", "1"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // L1: N exact multiplies by 1.0 on each dense FFN's chain. Bit-exact,
        // but the runs exist to be slower: fit the slope, never quote one.
        name: "dispatch_probe",
        env: "TURBOSPARK_DISPATCH_PROBE",
        values: &["<unset>", "4", "8"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35", "llama", "qwen3", "qwen2", "qwen3_vl"],
        quotable: false,
        doc: "docs/MLX_SERVE_KERNELS.md",
        engagement: &[on("*", "dispatch_probe")],
    },
    Knob {
        // L4: skip blocks in the decode trunk, a comma list of attn, mlp, gdn.
        // Output is wrong by design; the lines are checked per block.
        name: "standin",
        env: "TURBOSPARK_STANDIN",
        values: &["<unset>", "attn", "mlp", "gdn"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::NotComparable,
        families: &["qwen35", "llama", "qwen3", "qwen2", "qwen3_vl"],
        quotable: false,
        doc: "docs/MLX_SERVE_KERNELS.md",
        engagement: &[],
    },
    Knob {
        // Audio: Whisper opens on Metal unless this says cpu. A CPU arm
        // measures another implementation, so compare within one value.
        name: "whisper_device",
        env: "TURBOSPARK_WHISPER_DEVICE",
        values: &["<unset>", "cpu"],
        read_at: ReadAt::Open,
        exactness: Exactness::NotComparable,
        families: &["whisper"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Audio: Moonshine's device. CPU is its default until its Metal peak
        // qualifies (docs/SPEEDUP_PLAYBOOK.md, item 3).
        name: "moonshine_device",
        env: "TURBOSPARK_MOONSHINE_DEVICE",
        values: &["<unset>", "metal", "cpu"],
        read_at: ReadAt::Open,
        exactness: Exactness::NotComparable,
        families: &["moonshine"],
        quotable: true,
        doc: "crates/audio/src/stt/moonshine/README.md",
        engagement: &[],
    },
    Knob {
        // Flint's layer-batched chunk driver, the default since the clean
        // paired win (finding flint.chunked-prefill-clean-2026-10-28,
        // +38.2% prefill bit-exact); "0" restores the walk. Byte-identical to
        // the walk on the real install (`docs/FLINT.md`, the lab's bit-exact
        // gate).
        name: "flint_chunked_prefill",
        env: "TURBOSPARK_FLINT_CHUNKED_PREFILL",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["flint"],
        quotable: true,
        doc: "docs/FLINT.md",
        engagement: &[off("0", "flint_chunked_prefill")],
    },
    Knob {
        // LFM3's M-row chunk driver, default on for INT4 installs; `0` is the
        // per-token loop behind the same entry point. KL-bounded because the
        // default arm takes Steel and the rows attention: with
        // `prefill_steel=0` and `prefill_rows_attn=0` it is bit-exact.
        name: "lfm3_chunked_prefill",
        env: "TURBOSPARK_LFM3_CHUNKED_PREFILL",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &["lfm3"],
        quotable: true,
        doc: "docs/LFM3_PHASE0.md",
        engagement: &[off("0", "lfm3_chunked_prefill")],
    },
    Knob {
        // Decode's gate+up+silu as one INT4 GEMV; `0` keeps the chain. The
        // Q8_0 install never takes it, so its line is absent there.
        name: "lfm3_fused_ffn",
        env: "TURBOSPARK_LFM3_FUSED_FFN",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["lfm3"],
        quotable: true,
        doc: "docs/LFM3_PHASE0.md",
        engagement: &[off("0", "lfm3_fused_ffn")],
    },
    Knob {
        // The DFlash twin of `dflash_chunked_prefill`: off is the walk.
        name: "mtp_chunked_prefill",
        env: "TURBOSPARK_MTP_CHUNKED_PREFILL",
        values: &["<unset>", "0"],
        read_at: ReadAt::PerCall,
        exactness: Exactness::KlBounded,
        families: &["qwen35"],
        quotable: true,
        doc: "docs/MTP.md",
        engagement: &[off("0", "mtp_chunked_prefill")],
    },
    Knob {
        // Submits the shared expert on its own command buffer ahead of the
        // streamed reads. Scheduling only; declared KL-bounded because the
        // two command buffers can reorder the residual add.
        name: "qwen_shared_read_overlap",
        env: "TURBOSPARK_QWEN_SHARED_READ_OVERLAP",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["qwen36"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Router-guess prefetch width for the next layer's experts. A
        // non-integer value panics at open by design.
        name: "qwen_prefetch",
        env: "TURBOSPARK_QWEN_PREFETCH",
        values: &["<unset>", "2", "4", "8"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen36"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "mapped_demand_prep",
        env: "TURBOSPARK_MAPPED_DEMAND_PREP",
        values: &["<unset>", "advice", "touch"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen36"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Forces dense attention past the sparse indexer's budget.
        name: "qsa_force_dense",
        env: "TURBOSPARK_QSA_FORCE_DENSE",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["qwen4exp"],
        quotable: true,
        doc: "docs/QWEN4_EXP.md",
        engagement: &[],
    },
    Knob {
        name: "read_qos",
        env: "TURBOSPARK_READ_QOS",
        values: &["<unset>", "utility"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Compiles every shader without fast math: numerics move everywhere.
        name: "metal_precise_math",
        env: "TURBOSPARK_METAL_PRECISE_MATH",
        values: &["<unset>", "1"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &[],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // One-layer-ahead router probe: a measurement, never a lane.
        name: "pilot_probe",
        env: "TURBOSPARK_PILOT_PROBE",
        values: &["<unset>", "1", "self"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen36"],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    // Diagnostics and captures. Setting any of these keeps the run but
    // makes it unquotable: they add I/O or host work inside the timed loop.
    Knob {
        name: "phases",
        env: "TURBOSPARK_PHASES",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/BENCHMARKING.md",
        engagement: &[],
    },
    Knob {
        name: "router_hist",
        env: "TURBOSPARK_ROUTER_HIST",
        values: &["<unset>", "<path>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Per-batch router capture; adds host work inside the timed loop.
        name: "router_batches",
        env: "TURBOSPARK_ROUTER_BATCHES",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "router_trace",
        env: "TURBOSPARK_ROUTER_TRACE",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/CLI.md",
        engagement: &[],
    },
    Knob {
        name: "ffn_hist",
        env: "TURBOSPARK_FFN_HIST",
        values: &["<unset>", "<path>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "resid_capture",
        env: "TURBOSPARK_RESID_CAPTURE",
        values: &["<unset>", "<path>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "cb1_trace",
        env: "TURBOSPARK_CB1_TRACE",
        values: &["<unset>", "<path>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen4exp"],
        quotable: false,
        doc: "docs/EXPERT_ROUTING.md",
        engagement: &[],
    },
    Knob {
        // Wires (`mlock`) the streamed slot cache at open. Unset is the
        // `auto` request, which wires qwen4exp at twice the slot budget and
        // leaves every other family unwired; `1` forces it on any streamed
        // family and `0` turns it off (QWEN4_EXP.md, "Wired expert slots").
        name: "expert_slots_wired",
        env: "TURBOSPARK_EXPERT_SLOTS_WIRED",
        values: &["<unset>", "1", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen4exp"],
        quotable: true,
        doc: "docs/ENV.md",
        engagement: &[
            on("1", "expert_slots_wired"),
            off("0", "expert_slots_wired"),
        ],
    },
    Knob {
        // The DFlash2 drafter's context features: per decoded position, the
        // hyper-connection stream mean after the tapped layers. Forces the
        // token-by-token prefill, so never a timing arm.
        name: "qwen4_tap_file",
        env: "TURBOSPARK_QWEN4_TAP_FILE",
        values: &["<unset>", "<path>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen4exp"],
        quotable: false,
        doc: "docs/QWEN4_EXP.md",
        engagement: &[],
    },
    Knob {
        name: "mtp_dump",
        env: "TURBOSPARK_MTP_DUMP",
        values: &["<unset>", "<path>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen35"],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "flint_debug",
        env: "TURBOSPARK_FLINT_DEBUG",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["flint"],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "flint_router_dump",
        env: "TURBOSPARK_FLINT_ROUTER_DUMP",
        values: &["<unset>", "<path>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["flint"],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "ane_prefill_stats",
        env: "TURBOSPARK_ANE_PREFILL_STATS",
        values: &["<unset>", "1"],
        read_at: ReadAt::PerCall,
        exactness: Exactness::BitExact,
        families: &["qwen35"],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // Fails one prediction to exercise the Steel redo. A fault
        // injection, never a timing arm.
        name: "ane_prefill_fault",
        env: "TURBOSPARK_ANE_PREFILL_FAULT",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::NotComparable,
        families: &["qwen35"],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        // F_NOCACHE on expert reads: a cold-disk measurement condition.
        name: "expert_nocache",
        env: "TURBOSPARK_EXPERT_NOCACHE",
        values: &["<unset>", "1"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "router_batch_limit",
        env: "TURBOSPARK_ROUTER_BATCH_LIMIT",
        values: &["<unset>"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "whisper_profile",
        env: "TURBOSPARK_WHISPER_PROFILE",
        values: &["<unset>", "1"],
        read_at: ReadAt::PerCall,
        exactness: Exactness::BitExact,
        families: &["whisper"],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "moonshine_profile",
        env: "TURBOSPARK_MOONSHINE_PROFILE",
        values: &["<unset>", "1"],
        read_at: ReadAt::PerCall,
        exactness: Exactness::BitExact,
        families: &["moonshine"],
        quotable: false,
        doc: "docs/BENCHMARKING.md",
        engagement: &[],
    },
    Knob {
        name: "qwen3_asr_profile",
        env: "TURBOSPARK_QWEN3_ASR_PROFILE",
        values: &["<unset>", "1"],
        read_at: ReadAt::PerCall,
        exactness: Exactness::BitExact,
        families: &["qwen3_asr"],
        quotable: false,
        doc: "docs/BENCHMARKING.md",
        engagement: &[],
    },
    Knob {
        name: "dispatch_profile",
        env: "TURBOSPARK_DISPATCH_PROFILE",
        values: &["<unset>", "1"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
    Knob {
        name: "expert_disk_io",
        env: "TURBOSPARK_EXPERT_DISK_IO",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &[],
        quotable: false,
        doc: "docs/ENV.md",
        engagement: &[],
    },
];

/// How a lane is compared with the path it replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "how", rename_all = "snake_case")]
pub enum Compare {
    /// Two arms of one bench in one experiment.
    Arms {
        /// The path the lane replaces.
        baseline: &'static str,
        /// The lane's own arm.
        lane: &'static str,
    },
    /// One arm timed in two `tslab kernel` processes that differ in one
    /// environment assignment, because the gpu crate reads the switch once
    /// per process and a bench calls the kernels directly. `tslab project
    /// --default-from A --mode-from B` reads the pair.
    Process {
        arm: &'static str,
        /// `VAR=value` to put on the process that times the baseline.
        baseline_env: &'static str,
    },
}

/// What makes a knob a kernel mode (`docs/KERNEL_MODES.md`): the bench that
/// prices it, how the bench separates it from the path it replaces, and the
/// tests that hold it to the default's bits or to a stated bound.
/// `tests/knob_registry.rs` checks that each of those still exists.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Lane {
    /// A `KNOBS` row's `name`.
    pub knob: &'static str,
    /// The default or older path, in words.
    pub replaces: &'static str,
    /// A `kernel::registry()` bench name.
    pub bench: &'static str,
    pub compare: Compare,
    /// Repo-relative parity test files (or the module holding the unit test).
    pub parity: &'static [&'static str],
}

/// The knobs that are kernel modes. A knob without a bench to price it is not
/// here yet: the others are listed in `docs/KERNEL_MODES.md` with what is
/// missing.
pub const LANES: &[Lane] = &[
    Lane {
        knob: "int4_mma8",
        replaces: "INT4 GEMV (one row) and the scalar batched GEMM (verify)",
        bench: "int4_matmul",
        compare: Compare::Arms {
            baseline: "gemv_per_row",
            lane: "mma8",
        },
        parity: &[
            "crates/gpu/tests/dequant_int4_mma8_parity.rs",
            "crates/gpu/tests/qwen38_decode_verify_parity.rs",
        ],
    },
    Lane {
        knob: "gdn_lazy_tape",
        replaces: "the eager GDN verify that advances state in place",
        bench: "gdn_delta",
        // The lane also pays `state_fold` at the round's end; price the pair.
        compare: Compare::Arms {
            baseline: "prefill_eager",
            lane: "prefill_lazy",
        },
        parity: &["crates/gpu/tests/gdn_lazy_fold.rs"],
    },
    Lane {
        knob: "gdn_dk128",
        replaces: "the runtime-Dk delta kernels",
        bench: "gdn_delta",
        compare: Compare::Process {
            arm: "prefill_lazy",
            baseline_env: "TURBOSPARK_GDN_DK128=0",
        },
        parity: &["crates/gpu/src/gdn_tests.rs"],
    },
    Lane {
        knob: "attn_decode_kernel",
        replaces: "the barrier decode attention kernel",
        bench: "attn_decode",
        compare: Compare::Arms {
            baseline: "barrier",
            lane: "gqa",
        },
        parity: &["crates/gpu/tests/attention_decode_gqa_parity.rs"],
    },
    Lane {
        knob: "prefill_small_n",
        replaces: "the scalar batched GEMM in 16-row slices on INT4 matrices too narrow for a whole Steel tile (the 48-row GDN a/b)",
        bench: "int4_matmul",
        // The `small_n` arm is the shipped routing (8-row tile up to 96 rows
        // of batch, Steel edge tile above), so price it at the prefill widths.
        compare: Compare::Arms {
            baseline: "scalar_gemm",
            lane: "small_n",
        },
        parity: &[
            "crates/gpu/tests/dequant_int4_steel_parity.rs",
            "crates/gpu/tests/dequant_int4_mma8_parity.rs",
        ],
    },
];

pub fn lane_of(knob: &str) -> Option<&'static Lane> {
    LANES.iter().find(|l| l.knob == knob)
}

pub fn by_name(name: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.name == name)
}

pub fn by_env(env: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.env == env)
}

/// The engagement lines a run printed (`[engaged] NAME`, without prefix).
pub fn engaged_lines(stderr: &str) -> Vec<String> {
    let mut out: Vec<String> = stderr
        .lines()
        .filter_map(|l| l.strip_prefix("[engaged] "))
        .map(|l| l.trim().to_string())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// What the arm's environment promises against what the run printed. Each
/// violation names the variable, its value and the line. `extra` are lines
/// a spec's arm requires for a new switch the table does not know yet.
pub fn engagement_violations(
    env: &std::collections::BTreeMap<String, String>,
    extra: &[String],
    lines: &[String],
) -> Vec<String> {
    let has = |l: &str| lines.iter().any(|x| x == l);
    let mut out = Vec::new();
    for (k, v) in env {
        if k == "TURBOSPARK_STANDIN" {
            for b in v.split(',').map(str::trim).filter(|b| !b.is_empty()) {
                let line = format!("standin_{b}");
                if !has(&line) {
                    out.push(format!("{k}={v}: no [engaged] {line}"));
                }
            }
            continue;
        }
        let Some(knob) = by_env(k) else { continue };
        for e in knob.engagement {
            if e.value != "*" && e.value != v {
                continue;
            }
            match (e.present, has(e.line)) {
                (true, false) => out.push(format!("{k}={v}: no [engaged] {}", e.line)),
                (false, true) => out.push(format!(
                    "{k}={v}: [engaged] {} printed, the switch did not take",
                    e.line
                )),
                _ => {}
            }
        }
    }
    for line in extra {
        if !has(line) {
            out.push(format!(
                "arm requires [engaged] {line}, which the run did not print"
            ));
        }
    }
    out
}

/// Variables that make a run unquotable when set (to anything).
pub fn unquotable_envs() -> impl Iterator<Item = &'static str> {
    KNOBS.iter().filter(|k| !k.quotable).map(|k| k.env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_envs_are_unique_and_prefixed() {
        let mut names: Vec<_> = KNOBS.iter().map(|k| k.name).collect();
        let mut envs: Vec<_> = KNOBS.iter().map(|k| k.env).collect();
        names.sort();
        names.dedup();
        envs.sort();
        envs.dedup();
        assert_eq!(names.len(), KNOBS.len());
        assert_eq!(envs.len(), KNOBS.len());
        assert!(KNOBS.iter().all(|k| k.env.starts_with("TURBOSPARK_")));
        assert!(KNOBS.iter().all(|k| k.values.first() == Some(&"<unset>")));
    }

    #[test]
    fn engagement_is_checked_both_ways() {
        let env =
            |k: &str, v: &str| std::collections::BTreeMap::from([(k.to_string(), v.to_string())]);
        let lines = engaged_lines("x\n[engaged] int4_mma8\n[engaged] attn_decode_gqa\n");
        assert_eq!(lines, ["attn_decode_gqa", "int4_mma8"]);
        assert!(engagement_violations(&env("TURBOSPARK_INT4_MMA8", "1"), &[], &lines).is_empty());
        // A forced barrier that still ran the GQA kernel did not take.
        let v = engagement_violations(
            &env("TURBOSPARK_ATTN_DECODE_KERNEL", "barrier"),
            &[],
            &lines,
        );
        assert_eq!(v.len(), 2, "{v:?}");
        // A kill switch whose lane still ran.
        let v = engagement_violations(
            &env("TURBOSPARK_PREFILL_STEEL", "0"),
            &[],
            &["prefill_steel".into()],
        );
        assert_eq!(v.len(), 1);
        let v = engagement_violations(
            &env("TURBOSPARK_STANDIN", "attn,mlp"),
            &[],
            &["standin_attn".into()],
        );
        assert_eq!(v, ["TURBOSPARK_STANDIN=attn,mlp: no [engaged] standin_mlp"]);
        let v = engagement_violations(&Default::default(), &["new_lane".into()], &lines);
        assert_eq!(v.len(), 1);
    }
}
