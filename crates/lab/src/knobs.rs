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

/// How the runtime reads the knob. tslab gives every run a fresh process, so
/// this is information for the reader, not a branch in the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadAt {
    /// Captured into the runner at open (a pub setter usually exists too).
    Open,
    /// A process-wide `OnceLock`: changing it needs a new process.
    ProcessCached,
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
}

pub const KNOBS: &[Knob] = &[
    Knob {
        name: "batched_gemv",
        env: "TURBOSPARK_BATCHED_GEMV",
        values: &["<unset>", "1", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &[
            "qwen35", "qwen36", "gemma4", "llama", "qwen3moe", "qwen3", "qwen2",
        ],
        quotable: true,
        doc: "docs/ENV.md",
    },
    Knob {
        name: "prefill_steel",
        env: "TURBOSPARK_PREFILL_STEEL",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["qwen35", "qwen36"],
        quotable: true,
        doc: "docs/BATCHED_PREFILL.md",
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
    },
    Knob {
        name: "qwen_batched_row_ops",
        env: "TURBOSPARK_QWEN_BATCHED_ROW_OPS",
        values: &["<unset>", "0"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::BitExact,
        families: &["qwen35", "qwen36"],
        quotable: true,
        doc: "docs/SPEEDUP_PLAYBOOK.md",
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
    },
    Knob {
        name: "attn_decode_kernel",
        env: "TURBOSPARK_ATTN_DECODE_KERNEL",
        values: &["<unset>", "barrier", "gqa"],
        read_at: ReadAt::ProcessCached,
        exactness: Exactness::KlBounded,
        families: &[],
        quotable: true,
        doc: "docs/SPEEDUP_PLAYBOOK.md",
    },
    Knob {
        name: "routed_batch",
        env: "TURBOSPARK_ROUTED_BATCH",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::KlBounded,
        families: &["gemma4", "gptOss", "qwen3moe", "llama"],
        quotable: true,
        doc: "docs/ENV.md",
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
    },
    Knob {
        name: "shared_cb",
        env: "TURBOSPARK_SHARED_CB",
        values: &["<unset>", "0"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["gemma4"],
        quotable: true,
        doc: "docs/ENV.md",
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
    },
    Knob {
        name: "metal_kernel_warmup",
        env: "TURBOSPARK_METAL_KERNEL_WARMUP",
        values: &["<unset>", "1"],
        read_at: ReadAt::Open,
        exactness: Exactness::BitExact,
        families: &["qwen36"],
        quotable: true,
        doc: "docs/ENV.md",
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
    },
];

pub fn by_name(name: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.name == name)
}

pub fn by_env(env: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.env == env)
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
}
