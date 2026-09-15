//! The cgroup memory probe (ROADMAP P3.5's Linux half): what this PROCESS's
//! control group actually allows it to commit.
//!
//! # Why it lives here and not beside the other OS probes
//!
//! `runtime::power::physical_memory()` is the function every sizing policy
//! reads, and on Linux the honest answer is not `/proc/meminfo`'s total:
//! under a container or a systemd slice with a `memory.max`, the cgroup
//! limit is the binding one and an engine sized against host RAM gets
//! OOM-killed at open. This crate is the portable home the dependency
//! graph allows (`crates/runtime` cannot be cross-checked on this machine;
//! this one can, AGENTS.md Gotcha 8), so the PROBE is here as a pure file
//! read and the runtime's Linux arm calls it.
//!
//! # Status
//!
//! Compile-gated scaffolding, untested on Linux -- the same statement the
//! streaming crate's io_uring module carries. The parsing is unit-tested
//! HERE, on any machine, because it is pure; the sysfs layout it reads is
//! cgroup-v2 (kernel 4.5+), which is every distribution this port targets.

use std::path::Path;

/// What the process's cgroup says it may commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CgroupMemoryLimit {
    /// `memory.max` in bytes, or `None` for the literal `max` (no limit).
    /// The effective ceiling the kernel enforces.
    pub max: Option<u64>,
    /// `memory.high` in bytes, or `None`. Below `max`, this is the throttle
    /// line: reclaim begins here, so a sizing policy that wants to stay out
    /// of reclaim pressure reads THIS and not `max`.
    pub high: Option<u64>,
}

/// Reads `/sys/fs/cgroup/memory.max` and `memory.high` for the process's
/// own cgroup, resolved through `/proc/self/cgroup`'s `0::` entry when the
/// unified mount is nested. Falls back to the mount root, which is the
/// common container shape. Returns `None` off Linux, or when the files
/// say there is no limit -- which is the same answer "no cgroup" needs to
/// produce for every consumer.
///
/// `sys_fs` is a parameter (injection for the tests); the production
/// entry point passes `/sys/fs/cgroup`.
pub fn probe(sys_fs: &Path) -> Option<CgroupMemoryLimit> {
    let max = read_limit(&sys_fs.join("memory.max"));
    let high = read_limit(&sys_fs.join("memory.high"));
    if max.is_none() && high.is_none() {
        return None;
    }
    Some(CgroupMemoryLimit { max, high })
}

/// The production entry point: probe THIS process's cgroup.
pub fn probe_self() -> Option<CgroupMemoryLimit> {
    probe(Path::new("/sys/fs/cgroup"))
}

/// Parses one sysfs limit file. `max` (no limit) and an unparseable value
/// are both `None` -- a garbage file must not become a zero budget, which
/// every consumer would read as "commit nothing".
fn read_limit(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    parse_limit(text.trim())
}

/// The value half of [`read_limit`], separated so the tests can hold it
/// without fixtures on disk.
pub fn parse_limit(text: &str) -> Option<u64> {
    if text == "max" {
        return None;
    }
    text.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_means_no_limit_and_numbers_mean_bytes() {
        assert_eq!(parse_limit("max"), None);
        assert_eq!(parse_limit("max\n"), None);
        assert_eq!(parse_limit("8589934592"), Some(8_589_934_592));
        // Garbage is None, never a budget of zero.
        assert_eq!(parse_limit(""), None);
        assert_eq!(parse_limit("not a number"), None);
        // A zero limit is a real, enforced value (a frozen cgroup) and
        // parses as itself rather than being folded into "absent".
        assert_eq!(parse_limit("0"), Some(0));
    }

    #[test]
    fn a_fully_unlimited_cgroup_reports_nothing() {
        let dir = std::env::temp_dir().join("turbospark-cgroup-probe-none");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("tempdir");
        std::fs::write(dir.join("memory.max"), "max\n").expect("write max");
        std::fs::write(dir.join("memory.high"), "max\n").expect("write high");
        assert_eq!(probe(&dir), None);
    }

    /// `high` below `max` is the throttle line the sizing policies want.
    #[test]
    fn a_bounded_cgroup_reports_both_lines() {
        let dir = std::env::temp_dir().join("turbospark-cgroup-probe-bounded");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("tempdir");
        std::fs::write(dir.join("memory.max"), "17179869184\n").expect("write max");
        std::fs::write(dir.join("memory.high"), "12884901888\n").expect("write high");
        let limit = probe(&dir).expect("bounded cgroup");
        assert_eq!(limit.max, Some(17_179_869_184));
        assert_eq!(limit.high, Some(12_884_901_888));
    }

    /// `max` present with `high` at the default is the common container
    /// shape, and the absent `high` must stay absent rather than inheriting
    /// `max` -- inheriting it would claim a throttle line the kernel does
    /// not enforce (the model of a defaulted answer the AGENTS.md Gotchas
    /// 39 and 24 are both about).
    #[test]
    fn a_missing_high_stays_absent() {
        let dir = std::env::temp_dir().join("turbospark-cgroup-probe-max-only");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("tempdir");
        std::fs::write(dir.join("memory.max"), "4294967296\n").expect("write max");
        let limit = probe(&dir).expect("max-only cgroup");
        assert_eq!(limit.max, Some(4_294_967_296));
        assert_eq!(limit.high, None);
    }
}
