//! Physical-disk-read accounting, and the seam that forces a disk-bound
//! read.
//!
//! Both exist to answer a question this crate has been ASSERTING rather
//! than measuring: whether an expert `pread` actually reached the device.
//! `crates/streaming/CLAUDE.md` Gotcha 3 says the read is a page-cache
//! memcpy (125 MiB/token in 5.26 ms on the real 26B, far past any SSD) and
//! that a cold or memory-tight host makes it genuinely disk-bound instead.
//! Nothing here could tell those two apart, which matters because
//! `docs/EXPERT_ROUTING.md` names the second one as the condition that
//! would REVERSE the router-lookahead prefetch decision.
//!
//! Borrowed from `garnermccloud/sglang-ssd-stream`, whose native reader
//! reports `submitted_bytes` against a `physical_bytes` delta taken from
//! `/proc/self/io`'s `read_bytes:` around each gather. `proc_pid_rusage`'s
//! `ri_diskio_bytesread` is the Darwin counterpart. Their flag is off by
//! default and so is this one, for the same reason: it is a syscall pair
//! per read batch on the decode critical path.

use std::sync::OnceLock;

/// Bytes this PROCESS has read from disk since it started, or `None` where
/// the probe is unavailable (every non-macOS target, and a failing call).
///
/// Process-wide, not per-descriptor: it counts the weight mapping's faults
/// and every other read in flight, exactly as the `/proc/self/io` counter
/// this is modelled on does. A caller wanting the cost of one operation
/// takes a DELTA around it and accepts that a concurrent reader inflates
/// the answer.
///
/// **`ri_diskio_bytesread` is a fixed-size struct copy, not a truncatable
/// one.** `proc_pid_rusage` takes no count parameter (`task_info`, which
/// `crates/bench/src/memory.rs` uses, does -- hence the prefix struct that
/// is correct THERE), so the kernel writes the whole flavor unconditionally
/// and a shortened struct is a stack overwrite rather than a smaller
/// answer. `libc::rusage_info_v2` is the complete binding and is used for
/// exactly that reason; V2 is the earliest flavor carrying the field.
#[cfg(target_os = "macos")]
pub(crate) fn process_disk_bytes_read() -> Option<u64> {
    // SAFETY: `rusage_info_v2` is plain old data, so an all-zero bit
    // pattern is a valid value of it.
    #[allow(unsafe_code)]
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    // SAFETY: the kernel writes exactly `size_of::<rusage_info_v2>()`
    // bytes for `RUSAGE_INFO_V2` into `info`, which is that size, and
    // retains the pointer no longer than the call. The cast to
    // `rusage_info_t` (`*mut c_void`) is Darwin's own calling convention
    // for this function.
    #[allow(unsafe_code)]
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V2,
            (&mut info as *mut libc::rusage_info_v2).cast::<libc::rusage_info_t>(),
        )
    };
    if rc == 0 {
        Some(info.ri_diskio_bytesread)
    } else {
        None
    }
}

/// Documented no-op off macOS: `proc_pid_rusage` is a `libproc` call and
/// has no portable counterpart, so callers get `None` and report the
/// measurement as unavailable rather than as zero.
#[cfg(not(target_os = "macos"))]
pub(crate) fn process_disk_bytes_read() -> Option<u64> {
    None
}

/// Whether to sample [`process_disk_bytes_read`] around each read batch
/// (`MFERENCE_EXPERT_DISK_IO=1`).
///
/// OFF by default, and read ONCE: this is a syscall pair per batch on the
/// decode critical path, and there are 30 layers per token on the real 26B
/// install. Same shape as `read_pool`'s `MFERENCE_READ_QOS` seam, including
/// the read-once part -- setting the variable after the streamer is open
/// does nothing.
pub(crate) fn measure_physical_io() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MFERENCE_EXPERT_DISK_IO").as_deref() == Ok("1"))
}

/// Whether to open the expert blob with the unified buffer cache bypassed
/// (`MFERENCE_EXPERT_NOCACHE=1`).
///
/// This is an EXPERIMENTAL CONDITION, not an optimization: it makes the
/// disk-bound arm that `docs/EXPERT_ROUTING.md` names as the one that would
/// reverse the prefetch decision reproducible on demand, in the shape of
/// `scripts/power.sh COOLING=max` (AGENTS.md Gotcha 28 -- a missing
/// condition rather than an unmeasurable one). Expect it to be SLOWER; that
/// is the point.
///
/// macOS has no `POSIX_FADV_DONTNEED`, so this is `F_NOCACHE` rather than an
/// eviction, and **the difference is not academic: `F_NOCACHE` DOES NOT
/// EVICT PAGES THAT ARE ALREADY RESIDENT.** Measured while writing
/// `the_disk_read_counter_moves_on_a_bypassed_read`: an 8 MiB file written,
/// `fsync`ed, closed and reopened with `F_NOCACHE` read back with a disk
/// delta of **0**, because the write had left every page in the buffer
/// cache. Setting `F_NOCACHE` on the WRITE descriptor as well took the same
/// read to a delta of exactly 8,388,608, stable across three runs.
///
/// The consequence for an operator is the whole usage protocol. An expert
/// blob that a previous run already faulted in stays resident, so
/// `MFERENCE_EXPERT_NOCACHE=1` on its own does NOT guarantee a disk-bound
/// arm on a warm machine -- it guarantees only that THIS process stops
/// adding to the cache. Pair it with `sudo purge` (or a fresh boot) when the
/// install has been read recently, and confirm with a non-zero
/// `bytes_physical` from [`process_disk_bytes_read`] rather than trusting
/// the flag. That confirmation is why the two seams landed together.
pub(crate) fn nocache_requested() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MFERENCE_EXPERT_NOCACHE").as_deref() == Ok("1"))
}

/// Turns off the unified buffer cache for `fd`. Returns whether the kernel
/// accepted it, so a caller can report the difference between "bypassed"
/// and "asked to bypass and was refused" instead of silently measuring the
/// warm path under a flag that says otherwise.
#[cfg(target_os = "macos")]
pub(crate) fn set_nocache(fd: std::os::unix::io::RawFd) -> bool {
    // SAFETY: `fd` is a valid, caller-owned descriptor for the lifetime of
    // this call. `F_NOCACHE` takes an int by value and retains nothing.
    #[allow(unsafe_code)]
    unsafe {
        libc::fcntl(fd, libc::F_NOCACHE, 1) == 0
    }
}

/// Documented no-op off macOS, matching `rdadvice`'s arm: `F_NOCACHE` is a
/// Darwin `fcntl`, and reporting `false` keeps the caller honest about
/// having failed to establish the condition.
#[cfg(not(target_os = "macos"))]
pub(crate) fn set_nocache(_fd: std::os::unix::io::RawFd) -> bool {
    false
}

/// Cumulative byte accounting for one streamer's cache-miss reads.
///
/// `requested` is free and always collected; `physical` and `samples` are
/// zero unless [`measure_physical_io`] is on, which is why a reader must
/// check `samples` before dividing rather than reading `physical == 0` as
/// "nothing hit the disk".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertIoStats {
    /// Bytes this streamer asked for: one expert stride per cache miss.
    /// Hits are excluded, since a hit reads nothing.
    pub bytes_requested: u64,
    /// Process-wide disk bytes attributed to those reads, summed over the
    /// per-batch deltas. Meaningless without `samples > 0`.
    pub bytes_physical: u64,
    /// Read batches executed (one per non-empty cache plan).
    pub batches: u64,
    /// Batches where a physical-I/O delta was successfully sampled.
    pub samples: u64,
}

impl ExpertIoStats {
    /// Folds `other` into this total, for summing across a model's
    /// per-layer streamers.
    pub fn accumulate(&mut self, other: &ExpertIoStats) {
        self.bytes_requested += other.bytes_requested;
        self.bytes_physical += other.bytes_physical;
        self.batches += other.batches;
        self.samples += other.samples;
    }

    /// Physical bytes per requested byte, or `None` when nothing was
    /// sampled or nothing was requested.
    ///
    /// Above 1.0 is read amplification (`F_RDADVISE` readahead pulling more
    /// than the stride, or another reader in the process). Near 0.0 is the
    /// warm page-cache case Gotcha 3 describes. There is no "expected"
    /// value: this is the instrument that decides which regime a run was
    /// in, so read it rather than predicting it.
    pub fn amplification(&self) -> Option<f64> {
        if self.samples == 0 || self.bytes_requested == 0 {
            return None;
        }
        Some(self.bytes_physical as f64 / self.bytes_requested as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe answers at all. This is the whole load-bearing claim of
    /// the file: `proc_pid_rusage` takes no size argument, so a wrong
    /// flavor constant or a wrong cast shows up as a non-zero return and
    /// `None` here rather than as a compile error.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_disk_read_probe_is_available_on_this_platform() {
        assert!(
            process_disk_bytes_read().is_some(),
            "proc_pid_rusage(RUSAGE_INFO_V2) failed; the flavor or the \
             pointer cast is wrong"
        );
    }

    /// A byte counter that went backwards would make every delta in
    /// `pread_streamer` a saturating zero, which reads as "nothing hit the
    /// disk" -- the exact false negative this instrument exists to prevent.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_disk_read_counter_never_goes_backwards() {
        let first = process_disk_bytes_read().expect("probe available");
        let second = process_disk_bytes_read().expect("probe available");
        assert!(second >= first, "{second} < {first}");
    }

    /// The counter MOVES when bytes actually come off the device, which is
    /// the only assertion that pins the `RUSAGE_INFO_V2` flavor.
    ///
    /// Written because the obvious pair of probe tests above BOTH SURVIVE a
    /// mutation to `RUSAGE_INFO_V0` (AGENTS.md's rule that a survivor whose
    /// mutation applied is a missing test). V0 carries no
    /// `ri_diskio_bytesread`, so the kernel leaves the zeroed field alone
    /// and still returns success: the probe answers `Some(0)` forever,
    /// availability passes, monotonicity passes, and every run reads as
    /// perfectly cache-resident. That is Gotcha 59's shape exactly -- a
    /// degenerate value scoring the instrument's best possible result.
    ///
    /// This also doubles as the only automated check that `set_nocache`
    /// does anything, since a bypassed read is what makes the delta
    /// non-zero at all.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_disk_read_counter_moves_on_a_bypassed_read() {
        use std::io::Write;
        use std::os::unix::fs::FileExt;
        use std::os::unix::io::AsRawFd;

        // Large enough that no readahead heuristic can serve it as a
        // rounding error, small enough to stay cheap.
        const BYTES: usize = 8 << 20;
        let dir = std::env::temp_dir().join(format!("turbospark-diskio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        {
            let mut file = std::fs::File::create(&path).unwrap();
            // Bypass on the WRITE side too, or the bytes stay resident in
            // the buffer cache and the read below is served from memory
            // however the read descriptor is configured. Asserted rather
            // than ignored: a silent failure here lands as a zero delta
            // below, whose message blames the rusage flavor instead.
            assert!(
                set_nocache(file.as_raw_fd()),
                "F_NOCACHE refused on the write descriptor"
            );
            file.write_all(&vec![7u8; BYTES]).unwrap();
            file.sync_all().unwrap();
        }

        let file = std::fs::File::open(&path).unwrap();
        assert!(
            set_nocache(file.as_raw_fd()),
            "F_NOCACHE refused; cannot establish a disk-bound read"
        );
        let before = process_disk_bytes_read().expect("probe available");
        let mut buffer = vec![0u8; BYTES];
        file.read_exact_at(&mut buffer, 0).unwrap();
        let after = process_disk_bytes_read().expect("probe available");
        let _ = std::fs::remove_dir_all(&dir);

        // EXACT rather than a threshold: with both descriptors bypassed
        // the bytes come off the device once and nothing else in this
        // process is reading, so the delta is the read size to the byte.
        // Measured stable at 8,388,608 across three consecutive runs.
        assert_eq!(
            after - before,
            BYTES as u64,
            "read {BYTES} bytes with the cache bypassed; the counter moved \
             by {}, so the rusage flavor is probably wrong",
            after - before
        );
    }

    #[test]
    fn accumulate_sums_every_field() {
        let mut total = ExpertIoStats {
            bytes_requested: 10,
            bytes_physical: 4,
            batches: 1,
            samples: 1,
        };
        total.accumulate(&ExpertIoStats {
            bytes_requested: 5,
            bytes_physical: 3,
            batches: 2,
            samples: 1,
        });
        assert_eq!(
            total,
            ExpertIoStats {
                bytes_requested: 15,
                bytes_physical: 7,
                batches: 3,
                samples: 2,
            }
        );
    }

    /// Unsampled is UNKNOWN, not zero. Returning `Some(0.0)` here would
    /// report every warm-by-default run as proven cache-resident, which is
    /// a claim the run did not make.
    #[test]
    fn amplification_is_none_without_a_sample() {
        let stats = ExpertIoStats {
            bytes_requested: 1024,
            bytes_physical: 0,
            batches: 4,
            samples: 0,
        };
        assert_eq!(stats.amplification(), None);
    }

    #[test]
    fn amplification_is_none_when_nothing_was_requested() {
        let stats = ExpertIoStats {
            bytes_requested: 0,
            bytes_physical: 0,
            batches: 0,
            samples: 3,
        };
        assert_eq!(stats.amplification(), None);
    }

    #[test]
    fn amplification_divides_physical_by_requested() {
        let stats = ExpertIoStats {
            bytes_requested: 400,
            bytes_physical: 600,
            batches: 2,
            samples: 2,
        };
        assert_eq!(stats.amplification(), Some(1.5));
    }
}
