//! Chunk range partitioning and concurrent chunk retrieval.

use super::DownloadError;

/// The largest single `Range` GET this client will issue. A GGUF's resident
/// core contains whole tensors far larger than this (Gemma 4's Q8_0
/// embedding table is 785 MB in ONE tensor), and a single response body that
/// long is where a CDN drops the connection: the first attempt at the real
/// 26.9 GB Q8_0 checkpoint died with "error decoding response body" ~2.5 GB
/// in. Splitting bounds what a retry has to re-fetch as well as making the
/// drop less likely.
pub const MAX_RANGE_BYTES: u64 = 16 * 1024 * 1024;

/// Attempts per chunk before giving up. Transport failures on a multi-GB
/// walk are expected rather than exceptional; a format error is not retried
/// because it will not change.
///
/// Raised from 4 to 8 in ROADMAP Phase M2. It did NOT fix the failure it was
/// raised for (a 26 GB Mixtral walk that died three times at the same layer,
/// ~19 GB in), and it is kept only because a longer walk deserves a longer
/// budget: the cost of being wrong is asymmetric, since another four attempts
/// cost seconds of backoff while giving up costs the whole walk, which has no
/// resume. Three hypotheses about that failure have been raised and none is
/// the fix. Two were tested and refuted -- bad offsets (the ranges end exactly
/// at EOF and `curl` fetched every failing 64 MiB chunk at HTTP 206) and
/// connection reuse (disabling pooling changed nothing). The third is
/// UNTESTED and is only a candidate: every one of these URLs is served by the
/// Xet LFS bridge, which is a single CloudFront edge per connection with a
/// documented per-edge rate cap (AGENTS.md Gotcha 46), so a long-lived
/// single-stream body is exactly the shape a CDN drops. Nothing here has
/// measured that, so do not read this constant as the fix either.
pub const RANGE_ATTEMPTS: usize = 8;

/// Concurrent chunk GETs per [`super::RangeSource::read_range`] call.
///
/// The walk reads a 4-27 GB checkpoint once, sequentially, and until this
/// existed it did so one [`MAX_RANGE_BYTES`] GET at a time. That is one
/// connection, hence one CloudFront edge of the Xet bridge, hence one
/// per-edge rate cap. Measured 2026-08-14 on AC against the real
/// `gpt-oss-20b-MXFP4.gguf`, at the former 64 MiB chunk size, over a 512 MiB
/// span (the size of one routed tensor):
///
/// | | wall clock | rate |
/// |---|---|---|
/// | serial, 8 x 64 MiB | 60.4 s | 8.9 MB/s |
/// | 8-way, same 512 MiB | 17.6 s | 30.5 MB/s |
///
/// 3.4x, and the serial arm landing on 8.9 MB/s is itself the finding: the
/// per-edge cap `xet-core` #821 documents is 8.7. Scaling is sublinear. A
/// 16 MiB sweep the same day read 10.4 / 16.0 / 23.4 MB/s at 1 / 4 / 8. The
/// 16 MiB cap is now used so dense checkpoints whose individual tensors sit
/// below 64 MiB can use the same eight-way path, while a failed request
/// re-fetches one quarter as much data. The concurrency stays at 8 because
/// the shared link is the limit and more streams buy only more sockets to
/// drop. These are cross-session
/// NETWORK numbers, unlike the two constants in
/// `crates/streaming/src/read_pool.rs` whose sweeps measure this machine's
/// page cache: read the shape, re-measure before quoting an absolute.
///
/// **KNOW WHICH WALKS THIS TOUCHES.** It engages only when ONE `read_range`
/// exceeds [`MAX_RANGE_BYTES`], and `gguf_checkpoint::read_tensor` issues one
/// call per TENSOR. So it is worth the 3.4x on an MoE checkpoint, whose
/// routed tensors are the whole expert table for a layer (Gemma's
/// `ffn_gate_up_exps` is ~410 MiB, i.e. 7 chunks) and are the dominant share
/// of its bytes, and worth almost NOTHING on a dense one, whose largest
/// tensor was under the former 64 MiB cap: TinyLlama re-streamed in 3:28
/// against a recorded ~3 min, unchanged. The smaller cap addresses that
/// dispatch gap, but it is not a new end-to-end dense-model benchmark.
///
/// This is worth nothing on its own: see [`super::HttpRangeSource::new`] for the
/// client setting that makes these separate connections rather than one
/// multiplexed HTTP/2 one.
pub const RANGE_CONCURRENCY: usize = 8;

/// Splits `[start, end_exclusive)` into successive chunks of at most `cap`
/// bytes. Pure, so the boundary arithmetic is testable without a network.
pub fn chunk_ranges(start: u64, end_exclusive: u64, cap: u64) -> Vec<(u64, u64)> {
    assert!(cap > 0, "chunk cap must be positive");
    let mut out = Vec::new();
    let mut at = start;
    while at < end_exclusive {
        let next = at.saturating_add(cap).min(end_exclusive);
        out.push((at, next));
        at = next;
    }
    out
}

/// One chunk's index, its half-open byte range, and the slice of the output
/// buffer it owns.
type ChunkJob<'a> = (usize, u64, u64, &'a mut [u8]);

/// Fills `out` by running `fetch` over `chunks`, up to `concurrency` at a
/// time. `chunks` must partition `out` in order, which is what
/// [`chunk_ranges`] produces.
///
/// Each worker gets a DISJOINT `&mut [u8]` carved out of `out` up front
/// rather than returning a `Vec` to be concatenated. That is why this needs
/// no `unsafe` in a crate that forbids it, why ordering is structural rather
/// than something the caller has to reassemble correctly, and why peak memory
/// is unchanged: collecting [`RANGE_CONCURRENCY`] separate
/// [`MAX_RANGE_BYTES`] buffers in flight would add half a gigabyte on top of
/// an allocation the caller has already made.
///
/// Generic over `fetch` so the two properties worth asserting -- that chunks
/// land at their own offsets, and that a multi-failure read reports the
/// LOWEST-indexed failure the way a serial one would -- are testable with no
/// server.
pub fn fill_chunks<F>(
    out: &mut [u8],
    chunks: &[(u64, u64)],
    concurrency: usize,
    fetch: F,
) -> Result<(), DownloadError>
where
    F: Fn(u64, u64, &mut [u8]) -> Result<(), DownloadError> + Sync,
{
    let mut rest: &mut [u8] = out;
    let mut jobs: Vec<ChunkJob<'_>> = Vec::with_capacity(chunks.len());
    for (index, &(start, end_exclusive)) in chunks.iter().enumerate() {
        let (head, tail) = rest.split_at_mut((end_exclusive - start) as usize);
        jobs.push((index, start, end_exclusive, head));
        rest = tail;
    }

    // A single chunk is the COMMON case, not a degenerate one:
    // `fetch_gguf_header` reads a megabyte at a time and every norm and
    // router tensor in a walk is far under the cap. It must not pay for a
    // thread, and it must keep reporting its error directly.
    if jobs.len() <= 1 {
        for (_, start, end_exclusive, dst) in jobs {
            fetch(start, end_exclusive, dst)?;
        }
        return Ok(());
    }

    let queue = std::sync::Mutex::new(jobs);
    let failures = std::sync::Mutex::new(Vec::<(usize, DownloadError)>::new());
    let workers = concurrency.clamp(1, chunks.len());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                // Pop under the lock and fetch outside it, so a stalled
                // chunk holds up nothing but itself. A fixed split of the
                // job list would instead let one slow edge idle a worker.
                while let Some((index, start, end_exclusive, dst)) =
                    queue.lock().expect("chunk queue lock").pop()
                {
                    if let Err(err) = fetch(start, end_exclusive, dst) {
                        failures
                            .lock()
                            .expect("chunk failure lock")
                            .push((index, err));
                    }
                }
            });
        }
    });

    let mut failures = failures.into_inner().expect("chunk failure lock");
    failures.sort_by_key(|(index, _)| *index);
    match failures.into_iter().next() {
        Some((_, err)) => Err(err),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_range_exactly_and_in_order() {
        // The property that matters: concatenating the chunks reproduces the
        // original range with no gap, no overlap, and none over the cap.
        for (start, end, cap) in [(0, 0, 8), (0, 1, 8), (7, 8, 8), (0, 24, 8), (5, 23, 7)] {
            let chunks = chunk_ranges(start, end, cap);
            assert_eq!(
                chunks.iter().map(|(a, b)| b - a).sum::<u64>(),
                end - start,
                "total length for {start}..{end} cap {cap}"
            );
            let mut at = start;
            for (a, b) in &chunks {
                assert_eq!(*a, at, "gap or overlap in {start}..{end} cap {cap}");
                assert!(b - a <= cap && b > a);
                at = *b;
            }
            assert_eq!(at, end);
        }
        assert!(
            chunk_ranges(4, 4, 8).is_empty(),
            "empty range yields no GET"
        );
    }

    /// Hostile GGUF offsets can place a short, otherwise valid range next to
    /// `u64::MAX`. Chunking must terminate there instead of wrapping around
    /// and growing the descriptor vector until the process aborts.
    #[test]
    fn chunks_a_range_at_the_end_of_the_address_space() {
        assert_eq!(
            chunk_ranges(u64::MAX - 7, u64::MAX, 64),
            vec![(u64::MAX - 7, u64::MAX)]
        );
    }

    /// The property a parallel fill can break and a serial one cannot:
    /// every chunk must land at ITS OWN offset. Each fetch writes its
    /// absolute file position, so any permutation or overlap of the
    /// destination slices shows up as a value mismatch rather than as a
    /// length one.
    #[test]
    fn a_parallel_fill_lands_every_chunk_at_its_own_offset() {
        let total = 4096u64;
        let chunks = chunk_ranges(0, total, 100);
        assert!(chunks.len() > 8, "want more chunks than workers");
        let mut out = vec![0u8; total as usize];
        fill_chunks(&mut out, &chunks, 8, |start, end_exclusive, dst| {
            assert_eq!(dst.len() as u64, end_exclusive - start, "slice width");
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = (start + i as u64) as u8;
            }
            Ok(())
        })
        .expect("every chunk succeeds");
        let expected: Vec<u8> = (0..total).map(|i| i as u8).collect();
        assert_eq!(out, expected);
    }

    /// A serial read reports the FIRST chunk that failed, and the parallel
    /// one has to keep reporting the same chunk however the workers happen
    /// to interleave. Without the sort this is whichever thread lost the
    /// race, i.e. a flaky multi-GB walk that blames a different offset every
    /// time it dies.
    #[test]
    fn a_parallel_fill_reports_the_lowest_indexed_failure() {
        let chunks = chunk_ranges(0, 1000, 10);
        let mut out = vec![0u8; 1000];
        let err = fill_chunks(&mut out, &chunks, 8, |start, _end, _dst| {
            if start == 70 || start == 320 {
                return Err(DownloadError::Request(format!("boom at {start}")));
            }
            Ok(())
        })
        .expect_err("two chunks fail");
        assert_eq!(err, DownloadError::Request("boom at 70".to_string()));
    }

    /// `fetch_gguf_header` calls `read_range` once per growth step and every
    /// small tensor in a walk is one chunk, so the single-chunk case is the
    /// common one and must not spawn.
    #[test]
    fn a_single_chunk_range_is_filled_on_the_calling_thread() {
        let caller = std::thread::current().id();
        let seen = std::sync::Mutex::new(None);
        let chunks = chunk_ranges(0, 8, 64);
        assert_eq!(chunks.len(), 1);
        let mut out = vec![0u8; 8];
        fill_chunks(&mut out, &chunks, 8, |_start, _end, dst| {
            *seen.lock().unwrap() = Some(std::thread::current().id());
            dst.fill(7);
            Ok(())
        })
        .expect("the one chunk succeeds");
        assert_eq!(out, vec![7u8; 8]);
        assert_eq!(seen.into_inner().unwrap(), Some(caller));
    }

    /// An empty range is not an error and issues no fetch at all.
    #[test]
    fn an_empty_range_fetches_nothing() {
        let mut out = Vec::new();
        fill_chunks(&mut out, &chunk_ranges(4, 4, 8), 8, |_, _, _| {
            panic!("an empty range must not fetch")
        })
        .expect("empty range succeeds");
    }
}
