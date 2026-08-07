//! A process-wide pool of parked reader threads for expert-blob reads.
//!
//! Why this is not `std::thread::scope`: the streamer splits every cache
//! miss into chunks so a lone miss still reads wide (see
//! `pread_streamer::execute_expert_cache_plan`), which multiplies the
//! number of threads a layer wants. There are 30 layers per token and the
//! reads sit on the decode critical path, so spawning them per layer pays
//! thread creation hundreds of times per token for reads that are only
//! microseconds of actual copying each.
//!
//! One pool serves every layer's streamer. A per-streamer pool would mean
//! 30 idle pools on the real 26B install, and they would never run
//! concurrently anyway: layers are executed one at a time.
//!
//! The pool deliberately has no work-stealing scheduler, no priorities and
//! no dynamic sizing. It runs exactly one kind of job.

use std::fs::File;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use crate::error::StreamerError;

/// Parked reader threads. Sized for memory bandwidth, not core count:
/// these jobs are `pread` out of page cache, which saturates well before
/// it runs out of cores.
///
/// SWEPT 2026-08-06 on the real 26B install (2252-token prompt, 32 slots,
/// three interleaved rounds after a discarded warmup), `expert io`
/// ms/token: 4 threads 3.97-4.03, **8 threads 3.65-3.74**, 16 threads
/// 3.70-3.76. 8 beat 4 in every round; 8 against 16 is a wash, so the
/// smaller pool wins on parked-thread cost. Do not re-derive it.
const POOL_THREADS: usize = 8;

/// One unit of parallel read work: a byte range of the stream copied into
/// a byte range of a slot.
pub(crate) struct ReadChunk {
    pub(crate) dest: *mut u8,
    pub(crate) len: usize,
    pub(crate) file_offset: u64,
}

// SAFETY: every `ReadChunk` in a batch addresses a byte range no other
// chunk in that batch overlaps (distinct cache slots, and non-overlapping
// spans within one slot -- the streamer asserts the first and constructs
// the second). The backing allocations outlive the batch because
// `run_batch` blocks until every chunk is done.
#[allow(unsafe_code)]
unsafe impl Send for ReadChunk {}
#[allow(unsafe_code)]
unsafe impl Sync for ReadChunk {}

/// Shared state for one submitted batch. Workers pull chunk indices off
/// `cursor` until it runs past the end, then drop their claim.
struct Batch {
    chunks: *const ReadChunk,
    chunk_count: usize,
    fd: RawFd,
    cursor: AtomicUsize,
    /// Claims still running. The batch is complete at zero.
    outstanding: Mutex<usize>,
    finished: Condvar,
    error: Mutex<Option<StreamerError>>,
}

// SAFETY: `chunks` points at a slice owned by the submitting thread, which
// blocks inside `run_batch` until `outstanding` reaches zero, so the slice
// outlives every read of it. The chunks themselves are `Sync` (above).
#[allow(unsafe_code)]
unsafe impl Send for Batch {}
#[allow(unsafe_code)]
unsafe impl Sync for Batch {}

impl Batch {
    /// Works chunks until the batch is drained. Errors are recorded rather
    /// than propagated: a failed chunk must not strand the other workers
    /// or leave the submitter waiting forever.
    fn run(&self) {
        // SAFETY: `fd` is owned by the submitter's `File`, which outlives
        // this batch (`run_batch` blocks until every claim is dropped).
        // `ManuallyDrop` keeps this borrowed view from closing it.
        #[allow(unsafe_code)]
        let file = std::mem::ManuallyDrop::new(unsafe { File::from_raw_fd(self.fd) });
        loop {
            let index = self.cursor.fetch_add(1, Ordering::Relaxed);
            if index >= self.chunk_count {
                return;
            }
            // SAFETY: `index` is in range, and the slice outlives the batch.
            #[allow(unsafe_code)]
            let chunk = unsafe { &*self.chunks.add(index) };
            // SAFETY: chunk ranges within a batch are disjoint.
            #[allow(unsafe_code)]
            let dest = unsafe { std::slice::from_raw_parts_mut(chunk.dest, chunk.len) };
            if let Err(e) = crate::pread_streamer::read_full(&file, dest, chunk.file_offset) {
                let mut guard = self.error.lock().unwrap();
                if guard.is_none() {
                    *guard = Some(e);
                }
            }
        }
    }
}

/// A worker's claim on a batch. Dropping it signals completion, so a
/// panicking read cannot hang the submitter.
struct Claim(std::sync::Arc<Batch>);

impl Drop for Claim {
    fn drop(&mut self) {
        let mut outstanding = self.0.outstanding.lock().unwrap();
        *outstanding -= 1;
        if *outstanding == 0 {
            self.0.finished.notify_all();
        }
    }
}

struct Pool {
    sender: std::sync::mpsc::Sender<Claim>,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<Claim>();
        // std's receiver is single-consumer, so the workers share it under
        // a mutex. Contention is a handful of lock acquisitions per layer
        // against reads that copy hundreds of KiB each.
        let receiver = std::sync::Arc::new(Mutex::new(receiver));
        for _ in 0..POOL_THREADS {
            let receiver = std::sync::Arc::clone(&receiver);
            std::thread::Builder::new()
                .name("mrefrust-expert-read".to_string())
                .spawn(move || loop {
                    let claim = {
                        let guard = receiver.lock().unwrap();
                        guard.recv()
                    };
                    // The channel only closes at process teardown.
                    let Ok(claim) = claim else { return };
                    claim.0.run();
                    drop(claim);
                })
                .expect("spawn expert read worker");
        }
        Pool { sender }
    })
}

/// Reads every chunk, in parallel across the pool, and blocks until all of
/// them are done. Returns the first error any chunk hit.
///
/// Runs the work inline when there is only one chunk: handing a single
/// ~800 KiB copy to another thread and waiting for it is strictly slower
/// than doing it here.
pub(crate) fn run_batch(file: &File, chunks: &[ReadChunk]) -> Result<(), StreamerError> {
    if chunks.is_empty() {
        return Ok(());
    }
    if chunks.len() == 1 {
        // SAFETY: single-threaded, and the caller owns the destination.
        #[allow(unsafe_code)]
        let dest = unsafe { std::slice::from_raw_parts_mut(chunks[0].dest, chunks[0].len) };
        return crate::pread_streamer::read_full(file, dest, chunks[0].file_offset);
    }

    let claims = chunks.len().min(POOL_THREADS);
    // The caller's `file` is borrowed, not duplicated: `run_batch` blocks
    // until every claim is dropped, so the descriptor cannot close under a
    // worker. Duplicating would cost a `dup` and a `close` per layer per
    // token (30 of each on the real 26B) to buy nothing.
    let batch = std::sync::Arc::new(Batch {
        chunks: chunks.as_ptr(),
        chunk_count: chunks.len(),
        fd: file.as_raw_fd(),
        cursor: AtomicUsize::new(0),
        outstanding: Mutex::new(claims),
        finished: Condvar::new(),
        error: Mutex::new(None),
    });

    let pool = pool();
    for _ in 0..claims {
        // A full channel is not possible (unbounded), so this only fails
        // if every worker died, which cannot happen short of a panic in
        // the loop itself.
        pool.sender
            .send(Claim(std::sync::Arc::clone(&batch)))
            .expect("expert read workers alive");
    }

    let mut outstanding = batch.outstanding.lock().unwrap();
    while *outstanding > 0 {
        outstanding = batch.finished.wait(outstanding).unwrap();
    }
    drop(outstanding);

    let error = batch.error.lock().unwrap().take();
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
