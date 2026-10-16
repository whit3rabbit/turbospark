//! Router-lookahead expert prefetch for the Qwen MoE sequential decode path
//! (`TURBOSPARK_QWEN_PREFETCH=<k>`, k in 1..=8, off when unset or 0).
//!
//! EXPERIMENT. `docs/FISH_DRAMA3_PHASE0.md` ("Router lookahead, measured")
//! priced this at a 5 ms ceiling per Drama 3 frame for k=4 and set a stop
//! rule: keep it only if an interleaved A/B shows 2 ms per frame or more.
//! Delete this module if it does not.
//!
//! While layer L's experts load, a worker reads the experts layer L+1's
//! router names when run on layer L's post-attention residual (the pilot
//! probe's guess, `moe::encode_qwen_pilot_probe`). Each layer owns its own
//! streamer and slot buffers, so the worker takes layer L+1's streamer by
//! value and hands it back before that layer plans. Nothing else touches it
//! in between, and the GPU does not read layer L+1's slots until after that
//! plan, so no lock is needed.
//!
//! The guess only decides which bytes get read early. It never decides which
//! experts a layer runs, so the output cannot change (AGENTS.md Gotcha 27).

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::time::Instant;

use streaming::PreadExpertStreamer;

use crate::real_forward_dispatch::router_topk_gemma4;

/// What one prefetch asked of a streamer.
struct Job {
    layer: usize,
    streamer: PreadExpertStreamer,
    /// Router logits of `layer`, read from the probe buffer.
    logits: Vec<f32>,
}

struct Done {
    layer: usize,
    streamer: PreadExpertStreamer,
    /// Experts the guess named that were not resident.
    reads: u64,
    /// Experts the guess named that already were.
    skipped: u64,
}

#[derive(Default)]
struct Stats {
    jobs: u64,
    reads: u64,
    skipped: u64,
    /// Main-thread time spent waiting for the worker. Nonzero means the
    /// prefetch did not fit in the lookahead window.
    join_wait_nanos: u64,
}

/// Worker thread and the one job it can have in flight.
pub(crate) struct Prefetcher {
    k: usize,
    top_k: usize,
    tx: Mutex<Option<Sender<Job>>>,
    rx: Mutex<Receiver<Done>>,
    /// The layer whose streamer the worker holds, if any.
    pending: Mutex<Option<usize>>,
    stats: Mutex<Stats>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// The guess width asked for by `TURBOSPARK_QWEN_PREFETCH`, if any.
pub(crate) fn width_from_env(top_k: usize) -> Option<usize> {
    let raw = std::env::var("TURBOSPARK_QWEN_PREFETCH").ok()?;
    match raw.trim().parse::<usize>() {
        Ok(0) => None,
        Ok(k) => Some(k.min(top_k.max(1))),
        // An unparseable value must not quietly measure the off arm.
        Err(_) => panic!("TURBOSPARK_QWEN_PREFETCH must be an integer, got {raw:?}"),
    }
}

/// The first `k` experts of the router's ranking, which is the guess.
pub(crate) fn guess(logits: &[f32], top_k: usize, k: usize) -> Vec<usize> {
    let ones = vec![1.0f32; logits.len()];
    let (mut ranked, _) = router_topk_gemma4(logits, top_k, &ones);
    ranked.truncate(k);
    ranked
}

/// Reads `guess` into slots the next plan can still evict, leaving
/// `keep_evictable` of them alone so that plan always has room.
fn run_job(job: Job, top_k: usize, k: usize) -> Done {
    let Job {
        layer,
        mut streamer,
        logits,
    } = job;
    let wanted = guess(&logits, top_k, k);
    let missing = streamer.non_resident_experts(&wanted);
    let skipped = (wanted.len() - missing.len()) as u64;
    let reservation = streamer.reserve_speculative_slots(&missing, top_k);
    streamer.execute_speculative_reservation(&reservation);
    Done {
        layer,
        streamer,
        reads: reservation.len() as u64,
        skipped,
    }
}

impl Prefetcher {
    pub(crate) fn spawn(k: usize, top_k: usize) -> Self {
        let (tx, jobs) = channel::<Job>();
        let (done_tx, rx) = channel::<Done>();
        let worker = std::thread::Builder::new()
            .name("qwen-prefetch".to_string())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    if done_tx.send(run_job(job, top_k, k)).is_err() {
                        return;
                    }
                }
            })
            .expect("spawn the prefetch worker");
        Self {
            k,
            top_k,
            tx: Mutex::new(Some(tx)),
            rx: Mutex::new(rx),
            pending: Mutex::new(None),
            stats: Mutex::new(Stats::default()),
            worker: Mutex::new(Some(worker)),
        }
    }

    /// Hands `layer`'s streamer to the worker. A layer with no streamer
    /// (mapped residency) is skipped.
    pub(crate) fn submit(
        &self,
        layer: usize,
        streamers: &mut [Option<PreadExpertStreamer>],
        logits: Vec<f32>,
    ) {
        let Some(slot) = streamers.get_mut(layer) else {
            return;
        };
        let Some(streamer) = slot.take() else {
            return;
        };
        let mut pending = self.pending.lock().unwrap();
        debug_assert!(pending.is_none(), "one prefetch in flight at a time");
        let sent = self.tx.lock().unwrap().as_ref().map(|tx| {
            tx.send(Job {
                layer,
                streamer,
                logits,
            })
        });
        match sent {
            Some(Ok(())) => *pending = Some(layer),
            // The worker is gone. The streamer came back inside the failed
            // send, so put it back rather than lose the layer.
            Some(Err(failed)) => *slot = Some(failed.0.streamer),
            None => {}
        }
    }

    /// Blocks until the worker has finished and returns its streamer. Call
    /// before a layer plans, and at the start of a token so an error path
    /// that skipped it cannot strand a streamer.
    pub(crate) fn join(&self, streamers: &mut [Option<PreadExpertStreamer>]) {
        let Some(layer) = self.pending.lock().unwrap().take() else {
            return;
        };
        let started = Instant::now();
        let done = self
            .rx
            .lock()
            .unwrap()
            .recv()
            .expect("the prefetch worker died holding a streamer");
        debug_assert_eq!(done.layer, layer);
        streamers[done.layer] = Some(done.streamer);
        let mut stats = self.stats.lock().unwrap();
        stats.jobs += 1;
        stats.reads += done.reads;
        stats.skipped += done.skipped;
        stats.join_wait_nanos += started.elapsed().as_nanos() as u64;
    }

    /// Is a job in flight for `layer`?
    pub(crate) fn pending_for(&self, layer: usize) -> bool {
        *self.pending.lock().unwrap() == Some(layer)
    }

    pub(crate) fn width(&self) -> usize {
        self.k
    }

    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Closing the channel ends the worker's loop. A job in flight
        // finishes first, so its slot writes are over before the join
        // returns.
        self.tx.lock().unwrap().take();
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
        let stats = self.stats.lock().unwrap();
        if stats.jobs > 0 {
            eprintln!(
                "[qwen-prefetch] k={} jobs={} reads={} ({:.2}/job) already-resident={} \
                 main-thread join wait {:.2} ms total",
                self.k,
                stats.jobs,
                stats.reads,
                stats.reads as f64 / stats.jobs as f64,
                stats.skipped,
                stats.join_wait_nanos as f64 / 1e6
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use streaming::{ExpertCachePolicy, StreamLayout};

    #[test]
    fn the_guess_is_the_head_of_the_router_ranking() {
        let mut logits = vec![0.0f32; 16];
        logits[9] = 5.0;
        logits[3] = 4.0;
        logits[12] = 3.0;
        logits[0] = 2.0;
        assert_eq!(guess(&logits, 4, 2), vec![9, 3]);
        assert_eq!(guess(&logits, 4, 4), vec![9, 3, 12, 0]);
    }

    fn streamer(experts: usize, slots: usize) -> (PreadExpertStreamer, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "turbospark-qwen-prefetch-{}-{experts}-{slots}",
            std::process::id()
        ));
        let stride = 4096usize;
        let bytes: Vec<u8> = (0..experts)
            .flat_map(|e| std::iter::repeat(e as u8 + 1).take(stride))
            .collect();
        std::fs::write(&path, bytes).unwrap();
        let layout = StreamLayout {
            path: path.display().to_string(),
            stream_offset: 0,
            stream_size: (experts * stride) as u64,
            experts_per_layer: experts,
            expert_stride: stride as u64,
            expert_offsets: None,
        };
        (
            PreadExpertStreamer::open(layout, slots, ExpertCachePolicy::Lfu).unwrap(),
            path,
        )
    }

    #[test]
    fn a_job_loads_the_guess_and_hands_the_streamer_back() {
        let (s, path) = streamer(16, 8);
        let mut streamers = vec![None, Some(s)];
        let prefetch = Prefetcher::spawn(2, 4);
        let mut logits = vec![0.0f32; 16];
        logits[7] = 5.0;
        logits[2] = 4.0;
        logits[11] = 3.0;
        prefetch.submit(1, &mut streamers, logits);
        // The worker owns it now.
        assert!(streamers[1].is_none());
        assert!(prefetch.pending_for(1));
        prefetch.join(&mut streamers);
        assert!(!prefetch.pending_for(1));
        let s = streamers[1].as_ref().expect("returned");
        let resident: Vec<usize> = s
            .resident_experts_snapshot()
            .into_iter()
            .flatten()
            .collect();
        // Width 2 of the ranking: 7 and 2, not 11.
        assert!(resident.contains(&7) && resident.contains(&2));
        assert!(!resident.contains(&11));
        // The bytes in each slot are that expert's, not another's.
        for (slot, expert) in s.resident_experts_snapshot().iter().enumerate() {
            if let Some(e) = expert {
                assert!(s.slot_data(slot).iter().all(|&b| b == *e as u8 + 1));
            }
        }
        drop(prefetch);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_resident_guess_is_not_read_again_and_the_plan_still_fits() {
        let (mut s, path) = streamer(16, 8);
        s.load_experts_cached(&[7, 1, 4]).unwrap();
        let mut streamers = vec![Some(s)];
        let prefetch = Prefetcher::spawn(3, 4);
        let mut logits = vec![0.0f32; 16];
        logits[7] = 5.0;
        logits[2] = 4.0;
        logits[11] = 3.0;
        prefetch.submit(0, &mut streamers, logits);
        prefetch.join(&mut streamers);
        let s = streamers[0].as_mut().unwrap();
        let resident: Vec<usize> = s
            .resident_experts_snapshot()
            .into_iter()
            .flatten()
            .collect();
        assert!(resident.contains(&7) && resident.contains(&2) && resident.contains(&11));
        // `keep_evictable = top_k` leaves room for a full demand plan.
        let plan = s.plan_experts_cached(&[0, 3, 5, 6], &Default::default());
        assert_eq!(plan.misses.len(), 4);
        drop(prefetch);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_layer_with_no_streamer_is_skipped() {
        let prefetch = Prefetcher::spawn(2, 4);
        let mut streamers: Vec<Option<PreadExpertStreamer>> = vec![None];
        prefetch.submit(0, &mut streamers, vec![0.0; 16]);
        assert!(!prefetch.pending_for(0));
        prefetch.join(&mut streamers);
    }
}
