//! Process-wide serialization for heavyweight native work.
//!
//! Text sessions, image sessions, and the in-process server share one Metal
//! device. A GUI can own more than one session, so per-session mutexes are
//! not enough to keep two large workloads from overlapping.

use std::sync::{Condvar, Mutex, OnceLock};

#[derive(Default)]
struct State {
    busy: bool,
}

static STATE: OnceLock<(Mutex<State>, Condvar)> = OnceLock::new();

pub struct HeavyWorkGuard {
    _private: (),
}

impl HeavyWorkGuard {
    /// Nonblocking host bridge for MLX work that does not call the Rust generator.
    pub fn try_acquire() -> Option<Self> {
        let (lock, _) = STATE.get_or_init(|| (Mutex::new(State::default()), Condvar::new()));
        let mut state = lock.lock().unwrap_or_else(|p| p.into_inner());
        if state.busy {
            None
        } else {
            state.busy = true;
            Some(Self { _private: () })
        }
    }

    pub fn acquire() -> Self {
        Self::acquire_cancellable(|| false).expect("uncancelled gate wait")
    }
    /// Poll cancellation while queued; an active permit is released only by its owner.
    pub fn acquire_cancellable(cancelled: impl Fn() -> bool) -> Option<Self> {
        let (lock, ready) = STATE.get_or_init(|| (Mutex::new(State::default()), Condvar::new()));
        let mut state = lock.lock().unwrap_or_else(|p| p.into_inner());
        while state.busy {
            if cancelled() {
                return None;
            }
            state = ready
                .wait_timeout(state, std::time::Duration::from_millis(50))
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        if cancelled() {
            return None;
        }
        state.busy = true;
        Some(Self { _private: () })
    }
}

impl Drop for HeavyWorkGuard {
    fn drop(&mut self) {
        let Some((lock, ready)) = STATE.get() else {
            return;
        };
        let mut state = lock.lock().unwrap_or_else(|p| p.into_inner());
        state.busy = false;
        ready.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::HeavyWorkGuard;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn a_second_heavy_job_waits_until_the_first_releases() {
        let first = HeavyWorkGuard::acquire();
        let (started, received) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _second = HeavyWorkGuard::acquire();
            started.send(()).expect("test receiver must remain alive");
        });

        assert!(received.recv_timeout(Duration::from_millis(50)).is_err());
        drop(first);
        assert!(received.recv_timeout(Duration::from_secs(1)).is_ok());
        worker.join().expect("heavy worker must not panic");
    }
    #[test]
    fn queued_cancellation_does_not_release_active_permit() {
        let first = HeavyWorkGuard::acquire();
        let waiter = thread::spawn(|| HeavyWorkGuard::acquire_cancellable(|| true).is_none());
        assert!(waiter.join().unwrap());
        assert!(HeavyWorkGuard::try_acquire().is_none());
        drop(first);
    }
}
