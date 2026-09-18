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

pub struct HeavyWorkGuard;

impl HeavyWorkGuard {
    pub fn acquire() -> Self {
        let (lock, ready) = STATE.get_or_init(|| (Mutex::new(State::default()), Condvar::new()));
        let mut state = lock.lock().expect("heavy work gate mutex poisoned");
        while state.busy {
            state = ready.wait(state).expect("heavy work gate mutex poisoned");
        }
        state.busy = true;
        Self
    }
}

impl Drop for HeavyWorkGuard {
    fn drop(&mut self) {
        let Some((lock, ready)) = STATE.get() else {
            return;
        };
        let mut state = lock.lock().expect("heavy work gate mutex poisoned");
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
}
