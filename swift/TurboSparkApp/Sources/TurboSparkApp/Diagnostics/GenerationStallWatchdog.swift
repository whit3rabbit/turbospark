import Foundation

/// Pure state machine for the engine stall watchdog (OC-14).
///
/// "Stalled" means: a local generate call is open, the app is waiting for its
/// next stream event, and no event of any kind (prefill progress, content,
/// reasoning, tool call, stop, finish) has arrived for `window` seconds. A
/// prefill progress event is activity, so a long prefill that reports
/// progress never trips it. Whether one large prefill chunk can stay silent
/// longer than the window is UNVERIFIED (see the draft's Implementation
/// record); Keep Waiting is the answer to that case.
///
/// Time is a plain `TimeInterval` the caller supplies, taken from a monotonic
/// clock that does not advance while the Mac sleeps, so a lid close during a
/// turn is not read as a hang. Nothing here reads a clock, spawns a thread or
/// touches the engine, so tests drive it with literal numbers.
struct GenerationStallMonitor: Equatable {
    /// 90 seconds, fixed. A configurable window was cut in review (OC-14
    /// Implementation record): it invites a setting nobody can reason about.
    static let defaultWindow: TimeInterval = 90

    /// What a poll found.
    struct Stall: Equatable {
        /// Seconds since the last real stream event (or since the call
        /// opened), so a second report after Keep Waiting reads larger.
        var silenceSeconds: TimeInterval
    }

    let window: TimeInterval
    private(set) var isArmed = false
    private var lastActivity: TimeInterval = 0
    /// Where the current window starts: the later of the last event and the
    /// last Keep Waiting. Separate from `lastActivity` so the reported
    /// silence stays honest after a re-arm.
    private var windowStart: TimeInterval = 0
    private(set) var reported = false

    init(window: TimeInterval = GenerationStallMonitor.defaultWindow) {
        self.window = window
    }

    /// A generate call opened. Arms the monitor.
    mutating func begin(now: TimeInterval) {
        isArmed = true
        lastActivity = now
        windowStart = now
        reported = false
    }

    /// Any stream event arrived.
    mutating func noteActivity(now: TimeInterval) {
        guard isArmed else { return }
        lastActivity = now
        windowStart = now
        reported = false
    }

    /// The generate call ended (finished, cancelled or threw). Disarms.
    mutating func end() {
        isArmed = false
        reported = false
    }

    /// The user chose Keep Waiting: another full window before the next
    /// report.
    mutating func keepWaiting(now: TimeInterval) {
        guard isArmed else { return }
        windowStart = now
        reported = false
    }

    /// Returns a stall at most once per window. The caller polls about once a
    /// second.
    mutating func poll(now: TimeInterval) -> Stall? {
        guard isArmed, !reported, now - windowStart >= window else { return nil }
        reported = true
        return Stall(silenceSeconds: max(0, now - lastActivity))
    }
}

/// Owns one monitor plus the once-a-second poll that feeds it, on the main
/// actor.
///
/// The draft sketched a separate watchdog thread. The stream's events are
/// consumed on the main actor and a wedged ENGINE thread does not block it,
/// so a main-actor `Task` sees every event stamp and every tick without a
/// lock. A wedged MAIN thread is a different failure (the deferred
/// heartbeat in OC-14): nothing here can present UI then.
///
/// Safety: this only reads what the consumer loop tells it and posts
/// `onStall`. It never calls into the engine.
@MainActor
final class GenerationStallWatchdog {
    private var monitor: GenerationStallMonitor
    private let clock: () -> TimeInterval
    private let onStall: (TimeInterval) -> Void
    private let onRecovered: () -> Void
    private var pollTask: Task<Void, Never>?
    /// True between a report and the next event, Keep Waiting or end.
    private(set) var isShowingStall = false
    /// True between `begin` and `end`. Read by the arming-balance tests.
    var isArmed: Bool { monitor.isArmed }

    /// Seconds since boot, not counting sleep.
    nonisolated static let defaultClock: () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }

    init(
        window: TimeInterval = GenerationStallMonitor.defaultWindow,
        clock: @escaping () -> TimeInterval = GenerationStallWatchdog.defaultClock,
        onStall: @escaping (TimeInterval) -> Void,
        onRecovered: @escaping () -> Void
    ) {
        self.monitor = GenerationStallMonitor(window: window)
        self.clock = clock
        self.onStall = onStall
        self.onRecovered = onRecovered
    }

    deinit { pollTask?.cancel() }

    /// A generate call opened. `pollInterval` nil means the caller drives
    /// `tick()` itself (tests).
    func begin(pollInterval: TimeInterval? = 1) {
        monitor.begin(now: clock())
        pollTask?.cancel()
        pollTask = nil
        guard let pollInterval else { return }
        pollTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(pollInterval))
                if Task.isCancelled { return }
                self?.tick()
            }
        }
    }

    /// One stream event arrived (any kind).
    func noteActivity() {
        monitor.noteActivity(now: clock())
        clearShown()
    }

    /// The generate call ended.
    func end() {
        monitor.end()
        pollTask?.cancel()
        pollTask = nil
        clearShown()
    }

    /// Keep Waiting: hide the notice and watch for another window.
    func keepWaiting() {
        monitor.keepWaiting(now: clock())
        isShowingStall = false
    }

    /// One poll. Public so tests and the task share the path.
    func tick() {
        // Re-checked here, on the same actor that applies events, so a turn
        // that ended between two ticks can never present a stale notice.
        guard let stall = monitor.poll(now: clock()) else { return }
        isShowingStall = true
        onStall(stall.silenceSeconds)
    }

    private func clearShown() {
        guard isShowingStall else { return }
        isShowingStall = false
        onRecovered()
    }
}
