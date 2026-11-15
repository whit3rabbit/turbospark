import Foundation

/// Bounds how often the app may relaunch itself, across processes (OC-14).
///
/// A relaunch is always the user's click, but a build that stalls again right
/// after the relaunch would invite a click loop, and a faulty caller must not
/// be able to spin one. The record is a small file under the store root: the
/// times of recent attempts, nothing else (no pid, path or chat data, never
/// sent anywhere).
///
/// Policy: at most `maxAttempts` within `windowSeconds`. A record that cannot
/// be WRITTEN refuses the attempt (fail closed: without it the bound cannot
/// be proven). A record that cannot be READ is replaced and the attempt
/// allowed once; the fresh file then enforces the cap, so a corrupt file
/// cannot allow a storm.
final class RelaunchGuard: @unchecked Sendable {
    static let fileName = "relaunch-guard.json"
    static let maxAttempts = 2
    static let windowSeconds: TimeInterval = 10 * 60

    struct Record: Codable, Equatable {
        var version: Int
        var attempts: [Date]
    }

    private let url: URL
    private let maxAttempts: Int
    private let window: TimeInterval
    private let lock = NSLock()

    init(
        directory: URL,
        maxAttempts: Int = RelaunchGuard.maxAttempts,
        window: TimeInterval = RelaunchGuard.windowSeconds
    ) {
        self.url = directory.appendingPathComponent(Self.fileName)
        self.maxAttempts = maxAttempts
        self.window = window
    }

    /// Records an attempt and returns true when it is within the cap.
    func recordAttempt(now: Date) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        var recent = readAttempts().filter { now.timeIntervalSince($0) < window && $0 <= now }
        guard recent.count < maxAttempts else { return false }
        recent.append(now)
        let record = Record(version: 1, attempts: recent)
        guard let data = try? Self.encoder.encode(record),
              (try? data.write(to: url, options: .atomic)) != nil
        else { return false }
        return true
    }

    private func readAttempts() -> [Date] {
        guard let data = try? Data(contentsOf: url),
              let record = try? Self.decoder.decode(Record.self, from: data)
        else { return [] }
        return record.attempts
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()
    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}

/// Persist first, then relaunch, with a hard bound on how long quitting can
/// take (OC-14).
///
/// `AppModel.shutdown()` is one synchronous main-actor function, so a step
/// that blocks cannot be timed out or skipped from inside it. The guard is
/// therefore applied around it:
///
/// 1. the attempt is counted (`RelaunchGuard`), refusing past the cap;
/// 2. a detached helper is spawned that waits (bounded) for this pid to
///    exit and then opens the bundle;
/// 3. a hard exit is armed for `hardExitAfter` seconds on a background
///    queue, the one thing that still runs if the main thread wedges;
/// 4. the chats and settings are flushed (`flush`), the part that matters;
/// 5. the ordinary quit path runs (`terminate`), which clears the crash
///    session marker last.
///
/// The hard exit clears the marker only when the flush finished, so a quit
/// that lost data still produces next launch's "did not quit cleanly"
/// notice. Every side effect is a seam so tests never relaunch or exit the
/// test process.
@MainActor
final class GuardedRelaunch {
    enum Outcome: Equatable {
        /// The helper is spawned and the quit has started.
        case relaunching
        /// Too many recent attempts, or the attempt could not be recorded.
        case refusedTooManyAttempts
        /// No relaunch is possible (not an app bundle, or the helper would
        /// not start). Nothing was flushed or terminated.
        case unavailable
        /// A relaunch is already under way; this click did nothing.
        case alreadyInProgress
    }

    static let hardExitAfter: TimeInterval = 10

    struct Seams {
        var now: () -> Date
        var canRelaunch: () -> Bool
        var spawnHelper: () throws -> Void
        var flush: () -> Void
        var terminate: () -> Void
        var endSession: () -> Void
        var forceExit: () -> Void
        /// Runs the closure after the delay on a queue other than main.
        var scheduleHardExit: (TimeInterval, @escaping @Sendable () -> Void) -> Void
    }

    private let attemptGuard: RelaunchGuard
    private let seams: Seams
    private var inProgress = false

    init(attemptGuard: RelaunchGuard, seams: Seams) {
        self.attemptGuard = attemptGuard
        self.seams = seams
    }

    func perform() -> Outcome {
        guard !inProgress else { return .alreadyInProgress }
        guard seams.canRelaunch() else { return .unavailable }
        guard attemptGuard.recordAttempt(now: seams.now()) else { return .refusedTooManyAttempts }
        inProgress = true
        do {
            try seams.spawnHelper()
        } catch {
            inProgress = false
            return .unavailable
        }

        let flushed = FlushFlag()
        let endSession = seams.endSession
        let forceExit = seams.forceExit
        seams.scheduleHardExit(Self.hardExitAfter) {
            if flushed.isSet { endSession() }
            forceExit()
        }
        seams.flush()
        flushed.set()
        seams.terminate()
        return .relaunching
    }

    /// Set once from the main actor, read from the hard-exit queue.
    private final class FlushFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
        func set() { lock.lock(); value = true; lock.unlock() }
    }
}

extension GuardedRelaunch.Seams {
    /// The real thing: `/bin/sh` waits for this pid (at most 30 s) and then
    /// opens the bundle with `/usr/bin/open`. Plain `open`, never `-n`, so a
    /// helper that outlives a stuck process only re-activates it instead of
    /// starting a second instance on the same store root. The pid and path
    /// are positional arguments, never spliced into the script text.
    @MainActor
    static func live(
        flush: @escaping () -> Void,
        terminate: @escaping () -> Void,
        endSession: @escaping () -> Void
    ) -> GuardedRelaunch.Seams {
        let bundleURL = Bundle.main.bundleURL
        return GuardedRelaunch.Seams(
            now: { Date() },
            // `swift run` is a bare executable; opening it by path starts
            // nothing useful.
            canRelaunch: { bundleURL.pathExtension == "app" },
            spawnHelper: {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/bin/sh")
                process.arguments = [
                    "-c",
                    "i=0; while kill -0 \"$1\" 2>/dev/null && [ \"$i\" -lt 300 ]; do sleep 0.1; i=$((i+1)); done; exec /usr/bin/open \"$2\"",
                    "turbospark-relaunch",
                    String(ProcessInfo.processInfo.processIdentifier),
                    bundleURL.path,
                ]
                process.standardInput = FileHandle.nullDevice
                process.standardOutput = FileHandle.nullDevice
                process.standardError = FileHandle.nullDevice
                try process.run()
            },
            flush: flush,
            terminate: terminate,
            endSession: endSession,
            // `_exit`: `exit` runs atexit handlers that can wait on the very
            // locks a wedged main thread holds.
            forceExit: { _exit(0) },
            scheduleHardExit: { delay, work in
                DispatchQueue.global(qos: .userInitiated).asyncAfter(
                    deadline: .now() + delay, execute: work)
            }
        )
    }
}
