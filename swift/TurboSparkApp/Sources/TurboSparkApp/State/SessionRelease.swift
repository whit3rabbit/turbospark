import Foundation

/// A weak reference to any class instance, so a caller can ask "is it gone?"
/// without keeping it alive.
final class WeakObject {
    weak var value: AnyObject?
    init(_ value: AnyObject?) { self.value = value }
}

/// Waiting until a released object is REALLY gone.
///
/// Setting a property to nil releases only the owner's reference. A task that
/// captured the object keeps it (and, for an engine session, the mapped
/// weights) alive until that task ends, so "I dropped it" proves nothing about
/// the drive being free. The truth is whether the weak reference is nil.
enum SessionRelease {
    static let defaultInterval: Duration = .milliseconds(25)

    /// Polls until `ref` is nil. `settle` runs before every look, so work that
    /// keeps the object alive is cancelled again if more of it started. True
    /// when the object went away before the patience ran out.
    @MainActor
    static func waitUntilGone(
        _ ref: WeakObject, patience: Duration, interval: Duration = defaultInterval,
        settle: () -> Void = {}
    ) async -> Bool {
        let deadline = ContinuousClock.now + patience
        while true {
            settle()
            if ref.value == nil { return true }
            // A cancelled caller must not spin: `Task.sleep` would throw at once.
            if ContinuousClock.now >= deadline || Task.isCancelled { return false }
            try? await Task.sleep(for: interval)
        }
    }

    /// Takes the strong reference `get` returns out of its owner with `clear`,
    /// then waits for the object to go. Returns nil when it did, and the weak
    /// reference to what is still holding on when it did not.
    @MainActor
    static func dropAndWait<T: AnyObject>(
        get: () -> T?, clear: () -> Void, patience: Duration,
        interval: Duration = defaultInterval, settle: () -> Void = {}
    ) async -> WeakObject? {
        let ref = WeakObject(get())
        clear()
        if await waitUntilGone(ref, patience: patience, interval: interval, settle: settle) { return nil }
        return ref
    }
}

/// Tries a step again, for a bounded time, for every id that has not yet
/// succeeded.
enum BoundedRetry {
    /// Calls `attempt` for each pending id, then again every `interval` until
    /// every id has succeeded or `patience` has passed. Returns the ids that
    /// never did. The first attempt is immediate.
    @MainActor
    static func drain(
        _ ids: [String], patience: Duration, interval: Duration,
        attempt: (String) -> Bool
    ) async -> [String] {
        var pending = ids
        let deadline = ContinuousClock.now + patience
        while true {
            pending = pending.filter { !attempt($0) }
            if pending.isEmpty { return [] }
            if ContinuousClock.now >= deadline || Task.isCancelled { return pending }
            try? await Task.sleep(for: interval)
        }
    }
}

/// A session taken out of its owner but not closed yet. The owner no longer
/// lists it; the object stays alive for as long as something is closing it, so
/// a weak reference to this says whether the close has really finished.
final class DetachedSession: @unchecked Sendable {
    private let closer: @Sendable () async -> Void

    init(close: @escaping @Sendable () async -> Void) {
        closer = close
    }

    func close() async { await closer() }
}

/// Where the chat session lives, as the eject path reads and clears it. The
/// app's is `AppModel.session`; a test's is a probe object, so the live release
/// code can run without a multi-gigabyte checkpoint.
@MainActor
struct ChatSessionSlot {
    /// The path of the model the session was opened from, nil when none is open.
    var loadedPath: () -> String?
    /// The session object itself (a strong reference), nil when none is open.
    var object: () -> AnyObject?
    /// Drops the owner's reference. Not a release: whatever else holds the
    /// session keeps it alive.
    var clear: () -> Void
}

/// The audio workspace's resident session, as the eject path reads it.
@MainActor
struct AudioSessionSlot {
    var residentPath: () -> String?
    /// Takes the session out of its owner at once; the caller closes it.
    var detach: () -> DetachedSession?
}

/// Resumes a continuation exactly once, from whichever caller gets there
/// first. The loser is ignored.
private final class ResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Bool, Never>?

    init(_ continuation: CheckedContinuation<Bool, Never>) { self.continuation = continuation }

    func resume(_ value: Bool) {
        lock.lock()
        let waiting = continuation
        continuation = nil
        lock.unlock()
        waiting?.resume(returning: value)
    }
}

/// Waiting for work that may never return, for a bounded time.
enum BoundedAwait {
    /// Runs `work` and returns true if it finished within `patience`, false if
    /// the wait ran out first. The work is NOT cancelled when the wait runs
    /// out: it may be blocked in native code that cannot be interrupted (an
    /// audio session's `close()` waits for the running job), so it carries on
    /// in the background and the caller decides what a late finish means.
    static func finishes(within patience: Duration, _ work: @escaping @Sendable () async -> Void) async -> Bool {
        await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            let outcome = ResumeOnce(continuation)
            let timer = Task {
                // A cancelled timer (the work finished) throws and says nothing.
                do {
                    try await Task.sleep(for: patience)
                    outcome.resume(false)
                } catch {}
            }
            Task {
                await work()
                // Answer before cancelling the timer: a cancelled sleep wakes
                // at once and must not get to answer first.
                outcome.resume(true)
                timer.cancel()
            }
        }
    }
}

/// Sessions the app dropped but something still holds, found by their weak
/// references. The eject coordinator keeps listing them as loaded, so a model
/// that is still mapped is never reported as unloaded.
@MainActor
final class RetiredSessionLedger {
    struct Entry {
        let ref: WeakObject
        let path: String
        let category: LoadedModelCategory
    }

    private var entries: [Entry] = []

    func add(_ ref: WeakObject, path: String, category: LoadedModelCategory = .chat) {
        entries.append(Entry(ref: ref, path: path, category: category))
    }

    /// The entries whose object is still alive (of one category, or all of
    /// them). Forgets the rest.
    func alive(_ category: LoadedModelCategory? = nil) -> [Entry] {
        entries.removeAll { $0.ref.value == nil }
        guard let category else { return entries }
        return entries.filter { $0.category == category }
    }
}
