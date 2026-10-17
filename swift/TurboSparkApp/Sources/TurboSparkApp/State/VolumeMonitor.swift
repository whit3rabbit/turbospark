import AppKit
import Foundation

/// One mounted volume, as far as the system lets the monitor cheaply say.
struct VolumeDescriptor: Equatable, Sendable {
    let url: URL
    /// The user-visible name from the workspace notification.
    let name: String?
    /// The BSD name of the volume's device ("disk4s1"), when it was readable.
    let bsdName: String?
    /// The volume UUID string, when it was readable.
    let uuid: String?
}

/// What happened to a volume. The monitor only reports; the debounce, the
/// refresh and every call into the engine belong to the consumer (task 5.6).
enum VolumeEvent: Equatable, Sendable {
    case mounted(VolumeDescriptor)
    /// The OS is about to unmount. Advisory: it cannot hold the unmount (that
    /// is the approval callback's job). It fires on EVERY unmount attempt,
    /// including one the approval callback dissents from, so the volume may
    /// stay mounted. Only `didUnmount` means the volume is gone.
    case willUnmount(VolumeDescriptor)
    /// The volume is gone. The descriptor carries what the monitor learned
    /// while the volume was still readable, because nothing can be read now.
    case didUnmount(VolumeDescriptor)
    case renamed(old: VolumeDescriptor, new: VolumeDescriptor)
}

/// What a read of a volume can return. Both fields are best effort.
struct VolumeDetails: Equatable, Sendable {
    var bsdName: String?
    var uuid: String?
}

/// The disk an unmount request is about, as DiskArbitration names it.
struct DiskIdentity: Equatable, Sendable {
    /// "disk4s1": the volume's own device.
    let bsdName: String?
    /// "disk4": the disk the eject coordinator ejects as a whole. For an APFS
    /// volume this is the synthesized container, not the physical store.
    let wholeDiskBSDName: String?
    let volumeURL: URL?
}

/// An objection to an unmount. `reason` is for the system's own dissent text.
struct UnmountDissent: Equatable, Sendable {
    let reason: String
}

/// Decides whether an unmount may go ahead (the eject coordinator, task 5.11).
///
/// **Threading.** `unmountRequested` is called synchronously on a private
/// worker queue, never on the main thread and never on the DiskArbitration
/// queue, and the caller waits only `VolumeMonitor.approvalTimeout` (100 ms)
/// for the answer. A delegate that needs the main actor should hop to it and
/// wait for it: if the main actor is busy the wait is abandoned and the unmount
/// is approved, so it can never deadlock and can never hold a drive hostage.
/// Dissent only when the answer is already known to be "this disk backs a
/// loaded model".
protocol UnmountApprovalDelegate: AnyObject, Sendable {
    func unmountRequested(for disk: DiskIdentity) -> UnmountDissent?
}

/// The system side of unmount approval, behind a seam so tests can play the
/// DiskArbitration callback. `DiskArbitrationApprovalSource` is the live one.
protocol UnmountApprovalSource: AnyObject, Sendable {
    /// Starts asking `handler` about each unmount. The handler may run on any
    /// thread and must answer within about 100 ms. Starting twice is a no-op.
    func start(_ handler: @escaping @Sendable (DiskIdentity) -> UnmountDissent?)
    /// Unregisters. Once this returns the handler is never called again.
    func stop()
}

/// Turns OS volume events into a typed stream and routes unmount approval to a
/// delegate (requirements 4.1, 4.6, 6.1 to 6.3).
///
/// It is deliberately thin: it never touches the engine and never touches the
/// main actor. Workspace notifications are handled on the monitor's own serial
/// queue, in the order they were posted, and published to every stream handed
/// out by `events()`.
final class VolumeMonitor: @unchecked Sendable {
    /// How long the approval callback waits for the delegate. The timeout path
    /// APPROVES: a stuck delegate must never block an unmount.
    static let approvalTimeout: TimeInterval = 0.1

    private static let notificationNames: [Notification.Name] = [
        NSWorkspace.didMountNotification,
        NSWorkspace.willUnmountNotification,
        NSWorkspace.didUnmountNotification,
        NSWorkspace.didRenameVolumeNotification,
    ]

    private let center: NotificationCenter
    private let source: UnmountApprovalSource
    private let core: Core
    private let queue: OperationQueue
    private let lock = NSLock()
    private var observers: [NSObjectProtocol] = []
    private var running = false

    /// Asked about each unmount while the monitor runs. Held weakly: the
    /// coordinator that owns the monitor is usually the delegate.
    var delegate: UnmountApprovalDelegate? {
        get { core.delegate }
        set { core.delegate = newValue }
    }

    /// - Parameters:
    ///   - notificationCenter: `NSWorkspace.shared.notificationCenter` in the
    ///     app. Tests pass their own center and post synthetic notifications.
    ///   - approvalSource: the DiskArbitration session in the app, a fake in tests.
    ///   - readDetails: reads a mounted volume's BSD name and UUID.
    init(
        notificationCenter: NotificationCenter = NSWorkspace.shared.notificationCenter,
        approvalSource: UnmountApprovalSource = DiskArbitrationApprovalSource(),
        readDetails: @escaping @Sendable (URL) -> VolumeDetails = { VolumeMonitor.defaultReadDetails($0) },
        approvalTimeout: TimeInterval = VolumeMonitor.approvalTimeout
    ) {
        center = notificationCenter
        source = approvalSource
        core = Core(readDetails: readDetails, approvalTimeout: approvalTimeout)
        queue = OperationQueue()
        queue.name = "app.turbospark.volume-monitor"
        // One at a time keeps events in the order the OS posted them.
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .utility
    }

    deinit {
        stop()
        core.hub.finish()
    }

    /// A new stream of every event from now on. Each caller gets its own, so
    /// the library state and the eject coordinator can both listen. A stream
    /// ends when its consumer stops iterating (do not reuse one after a
    /// `break`), or when the monitor is deallocated.
    func events() -> AsyncStream<VolumeEvent> {
        core.hub.stream()
    }

    /// Starts observing. Calling it while running does nothing.
    func start() {
        lock.lock(); defer { lock.unlock() }
        guard !running else { return }
        running = true
        let generation = core.beginRun()
        // The handlers capture the core, never `self`: the monitor must be
        // able to deallocate (and unregister in deinit) while started.
        let core = self.core
        observers = Self.notificationNames.map { name in
            center.addObserver(forName: name, object: nil, queue: queue) { note in
                core.handle(note, generation: generation)
            }
        }
        source.start { disk in core.decideUnmount(disk) }
    }

    /// Stops observing and unregisters the approval callback. Safe to call
    /// repeatedly, and to start again afterwards.
    func stop() {
        lock.lock(); defer { lock.unlock() }
        guard running else { return }
        running = false
        for observer in observers { center.removeObserver(observer) }
        observers = []
        // Before the source stops, so a callback still in flight approves
        // instead of asking a delegate of a monitor that has been stopped.
        core.endRun()
        source.stop()
    }

    // MARK: Default volume reader

    /// The BSD name from `statfs` and the UUID from the volume's resource
    /// values. Both are cheap and neither is available once the volume is gone.
    static func defaultReadDetails(_ url: URL) -> VolumeDetails {
        let uuid = (try? url.resourceValues(forKeys: [.volumeUUIDStringKey]))?.volumeUUIDString
        var info = statfs()
        var bsdName: String?
        if statfs(url.path, &info) == 0 {
            var from = info.f_mntfromname
            let device = withUnsafeBytes(of: &from) { raw in
                String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
            }
            // Only a real device names a BSD disk; "map auto_home" does not.
            if device.hasPrefix("/dev/") { bsdName = String(device.dropFirst("/dev/".count)) }
        }
        return VolumeDetails(bsdName: bsdName, uuid: uuid)
    }
}

// MARK: - Shared state

private extension VolumeMonitor {
    /// Everything the notification handlers and the approval callback share.
    /// It never references the monitor, so the monitor can deallocate.
    final class Core: @unchecked Sendable {
        let hub = EventHub()
        private let readDetails: @Sendable (URL) -> VolumeDetails
        private let approvalTimeout: TimeInterval
        /// Delegate calls run here, so a slow delegate holds a worker thread
        /// and nothing else.
        private let workers = DispatchQueue(
            label: "app.turbospark.volume-monitor.approval-workers", qos: .userInitiated, attributes: .concurrent)

        private let lock = NSLock()
        private var generation = 0
        private var active = false
        private weak var weakDelegate: UnmountApprovalDelegate?
        /// What was learned about each mounted volume, by mount path.
        private var known: [String: VolumeDescriptor] = [:]

        init(readDetails: @escaping @Sendable (URL) -> VolumeDetails, approvalTimeout: TimeInterval) {
            self.readDetails = readDetails
            self.approvalTimeout = approvalTimeout
        }

        var delegate: UnmountApprovalDelegate? {
            get { lock.lock(); defer { lock.unlock() }; return weakDelegate }
            set { lock.lock(); weakDelegate = newValue; lock.unlock() }
        }

        /// Opens a run and returns its generation. Work from an earlier run
        /// that is still queued carries an old generation and is dropped.
        func beginRun() -> Int {
            lock.lock(); defer { lock.unlock() }
            generation += 1
            active = true
            known.removeAll()
            return generation
        }

        func endRun() {
            lock.lock(); defer { lock.unlock() }
            generation += 1
            active = false
            known.removeAll()
        }

        private func isCurrent(_ generation: Int) -> Bool {
            lock.lock(); defer { lock.unlock() }
            return active && generation == self.generation
        }

        /// Publishes under the same lock `endRun` takes, so no event can be
        /// yielded after `stop()` has returned.
        private func emit(_ event: VolumeEvent, generation: Int) {
            lock.lock(); defer { lock.unlock() }
            guard active, generation == self.generation else { return }
            hub.yield(event)
        }

        // MARK: Workspace notifications

        func handle(_ note: Notification, generation: Int) {
            guard isCurrent(generation) else { return }
            let info = note.userInfo
            guard let url = info?[NSWorkspace.volumeURLUserInfoKey] as? URL else { return }
            let name = info?[NSWorkspace.localizedVolumeNameUserInfoKey] as? String

            switch note.name {
            case NSWorkspace.didMountNotification:
                // A fresh volume: nothing known about it can be trusted.
                let volume = describe(url, name: name, fallback: nil)
                remember(volume)
                emit(.mounted(volume), generation: generation)

            case NSWorkspace.willUnmountNotification:
                // Still mounted, so this is the last chance to read it.
                let volume = describe(url, name: name, fallback: recall(url))
                remember(volume)
                emit(.willUnmount(volume), generation: generation)

            case NSWorkspace.didUnmountNotification:
                let volume = forget(url, name: name)
                emit(.didUnmount(volume), generation: generation)

            case NSWorkspace.didRenameVolumeNotification:
                // Both keys are documented; a missing old URL means only the
                // name changed.
                let oldURL = info?[NSWorkspace.oldVolumeURLUserInfoKey] as? URL ?? url
                let oldName = info?[NSWorkspace.oldLocalizedVolumeNameUserInfoKey] as? String
                let old = forget(oldURL, name: oldName)
                let new = describe(url, name: name, fallback: old)
                remember(new)
                emit(.renamed(old: old, new: new), generation: generation)

            default:
                break
            }
        }

        private func describe(_ url: URL, name: String?, fallback: VolumeDescriptor?) -> VolumeDescriptor {
            let read = readDetails(url)
            return VolumeDescriptor(
                url: url,
                name: name ?? fallback?.name,
                bsdName: read.bsdName ?? fallback?.bsdName,
                uuid: read.uuid ?? fallback?.uuid)
        }

        private func recall(_ url: URL) -> VolumeDescriptor? {
            lock.lock(); defer { lock.unlock() }
            return known[url.path]
        }

        private func remember(_ volume: VolumeDescriptor) {
            lock.lock(); defer { lock.unlock() }
            known[volume.url.path] = volume
        }

        /// What the monitor knew about a volume that is leaving, with the
        /// notification's own URL and name winning.
        private func forget(_ url: URL, name: String?) -> VolumeDescriptor {
            lock.lock(); defer { lock.unlock() }
            let remembered = known.removeValue(forKey: url.path)
            return VolumeDescriptor(
                url: url, name: name ?? remembered?.name, bsdName: remembered?.bsdName, uuid: remembered?.uuid)
        }

        // MARK: Unmount approval

        /// The body of the DiskArbitration callback. Runs on the DA queue and
        /// returns within `approvalTimeout` whatever the delegate does.
        func decideUnmount(_ disk: DiskIdentity) -> UnmountDissent? {
            // Stopped, or no delegate (or one that is gone): nothing objects.
            guard let delegate = activeDelegate() else { return nil }
            let slot = AnswerSlot()
            // The delegate runs on a worker, never here and never on main, and
            // this thread waits a bounded time for it. If the delegate needs
            // the main actor and main is busy, the wait ends and the unmount
            // is approved; main is never blocked by this code.
            workers.async { slot.fill(delegate.unmountRequested(for: disk)) }
            return slot.answer(waitingUpTo: approvalTimeout)
        }

        private func activeDelegate() -> UnmountApprovalDelegate? {
            lock.lock(); defer { lock.unlock() }
            return active ? weakDelegate : nil
        }
    }

    /// One delegate answer, or the lack of one. Whichever of the answer and
    /// the deadline comes first decides, atomically.
    final class AnswerSlot: @unchecked Sendable {
        private let lock = NSLock()
        private let ready = DispatchSemaphore(value: 0)
        /// `nil`: no answer yet. `.some(nil)`: the delegate approved.
        private var stored: UnmountDissent??
        private var closed = false

        func fill(_ answer: UnmountDissent?) {
            lock.lock()
            if !closed { stored = .some(answer) }
            lock.unlock()
            ready.signal()
        }

        /// Waits up to `timeout`, then closes the slot. A late answer is
        /// dropped, and a missing one means approve (nil).
        func answer(waitingUpTo timeout: TimeInterval) -> UnmountDissent? {
            _ = ready.wait(timeout: .now() + timeout)
            lock.lock(); defer { lock.unlock() }
            closed = true
            return stored ?? nil
        }
    }

    /// Fans events out to any number of streams.
    final class EventHub: @unchecked Sendable {
        private let lock = NSLock()
        private var continuations: [UUID: AsyncStream<VolumeEvent>.Continuation] = [:]
        private var finished = false

        func stream() -> AsyncStream<VolumeEvent> {
            AsyncStream { continuation in
                let id = UUID()
                lock.lock()
                if finished {
                    lock.unlock()
                    continuation.finish()
                    return
                }
                continuations[id] = continuation
                lock.unlock()
                continuation.onTermination = { [weak self] _ in self?.remove(id) }
            }
        }

        func yield(_ event: VolumeEvent) {
            lock.lock(); defer { lock.unlock() }
            for continuation in continuations.values { continuation.yield(event) }
        }

        func finish() {
            lock.lock()
            finished = true
            let all = Array(continuations.values)
            continuations.removeAll()
            lock.unlock()
            for continuation in all { continuation.finish() }
        }

        private func remove(_ id: UUID) {
            lock.lock(); continuations.removeValue(forKey: id); lock.unlock()
        }
    }
}
