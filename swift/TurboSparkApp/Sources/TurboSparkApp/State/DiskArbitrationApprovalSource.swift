import DiskArbitration
import Foundation

/// The live unmount-approval source: a DiskArbitration session whose
/// `DADiskUnmountApprovalCallback` asks a handler about every unmount.
///
/// **Threads.** The session is scheduled on one private serial queue, and the
/// callback runs there. The handler is expected to answer within about 100 ms
/// (`VolumeMonitor` enforces that), because DiskArbitration holds the unmount
/// until every approval callback has returned.
///
/// **Lifetime.** The callback's context pointer is an unmanaged, retained
/// `Context` box. `stop()` deactivates the box (so a callback already in
/// flight answers "approve"), unregisters and unschedules the session on the
/// DA queue itself (so no callback runs concurrently with the teardown or
/// after it), and only then releases the box.
///
/// **Permissions.** DiskArbitration needs no entitlement in an unsandboxed
/// app, and TurboSpark is not sandboxed (the bundle is signed with no
/// entitlements file; see the task 5.3 notes).
final class DiskArbitrationApprovalSource: UnmountApprovalSource, @unchecked Sendable {
    /// What the retained context pointer points at. The handler is dropped on
    /// `deactivate()`, which also breaks any cycle through its captures.
    final class Context: @unchecked Sendable {
        private let lock = NSLock()
        private var handler: (@Sendable (DiskIdentity) -> UnmountDissent?)?

        init(_ handler: @escaping @Sendable (DiskIdentity) -> UnmountDissent?) {
            self.handler = handler
        }

        /// A deactivated context approves everything.
        func decide(_ disk: DiskIdentity) -> UnmountDissent? {
            lock.lock()
            let current = handler
            lock.unlock()
            return current?(disk)
        }

        func deactivate() {
            lock.lock(); handler = nil; lock.unlock()
        }
    }

    /// A C function pointer, so it cannot capture: everything it needs comes
    /// through the context. It answers a retained dissenter (DiskArbitration
    /// releases it) or nil to approve.
    static let unmountApprovalCallback: DADiskUnmountApprovalCallback = { disk, context in
        guard let context else { return nil }
        let box = Unmanaged<Context>.fromOpaque(context).takeUnretainedValue()
        guard let dissent = box.decide(DiskIdentity(disk: disk)) else { return nil }
        // Busy is what a process holding the volume open would cause, and is
        // the status the Finder reports as "in use".
        let dissenter = DADissenterCreate(kCFAllocatorDefault, DAReturn(kDAReturnBusy), dissent.reason as CFString)
        return Unmanaged.passRetained(dissenter)
    }

    private struct Registration {
        let session: DASession
        /// +1 retained `Context`. The box has no other owner.
        let pointer: UnsafeMutableRawPointer
    }

    private static let queueKey = DispatchSpecificKey<Void>()
    private let queue = DispatchQueue(label: "app.turbospark.volume-monitor.disk-arbitration", qos: .userInitiated)
    private let lock = NSLock()
    private var registration: Registration?

    init() {
        queue.setSpecific(key: Self.queueKey, value: ())
    }

    deinit {
        stop()
    }

    /// The live context, for tests.
    var activeContext: Context? {
        lock.lock(); defer { lock.unlock() }
        return registration.map { Unmanaged<Context>.fromOpaque($0.pointer).takeUnretainedValue() }
    }

    func start(_ handler: @escaping @Sendable (DiskIdentity) -> UnmountDissent?) {
        lock.lock(); defer { lock.unlock() }
        guard registration == nil else { return }
        // No session means no approvals; every unmount then proceeds, which is
        // the same as not being installed at all.
        guard let session = DASessionCreate(kCFAllocatorDefault) else { return }
        let pointer = Unmanaged.passRetained(Context(handler)).toOpaque()
        DARegisterDiskUnmountApprovalCallback(session, nil, Self.unmountApprovalCallback, pointer)
        DASessionSetDispatchQueue(session, queue)
        registration = Registration(session: session, pointer: pointer)
    }

    func stop() {
        lock.lock()
        let current = registration
        registration = nil
        lock.unlock()
        guard let current else { return }

        Unmanaged<Context>.fromOpaque(current.pointer).takeUnretainedValue().deactivate()
        let teardown = {
            // On the DA queue, so it is serialized with callback delivery:
            // after this block nothing can call into the context.
            DAUnregisterCallback(
                current.session, unsafeBitCast(Self.unmountApprovalCallback, to: UnsafeMutableRawPointer.self),
                current.pointer)
            DASessionSetDispatchQueue(current.session, nil)
        }
        if DispatchQueue.getSpecific(key: Self.queueKey) != nil {
            // Called from inside a callback: a sync hop would deadlock.
            teardown()
        } else {
            // This waits for a callback in flight, at most the approval
            // timeout, because the monitor deactivates the handler first.
            queue.sync(execute: teardown)
        }
        Unmanaged<Context>.fromOpaque(current.pointer).release()
    }
}

extension DiskIdentity {
    /// Reads a disk's names from the DADisk the callback received. All of it
    /// is held by the disk object already, so nothing here blocks.
    init(disk: DADisk) {
        let bsdName = DADiskGetBSDName(disk).map { String(cString: $0) }
        // The C string belongs to the DADisk, so the disk must outlive the
        // copy. Without this the whole disk was released first and the name
        // read back as garbage (found by the live eject smoke).
        var wholeDisk: String?
        if let whole = DADiskCopyWholeDisk(disk) {
            wholeDisk = withExtendedLifetime(whole) { DADiskGetBSDName(whole).map { String(cString: $0) } }
        }
        let description = DADiskCopyDescription(disk) as? [String: Any]
        let volumeURL = description?[kDADiskDescriptionVolumePathKey as String] as? URL
        self.init(bsdName: bsdName, wholeDiskBSDName: wholeDisk, volumeURL: volumeURL)
    }
}
