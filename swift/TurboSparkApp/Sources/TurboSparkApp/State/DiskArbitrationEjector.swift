import DiskArbitration
import Foundation
import IOKit

/// BSD disk names, which are only ever "disk" and numbers.
enum DiskNaming {
    /// "disk5s1" and "disk12s3s1" give "disk5" and "disk12"; a whole disk gives
    /// itself. Anything that is not a disk name gives nil.
    static func wholeDisk(ofSlice name: String) -> String? {
        let prefix = "disk"
        guard name.hasPrefix(prefix) else { return nil }
        let rest = name.dropFirst(prefix.count)
        let digits = rest.prefix { $0.isASCII && $0.isNumber }
        guard !digits.isEmpty else { return nil }
        // What follows the number must be nothing, or slices ("s1", "s1s2").
        var tail = rest.dropFirst(digits.count)
        while !tail.isEmpty {
            guard tail.first == "s" else { return nil }
            let slice = tail.dropFirst().prefix { $0.isASCII && $0.isNumber }
            guard !slice.isEmpty else { return nil }
            tail = tail.dropFirst(1 + slice.count)
        }
        return prefix + digits
    }
}

/// Finds the disk a volume's storage really sits on.
///
/// An APFS volume's "whole disk" is the synthesized container (`disk5` for
/// `disk5s1`), which is not ejectable: the drive is the physical store the
/// container sits on (`disk4`). Blocking, so only off the main actor.
protocol PhysicalDiskResolving: Sendable {
    /// The whole disk under `bsdName` (a whole disk or a slice), or nil.
    func physicalWholeDisk(of bsdName: String) -> String?
}

/// The IORegistry walk: from the media of the disk named, up the IOService
/// plane through every `IOMedia` ancestor, to the topmost one. For an exFAT
/// partition that is its own whole disk; for an APFS container it is the
/// physical store's whole disk.
///
/// The first parent is followed at each step (an `IOMedia` chain has one), the
/// walk is bounded, and every registry entry is released on every path.
struct IOKitPhysicalDiskResolver: PhysicalDiskResolving {
    static let maximumDepth = 24

    func physicalWholeDisk(of bsdName: String) -> String? {
        guard let session = DASessionCreate(kCFAllocatorDefault),
            let disk = bsdName.withCString({ DADiskCreateFromBSDName(kCFAllocatorDefault, session, $0) })
        else { return nil }
        var entry = DADiskCopyIOMedia(disk)
        guard entry != 0 else { return nil }
        var topmost: String?
        for _ in 0..<Self.maximumDepth {
            if IOObjectConformsTo(entry, "IOMedia") != 0, let name = Self.bsdName(of: entry) { topmost = name }
            var parent: io_registry_entry_t = 0
            let status = IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent)
            IOObjectRelease(entry)
            guard status == KERN_SUCCESS, parent != 0 else { return topmost.flatMap(DiskNaming.wholeDisk) }
            entry = parent
        }
        IOObjectRelease(entry)
        return topmost.flatMap(DiskNaming.wholeDisk)
    }

    private static func bsdName(of entry: io_registry_entry_t) -> String? {
        IORegistryEntryCreateCFProperty(entry, "BSD Name" as CFString, kCFAllocatorDefault, 0)?
            .takeRetainedValue() as? String
    }
}

/// Reads where a model's files are, off the main actor, for the coordinator's
/// snapshot: the link-resolved path, the volume's mount point and device
/// (`statfs`), and the disks it sits on.
struct LiveModelVolumeLocator: ModelVolumeLocating {
    var physical: PhysicalDiskResolving = IOKitPhysicalDiskResolver()

    func locate(path: String) -> ModelVolume {
        var real = path
        if let resolved = realpath(path, nil) {
            real = String(cString: resolved)
            free(resolved)
        }
        var info = statfs()
        guard statfs(real, &info) == 0 else { return .unresolved(real) }
        let mount = Self.string(from: &info.f_mntonname)
        let device = Self.string(from: &info.f_mntfromname)
        // Only a real device names a BSD disk ("map auto_home" does not).
        let bsd = device.hasPrefix("/dev/") ? String(device.dropFirst("/dev/".count)) : nil
        var disks: Set<String> = []
        if let bsd, let whole = DiskNaming.wholeDisk(ofSlice: bsd) {
            disks.insert(whole)
            // The IORegistry is read only for a drive: the internal volumes can
            // never match, and the walk is the slowest part of a locate.
            if !EjectDecisionState.isInternalMount(EjectDecisionState.normalized(mount)),
                let store = physical.physicalWholeDisk(of: whole)
            {
                disks.insert(store)
            }
        }
        return ModelVolume(realPath: real, mountPoint: mount, bsdName: bsd, diskNames: disks)
    }

    private static func string<T>(from field: inout T) -> String {
        withUnsafeBytes(of: &field) { raw in
            String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
        }
    }
}

/// Unmounts and ejects a disk with DiskArbitration (requirement 6.1).
///
/// **THE RULE.** The coordinator hands over the disk the unmount request named.
/// 1. Unmount the whole disk that request named (`kDADiskUnmountOptionWhole`),
///    which for an APFS volume is the synthesized container: every volume on it
///    goes together.
/// 2. If the drive is a different, physical disk (resolved through the
///    IORegistry, see `IOKitPhysicalDiskResolver`), unmount that too, so a
///    second partition on the same drive does not keep the eject busy.
/// 3. Eject the physical disk (the resolved one, else the one the request
///    named) when its media is ejectable. A fixed disk that is not ejectable is
///    done after step 1 or 2, as the Finder treats it.
/// A step the system refuses ends the sequence with the system's own status
/// text, and nothing is retried.
///
/// **FIXED INTERNAL DEVICES ARE THE EXCEPTION** (review N2). When
/// DiskArbitration says any disk of the sequence is internal
/// (`kDADiskDescriptionDeviceInternalKey`) AND its media is not ejectable
/// (`treatsAsInternal`), only the requested VOLUME is unmounted: no whole-disk
/// unmount, no eject. The result then says so (`internalDiskNotice`), as a
/// failure to eject. An SD card in a built-in reader is an internal device
/// with ejectable media, and ejects like any other card.
///
/// **THREADS.** The DASession is scheduled on a private serial queue, and each
/// DiskArbitration callback runs there. It only completes a one-shot box, which
/// resumes the awaiting task: it takes no lock another thread holds while
/// waiting, and does no work. The awaiting code runs on the cooperative pool,
/// never the main thread.
///
/// **NEVER HANGS.** Each step has a generous timeout (the one-shot box is
/// completed by whichever of the callback and the timer comes first). A step
/// that times out ends the sequence with a failure; its late callback finds
/// the box finished and is ignored, and the session's queue is cleared before
/// the function returns so none arrives after.
///
/// **OBJECT LIFETIMES.** DiskArbitration objects are managed by ARC. The
/// completion box handed to the C callback is retained once for it and released
/// in it; the only leak is that one small box when a callback never fires.
final class DiskArbitrationEjector: DiskEjecting, @unchecked Sendable {
    static let defaultTimeout: TimeInterval = 60

    private let timeout: TimeInterval
    private let physical: PhysicalDiskResolving
    private let queue = DispatchQueue(label: "app.turbospark.eject.disk-arbitration", qos: .userInitiated)

    init(timeout: TimeInterval = DiskArbitrationEjector.defaultTimeout, physical: PhysicalDiskResolving = IOKitPhysicalDiskResolver()) {
        self.timeout = timeout
        self.physical = physical
    }

    // MARK: The sequence

    func unmountAndEject(_ disk: DiskIdentity) async -> EjectResult {
        guard let session = DASessionCreate(kCFAllocatorDefault) else {
            return .failed("DiskArbitration is not available")
        }
        DASessionSetDispatchQueue(session, queue)
        defer { DASessionSetDispatchQueue(session, nil) }
        guard let target = Self.target(for: disk, session: session, physical: physical) else {
            return .failed("the disk could not be found")
        }

        let steps = Self.plan(isInternal: target.isInternal, hasStore: target.store != nil, ejectable: target.ejectable)
        for step in steps {
            let failure: String?
            switch step {
            case .unmountVolume:
                guard let volume = target.volume else { return .failed("the volume could not be found") }
                failure = await unmount(volume, whole: false)
            case .unmountWhole:
                failure = await unmount(target.named, whole: true)
            case .unmountStore:
                guard let store = target.store else { return .failed("the disk could not be found") }
                failure = await unmount(store, whole: true)
            case .eject:
                let result = await perform { context in
                    DADiskEject(
                        target.store ?? target.named, DADiskEjectOptions(kDADiskEjectOptionDefault),
                        Self.ejectCallback, context)
                }
                failure = result.failureText
            }
            if let failure { return .failed(failure) }
        }
        return Self.treatsAsInternal(isInternal: target.isInternal, ejectable: target.ejectable)
            ? .failed(Self.internalDiskNotice) : .ejected
    }

    /// What the toast says after the volume of an internal disk was unmounted
    /// and nothing more. A system-style string like the DiskArbitration ones,
    /// shown after "could not be ejected:".
    static let internalDiskNotice = "this is an internal disk, so it was unmounted but not ejected"

    /// One step of the sequence.
    enum EjectStep: Equatable, Sendable {
        /// The one volume the request named, and only it.
        case unmountVolume
        /// The whole disk the request named (an APFS container, every volume).
        case unmountWhole
        /// The physical disk under that container, when it is another disk.
        case unmountStore
        case eject
    }

    /// The steps for a disk, decided from facts that were read from the system.
    ///
    /// **A FIXED DEVICE INSIDE THIS MAC GETS ITS VOLUME UNMOUNTED AND NOTHING
    /// MORE.** The request path alone cannot tell (`/Volumes/Data2` on a
    /// second internal SSD looks like any drive), and unmounting a whole
    /// internal disk takes every volume on it away, system volumes included.
    /// The volume the user asked to unmount is unmounted, which is what they
    /// asked for once the models are off it; the disk itself is left alone.
    /// Ejectable media on an internal device (a card in the built-in reader)
    /// is not fixed and takes the ordinary sequence.
    static func plan(isInternal: Bool, hasStore: Bool, ejectable: Bool) -> [EjectStep] {
        if treatsAsInternal(isInternal: isInternal, ejectable: ejectable) { return [.unmountVolume] }
        var steps: [EjectStep] = [.unmountWhole]
        if hasStore { steps.append(.unmountStore) }
        if ejectable { steps.append(.eject) }
        return steps
    }

    /// Whether a disk is a fixed device inside this Mac: the device says it is
    /// internal AND its media is not ejectable. A built-in SD card reader
    /// reports an internal device, but the card in it is ejectable media and
    /// should eject like any other.
    static func treatsAsInternal(isInternal: Bool, ejectable: Bool) -> Bool {
        isInternal && !ejectable
    }

    /// The disks the sequence works on, resolved while the volume is mounted.
    struct Target {
        /// The volume the request named, when it could be resolved.
        let volume: DADisk?
        /// The whole disk the unmount request named.
        let named: DADisk
        /// The physical disk when it is a different one.
        let store: DADisk?
        let ejectable: Bool
        /// DiskArbitration says the device is inside this Mac.
        let isInternal: Bool
    }

    static func target(for disk: DiskIdentity, session: DASession, physical: PhysicalDiskResolving) -> Target? {
        func byName(_ name: String?) -> DADisk? {
            name.flatMap { $0.withCString { DADiskCreateFromBSDName(kCFAllocatorDefault, session, $0) } }
        }
        let volume = byName(disk.bsdName)
            ?? disk.volumeURL.flatMap { DADiskCreateFromVolumePath(kCFAllocatorDefault, session, $0 as CFURL) }
        guard let named = byName(disk.wholeDiskBSDName) ?? volume.flatMap({ DADiskCopyWholeDisk($0) }) else {
            return nil
        }
        // The BSD name belongs to the disk object, so it is read inside the
        // extended lifetime of that object.
        let namedName = withExtendedLifetime(named) { DADiskGetBSDName(named).map { String(cString: $0) } }
        let storeName = namedName.flatMap { physical.physicalWholeDisk(of: $0) }
        let store = storeName.flatMap { $0 == namedName ? nil : byName($0) }
        let ejectDisk = store ?? named
        // Any of the disks the sequence would touch being internal is enough:
        // the synthesized APFS container, the physical store under it, and
        // the volume itself are all asked, so one that is internal cannot be
        // missed by reading only another.
        let isInternalDevice = [volume, named, store].contains { disk in
            guard let disk else { return false }
            let found = DADiskCopyDescription(disk) as? [String: Any]
            return found?[kDADiskDescriptionDeviceInternalKey as String] as? Bool == true
        }
        let description = DADiskCopyDescription(ejectDisk) as? [String: Any]
        // **AN UNREADABLE ANSWER ERRS TOWARD NOT EJECTING AN INTERNAL DEVICE.**
        // Media that says nothing about being ejectable is taken as ejectable
        // for an external drive (the Finder's behavior) and as fixed for a
        // device inside this Mac, which `treatsAsInternal` then guards.
        let ejectable =
            description?[kDADiskDescriptionMediaEjectableKey as String] as? Bool ?? !isInternalDevice
        return Target(volume: volume, named: named, store: store, ejectable: ejectable, isInternal: isInternalDevice)
    }

    private func unmount(_ disk: DADisk, whole: Bool) async -> String? {
        let options = whole ? kDADiskUnmountOptionWhole : kDADiskUnmountOptionDefault
        let step = await perform { context in
            DADiskUnmount(disk, DADiskUnmountOptions(options), Self.unmountCallback, context)
        }
        return step.failureText
    }

    // MARK: One DiskArbitration call, awaited

    /// How one call ended.
    enum Step: Equatable, Sendable {
        case done
        /// The system dissented, with its words.
        case refused(String)
        case timedOut

        var failureText: String? {
            switch self {
            case .done: return nil
            case .refused(let reason): return reason
            case .timedOut: return "the system did not answer in time"
            }
        }
    }

    /// Completed by the DiskArbitration callback or by the timer, whichever
    /// is first; the other is ignored.
    final class Completion: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Step, Never>?
        private var timer: DispatchWorkItem?
        private var finished = false

        init(_ continuation: CheckedContinuation<Step, Never>) { self.continuation = continuation }

        func arm(_ timer: DispatchWorkItem) {
            lock.lock(); defer { lock.unlock() }
            if finished { timer.cancel() } else { self.timer = timer }
        }

        func finish(_ step: Step) {
            lock.lock()
            guard !finished else { lock.unlock(); return }
            finished = true
            let waiting = continuation
            continuation = nil
            let pending = timer
            timer = nil
            lock.unlock()
            pending?.cancel()
            waiting?.resume(returning: step)
        }
    }

    private func perform(_ start: (UnsafeMutableRawPointer) -> Void) async -> Step {
        await withCheckedContinuation { (continuation: CheckedContinuation<Step, Never>) in
            let completion = Completion(continuation)
            let timer = DispatchWorkItem { completion.finish(.timedOut) }
            completion.arm(timer)
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + timeout, execute: timer)
            // One retain for the callback, which releases it. If the call
            // never calls back, this box is the only thing that leaks.
            start(Unmanaged.passRetained(completion).toOpaque())
        }
    }

    private static let unmountCallback: DADiskUnmountCallback = { _, dissenter, context in
        guard let context else { return }
        let completion = Unmanaged<Completion>.fromOpaque(context).takeRetainedValue()
        completion.finish(dissenter.map { Step.refused(DiskArbitrationEjector.message(for: $0)) } ?? Step.done)
    }

    private static let ejectCallback: DADiskEjectCallback = { _, dissenter, context in
        guard let context else { return }
        let completion = Unmanaged<Completion>.fromOpaque(context).takeRetainedValue()
        completion.finish(dissenter.map { Step.refused(DiskArbitrationEjector.message(for: $0)) } ?? Step.done)
    }

    // MARK: Words

    /// The system's text for a dissent, or its status code when it has none.
    static func message(for dissenter: DADissenter) -> String {
        message(status: DADissenterGetStatus(dissenter), text: DADissenterGetStatusString(dissenter) as String?)
    }

    /// The system's own text, else the C library's text for a Unix error the
    /// status carries (a busy volume arrives as `unix_err(EBUSY)`, 0xC010, with
    /// no string of its own, measured live), else the status code.
    static func message(status: Int32, text: String?) -> String {
        if let text = text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty { return text }
        let bits = UInt32(bitPattern: status)
        // `unix_err(e)` is sub-system 3 of the kernel system: (3 << 14) | e.
        if bits >> 14 == 3, bits & 0x3FFF != 0, let described = strerror(Int32(bits & 0x3FFF)) {
            return String(cString: described)
        }
        return "DiskArbitration error 0x" + String(bits, radix: 16)
    }
}
