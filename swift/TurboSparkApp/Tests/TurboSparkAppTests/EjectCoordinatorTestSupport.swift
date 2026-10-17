import Foundation
import XCTest

@testable import TurboSparkApp

// Fakes for the eject coordinator suites (tasks 5.11 and 5.12). No real disk,
// no real session and no DiskArbitration object is ever created here.

/// An ordered log any thread can append to. The suites compare the whole
/// sequence, so the ORDER of a stop, an unload, an eject and a notice is what
/// they assert.
final class EjectLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ entry: String) { lock.lock(); items.append(entry); lock.unlock() }
    var entries: [String] { lock.lock(); defer { lock.unlock() }; return items }
    func count(of entry: String) -> Int { entries.filter { $0 == entry }.count }
}

/// Holds a step open until the test lets it go, so a test can look at the world
/// while a cycle is parked mid-way.
actor AsyncGate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private(set) var arrivals = 0

    func pass() async {
        arrivals += 1
        if open { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        open = true
        let all = waiters
        waiters = []
        for waiter in all { waiter.resume() }
    }

    /// True once `count` callers have reached the gate.
    func hasArrivals(_ count: Int = 1) -> Bool { arrivals >= count }
}

/// Polls an async condition until it holds or the timeout passes.
func eventuallyAsync(timeout: TimeInterval = 5, _ condition: () async -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if await condition() { return true }
        try? await Task.sleep(nanoseconds: 5_000_000)
    }
    return await condition()
}

/// The sessions, as one list the test edits. An unload removes what it
/// released, unless the model is marked stubborn.
@MainActor
final class FakeLoadedModels: LoadedModelSource {
    var models: [LoadedModel]
    /// Paths whose session refuses to go (a request still holds it).
    var stubbornPaths: Set<String> = []
    var stopGate: AsyncGate?
    var unloadGate: AsyncGate?
    private let log: EjectLog
    private(set) var stopped: [LoadedModelCategory] = []
    private(set) var unloaded: [LoadedModelCategory] = []
    private(set) var readCount = 0
    /// Releases begun and not yet ended. Zero when the coordinator balanced
    /// every `beginRelease`.
    private(set) var holdDepth = 0
    private(set) var beginCount = 0
    /// The release bracket and the steps inside it, in order. Kept apart from
    /// the shared log so the suites that compare that whole log are unchanged.
    private(set) var trace: [String] = []

    init(_ models: [LoadedModel], log: EjectLog) {
        self.models = models
        self.log = log
    }

    func loadedModels() -> [LoadedModel] {
        readCount += 1
        trace.append("read")
        return models
    }

    func beginRelease(of models: [LoadedModel]) {
        holdDepth += 1
        beginCount += 1
        trace.append("begin")
    }

    func endRelease(of models: [LoadedModel]) {
        holdDepth -= 1
        trace.append("end")
    }

    func stopGeneration(in category: LoadedModelCategory, models: [LoadedModel]) async {
        log.add("stop:\(category)")
        trace.append("stop:\(category)")
        stopped.append(category)
        await stopGate?.pass()
    }

    func unload(_ category: LoadedModelCategory, models released: [LoadedModel]) async {
        log.add("unload:\(category)")
        trace.append("unload:\(category)")
        unloaded.append(category)
        let paths = Set(released.map(\.path))
        models.removeAll { $0.category == category && paths.contains($0.path) && !stubbornPaths.contains($0.path) }
        await unloadGate?.pass()
    }
}

/// A disk ejector that records its calls. `duringEject` runs while the
/// "unmount" is in flight, which is when DiskArbitration would ask the
/// coordinator to approve its own unmount.
final class FakeEjector: DiskEjecting, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [DiskIdentity] = []
    private var result: EjectResult
    private let log: EjectLog
    var duringEject: (@Sendable (DiskIdentity) async -> Void)?

    init(result: EjectResult = .ejected, log: EjectLog) {
        self.result = result
        self.log = log
    }

    var calls: [DiskIdentity] { lock.lock(); defer { lock.unlock() }; return recorded }
    func setResult(_ new: EjectResult) { lock.lock(); result = new; lock.unlock() }

    func unmountAndEject(_ disk: DiskIdentity) async -> EjectResult {
        lock.lock()
        recorded.append(disk)
        let answer = result
        lock.unlock()
        log.add("eject")
        await duringEject?(disk)
        return answer
    }
}

/// Maps a model path to the volume it is on without reading a disk: the
/// volume folder under /Volumes decides.
struct FakeVolumeLocator: ModelVolumeLocating {
    /// Volume folder name to what the system would report for it.
    var volumes: [String: ModelVolume] = [
        "T7": ModelVolume(
            realPath: "", mountPoint: "/Volumes/T7", bsdName: "disk5s1", diskNames: ["disk5", "disk4"]),
        "T7 Data": ModelVolume(
            realPath: "", mountPoint: "/Volumes/T7 Data", bsdName: "disk5s2", diskNames: ["disk5", "disk4"]),
        "Spare": ModelVolume(
            realPath: "", mountPoint: "/Volumes/Spare", bsdName: "disk7s1", diskNames: ["disk7", "disk6"]),
    ]

    func locate(path: String) -> ModelVolume {
        let parts = URL(fileURLWithPath: path).standardizedFileURL.pathComponents
        if parts.count > 2, parts[1] == "Volumes", let known = volumes[parts[2]] {
            return ModelVolume(
                realPath: path, mountPoint: known.mountPoint, bsdName: known.bsdName,
                diskNames: known.diskNames)
        }
        // Everything else is the internal disk.
        return ModelVolume(
            realPath: path, mountPoint: "/System/Volumes/Data", bsdName: "disk3s5", diskNames: ["disk3"])
    }
}

/// Collects what the coordinator asked the user to be told.
@MainActor
final class NoticeRecorder {
    private(set) var all: [EjectNotice] = []
    private let log: EjectLog
    init(log: EjectLog) { self.log = log }

    func add(_ notice: EjectNotice) {
        all.append(notice)
        switch notice {
        case .ejected: log.add("notice:ejected")
        case .ejectFailed: log.add("notice:failed")
        case .removedWithoutEject: log.add("notice:removed")
        case .unloadBlocked: log.add("notice:blocked")
        }
    }
}

/// A host for the approval delegate that counts nothing but its own registration.
/// `stopCount` stays zero in every test: the coordinator must never stop the
/// monitor (the 10.5 s unmount stall).
final class FakeApprovalHost: UnmountApprovalHosting {
    var delegate: UnmountApprovalDelegate?
    private(set) var stopCount = 0
    func stop() { stopCount += 1 }
}

/// A value any thread can set and read.
final class EjectBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value
    init(_ value: Value) { stored = value }
    var value: Value {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}

enum EjectFixtures {
    static let t7Disk = DiskIdentity(
        bsdName: "disk5s1", wholeDiskBSDName: "disk5",
        volumeURL: URL(fileURLWithPath: "/Volumes/T7", isDirectory: true))
    /// The second volume of the same drive.
    static let t7DataDisk = DiskIdentity(
        bsdName: "disk5s2", wholeDiskBSDName: "disk5",
        volumeURL: URL(fileURLWithPath: "/Volumes/T7 Data", isDirectory: true))
    static let spareDisk = DiskIdentity(
        bsdName: "disk7s1", wholeDiskBSDName: "disk7",
        volumeURL: URL(fileURLWithPath: "/Volumes/Spare", isDirectory: true))
    static let dataVolume = DiskIdentity(
        bsdName: "disk3s5", wholeDiskBSDName: "disk3",
        volumeURL: URL(fileURLWithPath: "/System/Volumes/Data", isDirectory: true))
    static let rootVolume = DiskIdentity(
        bsdName: "disk3s1s1", wholeDiskBSDName: "disk3", volumeURL: URL(fileURLWithPath: "/", isDirectory: true))

    static let t7Volume = VolumeDescriptor(
        url: URL(fileURLWithPath: "/Volumes/T7", isDirectory: true), name: "T7", bsdName: "disk5s1", uuid: "UUID-T7")
    static let spareVolume = VolumeDescriptor(
        url: URL(fileURLWithPath: "/Volumes/Spare", isDirectory: true), name: "Spare", bsdName: "disk7s1",
        uuid: "UUID-SPARE")

    static let chat = LoadedModel(
        category: .chat, name: "qwen", path: "/Volumes/T7/TurboSpark/text/qwen.gturbo")
    static let server = LoadedModel(
        category: .server, name: "gemma", path: "/Volumes/T7/TurboSpark/text/gemma.gturbo")
    static let audio = LoadedModel(
        category: .audio, name: "speech", path: "/Volumes/T7/TurboSpark/audio/speech.gturbo")
    static let onSecondVolume = LoadedModel(
        category: .server, name: "phi", path: "/Volumes/T7 Data/TurboSpark/text/phi.gturbo")
    static let onSpare = LoadedModel(
        category: .server, name: "other", path: "/Volumes/Spare/TurboSpark/text/other.gturbo")
    static let onThisMac = LoadedModel(
        category: .chat, name: "local", path: "/Users/test/.turbospark/models/text/local.gturbo")
}
