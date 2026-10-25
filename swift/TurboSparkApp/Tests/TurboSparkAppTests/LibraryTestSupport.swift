import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

// Shared fixtures for the library state suites (tasks 5.6 to 5.8).
//
// **The app suites link the RELEASE engine archive**, where the location
// overrides the Rust tests use to build fixture libraries do not exist, so no
// test here may reach the real engine for libraries. Everything goes through
// `FakeModelLibraryService`.

enum FakeServiceError: Error, Equatable {
    case scripted(String)
    case unscripted(String)
}

/// A `ModelLibraryService` whose answers are scripted and whose calls are
/// recorded, including which thread made them.
final class FakeModelLibraryService: ModelLibraryService, @unchecked Sendable {
    struct State {
        var libraries: [ModelLibrary] = []
        var installed: [InstalledModel] = []
        var listError: Error?
        var installedError: Error?
        /// Volume root path to the marker a `discover` there finds.
        var markers: [String: LibraryMarkerSummary] = [:]
        var discoverError: Error?
        var addError: Error?
        var listCalls = 0
        var installedCalls = 0
        var discoverRoots: [String] = []
        var addCalls: [String] = []
        /// Roots `importMovedRoot` was asked for, in order.
        var importRoots: [String] = []
        /// Answers an import. It runs OUTSIDE the lock, on the calling thread,
        /// so a test can park an import in flight. Nil: an unscripted call.
        var importHandler: (@Sendable (String) throws -> ImportedStore)?
        var mainThreadCalls: [String] = []
        /// Changes to the registry that a test did not script: a remove, a
        /// default change or a forget. A surprise removal must make none.
        var writeCalls: [String] = []
        /// Scripted failures of `remove` and `setDefault`.
        var removeError: Error?
        var setDefaultError: Error?
        /// When true `remove` and `setDefault` change the scripted list, as the
        /// engine does, instead of recording the call and failing as unscripted.
        var appliesWrites = false
        /// One record per `deleteInstall`, `deleteImageInstall` or `forget`:
        /// the call, the install id and whether the calling thread was main.
        var deleteCalls: [DeleteCall] = []
        /// Scripted failure of a delete or a forget; nil lets it run.
        var deleteError: Error?
        var forgetError: Error?
        /// Parks a delete INSIDE the engine call until a test signals it, so
        /// the test can look at the app while the engine is "busy removing
        /// gigabytes". `deleteEntered` is signalled when the call arrives.
        var deleteGate: DispatchSemaphore?
        var deleteEntered: DispatchSemaphore?
        /// Parks an `add` INSIDE the engine call until a test signals it, so a
        /// test can look at the app while the drive is "being added". The call
        /// is recorded in `addCalls` before it waits.
        var addGate: DispatchSemaphore?
        /// Whether the thread that ran each `add` was the main thread, in order.
        var addOnMainThread: [Bool] = []
        /// Parks `libraries()` INSIDE the engine call until a test signals it, as a
        /// drive that was just pulled can hold a real listing. `listEntered` is
        /// signalled when a call arrives.
        var listGate: DispatchSemaphore?
        var listEntered: DispatchSemaphore?
        /// Every install the app started, in order, with the destination it asked
        /// for (nil is This Mac). Recorded before any script runs, so a refused
        /// install is on the list too.
        var installCalls: [InstallCall] = []
        /// Answers a text, repo or image install. Nil: the stream fails as
        /// unscripted, after the call is recorded.
        var textInstallScript: (@Sendable (InstallCall) -> AsyncThrowingStream<InstallEvent, Error>)?
        var imageInstallScript: (@Sendable (InstallCall) -> AsyncThrowingStream<ImageInstallEvent, Error>)?
        /// Answers an audio install. It runs on the calling thread, OUTSIDE the
        /// lock, so a test can park it.
        var audioInstallScript: (@Sendable (InstallCall) throws -> AudioInstalledRecord)?
        /// Every move the app asked the engine for, in order, recorded before any
        /// script runs.
        var moveCalls: [MoveCall] = []
        /// Answers a move. It runs OUTSIDE the lock, on whatever thread the async
        /// call landed on, and gets the progress callback the app passed. It may
        /// park (`MoveGate`) and may throw the engine's error. Nil: an
        /// unscripted call, which throws.
        var moveScript: MoveScript?
        /// When true a move that succeeds relocates the row in `installed` to the
        /// destination, as the engine rewrites its registry.
        var appliesMoves = false
        /// Every `cancelMove`, with the thread that made it.
        var cancelCalls: [CancelCall] = []
        /// What `cancelMove` answers, unless `cancelHandler` is set.
        var cancelAnswer = true
        /// Runs on a `cancelMove`, outside the lock, and gives its answer. A test
        /// uses it to let a parked move see the cancel.
        var cancelHandler: (@Sendable () -> Bool)?
    }

    typealias MoveScript = @Sendable (MoveCall, @escaping @Sendable (UInt64, UInt64) -> Void) async throws
        -> MoveOutcome

    /// One move as the engine would have seen it.
    struct MoveCall: Equatable {
        let installId: String
        let libraryId: String
        /// Exactly what the app passed: nil is the ABI's "could not measure".
        let freeBytes: Int64?
        let onMainThread: Bool
        /// The Mach thread the call started on (before its first suspension).
        let threadID: UInt32
    }

    struct CancelCall: Equatable {
        let onMainThread: Bool
        let threadID: UInt32
    }

    static func currentThreadID() -> UInt32 { pthread_mach_thread_np(pthread_self()) }

    /// One install as the engine would have seen it.
    struct InstallCall: Equatable {
        enum Kind: Equatable { case text, repo, image, audio }
        let kind: Kind
        let alias: String
        /// The `library:` argument, exactly as the app passed it.
        let library: String?
        let repo: String?
        let acknowledged: [String]
        let onMainThread: Bool
    }

    /// One delete-family call as the engine would have seen it.
    struct DeleteCall: Equatable {
        enum Kind: Equatable { case text, image, forget }
        let kind: Kind
        let installId: String
        let onMainThread: Bool
    }

    private let lock = NSLock()
    private var state = State()

    init(libraries: [ModelLibrary] = [], installed: [InstalledModel] = []) {
        state.libraries = libraries
        state.installed = installed
    }

    /// Reads or changes the script under the lock.
    func with<T>(_ body: (inout State) throws -> T) rethrows -> T {
        lock.lock(); defer { lock.unlock() }
        return try body(&state)
    }

    var listCalls: Int { with { $0.listCalls } }
    var installedCalls: Int { with { $0.installedCalls } }
    var discoverRoots: [String] { with { $0.discoverRoots } }
    var addCalls: [String] { with { $0.addCalls } }
    var addOnMainThread: [Bool] { with { $0.addOnMainThread } }
    var importRoots: [String] { with { $0.importRoots } }
    /// Names of the calls that ran on the main thread. The engine reads the
    /// mount table and probes drives, so every one of these is a defect.
    var mainThreadCalls: [String] { with { $0.mainThreadCalls } }

    private func note(_ name: String) {
        if Thread.isMainThread { with { $0.mainThreadCalls.append(name) } }
    }

    // MARK: Reads

    func libraries() throws -> [ModelLibrary] {
        note("libraries")
        // Parked OUTSIDE the lock so a test can read the fake while it waits.
        let (gate, entered) = with { state in (state.listGate, state.listEntered) }
        entered?.signal()
        gate?.wait()
        return try with { state -> [ModelLibrary] in
            state.listCalls += 1
            if let error = state.listError { throw error }
            return state.libraries
        }
    }

    func installedWithAvailability() throws -> [InstalledModel] {
        note("installedWithAvailability")
        return try with { state -> [InstalledModel] in
            state.installedCalls += 1
            if let error = state.installedError { throw error }
            return state.installed
        }
    }

    func discover(volumeRoot: String) throws -> LibraryMarkerSummary? {
        note("discover")
        return try with { state -> LibraryMarkerSummary? in
            state.discoverRoots.append(volumeRoot)
            if let error = state.discoverError { throw error }
            return state.markers[volumeRoot]
        }
    }

    // MARK: Changes

    func add(path: String, name: String?) throws -> AddedLibrary {
        note("add")
        let onMain = Thread.isMainThread
        // Recorded first, then parked OUTSIDE the lock so a test can read the
        // fake while the call is in flight.
        let gate = with { state -> DispatchSemaphore? in
            state.addCalls.append(path)
            state.addOnMainThread.append(onMain)
            return state.addGate
        }
        gate?.wait()
        return try with { state -> AddedLibrary in
            if let error = state.addError { throw error }
            guard let marker = state.markers[path] else { throw FakeServiceError.unscripted("add \(path)") }
            let library = ModelLibrary(
                id: marker.id, name: name ?? marker.name, root: path + "/TurboSpark",
                available: true, modelCount: marker.modelCount)
            state.libraries.append(library)
            return try LibraryFixtures.added(library)
        }
    }

    var writeCalls: [String] { with { $0.writeCalls } }

    func remove(id: String) throws {
        note("remove")
        try with { state in
            state.writeCalls.append("remove \(id)")
            if let error = state.removeError { throw error }
            guard state.appliesWrites else { throw FakeServiceError.unscripted("remove") }
            if id == "internal" {
                throw TurboSparkError(code: .invalidArgument, message: "internal_library: This Mac cannot be removed")
            }
            guard let removed = state.libraries.first(where: { $0.id == id }) else {
                throw TurboSparkError(code: .invalidArgument, message: "unknown_library: no library has the id \(id)")
            }
            state.libraries.removeAll { $0.id == id }
            state.installed.removeAll { $0.library == id }
            // Removing the default hands it back to This Mac, as the engine does.
            if removed.isDefault {
                state.libraries = state.libraries.map { Self.copy($0, isDefault: $0.isInternal) }
            }
        }
    }

    func setDefault(id: String) throws {
        note("setDefault")
        try with { state in
            state.writeCalls.append("setDefault \(id)")
            if let error = state.setDefaultError { throw error }
            guard state.appliesWrites else { throw FakeServiceError.unscripted("setDefault") }
            guard state.libraries.contains(where: { $0.id == id }) else {
                throw TurboSparkError(code: .invalidArgument, message: "unknown_library: no library has the id \(id)")
            }
            state.libraries = state.libraries.map { Self.copy($0, isDefault: $0.id == id) }
        }
    }

    private static func copy(_ library: ModelLibrary, isDefault: Bool) -> ModelLibrary {
        ModelLibrary(
            id: library.id, name: library.name, root: library.root, available: library.available,
            duplicate: library.duplicate, modelCount: library.modelCount, isDefault: isDefault,
            lastSeen: library.lastSeen, isInternal: library.isInternal)
    }
    func forget(installId: String) throws {
        let onMain = Thread.isMainThread
        try with { state in
            state.writeCalls.append("forget \(installId)")
            state.deleteCalls.append(DeleteCall(kind: .forget, installId: installId, onMainThread: onMain))
            if let error = state.forgetError { throw error }
            guard state.appliesWrites else { throw FakeServiceError.unscripted("forget") }
            state.installed.removeAll { $0.installId == installId }
        }
    }

    var deleteCalls: [DeleteCall] { with { $0.deleteCalls } }

    func deleteInstall(installId: String) throws {
        try recordDelete(.text, installId: installId)
    }

    func deleteImageInstall(installId: String) throws {
        try recordDelete(.image, installId: installId)
    }

    /// Records the call, parks it if a gate is set, then answers the script.
    /// The gate wait runs OUTSIDE the lock so the test can read the fake.
    private func recordDelete(_ kind: DeleteCall.Kind, installId: String) throws {
        let onMain = Thread.isMainThread
        let (gate, entered) = with { state -> (DispatchSemaphore?, DispatchSemaphore?) in
            state.deleteCalls.append(DeleteCall(kind: kind, installId: installId, onMainThread: onMain))
            return (state.deleteGate, state.deleteEntered)
        }
        entered?.signal()
        gate?.wait()
        try with { state in
            if let error = state.deleteError { throw error }
            guard state.appliesWrites else { throw FakeServiceError.unscripted("delete \(installId)") }
            state.installed.removeAll { $0.installId == installId }
        }
    }
    func importMovedRoot(_ root: String) throws -> ImportedStore {
        note("importMovedRoot")
        let handler = with { state -> (@Sendable (String) throws -> ImportedStore)? in
            state.importRoots.append(root)
            return state.importHandler
        }
        guard let handler else { throw FakeServiceError.unscripted("import") }
        return try handler(root)
    }

    var moveCalls: [MoveCall] { with { $0.moveCalls } }
    var cancelCalls: [CancelCall] { with { $0.cancelCalls } }

    func move(
        installId: String, to libraryId: String, freeBytes: Int64?,
        progress: @escaping @Sendable (UInt64, UInt64) -> Void
    ) async throws -> MoveOutcome {
        // Recorded before the first suspension, so `threadID` is the thread the
        // engine call would have been entered on.
        let call = MoveCall(
            installId: installId, libraryId: libraryId, freeBytes: freeBytes,
            onMainThread: Thread.isMainThread, threadID: Self.currentThreadID())
        let (script, applies) = with { state -> (MoveScript?, Bool) in
            state.moveCalls.append(call)
            return (state.moveScript, state.appliesMoves)
        }
        guard let script else { throw FakeServiceError.unscripted("move") }
        let outcome = try await script(call, progress)
        if applies {
            with { state in
                guard let index = state.installed.firstIndex(where: { $0.installId == installId }) else { return }
                let destination = state.libraries.first { $0.id == libraryId }
                state.installed[index] = Self.relocated(
                    state.installed[index], to: destination, newInstallId: outcome.installId)
            }
        }
        return outcome
    }

    /// The row as the engine lists it after a move: the destination's library,
    /// root, id and kind, everything else as it was.
    static func relocated(_ row: InstalledModel, to library: ModelLibrary?, newInstallId: String) -> InstalledModel {
        let isInternal = library?.isInternal ?? true
        let root = library?.root ?? "/Users/test/.turbospark/models"
        return InstalledModel(
            alias: row.alias, repo: row.repo, revision: row.revision,
            path: "\(root)/\(row.modality)/\(row.alias).gturbo", family: row.family,
            modality: row.modality, installBytes: row.installBytes, installedOn: row.installedOn,
            status: row.status, kind: row.kind, variant: row.variant,
            library: isInternal ? "internal" : library?.id, libraryName: library?.name,
            installId: newInstallId, availability: .available,
            locationKind: isInternal ? .thisMac : .library)
    }

    func cancelMove() -> Bool {
        let call = CancelCall(onMainThread: Thread.isMainThread, threadID: Self.currentThreadID())
        let (handler, answer) = with { state -> ((@Sendable () -> Bool)?, Bool) in
            state.cancelCalls.append(call)
            return (state.cancelHandler, state.cancelAnswer)
        }
        return handler?() ?? answer
    }

    var installCalls: [InstallCall] { with { $0.installCalls } }

    /// Records the call and returns the script that answers it, if any.
    private func recordInstall(_ call: InstallCall) -> State {
        with { state in
            state.installCalls.append(call)
            return state
        }
    }

    func installText(alias: String, library: String?) -> AsyncThrowingStream<InstallEvent, Error> {
        let call = InstallCall(
            kind: .text, alias: alias, library: library, repo: nil, acknowledged: [],
            onMainThread: Thread.isMainThread)
        guard let script = recordInstall(call).textInstallScript else {
            return AsyncThrowingStream { $0.finish(throwing: FakeServiceError.unscripted("installText")) }
        }
        return script(call)
    }

    func installRepo(
        repo: String, alias: String, file: String?, sidecarRepo: String?, library: String?
    ) -> AsyncThrowingStream<InstallEvent, Error> {
        let call = InstallCall(
            kind: .repo, alias: alias, library: library, repo: repo, acknowledged: [],
            onMainThread: Thread.isMainThread)
        guard let script = recordInstall(call).textInstallScript else {
            return AsyncThrowingStream { $0.finish(throwing: FakeServiceError.unscripted("installRepo")) }
        }
        return script(call)
    }

    func installImage(alias: String, library: String?) -> AsyncThrowingStream<ImageInstallEvent, Error> {
        let call = InstallCall(
            kind: .image, alias: alias, library: library, repo: nil, acknowledged: [],
            onMainThread: Thread.isMainThread)
        guard let script = recordInstall(call).imageInstallScript else {
            return AsyncThrowingStream { $0.finish(throwing: FakeServiceError.unscripted("installImage")) }
        }
        return script(call)
    }

    func installAudio(
        _ identity: AudioProfileIdentity, acknowledged: [String], library: String?,
        progress: @escaping @Sendable (AudioDownloadProgress) -> Void
    ) throws -> AudioInstalledRecord {
        let call = InstallCall(
            kind: .audio, alias: identity.alias, library: library, repo: nil, acknowledged: acknowledged,
            onMainThread: Thread.isMainThread)
        guard let script = recordInstall(call).audioInstallScript else {
            throw FakeServiceError.unscripted("installAudio")
        }
        return try script(call)
    }
}

/// A gate an async script waits on without holding a thread, so a test can look
/// at the app while a scripted engine call is "copying gigabytes".
final class MoveGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private var arrivals = 0

    /// How many callers have reached `wait()`, whether or not it has opened.
    var arrived: Int { lock.lock(); defer { lock.unlock() }; return arrivals }

    func wait() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.lock()
            arrivals += 1
            if opened {
                lock.unlock()
                continuation.resume()
            } else {
                waiters.append(continuation)
                lock.unlock()
            }
        }
    }

    func open() {
        lock.lock()
        opened = true
        let parked = waiters
        waiters = []
        lock.unlock()
        for continuation in parked { continuation.resume() }
    }
}

/// A link probe that answers from a closure and records what it was asked.
final class FakeDriveProbe: DriveLinkProbing, @unchecked Sendable {
    private let lock = NSLock()
    private var asked: [String] = []
    private var dropped: [String] = []
    private var provider: @Sendable (URL) -> DriveConnection

    /// A 10 Gb/s USB link whose volume UUID names the volume folder, unless a
    /// test says otherwise.
    static let defaultProvider: @Sendable (URL) -> DriveConnection = { url in
        let parts = url.standardizedFileURL.pathComponents
        let volume = parts.count > 2 && parts[1] == "Volumes" ? parts[2] : "root"
        return DriveConnection(
            kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "UUID-\(volume)", isLocal: true)
    }

    init(provider: @escaping @Sendable (URL) -> DriveConnection = FakeDriveProbe.defaultProvider) {
        self.provider = provider
    }

    func setProvider(_ provider: @escaping @Sendable (URL) -> DriveConnection) {
        lock.lock(); defer { lock.unlock() }
        self.provider = provider
    }

    var connectionRequests: [String] { lock.lock(); defer { lock.unlock() }; return asked }
    var invalidations: [String] { lock.lock(); defer { lock.unlock() }; return dropped }

    func connection(for volume: URL) async -> DriveConnection {
        lock.lock()
        asked.append(volume.standardizedFileURL.path)
        let answer = provider
        lock.unlock()
        return answer(volume)
    }

    func invalidate(volume: URL) async {
        lock.lock(); defer { lock.unlock() }
        dropped.append(volume.standardizedFileURL.path)
    }
}

/// A debounce timer the test fires by hand. A cancelled sleeper throws, like
/// `Task.sleep`, so a superseded debounce never fires.
final class ManualSleeper: @unchecked Sendable {
    private let lock = NSLock()
    private var waiters: [UUID: CheckedContinuation<Void, Error>] = [:]
    private var started = 0
    private var requested: [Duration] = []

    /// How many sleeps were started, cancelled ones included.
    var sleepCount: Int { lock.lock(); defer { lock.unlock() }; return started }
    /// Sleeps that are still waiting to be fired.
    var pendingCount: Int { lock.lock(); defer { lock.unlock() }; return waiters.count }
    var durations: [Duration] { lock.lock(); defer { lock.unlock() }; return requested }

    func sleep(_ duration: Duration) async throws {
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                lock.lock()
                started += 1
                requested.append(duration)
                if Task.isCancelled {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                waiters[id] = continuation
                lock.unlock()
            }
        } onCancel: {
            lock.lock()
            let waiting = waiters.removeValue(forKey: id)
            lock.unlock()
            waiting?.resume(throwing: CancellationError())
        }
    }

    /// Lets every live sleep finish.
    func fire() {
        lock.lock()
        let live = waiters
        waiters = [:]
        lock.unlock()
        for continuation in live.values { continuation.resume() }
    }
}

enum LibraryFixtures {
    static let internalLibrary = ModelLibrary(
        id: "internal", name: "This Mac", root: "/Users/test/.turbospark/models",
        available: true, modelCount: 2, isDefault: true, isInternal: true)

    static func external(
        _ id: String, name: String? = nil, volume: String? = nil, available: Bool = true,
        isDefault: Bool = false, modelCount: Int = 3
    ) -> ModelLibrary {
        let mount = volume ?? name ?? id
        return ModelLibrary(
            id: id, name: name ?? id, root: available ? "/Volumes/\(mount)/TurboSpark" : nil,
            available: available, modelCount: modelCount, isDefault: isDefault,
            lastSeen: available ? nil : "2026-10-01")
    }

    static func row(
        _ alias: String, library: ModelLibrary? = nil, availability: LibraryAvailability = .available
    ) -> InstalledModel {
        let id = library?.id ?? "internal"
        let root = library?.root ?? "/Volumes/\(id)/TurboSpark"
        return InstalledModel(
            alias: alias, repo: "owner/\(alias)",
            path: library?.isInternal ?? true
                ? "/Users/test/.turbospark/models/text/\(alias).gturbo" : "\(root)/text/\(alias).gturbo",
            family: "qwen", library: library == nil ? "internal" : id,
            libraryName: library?.name ?? "This Mac", installId: "\(id):text/\(alias)",
            availability: availability,
            locationKind: (library?.isInternal ?? true) ? .thisMac : .library,
            connectName: availability == .notConnected ? library?.name : nil)
    }

    static func marker(
        _ id: String, name: String = "Spare", modelCount: Int = 4, readOnly: Bool = false
    ) throws -> LibraryMarkerSummary {
        let json = """
            {"id":"\(id)","name":"\(name)","createdOn":"2026-10-01","modelCount":\(modelCount),
             "readOnly":\(readOnly)}
            """
        return try JSONDecoder().decode(LibraryMarkerSummary.self, from: Data(json.utf8))
    }

    static func added(_ library: ModelLibrary) throws -> AddedLibrary {
        let root = library.root.map { "\"\($0)\"" } ?? "null"
        let json = """
            {"library":{"id":"\(library.id)","name":"\(library.name)","root":\(root),
             "available":\(library.available),"duplicate":false,"modelCount":\(library.modelCount),
             "isDefault":\(library.isDefault),"lastSeen":null,"isInternal":\(library.isInternal)},
             "skipped":[]}
            """
        return try JSONDecoder().decode(AddedLibrary.self, from: Data(json.utf8))
    }

    /// What `ts_store_import_moved_root` answers for `library`, through the
    /// package's own decoder (`ImportedStore` has no public initializer).
    static func imported(
        _ library: ModelLibrary, copyFailures: [(key: String, reason: String)] = [],
        needsAdoption: [String] = [], skipped: [String] = []
    ) throws -> ImportedStore {
        let root = library.root.map { "\"\($0)\"" } ?? "null"
        func array(_ values: [String]) -> String {
            "[" + values.map { "\"\($0)\"" }.joined(separator: ",") + "]"
        }
        let failures = copyFailures
            .map { "{\"key\":\"\($0.key)\",\"reason\":\"\($0.reason)\"}" }
            .joined(separator: ",")
        let json = """
            {"library":{"id":"\(library.id)","name":"\(library.name)","root":\(root),
             "available":\(library.available),"duplicate":false,"modelCount":\(library.modelCount),
             "isDefault":\(library.isDefault),"lastSeen":null,"isInternal":\(library.isInternal)},
             "skipped":\(array(skipped)),"custom":[],"needsAdoption":\(array(needsAdoption)),
             "copiedToHome":[],"copyFailures":[\(failures)],"restoredCatalogOverride":false}
            """
        return try JSONDecoder().decode(ImportedStore.self, from: Data(json.utf8))
    }

    /// What `ts_model_move` answers, through the package's own decoder
    /// (`MoveOutcome` has no public initializer). `kind` is the wire outcome:
    /// `moved`, `source_marker_stale`, `source_cleanup_failed` or one this build
    /// does not know.
    static func moveOutcome(
        _ kind: String, installId: String, staleMarker: String? = nil, path: String? = nil
    ) throws -> MoveOutcome {
        func quoted(_ value: String?) -> String {
            guard let value else { return "null" }
            let data = try! JSONEncoder().encode(value)
            return String(decoding: data, as: UTF8.self)
        }
        let json = """
            {"outcome":"\(kind)","installId":\(quoted(installId)),"staleMarker":\(quoted(staleMarker)),
             "path":\(quoted(path))}
            """
        return try JSONDecoder().decode(MoveOutcome.self, from: Data(json.utf8))
    }

    /// The engine's error for a refused or failed move: the code first, then a
    /// sentence the app must never show.
    static func moveError(_ code: String, sentence: String = "engine sentence") -> TurboSparkError {
        let status: TurboSparkError.Code
        switch code {
        case "busy": status = .busy
        case "cancelled": status = .cancelled
        default: status = .invalidArgument
        }
        return TurboSparkError(code: status, message: "\(code): \(sentence)")
    }

    static func volume(
        _ name: String, uuid: String? = nil, path: String? = nil
    ) -> VolumeDescriptor {
        VolumeDescriptor(
            url: URL(fileURLWithPath: path ?? "/Volumes/\(name)", isDirectory: true),
            name: name, bsdName: nil, uuid: uuid)
    }
}

/// The common setup of the library state suites: a fake engine with This Mac
/// and one external library (T7), a fake probe, a hand-fired debounce timer,
/// and a dismissals file in a temp directory.
@MainActor
class LibraryStateTestCase: XCTestCase {
    var directory: URL!
    var service: FakeModelLibraryService!
    var probe: FakeDriveProbe!
    var sleeper: ManualSleeper!
    let t7 = LibraryFixtures.external("t7-id", name: "T7", volume: "T7")

    var dismissalsURL: URL { directory.appendingPathComponent("library_offer_dismissals.json") }

    static let freeSpace: @Sendable (URL) -> Int64? = { url in
        switch url.path {
        case "/Users/test/.turbospark/models": return 100_000_000_000
        case "/Volumes/T7/TurboSpark": return 900_000_000_000
        default: return nil
        }
    }

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("library-state-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        service = FakeModelLibraryService(
            libraries: [LibraryFixtures.internalLibrary, t7],
            installed: [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-t7", library: t7),
            ])
        probe = FakeDriveProbe()
        sleeper = ManualSleeper()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func makeState(
        dismissals: URL? = nil, freeSpace: (@Sendable (URL) -> Int64?)? = nil
    ) -> LibraryState {
        let sleeper = self.sleeper!
        return LibraryState(
            service: service, probe: probe, dismissalsURL: dismissals ?? dismissalsURL,
            sleep: { duration in try await sleeper.sleep(duration) },
            freeSpace: freeSpace ?? Self.freeSpace)
    }

    /// Fires the debounce and waits for the refresh it starts.
    ///
    /// The timer task is created by `handle` but only starts running when the
    /// test next suspends, so firing at once would find nothing waiting and
    /// the timer would then wait forever. Wait for it to arm first.
    func fireDebounce(_ state: LibraryState) async {
        let armed = await eventually { self.sleeper.pendingCount > 0 }
        XCTAssertTrue(armed, "no debounce timer is waiting to be fired")
        sleeper.fire()
        await state.finishScheduledRefresh()
    }

    /// An `AppModel` wired to `state`, with its start-up work finished.
    ///
    /// `AppModel.init` starts a scan of the user's model folders that writes
    /// `installed` when it lands. Waiting for it keeps a baseline taken now
    /// from moving under a test for a reason that has nothing to do with it.
    func makeModel(using state: LibraryState) async -> AppModel {
        let model = AppModel()
        model.libraryState = state
        await model.modelScanTask?.value
        return model
    }

    /// A moment for tasks that were created but have not started. Only for
    /// asserting that something did NOT happen.
    func letTasksStart() async {
        try? await Task.sleep(nanoseconds: 60_000_000)
    }
}

/// Polls `condition` on the main actor until it holds or the timeout passes.
@MainActor
func eventually(timeout: TimeInterval = 5, _ condition: () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if condition() { return true }
        try? await Task.sleep(nanoseconds: 5_000_000)
    }
    return condition()
}
