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
        return try with { state -> AddedLibrary in
            state.addCalls.append(path)
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
        with { $0.writeCalls.append("forget \(installId)") }
        throw FakeServiceError.unscripted("forget")
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

    func move(
        installId: String, to libraryId: String, freeBytes: Int64?,
        progress: @escaping @Sendable (UInt64, UInt64) -> Void
    ) async throws -> MoveOutcome {
        throw FakeServiceError.unscripted("move")
    }

    func cancelMove() -> Bool { false }

    func installText(alias: String, library: String?) -> AsyncThrowingStream<InstallEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: FakeServiceError.unscripted("installText")) }
    }

    func installRepo(
        repo: String, alias: String, file: String?, sidecarRepo: String?, library: String?
    ) -> AsyncThrowingStream<InstallEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: FakeServiceError.unscripted("installRepo")) }
    }

    func installImage(alias: String, library: String?) -> AsyncThrowingStream<ImageInstallEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: FakeServiceError.unscripted("installImage")) }
    }

    func installAudio(
        _ identity: AudioProfileIdentity, acknowledged: [String], library: String?,
        progress: @escaping @Sendable (AudioDownloadProgress) -> Void
    ) throws -> AudioInstalledRecord {
        throw FakeServiceError.unscripted("installAudio")
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
