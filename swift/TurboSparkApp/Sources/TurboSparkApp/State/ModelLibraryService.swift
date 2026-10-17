import Foundation
import TurboSpark

/// The one door from the app to the engine's model libraries (design.md
/// "Service seam").
///
/// `LibraryState` and every later library task (moves, installs into a chosen
/// library, the moved-store conversion) reach libraries only through this
/// protocol. The live implementation is a thin pass-through to
/// `TurboSparkLibraries`, `TurboSparkCatalog` and `AudioCatalog`; app tests
/// inject a fake.
///
/// **THE FAKE IS NOT A CONVENIENCE, IT IS THE ONLY WAY.** The app suites link
/// the RELEASE engine archive (`scripts/swift-lib.sh` builds `--release`),
/// where the location overrides the Rust tests use to build fixture libraries
/// do not exist, so a test cannot make a library through the real engine.
///
/// **THREADING.** Every method that is not `async` or a stream blocks: it
/// reads the mount table, probes a drive or writes the registry, and a drive
/// that is hung can hold it for a long time. Call them off the main actor
/// (`LibraryState` does, with a detached task); the protocol deliberately does
/// not hide that behind `async` so a call site cannot forget it is there.
///
/// Errors are `TurboSparkError`s. Read the engine's code through
/// `error.libraryFault`, never from the message text.
protocol ModelLibraryService: Sendable {
    // MARK: Reads

    /// Every library, This Mac first, after locating each one. A library that
    /// is not connected is listed with no root; that is not an error.
    func libraries() throws -> [ModelLibrary]

    /// Every installed text model, including rows whose library is not
    /// connected (`.notConnected`) or whose folder is gone (`.missing`).
    func installedWithAvailability() throws -> [InstalledModel]

    /// The library marker at the top of the volume at `volumeRoot`, or nil.
    /// Reads one file and never walks the drive.
    func discover(volumeRoot: String) throws -> LibraryMarkerSummary?

    // MARK: Library changes

    func add(path: String, name: String?) throws -> AddedLibrary
    func remove(id: String) throws
    func setDefault(id: String) throws
    /// Stops listing an install and drops it from its library's marker.
    func forget(installId: String) throws
    func importMovedRoot(_ root: String) throws -> ImportedStore

    // MARK: Deletes

    /// Deletes a text install by the `installId` of its row: its folder, its
    /// registry row and its entry in its library's marker. The engine finds the
    /// install where its library is NOW and refuses a library that is not
    /// connected or is read-only, so this is the only delete a library row may
    /// use. A folder that is already gone is only forgotten (success).
    func deleteInstall(installId: String) throws
    /// The same for an image install (`<library>:image/<alias>`).
    func deleteImageInstall(installId: String) throws

    // MARK: Moves

    /// Moves an install to another library. Blocks for as long as the copy
    /// takes on a dedicated thread; the caller awaits.
    func move(
        installId: String, to libraryId: String, freeBytes: Int64?,
        progress: @escaping @Sendable (UInt64, UInt64) -> Void
    ) async throws -> MoveOutcome

    /// Asks the move in flight to stop. False when none was running.
    @discardableResult
    func cancelMove() -> Bool

    // MARK: Installs with a destination

    /// `library` is a `ModelLibrary.id`; nil or `internal` is This Mac.
    func installText(alias: String, library: String?) -> AsyncThrowingStream<InstallEvent, Error>

    func installRepo(
        repo: String, alias: String, file: String?, sidecarRepo: String?, library: String?
    ) -> AsyncThrowingStream<InstallEvent, Error>

    func installImage(alias: String, library: String?) -> AsyncThrowingStream<ImageInstallEvent, Error>

    /// Blocking transfer; call off the main actor.
    func installAudio(
        _ identity: AudioProfileIdentity, acknowledged: [String], library: String?,
        progress: @escaping @Sendable (AudioDownloadProgress) -> Void
    ) throws -> AudioInstalledRecord
}

/// The engine, through the Swift package. Stateless: a value can be copied to
/// any task.
struct LiveModelLibraryService: ModelLibraryService {
    init() {}

    func libraries() throws -> [ModelLibrary] {
        try TurboSparkLibraries.list()
    }

    func installedWithAvailability() throws -> [InstalledModel] {
        try TurboSparkCatalog.installedWithAvailability()
    }

    func discover(volumeRoot: String) throws -> LibraryMarkerSummary? {
        try TurboSparkLibraries.discover(volumeRoot: volumeRoot)
    }

    func add(path: String, name: String?) throws -> AddedLibrary {
        try TurboSparkLibraries.add(path: path, name: name)
    }

    func remove(id: String) throws {
        try TurboSparkLibraries.remove(id: id)
    }

    func setDefault(id: String) throws {
        try TurboSparkLibraries.setDefault(id: id)
    }

    func forget(installId: String) throws {
        try TurboSparkLibraries.forget(installId: installId)
    }

    func importMovedRoot(_ root: String) throws -> ImportedStore {
        try TurboSparkLibraries.importMovedRoot(root)
    }

    func deleteInstall(installId: String) throws {
        try TurboSparkCatalog.deleteInstall(installId: installId)
    }

    func deleteImageInstall(installId: String) throws {
        try TurboSparkCatalog.deleteImageInstall(installId: installId)
    }

    func move(
        installId: String, to libraryId: String, freeBytes: Int64?,
        progress: @escaping @Sendable (UInt64, UInt64) -> Void
    ) async throws -> MoveOutcome {
        try await TurboSparkLibraries.move(
            installId: installId, to: libraryId, freeBytes: freeBytes, progress: progress)
    }

    @discardableResult
    func cancelMove() -> Bool {
        TurboSparkLibraries.cancelMove()
    }

    func installText(alias: String, library: String?) -> AsyncThrowingStream<InstallEvent, Error> {
        TurboSparkCatalog.install(alias, library: library)
    }

    func installRepo(
        repo: String, alias: String, file: String?, sidecarRepo: String?, library: String?
    ) -> AsyncThrowingStream<InstallEvent, Error> {
        TurboSparkCatalog.install(
            repo: repo, alias: alias, file: file, sidecarRepo: sidecarRepo, library: library)
    }

    func installImage(alias: String, library: String?) -> AsyncThrowingStream<ImageInstallEvent, Error> {
        TurboSparkCatalog.installImage(alias, library: library)
    }

    func installAudio(
        _ identity: AudioProfileIdentity, acknowledged: [String], library: String?,
        progress: @escaping @Sendable (AudioDownloadProgress) -> Void
    ) throws -> AudioInstalledRecord {
        try AudioCatalog.install(
            identity, acknowledged: acknowledged, library: library, progress: progress)
    }
}
