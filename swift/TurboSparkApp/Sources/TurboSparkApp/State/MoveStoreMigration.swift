import Combine
import Foundation
import TurboSpark

/// The root a past Move Store moved the model store to, as the conversion sees
/// it. The live value is `ModelStoreRootRecord` (machine level); tests inject a
/// fake that can refuse to clear.
protocol MovedStoreRecord: Sendable {
    /// The recorded root, nil when none is recorded.
    func load() -> String?
    /// Ends the record. True when no root is recorded afterwards.
    func clear() -> Bool
}

struct LiveMovedStoreRecord: MovedStoreRecord {
    func load() -> String? { ModelStoreRootRecord.load() }
    func clear() -> Bool { ModelStoreRootRecord.clear() }
}

/// A moved store whose drive is not connected: what `LibraryState` lists as a
/// not-connected placeholder named after the folder (requirement 9.3).
struct PendingMovedStore: Equatable, Sendable {
    /// The recorded root, absolute and standardized.
    let root: String
    /// The root folder's name. Untrusted: it came from a path on a drive, so a
    /// view shows it through `UntrustedText.display`.
    let name: String
}

/// `moved_store_pending.json`, machine level: a conversion that did not finish.
///
/// Written for every unfinished outcome (drive away, home copies that failed,
/// a refusal, a record that would not clear) and removed when the conversion
/// completes. `notConnected` is true only when the engine found the root
/// absent, which is the one state shown as a placeholder: any other pending
/// state has a real library listed or a root that answered.
struct MovedStorePendingFile: Codable, Equatable, Sendable {
    var root: String
    var name: String
    var notConnected: Bool
}

/// Converts the store a past Move Store moved into a model library, once
/// (requirements 9.2, 9.3, design.md "MoveStoreMigration").
///
/// **THE INPUT IS THE MACHINE-LEVEL `ModelStoreRootRecord`, AND IT IS NEVER
/// APPLIED TO THE ENGINE.** The home is the engine's store root (requirement
/// 1.1). The conversion asks the engine to register `<root>/models` as a
/// library where it is (`importMovedRoot`), which moves no model file and writes
/// nothing under the moved root except the library marker the engine owns. The
/// only writes made HERE are the record, `moved_store_pending.json`, and
/// whatever the engine does (the marker, the registry, and the embedding and
/// OCR home copies).
///
/// **IT RUNS AT LAUNCH AND ON EACH MOUNT WHILE A CONVERSION IS PENDING.**
/// `start(events:)` takes one stream of the process-wide volume monitor and
/// drains it with one `for await`. It never creates a monitor and never stops
/// one.
///
/// **OUTCOMES.**
/// - Complete (`ImportedStore.isComplete`): the record is cleared and then the
///   pending file is removed, in that order, only after the import returned.
/// - Home copies failed (`copyFailures`): not complete. The record and a
///   pending file stay, the failures are state, and the next trigger imports
///   again (the engine retries only the failed copies).
/// - The root is not connected: the record stays, the pending file is written
///   and `LibraryState` lists a placeholder named after the folder.
/// - Anything else (a refusal, `io`, `busy`): the record and a pending file
///   stay, the failure is logged, and the next trigger tries again.
/// Nothing here is an app error or a toast: a drive that is away is a state
/// (requirements 3.3, 3.4), and a conversion that cannot finish yet is retried.
///
/// **A CRASH ANYWHERE IS SAFE.** The import is idempotent by marker id, the
/// record is cleared only after it returned a complete result, and a run that
/// finds no record removes a pending file left behind by a crash between the
/// two.
///
/// **ONE IMPORT AT A TIME.** A trigger that arrives while one is in flight
/// folds into a single follow-up, so a drive that mounts just after the engine
/// looked is not missed, and two mounts in a row never start two imports.
@MainActor
final class MoveStoreMigration: ObservableObject {
    nonisolated static let pendingFileName = "moved_store_pending.json"

    /// Machine level, like the record: the store was shared by every profile.
    nonisolated static var defaultPendingURL: URL {
        AppStorageRoot.machineRoot.appendingPathComponent(pendingFileName)
    }

    /// Shown only if the pending file cannot be read or saved. It has no
    /// catalog key yet (task 6 adds it with the other library strings).
    private static let pendingLabel: String.LocalizationValue = "Moved store conversion"

    // MARK: Published state

    /// The home copies the last import could not make
    /// (`embeddings/<alias>`, `ocr/<alias>`, `models.json`). Untrusted, and
    /// the reasons may echo drive paths: show them through
    /// `UntrustedText.display`. Not an app error. Empty once a retry succeeds.
    @Published private(set) var lastCopyFailures: [LibraryCopyFailure] = []
    /// What the last successful import reported, for `needsAdoption` (music
    /// installs the user must move to This Mac and adopt there), `skipped` and
    /// `custom`.
    @Published private(set) var lastImported: ImportedStore?

    /// A conversion is unfinished: the next launch or mount imports again.
    private(set) var isPending = false
    /// Why the last attempt did not finish, escaped for a log. Not shown.
    private(set) var lastFailure: String?
    /// Events fully handled so far. A test seam for waiting on the stream.
    private(set) var handledEventCount = 0

    // MARK: Collaborators

    private let service: ModelLibraryService
    private let libraryState: LibraryState
    private let record: MovedStoreRecord
    private let pendingURL: URL
    private let homeRoot: @Sendable () -> String
    private let onImported: @MainActor () -> Void

    // MARK: Work in flight

    private var eventTask: Task<Void, Never>?
    private var runner: Task<Void, Never>?
    private var rerunRequested = false
    /// What the pending file holds, as far as this process knows, so an
    /// unchanged state is not written again at every launch.
    private var writtenPending: MovedStorePendingFile?

    /// - Parameters:
    ///   - homeRoot: the engine's store root, which a record equal to is
    ///     nothing to convert.
    ///   - onImported: called after an import listed the moved store's library
    ///     (complete or not), so the picker list read at launch is read again.
    init(
        service: ModelLibraryService,
        libraryState: LibraryState,
        record: MovedStoreRecord = LiveMovedStoreRecord(),
        pendingURL: URL = MoveStoreMigration.defaultPendingURL,
        homeRoot: @escaping @Sendable () -> String = { ModelStorageManager.defaultTurboSparkStoreRoot },
        onImported: @escaping @MainActor () -> Void = {}
    ) {
        self.service = service
        self.libraryState = libraryState
        self.record = record
        self.pendingURL = pendingURL
        self.homeRoot = homeRoot
        self.onImported = onImported
    }

    // MARK: - Lifecycle

    /// Converts now (at launch), and again on each mount while pending.
    ///
    /// **ONE `for await` PER STREAM.** An `AsyncStream` ends when its consumer
    /// leaves the loop, so this takes the stream it will drain and a second
    /// call is ignored. It never stops the monitor: `VolumeMonitor.stop()`
    /// belongs to app termination alone.
    func start(events: AsyncStream<VolumeEvent>) {
        guard eventTask == nil else { return }
        eventTask = Task { [weak self] in
            for await event in events {
                guard let self else { return }
                self.handle(event)
            }
        }
        showPlaceholderFromThePendingFile()
        trigger()
    }

    /// Stops consuming events. A conversion already in flight finishes, and the
    /// monitor itself is left running.
    func stop() {
        eventTask?.cancel()
        eventTask = nil
    }

    /// Waits for the run in flight and any follow-up it folded in.
    func waitUntilIdle() async {
        while let runner { await runner.value }
    }

    // MARK: - Events

    /// Only a mount can connect a drive, and only a pending conversion has a
    /// reason to look: `willUnmount`, `didUnmount` and `renamed` never run it.
    private func handle(_ event: VolumeEvent) {
        defer { handledEventCount += 1 }
        guard case .mounted = event else { return }
        guard isPending || runner != nil else { return }
        trigger()
    }

    /// Starts a run, or folds the request into one follow-up when a run is in
    /// flight.
    private func trigger() {
        if runner != nil {
            rerunRequested = true
            return
        }
        runner = Task { [weak self] in await self?.drain() }
    }

    private func drain() async {
        repeat {
            rerunRequested = false
            await attempt()
        } while rerunRequested
        runner = nil
    }

    // MARK: - The conversion

    private func attempt() async {
        guard let recorded = record.load() else {
            becomeIdle()
            return
        }
        guard let root = Self.absolutePath(recorded) else {
            // Handed to the engine, a relative path would resolve against the
            // working directory. Move Store only ever recorded absolute paths.
            logLine("the recorded store root \"\(UntrustedText.display(recorded, limit: 200))\" is not an absolute path; it is left as it is")
            becomeIdle()
            return
        }
        if root == Self.absolutePath(homeRoot()) {
            // The home is This Mac already.
            becomeIdle()
            return
        }

        // The engine reads the mount table and the drive and writes the
        // registry, so none of it runs on the main actor.
        let service = self.service
        let outcome = await Task.detached(priority: .utility) { () -> Result<ImportedStore, Error> in
            Result { try service.importMovedRoot(root) }
        }.value

        switch outcome {
        case .success(let imported):
            await finish(imported, root: root)
        case .failure(let error):
            fail(error, root: root)
        }
    }

    private func finish(_ imported: ImportedStore, root: String) async {
        lastImported = imported
        lastCopyFailures = imported.copyFailures
        lastFailure = nil

        var complete = imported.isComplete
        var superseded = false
        if complete, let now = record.load(), Self.absolutePath(now) != root {
            // The record was rewritten while the import ran (the legacy Move
            // Store sheet saves a new root). Clearing it would drop that newer
            // store, so it is left alone and the follow-up run imports it.
            complete = false
            superseded = true
            rerunRequested = true
        }
        if complete {
            // The record goes only now that the import has returned a complete
            // result, and the pending file only after the record.
            if record.clear() {
                discardPendingFile()
            } else {
                // Converted, but the record still names the root: the next
                // trigger imports again (idempotent) and clears it.
                complete = false
                let note = "the moved store was converted but its record could not be cleared"
                lastFailure = note
                logLine(note)
            }
        }
        if complete {
            isPending = false
        } else {
            isPending = true
            // A superseded run leaves the pending file to the follow-up, which
            // names the newer root.
            if !superseded { keepPendingFile(root: root, notConnected: false) }
        }
        // The library is real and listed now, so no placeholder stands beside
        // it, whether or not the conversion is complete.
        libraryState.setPendingMovedStore(nil)
        await libraryState.refresh()
        onImported()
    }

    private func fail(_ error: Error, root: String) {
        isPending = true
        if (error as? TurboSparkError)?.libraryFault?.code == .notConnected {
            // The expected state for a drive that is unplugged: not logged,
            // not an error, and not a toast.
            lastFailure = nil
            keepPendingFile(root: root, notConnected: true)
            libraryState.setPendingMovedStore(
                PendingMovedStore(root: root, name: Self.folderName(of: root)))
            return
        }
        // A refusal (nested, marker), io, busy or anything else. Nothing is
        // cleared; the next trigger tries again.
        let line = UntrustedText.display("\(error)", limit: 300)
        lastFailure = line
        logLine("could not convert the moved store at \(UntrustedText.display(root, limit: 200)): \(line)")
        keepPendingFile(root: root, notConnected: false)
        libraryState.setPendingMovedStore(nil)
    }

    /// Nothing is recorded, or it names the home, or it is unusable: nothing is
    /// pending, and a pending file a crash left behind goes.
    private func becomeIdle() {
        isPending = false
        discardPendingFile()
        libraryState.setPendingMovedStore(nil)
    }

    // MARK: - The pending file

    /// At launch, a drive that was away last time is listed as away at once,
    /// before the engine has answered. The file counts only when it still
    /// matches the record and recorded an absent root; the name is derived from
    /// the root again rather than trusted from disk. An unreadable file is
    /// quarantined by `AppJSONStore`, and the run that follows derives
    /// everything from the record.
    private func showPlaceholderFromThePendingFile() {
        guard
            let file = AppJSONStore.load(
                MovedStorePendingFile.self, from: pendingURL, label: Self.pendingLabel)
        else { return }
        writtenPending = file
        guard
            file.notConnected, let recorded = record.load(),
            let root = Self.absolutePath(recorded), root == file.root
        else { return }
        libraryState.setPendingMovedStore(PendingMovedStore(root: root, name: Self.folderName(of: root)))
    }

    private func keepPendingFile(root: String, notConnected: Bool) {
        let file = MovedStorePendingFile(
            root: root, name: Self.folderName(of: root), notConnected: notConnected)
        guard file != writtenPending else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        if AppJSONStore.save(file, to: pendingURL, label: Self.pendingLabel, encoder: encoder) {
            writtenPending = file
        }
    }

    private func discardPendingFile() {
        writtenPending = nil
        try? FileManager.default.removeItem(at: pendingURL)
    }

    // MARK: - Paths

    /// `path` as an absolute, standardized path, or nil when it is empty or
    /// relative. A leading tilde is expanded.
    nonisolated static func absolutePath(_ path: String) -> String? {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        let expanded = (trimmed as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }

    /// The name of the root folder, which the engine also gives the library it
    /// registers there (`default_name` in the catalog's import), so the
    /// placeholder and the library it becomes read the same.
    nonisolated static func folderName(of root: String) -> String {
        let name = URL(fileURLWithPath: root).standardizedFileURL.lastPathComponent
        return name.isEmpty || name == "/" ? "TurboSpark" : name
    }

    private func logLine(_ message: String) {
        FileHandle.standardError.write(Data("TurboSpark: moved store conversion: \(message)\n".utf8))
    }
}
