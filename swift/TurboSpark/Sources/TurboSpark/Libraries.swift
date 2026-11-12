import CTurboSpark
import Foundation

// Model libraries: folders on directly connected drives that hold model files,
// with "This Mac" as the implicit one (id "internal").
//
// Every key here is camelCase, the wire shape the Rust side invented for this
// binding (crates/ffi/src/api/libraries.rs), so no CodingKeys are needed and
// a rename on either side fails a decode test rather than dropping a field.
//
// **Names, aliases and paths come from a drive anyone could have written.**
// The engine passes them through raw (JSON-escaped only) because they are
// identities; show them through `UntrustedText.display`. The English literal
// "This Mac" is a name, not a label: localize it from `isInternal`.

/// Whether an installed model can be opened now.
public enum LibraryAvailability: String, Decodable, Sendable, Equatable {
    case available
    /// Its library's drive is not connected. The row stays listed and names
    /// the library to connect; this is a state, not an error.
    case notConnected = "not_connected"
    /// The drive is connected but the model's folder is gone, or its library
    /// is no longer listed.
    case missing
}

/// Where an installed model is, told from the engine's ids and paths and
/// never from a name (a marker can name its library "This Mac").
public enum LibraryLocationKind: String, Decodable, Sendable, Equatable {
    /// This Mac's models folder. The wire spelling is `internal`.
    case thisMac = "internal"
    /// A listed library.
    case library
    /// A folder the user chose with `pull --out`, outside every library.
    case custom
    /// A library that is no longer listed.
    case unknown
}

/// One library as the list shows it.
public struct ModelLibrary: Decodable, Sendable, Identifiable, Equatable {
    /// `internal` for This Mac, otherwise the library's own id. Install,
    /// move and default calls take this, never the name.
    public let id: String
    /// The library's name. For This Mac it is the English literal "This Mac":
    /// show a localized label when `isInternal`. Untrusted otherwise.
    public let name: String
    /// Where the library is now, nil while its drive is not connected.
    public let root: String?
    public let available: Bool
    /// True for exactly one listing: another connected folder carries the
    /// same library marker and the first one found is used. Do not treat it
    /// as sticky.
    public let duplicate: Bool
    public let modelCount: Int
    /// Exactly one library is the default for new downloads.
    public let isDefault: Bool
    /// `YYYY-MM-DD`, nil when it was never recorded.
    public let lastSeen: String?
    public let isInternal: Bool

    public init(
        id: String, name: String, root: String?, available: Bool, duplicate: Bool = false,
        modelCount: Int = 0, isDefault: Bool = false, lastSeen: String? = nil,
        isInternal: Bool = false
    ) {
        self.id = id
        self.name = name
        self.root = root
        self.available = available
        self.duplicate = duplicate
        self.modelCount = modelCount
        self.isDefault = isDefault
        self.lastSeen = lastSeen
        self.isInternal = isInternal
    }
}

/// What `discover` found on a volume: the marker of the standard TurboSpark
/// folder at the top of it.
public struct LibraryMarkerSummary: Decodable, Sendable, Equatable {
    public let id: String
    /// Untrusted, like every name from a drive.
    public let name: String
    public let createdOn: String
    public let modelCount: Int
    /// A library a newer TurboSpark wrote: this build can list it, not change
    /// it.
    public let readOnly: Bool
}

/// The result of adding a library.
public struct AddedLibrary: Decodable, Sendable, Equatable {
    public let library: ModelLibrary
    /// Aliases of the marker's rows that were not recorded because the alias
    /// and kind already belong to another library.
    public let skipped: [String]
}

/// One home copy a store import could not make.
public struct LibraryCopyFailure: Decodable, Sendable, Equatable {
    /// `embeddings/<alias>`, `ocr/<alias>` or `models.json`. Untrusted.
    public let key: String
    /// Why, escaped and capped by the engine. May echo drive paths.
    public let reason: String
}

/// The result of registering the models folder of a store Move Store moved.
public struct ImportedStore: Decodable, Sendable, Equatable {
    public let library: ModelLibrary
    /// Rows not recorded because their alias already belongs to another place.
    public let skipped: [String]
    /// Rows outside the models folder, kept as custom locations.
    public let custom: [String]
    /// Music installs with no receipt. The app asks the user to move them to
    /// This Mac and adopt them there.
    public let needsAdoption: [String]
    public let copiedToHome: [String]
    public let copyFailures: [LibraryCopyFailure]
    public let restoredCatalogOverride: Bool

    /// False while any home copy failed. **The conversion is not done then:**
    /// the failed copies are retried only when the import is called again, so
    /// keep it pending (on mount and at launch) or show the failures.
    public var isComplete: Bool { copyFailures.isEmpty }
}

/// How a model move that put the install in its destination ended.
///
/// **Every case means the install is in the destination and listed there.**
/// None is a failure to show; the last two ask the user for something small.
public enum MoveOutcome: Decodable, Sendable, Equatable {
    /// Copied, checked, listed, and the original deleted.
    case moved(installId: String)
    /// The original library's marker still lists the install, which is
    /// harmless. `staleMarker` names that library (untrusted).
    case sourceMarkerStale(installId: String, staleMarker: String?)
    /// The original folder could not be deleted and is left over at `path`.
    /// Tell the user to delete it by hand.
    case sourceCleanupFailed(installId: String, staleMarker: String?, path: String)
    /// An outcome this build does not know. The install is in the
    /// destination, as for every outcome.
    case unrecognized(outcome: String, installId: String)

    /// The install's id in its new library.
    public var installId: String {
        switch self {
        case .moved(let id), .sourceMarkerStale(let id, _), .sourceCleanupFailed(let id, _, _),
            .unrecognized(_, let id):
            return id
        }
    }

    private struct Wire: Decodable {
        let outcome: String
        let installId: String
        let staleMarker: String?
        let path: String?
    }

    public init(from decoder: Decoder) throws {
        let wire = try Wire(from: decoder)
        switch wire.outcome {
        case "moved":
            self = .moved(installId: wire.installId)
        case "source_marker_stale":
            self = .sourceMarkerStale(installId: wire.installId, staleMarker: wire.staleMarker)
        case "source_cleanup_failed":
            guard let path = wire.path else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "source_cleanup_failed without the leftover path"))
            }
            self = .sourceCleanupFailed(
                installId: wire.installId, staleMarker: wire.staleMarker, path: path)
        default:
            self = .unrecognized(outcome: wire.outcome, installId: wire.installId)
        }
    }
}

/// Listing, adding, moving and removing model libraries.
///
/// Every call is synchronous except `move`. The list, `add`, `discover`,
/// `forget` and `importMovedRoot` read the mount table, probe a drive or write
/// the registry, so **call them off the main actor** (`Task.detached`, as the
/// audio screens do). A change is refused with the `busy` code while an
/// install or a move runs; the check is made once, so also stop queueing
/// installs while you change libraries.
public enum TurboSparkLibraries {
    /// Every library, This Mac first, after locating each one. A library that
    /// is not connected is listed with no `root`; that is not an error.
    public static func list() throws -> [ModelLibrary] {
        try decode([ModelLibrary].self, from: try takeString { ts_libraries_json($0) })
    }

    /// Adds the folder at `path` as a library (or the TurboSpark folder it
    /// holds), adopting the library already there with its models. A new one
    /// gets a `TurboSpark` folder and a marker written by this call. `name`
    /// nil or empty derives one: the marker's, else the drive's.
    ///
    /// Refusals are `LibraryErrorCode.refusal` plus `marker`, `io` and `busy`.
    public static func add(path: String, name: String? = nil) throws -> AddedLibrary {
        try decode(
            AddedLibrary.self,
            from: try takeString { out in
                path.withCString { p in
                    withOptionalCString(name) { n in ts_library_add(p, n, out) }
                }
            })
    }

    /// Stops listing the library and its models. Never touches a file: the
    /// folder and its marker stay, and adding the drive again lists its
    /// models again. Refusals: `unknownLibrary`, `internalLibrary`, `io`,
    /// `busy`.
    public static func remove(id: String) throws {
        try check(id.withCString { ts_library_remove($0) })
    }

    /// Makes `id`, or `internal` for This Mac, the only library new downloads
    /// go to. Refusals: `unknownLibrary`, `io`, `busy`.
    public static func setDefault(id: String) throws {
        try check(id.withCString { ts_library_set_default($0) })
    }

    /// The library marker at the top of the volume at `volumeRoot`, or nil
    /// when there is none, it cannot be read, or the path is not there. Reads
    /// one file and never walks the drive, so it is safe to call as a drive
    /// connects.
    public static func discover(volumeRoot: String) throws -> LibraryMarkerSummary? {
        try decode(
            LibraryMarkerSummary?.self,
            from: try takeString { out in
                volumeRoot.withCString { ts_library_discover_json($0, out) }
            })
    }

    /// Stops listing the install (`installId` of an installed row,
    /// `<library>:<modality>/<alias>`) and drops it from its library's
    /// marker. Deletes no file, and an id that is not listed is not an error.
    /// Refusals: `notConnected`, `libraryReadOnly`, `libraryDisconnected`,
    /// `homeOnly`, `io`, `busy`.
    public static func forget(installId: String) throws {
        try check(installId.withCString { ts_model_forget($0) })
    }

    /// Registers the `models` folder of a store Move Store moved to `root` as
    /// a library, keeping every model where it is, and copies its embedding
    /// and OCR models and `models.json` into the home. Safe to call again:
    /// it returns the same library and retries the copies that failed.
    ///
    /// **Not finished while `copyFailures` is non-empty** (`isComplete`).
    /// Refusals: `notConnected` (the root or its models folder is not there),
    /// `refusal(.nested)`, `marker`, `io`, `busy`.
    public static func importMovedRoot(_ root: String) throws -> ImportedStore {
        try decode(
            ImportedStore.self,
            from: try takeString { out in
                root.withCString { ts_store_import_moved_root($0, out) }
            })
    }

    /// Moves an install to another library: copies it, checks the copy, lists
    /// it under the destination, and only then deletes the original. A
    /// refused, failed or cancelled move leaves the original where it was and
    /// removes any partial copy.
    ///
    /// The engine call BLOCKS for as long as the copy takes, so this runs it
    /// on a dedicated thread (a move can last tens of minutes, and a shared
    /// worker parked that long would starve the process) and the caller
    /// awaits. `progress` receives `(done, total)` bytes on that thread: first
    /// `(0, total)`, then the cumulative bytes copied. Take the maximum if you
    /// drive a bar from it.
    ///
    /// - Parameters:
    ///   - installId: the `installId` of the row, `<library>:<modality>/<alias>`.
    ///   - libraryId: the destination's `id`, or `internal` for This Mac.
    ///   - freeBytes: the destination's free space as the app measured it.
    ///     Below the model's size the move is refused before anything is
    ///     copied (`insufficientSpace`). Pass nil only when it could not be
    ///     measured: the move is then refused as `freeSpaceUnknown`, before
    ///     anything else, even for a destination that is not connected.
    ///
    /// Cancel with `cancelMove()`; cancelling the awaiting `Task` does not
    /// stop the copy. The model must be unloaded first: this library cannot
    /// see sessions. Errors are `TurboSparkError`s whose `libraryFault` codes
    /// are listed in `docs/SWIFT_BINDINGS.md`: `notConnected`,
    /// `libraryReadOnly`, `homeOnly`, `sameLibrary`, `unknownInstall`,
    /// `unknownLibrary`, `destinationTaken`, `insufficientSpace`,
    /// `verifyFailed`, `freeSpaceUnknown`, `io`, `cancelled` and `busy`.
    public static func move(
        installId: String,
        to libraryId: String,
        freeBytes: Int64?,
        progress: @escaping @Sendable (_ done: UInt64, _ total: UInt64) -> Void = { _, _ in }
    ) async throws -> MoveOutcome {
        try await withCheckedThrowingContinuation { continuation in
            let thread = Thread {
                let box = MoveBox(progress)
                let userdata = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<MoveBox>.fromOpaque(userdata).release() }
                do {
                    // A negative figure is the ABI's "could not measure".
                    let free = freeBytes ?? -1
                    let json = try takeString { out in
                        installId.withCString { id in
                            libraryId.withCString { library in
                                ts_model_move(id, library, free, moveCallback, userdata, out)
                            }
                        }
                    }
                    continuation.resume(returning: try decode(MoveOutcome.self, from: json))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            thread.name = "com.turbospark.model-move"
            thread.start()
        }
    }

    /// Asks the move in flight to stop at its next file or chunk boundary.
    /// The `move` call then throws with the `cancelled` code, the original
    /// untouched and the partial copy removed. Returns true when a move was
    /// running and has been signalled, false when none was (not an error). A
    /// cancel that arrives after the copy was published cannot undo the move,
    /// which then ends `moved`. Safe from any thread, and it leaves install
    /// walks alone: `TurboSparkCatalog.cancelInstall()` stops those.
    @discardableResult
    public static func cancelMove() -> Bool {
        ts_model_move_cancel() != 0
    }
}

/// Carries a move's progress closure across the C boundary. The callback runs
/// on the move's own thread, between files and chunks. Internal so a test can
/// drive the callback without a drive to move between.
final class MoveBox: @unchecked Sendable {
    let progress: @Sendable (UInt64, UInt64) -> Void
    init(_ progress: @escaping @Sendable (UInt64, UInt64) -> Void) { self.progress = progress }
}

let moveCallback: TsInstallCallback = { userdata, kind, _, _, done, total in
    // Byte progress only: a move sends no stage lines.
    guard let userdata, kind == TS_INSTALL_BYTES else { return }
    Unmanaged<MoveBox>.fromOpaque(userdata).takeUnretainedValue().progress(done, total)
}
