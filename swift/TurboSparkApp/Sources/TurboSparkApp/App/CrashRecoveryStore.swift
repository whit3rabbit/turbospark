import Foundation

/// Detects that the previous run of the app did not exit through
/// `applicationWillTerminate` (ZP-17).
///
/// SwiftUI has no way to catch a trap (force unwrap, bounds, precondition,
/// out of memory) raised while a view renders, so the process cannot contain
/// one. What it can do is notice afterwards: a marker file is written when the
/// session starts and deleted on every orderly exit. A marker that is still
/// there at the next start means the last run was killed, crashed, lost power
/// or was force quit, and the app says so once.
///
/// The marker holds a start time and a format version. No pid (a reused pid
/// proves nothing), no chat id, no path, nothing the user typed, and it is
/// never sent anywhere. It is a plain file under the store root, not inside
/// the vault, so it can be read without unlocking a profile.
///
/// Known false positives, accepted: a second instance of the app on the same
/// store root, a reboot or power loss during a session, and a Force Quit. All
/// look like a crash from here, and the notice says "did not quit cleanly",
/// not "crashed".
public final class CrashRecoveryStore: @unchecked Sendable {
    public enum Launch: Equatable, Sendable {
        /// No marker was left: the last run exited in order, or this is the
        /// first run, or the marker could not be read for lack of a store.
        case clean
        /// A marker survived. `startedAt` is nil when the file was present
        /// but unreadable, which still proves the last run did not clear it.
        case previousEndedAbnormally(startedAt: Date?)
    }

    struct Marker: Codable, Equatable {
        var version: Int
        var startedAt: Date
    }

    static let fileName = "session-marker.json"
    static let currentVersion = 1

    /// The store for this run's profile root. Tests build their own over a
    /// temp directory.
    public static let shared = CrashRecoveryStore(directory: AppStorageRoot.directory)

    private let markerURL: URL
    private let lock = NSLock()
    private var begun = false

    public init(directory: URL) {
        self.markerURL = directory.appendingPathComponent(Self.fileName)
    }

    /// Reads any surviving marker, then writes a fresh one for this session.
    ///
    /// Reading and replacing happen in one call so a notice is shown once per
    /// marker: the next start finds only the new marker. A second call in the
    /// same process (the main view appears again after a vault lock) returns
    /// `.clean` without touching the file, otherwise the process would read
    /// its own marker as a crash.
    @discardableResult
    public func beginSession(now: Date = Date()) -> Launch {
        lock.lock()
        defer { lock.unlock() }
        guard !begun else { return .clean }
        begun = true

        let launch = readSurvivingMarker()
        let marker = Marker(version: Self.currentVersion, startedAt: now)
        // A failed write only costs the next run its notice. It must never
        // block launch or surface an error for a diagnostic aid.
        if let data = try? Self.encoder.encode(marker) {
            try? data.write(to: markerURL, options: .atomic)
        }
        return launch
    }

    /// Deletes the marker. Called on every orderly exit, and only there: a
    /// vault lock replaces the model but the process lives on, so it must not
    /// clear the marker.
    public func endSession() {
        lock.lock()
        defer { lock.unlock() }
        try? FileManager.default.removeItem(at: markerURL)
    }

    private func readSurvivingMarker() -> Launch {
        guard FileManager.default.fileExists(atPath: markerURL.path) else { return .clean }
        guard let data = try? Data(contentsOf: markerURL),
              let marker = try? Self.decoder.decode(Marker.self, from: data)
        else { return .previousEndedAbnormally(startedAt: nil) }
        return .previousEndedAbnormally(startedAt: marker.startedAt)
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
