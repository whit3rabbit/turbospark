import Foundation
import CryptoKit

/// Thread-safe file snapshot and content hash cache to protect against stale writes and race conditions.
public actor FileSnapshotStore {
    public static let shared = FileSnapshotStore()

    private var snapshots: [String: String] = [:]
    /// Insertion order, oldest first, so the map can be trimmed without
    /// hashing timestamps into every entry.
    private var insertionOrder: [String] = []

    /// Largest number of tracked files. Every file the model reads or writes
    /// adds one entry and nothing ever removed them: `reset()` had no callers
    /// at all, so the "e.g., when switching projects" in its own doc comment
    /// described something that did not happen. A cap is the fix that holds
    /// even if a future caller forgets to reset.
    static let maximumTrackedFiles = 512

    public init() {}

    /// Records the content hash of a file whose contents the caller already
    /// has.
    ///
    /// **Takes the content rather than re-reading the path**, which is not a
    /// micro-optimization: every caller reaches this immediately after
    /// loading the file, so the old signature took a second full copy of
    /// every file the model touched purely to hash bytes that were already in
    /// memory one frame up.
    public func recordSnapshot(url: URL, content: String) {
        record(path: url.standardizedFileURL.path, data: Data(content.utf8))
    }

    /// Records the content hash of a file at the given URL, reading it.
    ///
    /// For callers that do not already hold the content. Prefer the overload
    /// above wherever the bytes are in hand.
    public func recordSnapshot(url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        record(path: url.standardizedFileURL.path, data: data)
    }

    private func record(path: String, data: Data) {
        let hashString = SHA256.hash(data: data).compactMap { String(format: "%02x", $0) }.joined()
        if snapshots[path] == nil {
            insertionOrder.append(path)
        }
        snapshots[path] = hashString

        while insertionOrder.count > Self.maximumTrackedFiles {
            let oldest = insertionOrder.removeFirst()
            snapshots.removeValue(forKey: oldest)
        }
    }

    /// Verifies if a file's current disk content matches the previously recorded snapshot.
    public func isStale(url: URL) -> Bool {
        let path = url.standardizedFileURL.path
        guard let recordedHash = snapshots[path] else {
            return false // Not previously tracked, assume fresh
        }
        guard let data = try? Data(contentsOf: url) else {
            return true // Cannot read file, treated as stale/missing
        }
        let currentHash = SHA256.hash(data: data).compactMap { String(format: "%02x", $0) }.joined()
        return currentHash != recordedHash
    }

    /// Clears recorded snapshots. Called when the active project changes, so
    /// a stale-write check cannot carry a hash from one workspace into
    /// another.
    public func reset() {
        snapshots.removeAll()
        insertionOrder.removeAll()
    }

    /// How many files are currently tracked. For tests and diagnostics.
    public var trackedFileCount: Int { snapshots.count }
}
