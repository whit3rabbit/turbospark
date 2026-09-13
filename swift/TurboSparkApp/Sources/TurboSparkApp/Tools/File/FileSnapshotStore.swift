import Foundation
import CryptoKit

/// Thread-safe file snapshot and content hash cache to protect against stale writes and race conditions.
public actor FileSnapshotStore {
    public static let shared = FileSnapshotStore()

    private var snapshots: [String: String] = [:]
    /// Rollback backup cache for undo_edit capability.
    private var backups: [String: String] = [:]
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

    /// Records pre-modification file content for undo_edit rollback.
    public func recordBackup(url: URL, content: String) {
        backups[url.standardizedFileURL.path] = content
    }

    /// Retrieves pre-modification file content for undo_edit rollback.
    public func restoreBackup(url: URL) -> String? {
        backups[url.standardizedFileURL.path]
    }

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

    /// Whether the file has a recorded snapshot at all.
    ///
    /// `isStale` deliberately answers false for an untracked file, which is
    /// the right default for EDIT (nothing read, nothing to compare) but
    /// wrong for a destructive WRITE over an existing file the model never
    /// read: there the absence of a snapshot is the hazard, not safety.
    /// Callers that need the distinction ask this first.
    public func isTracked(url: URL) -> Bool {
        snapshots[url.standardizedFileURL.path] != nil
    }

    /// Forgets a file's snapshot. Called after a patch deletes the file, so
    /// the store does not keep a hash for a path that no longer exists (and
    /// so a later file of the same path starts untracked, not "fresh
    /// against a dead hash").
    public func removeSnapshot(url: URL) {
        let path = url.standardizedFileURL.path
        guard snapshots.removeValue(forKey: path) != nil else { return }
        insertionOrder.removeAll { $0 == path }
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
