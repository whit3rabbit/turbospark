import Foundation
import CryptoKit

/// Thread-safe file snapshot and content hash cache to protect against stale writes and race conditions.
public actor FileSnapshotStore {
    public static let shared = FileSnapshotStore()

    private var snapshots: [String: String] = [:]

    public init() {}

    /// Records the content hash of a file at the given URL.
    public func recordSnapshot(url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        let hash = SHA256.hash(data: data)
        let hashString = hash.compactMap { String(format: "%02x", $0) }.joined()
        snapshots[url.standardizedFileURL.path] = hashString
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

    /// Clears recorded snapshots (e.g., when switching projects).
    public func reset() {
        snapshots.removeAll()
    }
}
