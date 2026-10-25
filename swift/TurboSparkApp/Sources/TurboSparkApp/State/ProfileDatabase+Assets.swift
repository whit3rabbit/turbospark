import Foundation
import SQLCipher

/// Asset ids referenced only by in-memory state (ghost chats), which the
/// database cannot see. Reconciliation never collects a pinned asset.
enum ManagedAssetPins {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var byOwner: [UUID: Set<String>] = [:]

    static func set(_ ids: Set<String>, owner: UUID) {
        lock.lock()
        defer { lock.unlock() }
        byOwner[owner] = ids.isEmpty ? nil : ids
    }

    static func clear(owner: UUID) {
        lock.lock()
        defer { lock.unlock() }
        byOwner[owner] = nil
    }

    static func clearAll() {
        lock.lock()
        defer { lock.unlock() }
        byOwner.removeAll()
    }

    static func allPinned() -> Set<String> {
        lock.lock()
        defer { lock.unlock() }
        return byOwner.values.reduce(into: Set<String>()) { $0.formUnion($1) }
    }

    /// Asset ids a message list references.
    static func assetIDs(in messages: [AppChatMessage]) -> Set<String> {
        var ids: Set<String> = []
        func add(_ reference: String?) {
            if let reference, let id = ManagedAssetStore.assetID(from: reference) { ids.insert(id) }
        }
        func visit(_ message: AppChatMessage) {
            message.imagePaths.forEach { add($0) }
            for result in message.toolResults {
                result.mediaReferences?.forEach { add($0.assetReference) }
            }
            message.alternates.forEach(visit)
        }
        messages.forEach(visit)
        return ids
    }
}

extension ProfileDatabase {
    /// The payload and reference graph must commit together, or a chat save
    /// could collect a chunk whose manifest update failed.
    func saveAudioRecord(key: String, payload: Data, references: [String]) throws {
        guard key.hasPrefix("audio:item:") || key == "audio:preferences" else { throw CocoaError(.coderInvalidValue) }
        let ids = try Set(references.map { reference in
            guard let id = ManagedAssetStore.assetID(from: reference) else { throw CocoaError(.coderInvalidValue) }
            return id
        })
        try transaction {
            try saveRecord(key: key, payload: payload)
            try withStatement("DELETE FROM audio_asset_references WHERE record_key = ?1") { statement in
                try bind(key, at: 1, to: statement)
                try stepDone(statement)
            }
            for id in ids {
                try withStatement("INSERT INTO audio_asset_references(record_key, asset_id) VALUES(?1, ?2)") { statement in
                    try bind(key, at: 1, to: statement)
                    try bind(id, at: 2, to: statement)
                    try stepDone(statement)
                }
            }
        }
    }

    func collectUnusedAudioAssets() throws -> [String] {
        var unreachable: [String] = []
        try transaction {
            let archive = try loadChatArchive() ?? AppChatArchive(selectedChatID: UUID(), chats: [])
            unreachable = try reconcileAssetReferences(in: archive)
        }
        return unreachable
    }

    func deleteAudioRecord(key: String) throws -> [String] {
        guard key.hasPrefix("audio:item:") else { throw CocoaError(.coderInvalidValue) }
        var unreachable: [String] = []
        try transaction {
            try withStatement("DELETE FROM audio_asset_references WHERE record_key = ?1") { statement in
                try bind(key, at: 1, to: statement)
                try stepDone(statement)
            }
            try deleteRecord(key: key)
            let archive = try loadChatArchive() ?? AppChatArchive(selectedChatID: UUID(), chats: [])
            unreachable = try reconcileAssetReferences(in: archive)
        }
        return unreachable
    }
}
