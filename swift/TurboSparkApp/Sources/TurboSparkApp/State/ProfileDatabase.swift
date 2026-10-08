import Foundation
import SQLCipher

final class ProfileDatabase: @unchecked Sendable {
    struct AssetMetadata: Equatable, Sendable {
        var id: String
        var fileName: String
        var mimeType: String?
        var byteCount: Int64
        var referenceCount: Int
        var createdAt: Date
    }
    enum DatabaseError: Error, LocalizedError {
        case open(String)
        case statement(String)
        case bind(String)
        case step(String)
        case cipherUnavailable
        case closed

        var errorDescription: String? {
            switch self {
            case .open(let message): return "Profile database could not be opened: \(message)"
            case .statement(let message): return "Profile database statement failed: \(message)"
            case .bind(let message): return "Profile database value could not be bound: \(message)"
            case .step(let message): return "Profile database operation failed: \(message)"
            case .cipherUnavailable: return "SQLCipher is unavailable in this build."
            case .closed: return "The profile database is locked."
            }
        }
    }

    enum WorkflowSchemaStatus: Equatable, Sendable {
        case uninitialized
        case available(version: Int64)
        case unsupportedVersion(Int64)
        case unavailable(String)
    }

    enum WorkflowSavedDefinitionStorageError: Error, Equatable, Sendable {
        case schemaUnavailable(WorkflowSchemaStatus)
        case invalidUpdatedAt
        case malformedRow(id: String, reason: String)
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    let url: URL
    private let lock = NSRecursiveLock()
    private var handle: OpaquePointer?
    /// Row ids that failed to decode at the last load. They are not in memory,
    /// so saves must leave them alone instead of treating them as deleted.
    private let undecodableLock = NSLock()
    private var undecodableChatIDs: Set<String> = []
    private var undecodableProjectIDs: Set<String> = []
    private var workflowSchemaStatusStorage: WorkflowSchemaStatus = .uninitialized

    var workflowSchemaStatus: WorkflowSchemaStatus {
        lock.lock()
        defer { lock.unlock() }
        return workflowSchemaStatusStorage
    }

    init(url: URL, key: Data) throws {
        self.url = url
        var database: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let result = sqlite3_open_v2(url.path, &database, flags, nil)
        guard result == SQLITE_OK, let database else {
            let message = database.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown error"
            if let database { sqlite3_close(database) }
            throw DatabaseError.open(message)
        }
        handle = database

        let keyResult = key.withUnsafeBytes { bytes in
            sqlite3_key(database, bytes.baseAddress, Int32(key.count))
        }
        guard keyResult == SQLITE_OK else {
            close()
            throw DatabaseError.open("SQLCipher refused the database key")
        }

        do {
            guard !(try scalarText("PRAGMA cipher_version;") ?? "").isEmpty else {
                throw DatabaseError.cipherUnavailable
            }
            try execute("PRAGMA cipher_memory_security = ON;")
            try execute("PRAGMA foreign_keys = ON;")
            try execute("PRAGMA journal_mode = WAL;")
            try execute("PRAGMA synchronous = FULL;")
            try execute("PRAGMA secure_delete = ON;")
            try createSchema()
            try execute("""
                CREATE TABLE IF NOT EXISTS audio_asset_references(
                    record_key TEXT NOT NULL,
                    asset_id TEXT NOT NULL REFERENCES assets(id),
                    PRIMARY KEY(record_key, asset_id)
                );
                """)
            do {
                workflowSchemaStatusStorage = try migrateWorkflowSchema()
            } catch {
                // Workflow persistence is optional to core profile operation.
                workflowSchemaStatusStorage = .unavailable(error.localizedDescription)
            }
            try Self.applyFilePermissions(at: url)
        } catch {
            close()
            throw error
        }
    }

    deinit { close() }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else { return }
        sqlite3_wal_checkpoint_v2(handle, nil, SQLITE_CHECKPOINT_TRUNCATE, nil, nil)
        sqlite3_close_v2(handle)
        self.handle = nil
    }

    func loadRecord(key: String) throws -> Data? {
        try withStatement("SELECT payload FROM private_records WHERE key = ?1") { statement in
            try bind(key, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return columnData(statement, index: 0)
        }
    }

    func saveRecord(key: String, payload: Data) throws {
        try withStatement(
            """
            INSERT INTO private_records(key, payload, updated_at)
            VALUES(?1, ?2, ?3)
            ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
            """) { statement in
            try bind(key, at: 1, to: statement)
            try bind(payload, at: 2, to: statement)
            try bind(Date().timeIntervalSince1970, at: 3, to: statement)
            try stepDone(statement)
        }
        if key.hasPrefix("memory:file:") {
            let body = String(data: payload, encoding: .utf8) ?? ""
            try withStatement(
                """
                INSERT INTO memory(id, scope, title, body, updated_at, payload_version, payload)
                VALUES(?1, ?2, ?3, ?4, ?5, 1, ?6)
                ON CONFLICT(id) DO UPDATE SET body=excluded.body,
                    updated_at=excluded.updated_at, payload=excluded.payload
                """) { statement in
                try bind(key, at: 1, to: statement)
                try bind(key.contains("/projects/") ? "project" : "profile", at: 2, to: statement)
                try bind(key.split(separator: "/").last.map(String.init), at: 3, to: statement)
                try bind(body, at: 4, to: statement)
                try bind(Date().timeIntervalSince1970, at: 5, to: statement)
                try bind(payload, at: 6, to: statement)
                try stepDone(statement)
            }
            try withStatement("DELETE FROM memory_fts WHERE memory_id = ?1") { statement in
                try bind(key, at: 1, to: statement)
                try stepDone(statement)
            }
            try withStatement(
                "INSERT INTO memory_fts(memory_id, title, body) VALUES(?1, ?2, ?3)"
            ) { statement in
                try bind(key, at: 1, to: statement)
                try bind(key.split(separator: "/").last.map(String.init), at: 2, to: statement)
                try bind(body, at: 3, to: statement)
                try stepDone(statement)
            }
        } else if key.hasPrefix("json:") {
            try withStatement(
                """
                INSERT INTO profile_settings(key, payload_version, payload, updated_at)
                VALUES(?1, 1, ?2, ?3)
                ON CONFLICT(key) DO UPDATE SET payload=excluded.payload,
                    updated_at=excluded.updated_at
                """) { statement in
                try bind(key, at: 1, to: statement)
                try bind(payload, at: 2, to: statement)
                try bind(Date().timeIntervalSince1970, at: 3, to: statement)
                try stepDone(statement)
            }
        }
    }

    func deleteRecord(key: String) throws {
        try withStatement("DELETE FROM private_records WHERE key = ?1") { statement in
            try bind(key, at: 1, to: statement)
            try stepDone(statement)
        }
        if key.hasPrefix("memory:file:") {
            try withStatement("DELETE FROM memory WHERE id = ?1") { statement in
                try bind(key, at: 1, to: statement)
                try stepDone(statement)
            }
            try withStatement("DELETE FROM memory_fts WHERE memory_id = ?1") { statement in
                try bind(key, at: 1, to: statement)
                try stepDone(statement)
            }
        } else if key.hasPrefix("json:") {
            try withStatement("DELETE FROM profile_settings WHERE key = ?1") { statement in
                try bind(key, at: 1, to: statement)
                try stepDone(statement)
            }
        }
    }

    func loadRecords(prefix: String) throws -> [(String, Data)] {
        var result: [(String, Data)] = []
        try withStatement(
            "SELECT key, payload FROM private_records WHERE key LIKE ?1 ORDER BY key"
        ) { statement in
            try bind(prefix + "%", at: 1, to: statement)
            while sqlite3_step(statement) == SQLITE_ROW {
                if let key = columnText(statement, index: 0),
                   let data = columnData(statement, index: 1) {
                    result.append((key, data))
                }
            }
        }
        return result
    }

    func loadChatArchive() throws -> AppChatArchive? {
        let selected = try loadMetadata(key: "selected-chat-id")
        var chats: [AppChat] = []
        var skipped: Set<String> = []
        let decoder = JSONDecoder()
        try withStatement(
            "SELECT payload, title, title_provenance, title_generation_attempted, recovery_anchor, id FROM chats ORDER BY updated_at DESC, id ASC"
        ) { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                let rowID = columnText(statement, index: 5)
                guard let payload = columnData(statement, index: 0),
                      var chat = try? decoder.decode(AppChat.self, from: payload)
                else {
                    // A row this build cannot read (written by a newer build)
                    // is not in memory, so a later save must not mistake it
                    // for a chat the user deleted.
                    if let rowID { skipped.insert(rowID) }
                    continue
                }
                let payloadFields = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any]

                if payloadFields?["title"] == nil || payloadFields?["title"] is NSNull,
                   let projectedTitle = columnText(statement, index: 1)
                {
                    chat.title = projectedTitle
                }
                if payloadFields?["titleProvenance"] == nil
                    || payloadFields?["titleProvenance"] is NSNull,
                   let rawValue = columnText(statement, index: 2),
                   let provenance = AppChatTitleProvenance(rawValue: rawValue)
                {
                    chat.titleProvenance = provenance
                }
                if payloadFields?["titleGenerationAttempted"] == nil
                    || payloadFields?["titleGenerationAttempted"] is NSNull,
                   sqlite3_column_type(statement, 3) != SQLITE_NULL
                {
                    chat.titleGenerationAttempted = sqlite3_column_int64(statement, 3) != 0
                }
                if payloadFields?["recoveryAnchor"] == nil
                    || payloadFields?["recoveryAnchor"] is NSNull,
                   let projectedData = optionalData(statement, index: 4),
                   let anchor = try? decoder.decode(RecoveryAnchor.self, from: projectedData),
                   anchor.isValid(forMessageCount: chat.messages.count)
                {
                    chat.recoveryAnchor = anchor
                }
                chats.append(chat)
            }
        }
        undecodableLock.lock()
        undecodableChatIDs = skipped
        undecodableLock.unlock()
        guard selected != nil || !chats.isEmpty else { return nil }
        let selectedID = selected.flatMap(UUID.init(uuidString:)) ?? chats.first?.id ?? UUID()
        return AppChatArchive(selectedChatID: selectedID, chats: chats)
    }

    /// Saves the chat graph and returns asset payload IDs that became
    /// unreachable. Callers remove those files only after this transaction
    /// commits, so a failed save can never strand a live database reference.
    func saveChatArchive(_ archive: AppChatArchive) throws -> [String] {
        let encoder = JSONEncoder()
        // Dictionary key order is randomized per process, so unsorted output
        // made every chat with tool calls look changed after each launch.
        encoder.outputFormatting = [.sortedKeys]
        var unreachableAssetIDs: [String] = []
        try transaction {
            let existing = try chatIDs()
            let current = Set(archive.chats.map { $0.id.uuidString })
            for chat in archive.chats {
                let payload = try encoder.encode(chat)
                if try chatPayload(id: chat.id.uuidString) != payload {
                    try upsertChat(chat, payload: payload)
                    try replaceNormalizedChildren(for: chat, encoder: encoder)
                    try updateSearchIndex(for: chat)
                }
            }
            undecodableLock.lock()
            let protectedIDs = undecodableChatIDs
            undecodableLock.unlock()
            for removed in existing.subtracting(current).subtracting(protectedIDs) {
                try deleteChat(id: removed)
            }
            try saveMetadata(key: "selected-chat-id", value: archive.selectedChatID.uuidString)
            unreachableAssetIDs = try reconcileAssetReferences(in: archive)
        }
        try Self.applyFilePermissions(at: url)
        return unreachableAssetIDs
    }

    func loadProjectArchive() throws -> AppProjectArchive? {
        let selected = try loadMetadata(key: "selected-project-id")
        var projects: [AppProject] = []
        var skipped: Set<String> = []
        let decoder = JSONDecoder()
        try withStatement("SELECT payload, id FROM projects ORDER BY updated_at DESC, id ASC") { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                guard let payload = columnData(statement, index: 0),
                      let project = try? decoder.decode(AppProject.self, from: payload)
                else {
                    // Same rule as chats: unreadable is not deleted.
                    if let id = columnText(statement, index: 1) { skipped.insert(id) }
                    continue
                }
                projects.append(project)
            }
        }
        undecodableLock.lock()
        undecodableProjectIDs = skipped
        undecodableLock.unlock()
        guard selected != nil || !projects.isEmpty else { return nil }
        return AppProjectArchive(
            selectedProjectID: selected.flatMap(UUID.init(uuidString:)),
            projects: projects)
    }

    func saveProjectArchive(_ archive: AppProjectArchive) throws {
        let encoder = JSONEncoder()
        try transaction {
            var existing: Set<String> = []
            try withStatement("SELECT id FROM projects") { statement in
                while sqlite3_step(statement) == SQLITE_ROW {
                    if let id = columnText(statement, index: 0) { existing.insert(id) }
                }
            }
            let current = Set(archive.projects.map { $0.id.uuidString })
            for project in archive.projects {
                try withStatement(
                    """
                    INSERT INTO projects(id, name, updated_at, payload_version, payload)
                    VALUES(?1, ?2, ?3, 1, ?4)
                    ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                        updated_at=excluded.updated_at, payload=excluded.payload
                    """) { statement in
                    try bind(project.id.uuidString, at: 1, to: statement)
                    try bind(project.name, at: 2, to: statement)
                    try bind(project.updatedAt.timeIntervalSince1970, at: 3, to: statement)
                    try bind(try encoder.encode(project), at: 4, to: statement)
                    try stepDone(statement)
                }
            }
            undecodableLock.lock()
            let protectedIDs = undecodableProjectIDs
            undecodableLock.unlock()
            for removed in existing.subtracting(current).subtracting(protectedIDs) {
                try withStatement("DELETE FROM projects WHERE id = ?1") { statement in
                    try bind(removed, at: 1, to: statement)
                    try stepDone(statement)
                }
            }
            if let selected = archive.selectedProjectID {
                try saveMetadata(key: "selected-project-id", value: selected.uuidString)
            } else {
                try deleteMetadata(key: "selected-project-id")
            }
        }
        try Self.applyFilePermissions(at: url)
    }

    func assetMetadata(id: String) throws -> AssetMetadata? {
        try withStatement(
            "SELECT file_name, mime_type, byte_count, reference_count, created_at FROM assets WHERE id = ?1"
        ) { statement in
            try bind(id, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return AssetMetadata(
                id: id,
                fileName: columnText(statement, index: 0) ?? "asset",
                mimeType: columnText(statement, index: 1),
                byteCount: sqlite3_column_int64(statement, 2),
                referenceCount: Int(sqlite3_column_int64(statement, 3)),
                createdAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 4)))
        }
    }

    func allAssetMetadata() throws -> [AssetMetadata] {
        var result: [AssetMetadata] = []
        try withStatement(
            "SELECT id, file_name, mime_type, byte_count, reference_count, created_at FROM assets ORDER BY id"
        ) { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                guard let id = columnText(statement, index: 0) else { continue }
                result.append(AssetMetadata(
                    id: id,
                    fileName: columnText(statement, index: 1) ?? "asset",
                    mimeType: columnText(statement, index: 2),
                    byteCount: sqlite3_column_int64(statement, 3),
                    referenceCount: Int(sqlite3_column_int64(statement, 4)),
                    createdAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 5))))
            }
        }
        return result
    }

    func retainAsset(
        id: String,
        fileName: String,
        mimeType: String?,
        byteCount: Int64
    ) throws {
        try withStatement(
            """
            INSERT INTO assets(id, file_name, mime_type, byte_count, reference_count, created_at)
            VALUES(?1, ?2, ?3, ?4, 1, ?5)
            ON CONFLICT(id) DO UPDATE SET
                reference_count=assets.reference_count + 1,
                created_at=excluded.created_at
            """) { statement in
            try bind(id, at: 1, to: statement)
            try bind(fileName, at: 2, to: statement)
            try bind(mimeType, at: 3, to: statement)
            guard sqlite3_bind_int64(statement, 4, byteCount) == SQLITE_OK else {
                throw DatabaseError.bind(errorMessage)
            }
            try bind(Date().timeIntervalSince1970, at: 5, to: statement)
            try stepDone(statement)
        }
    }

    /// Returns true when the encrypted payload no longer has any references.
    func releaseAsset(id: String) throws -> Bool {
        try transaction {
            try withStatement(
                "UPDATE assets SET reference_count = reference_count - 1 WHERE id = ?1"
            ) { statement in
                try bind(id, at: 1, to: statement)
                try stepDone(statement)
            }
            try withStatement("DELETE FROM assets WHERE id = ?1 AND reference_count <= 0") { statement in
                try bind(id, at: 1, to: statement)
                try stepDone(statement)
            }
        }
        return try assetMetadata(id: id) == nil
    }

    private func reconcileAssetReferences(in archive: AppChatArchive) throws -> [String] {
        var counts: [String: Int] = [:]
        func count(_ reference: String?) {
            guard let reference, let id = ManagedAssetStore.assetID(from: reference) else { return }
            counts[id, default: 0] += 1
        }
        func count(_ message: AppChatMessage) {
            message.imagePaths.forEach { count($0) }
            for result in message.toolResults {
                result.mediaReferences?.forEach { count($0.assetReference) }
            }
            message.alternates.forEach(count)
        }
        // Files the legacy migration encrypted but no chat references (for
        // example generated images left behind by a deleted chat). The
        // plaintext originals are gone, so these are the only copies: keep
        // them reachable through the migration record instead of collecting
        // them a minute after the next save.
        if let payload = try loadRecord(key: ProfileRepository.legacyAssetsRecordKey),
           let descriptors = try? JSONDecoder().decode([ManagedAssetDescriptor].self, from: payload) {
            for descriptor in descriptors { counts[descriptor.id, default: 0] += 1 }
        }
        for chat in archive.chats {
            chat.draftAttachments.forEach { count($0.sourcePath) }
            chat.artifacts.forEach { count($0.path) }
            chat.messages.forEach(count)
        }

        // Audio outlives the chat that happened to trigger this collection.
        // Read normalized references, including records this build cannot decode.
        try withStatement("SELECT asset_id FROM audio_asset_references") { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                if let id = columnText(statement, index: 0) { counts[id, default: 0] += 1 }
            }
        }

        var unreachable: [String] = []
        let pendingWriteCutoff = Date().addingTimeInterval(-60)
        // Temporary (ghost) chats are never archived, so their images are
        // referenced only from memory. Without this the 60 s grace window
        // would expire and collect assets a live ghost chat still uses.
        let pinned = ManagedAssetPins.allPinned()
        for metadata in try allAssetMetadata() {
            guard let referenceCount = counts[metadata.id], referenceCount > 0 else {
                if pinned.contains(metadata.id) { continue }
                // File encryption and chat attachment are separate operations.
                // A concurrent chat save must not collect a payload still on
                // its way back from the import task to the main actor.
                guard metadata.createdAt <= pendingWriteCutoff else { continue }
                try withStatement("DELETE FROM assets WHERE id = ?1") { statement in
                    try bind(metadata.id, at: 1, to: statement)
                    try stepDone(statement)
                }
                unreachable.append(metadata.id)
                continue
            }
            // Skip rows that already carry the right count: this runs on every
            // debounced save and each update is a synchronous=FULL write.
            guard metadata.referenceCount != referenceCount else { continue }
            try withStatement("UPDATE assets SET reference_count = ?2 WHERE id = ?1") { statement in
                try bind(metadata.id, at: 1, to: statement)
                try bind(Int64(referenceCount), at: 2, to: statement)
                try stepDone(statement)
            }
        }
        return unreachable
    }

    func integrityCheck() throws -> Bool {
        try scalarText("PRAGMA integrity_check;") == "ok"
    }

    func checkpoint() throws {
        try execute("PRAGMA wal_checkpoint(FULL);")
        try Self.applyFilePermissions(at: url)
    }

    /// Re-encrypts every page under `key`. SQLCipher rewrites the file inside
    /// one transaction, so a crash leaves it wholly under the old or the new
    /// key; the caller records both keys durably before calling this.
    func rekey(to key: Data) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else { throw DatabaseError.closed }
        // Fold the WAL into the main file and leave WAL mode for the rewrite:
        // a rollback journal makes the rekey one atomic transaction, whereas
        // a crash with new-key frames in the -wal beside an old-key main file
        // would leave a file neither key opens.
        try execute("PRAGMA wal_checkpoint(TRUNCATE);")
        try execute("PRAGMA journal_mode = DELETE;")
        let status = key.withUnsafeBytes { bytes in
            sqlite3_rekey(handle, bytes.baseAddress, Int32(key.count))
        }
        let restored = (try? execute("PRAGMA journal_mode = WAL;")) != nil
        guard status == SQLITE_OK else {
            throw DatabaseError.open("SQLCipher refused the replacement key")
        }
        guard restored else { throw DatabaseError.open("could not return to WAL mode after rekey") }
        try Self.applyFilePermissions(at: url)
    }

    /// Produces a transactionally consistent encrypted snapshot using
    /// SQLite's online backup API. Copying the database file itself would
    /// race the WAL and can omit the newest committed rows.
    ///
    /// Returns the asset rows as of the snapshot. They are read under the same
    /// lock hold as the copy, so the list cannot disagree with the snapshot
    /// when a row is added or deleted a moment later.
    @discardableResult
    func backup(to destination: URL, key: Data) throws -> [AssetMetadata] {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else { throw DatabaseError.closed }
        try? FileManager.default.removeItem(at: destination)
        var target: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(destination.path, &target, flags, nil) == SQLITE_OK,
              let target
        else {
            let message = target.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown error"
            if let target { sqlite3_close(target) }
            throw DatabaseError.open(message)
        }
        defer { sqlite3_close(target) }
        let keyResult = key.withUnsafeBytes { bytes in
            sqlite3_key(target, bytes.baseAddress, Int32(key.count))
        }
        guard keyResult == SQLITE_OK else {
            throw DatabaseError.open("SQLCipher refused the snapshot key")
        }
        guard let backup = sqlite3_backup_init(target, "main", handle, "main") else {
            throw DatabaseError.open(String(cString: sqlite3_errmsg(target)))
        }
        let result = sqlite3_backup_step(backup, -1)
        let finish = sqlite3_backup_finish(backup)
        guard result == SQLITE_DONE, finish == SQLITE_OK else {
            throw DatabaseError.open(String(cString: sqlite3_errmsg(target)))
        }
        var integrity: OpaquePointer?
        guard sqlite3_prepare_v2(target, "PRAGMA integrity_check;", -1, &integrity, nil) == SQLITE_OK,
              let integrity
        else { throw DatabaseError.open(String(cString: sqlite3_errmsg(target))) }
        defer { sqlite3_finalize(integrity) }
        guard sqlite3_step(integrity) == SQLITE_ROW,
              sqlite3_column_text(integrity, 0).map({ String(cString: $0) }) == "ok"
        else { throw DatabaseError.open("snapshot integrity check failed") }
        try Self.applyFilePermissions(at: destination)
        return try allAssetMetadata()
    }

    /// Per-chat projection tables. Keys are scoped to the chat so two chats
    /// may legitimately share a row id (a branch, a duplicate, or a model that
    /// numbers its todos "1", "2"); a global key made the second save fail.
    /// Todos key on ordinal because model-chosen ids can repeat inside one chat.
    static let childTableDDL: [String: String] = [
        "messages": """
        CREATE TABLE IF NOT EXISTS messages(
            id TEXT NOT NULL,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            role TEXT NOT NULL,
            created_at REAL NOT NULL,
            payload_version INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY(chat_id, id)
        );
        """,
        "alternates": """
        CREATE TABLE IF NOT EXISTS alternates(
            id TEXT NOT NULL,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            payload_version INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY(chat_id, id)
        );
        """,
        "attachments": """
        CREATE TABLE IF NOT EXISTS attachments(
            id TEXT NOT NULL,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            asset_id TEXT,
            file_name TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            payload_version INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY(chat_id, id)
        );
        """,
        "artifacts": """
        CREATE TABLE IF NOT EXISTS artifacts(
            id TEXT NOT NULL,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            asset_id TEXT,
            title TEXT NOT NULL,
            payload_version INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY(chat_id, id)
        );
        """,
        "todos": """
        CREATE TABLE IF NOT EXISTS todos(
            id TEXT NOT NULL,
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            status TEXT NOT NULL,
            payload_version INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY(chat_id, ordinal)
        );
        """,
    ]

    private func createSchema() throws {
        try execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations(
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS private_records(
                key TEXT PRIMARY KEY,
                payload BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vault_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chats(
                id TEXT PRIMARY KEY,
                project_id TEXT,
                title TEXT NOT NULL,
                updated_at REAL NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chats_project_updated
                ON chats(project_id, updated_at DESC);
            \(Self.childTableDDL["messages"]!)
            CREATE INDEX IF NOT EXISTS messages_chat_ordinal ON messages(chat_id, ordinal);
            \(Self.childTableDDL["alternates"]!)
            CREATE TABLE IF NOT EXISTS drafts(
                chat_id TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects(
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                updated_at REAL NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            \(Self.childTableDDL["attachments"]!)
            \(Self.childTableDDL["artifacts"]!)
            CREATE TABLE IF NOT EXISTS memory(
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                title TEXT,
                body TEXT NOT NULL,
                updated_at REAL NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB
            );
            CREATE TABLE IF NOT EXISTS memory_system(
                id INTEGER PRIMARY KEY CHECK(id = 1),
                payload BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_claims(
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence TEXT NOT NULL,
                text TEXT NOT NULL,
                supersedes_claim_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_evidence(
                id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL REFERENCES memory_claims(id) ON DELETE CASCADE,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                quote TEXT NOT NULL,
                source_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memory_evidence_source
                ON memory_evidence(chat_id, message_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_claims_fts USING fts5(
                claim_id UNINDEXED, text, tokenize='unicode61'
            );
            CREATE TABLE IF NOT EXISTS memory_claim_embeddings(
                claim_id TEXT NOT NULL REFERENCES memory_claims(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY(claim_id, model)
            );
            CREATE TABLE IF NOT EXISTS embeddings(
                id TEXT PRIMARY KEY,
                memory_id TEXT REFERENCES memory(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                vector BLOB NOT NULL,
                payload_version INTEGER NOT NULL
            );
            \(Self.childTableDDL["todos"]!)
            CREATE TABLE IF NOT EXISTS profile_settings(
                key TEXT PRIMARY KEY,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chat_fts USING fts5(
                chat_id UNINDEXED,
                title,
                body,
                tokenize='unicode61'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                memory_id UNINDEXED,
                title,
                body,
                tokenize='unicode61'
            );
            CREATE TABLE IF NOT EXISTS assets(
                id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                mime_type TEXT,
                byte_count INTEGER NOT NULL,
                reference_count INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );
            INSERT OR IGNORE INTO schema_migrations(version, applied_at)
            VALUES(1, unixepoch());
            """)
        try migrateChildTableKeys()
        try migrateChatProjections()
    }

    /// Installs workflow-owned state independently from profile-wide migrations.
    /// All workflow objects and their version marker commit or roll back together.
    private func migrateWorkflowSchema() throws -> WorkflowSchemaStatus {
        var status: WorkflowSchemaStatus = .uninitialized
        try transaction {
            try execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_schema_migrations(
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                );
                """)

            let highestVersion = try highestWorkflowSchemaVersion()
            if let highestVersion, highestVersion > 1 {
                status = .unsupportedVersion(highestVersion)
                return
            }
            if highestVersion == 1 {
                status = .available(version: 1)
                return
            }

            let tableStatements = [
                """
                CREATE TABLE IF NOT EXISTS workflow_runs(
                    run_id TEXT PRIMARY KEY,
                    name TEXT,
                    state TEXT,
                    script_source TEXT,
                    script_hash TEXT,
                    facade_version INTEGER,
                    args_json TEXT,
                    parent_run_id TEXT,
                    launch_source TEXT,
                    canonicalization_version INTEGER,
                    usage_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS workflow_actors(
                    run_id TEXT,
                    actor_name TEXT,
                    usage_json TEXT,
                    transcript_json TEXT,
                    transcript_sha256 TEXT,
                    PRIMARY KEY(run_id, actor_name)
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS workflow_events(
                    run_id TEXT,
                    seq INTEGER,
                    kind TEXT,
                    lane TEXT,
                    site_index INTEGER,
                    site_ordinal INTEGER,
                    input_hash TEXT,
                    payload_json TEXT,
                    created_at TEXT,
                    PRIMARY KEY(run_id, seq)
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS workflow_artifacts(
                    run_id TEXT,
                    artifact_id TEXT,
                    version INTEGER,
                    kind TEXT,
                    is_primary INTEGER,
                    storage_path TEXT,
                    created_at TEXT,
                    PRIMARY KEY(run_id, artifact_id, version)
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS workflow_questions(
                    run_id TEXT,
                    question_id TEXT,
                    prompt TEXT,
                    answered INTEGER,
                    answer TEXT,
                    created_at TEXT,
                    resolved_at TEXT,
                    PRIMARY KEY(run_id, question_id)
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS saved_workflows(
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT,
                    scope TEXT,
                    script_source TEXT,
                    args_declaration_json TEXT,
                    description TEXT,
                    updated_at TEXT
                );
                """,
            ]
            for statement in tableStatements {
                try execute(statement)
            }
            try execute(
                """
                CREATE INDEX IF NOT EXISTS workflow_runs_state_updated
                    ON workflow_runs(state, updated_at);
                INSERT INTO workflow_schema_migrations(version, applied_at)
                VALUES(1, unixepoch());
                """)
            status = .available(version: 1)
        }
        return status
    }

    private func highestWorkflowSchemaVersion() throws -> Int64? {
        try withStatement("SELECT MAX(version) FROM workflow_schema_migrations") { statement in
            guard sqlite3_step(statement) == SQLITE_ROW else {
                throw DatabaseError.step(errorMessage)
            }
            guard sqlite3_column_type(statement, 0) != SQLITE_NULL else { return nil }
            return sqlite3_column_int64(statement, 0)
        }
    }

    /// Rewrites databases created with a global `id` primary key on the per-chat
    /// projection tables to the chat-scoped keys in `childTableDDL`. The tables
    /// are write-only projections of `chats.payload`, but the copy keeps every
    /// existing row anyway (OR IGNORE only guards against a pre-existing
    /// duplicate bricking the open). One transaction: all tables or none.
    private func migrateChildTableKeys() throws {
        let columns: [String: String] = [
            "messages": "id, chat_id, ordinal, role, created_at, payload_version, payload",
            "alternates": "id, chat_id, message_id, ordinal, payload_version, payload",
            "attachments": "id, chat_id, asset_id, file_name, extracted_text, payload_version, payload",
            "artifacts": "id, chat_id, asset_id, title, payload_version, payload",
            "todos": "id, chat_id, ordinal, status, payload_version, payload",
        ]
        var stale: [String] = []
        for table in columns.keys.sorted() where try !chatIDIsPrimaryKey(table: table) {
            stale.append(table)
        }
        guard !stale.isEmpty else { return }
        try transaction {
            for table in stale {
                try execute("ALTER TABLE \(table) RENAME TO \(table)_old;")
                if table == "messages" { try execute("DROP INDEX IF EXISTS messages_chat_ordinal;") }
                try execute(Self.childTableDDL[table]!)
                try execute(
                    "INSERT OR IGNORE INTO \(table)(\(columns[table]!)) SELECT \(columns[table]!) FROM \(table)_old;")
                try execute("DROP TABLE \(table)_old;")
            }
            try execute("CREATE INDEX IF NOT EXISTS messages_chat_ordinal ON messages(chat_id, ordinal);")
            try execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(3, unixepoch());")
        }
    }

    private func chatIDIsPrimaryKey(table: String) throws -> Bool {
        var isKey = false
        try withStatement("PRAGMA table_info(\(table));") { statement in
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                if columnText(statement, index: 1) == "chat_id", sqlite3_column_int(statement, 5) > 0 {
                    isKey = true
                }
            }
        }
        return isKey
    }

    /// Adds chat metadata projections for fast indexing and preserves them
    /// when older app versions rewrite payloads without knowing these fields.
    private func migrateChatProjections() throws {
        try transaction {
            let requiredColumns: [(String, String)] = [
                ("title_provenance", "TEXT"),
                ("title_generation_attempted", "INTEGER"),
                ("recovery_anchor", "BLOB"),
            ]
            let existingColumns = try chatColumnNames()
            let migrationAlreadyApplied = try hasMigration(version: 2)
            if migrationAlreadyApplied
                && requiredColumns.allSatisfy({ existingColumns.contains($0.0) })
            {
                return
            }

            var columns = existingColumns
            for (name, definition) in requiredColumns where !columns.contains(name) {
                try execute("ALTER TABLE chats ADD COLUMN \(name) \(definition);")
                columns.insert(name)
            }

            let rows = try chatProjectionMigrationRows()
            let decoder = JSONDecoder()
            let encoder = JSONEncoder()
            for row in rows {
                let decodedChat = try? decoder.decode(AppChat.self, from: row.payload)
                let payload = (try? JSONSerialization.jsonObject(with: row.payload)) as? [String: Any]
                let title = payload?["title"] as? String ?? row.title
                let inferredProvenance = AppChatTitleProvenance.inferred(from: title)
                let payloadProvenance = (payload?["titleProvenance"] as? String)
                    .flatMap { AppChatTitleProvenance(rawValue: $0)?.rawValue }
                let provenance = row.titleProvenance
                    ?? payloadProvenance
                    ?? inferredProvenance.rawValue
                let payloadAttempt = payload?["titleGenerationAttempted"] as? Bool
                let attempted = row.titleGenerationAttempted
                    ?? payloadAttempt.map { $0 ? 1 : 0 }
                    ?? (provenance == AppChatTitleProvenance.unclaimed.rawValue ? 0 : 1)
                let anchorData = row.recoveryAnchor
                    ?? decodedChat?.recoveryAnchor.flatMap { try? encoder.encode($0) }

                try withStatement(
                    """
                    UPDATE chats SET
                        title_provenance = COALESCE(title_provenance, ?2),
                        title_generation_attempted = COALESCE(title_generation_attempted, ?3),
                        recovery_anchor = COALESCE(recovery_anchor, ?4)
                    WHERE id = ?1
                    """) { statement in
                    try bind(row.id, at: 1, to: statement)
                    try bind(provenance, at: 2, to: statement)
                    try bind(attempted, at: 3, to: statement)
                    try bind(anchorData, at: 4, to: statement)
                    try stepDone(statement)
                }
            }
            try execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(2, unixepoch());")
        }
    }

    private struct ChatProjectionMigrationRow {
        var id: String
        var title: String
        var payload: Data
        var titleProvenance: String?
        var titleGenerationAttempted: Int64?
        var recoveryAnchor: Data?
    }

    private func chatColumnNames() throws -> Set<String> {
        var names = Set<String>()
        try withStatement("PRAGMA table_info(chats);") { statement in
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                if let name = columnText(statement, index: 1) { names.insert(name) }
            }
        }
        return names
    }

    private func hasMigration(version: Int64) throws -> Bool {
        try withStatement("SELECT 1 FROM schema_migrations WHERE version = ?1 LIMIT 1") { statement in
            try bind(version, at: 1, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_ROW { return true }
            if result == SQLITE_DONE { return false }
            throw DatabaseError.step(errorMessage)
        }
    }

    private func chatProjectionMigrationRows() throws -> [ChatProjectionMigrationRow] {
        var rows: [ChatProjectionMigrationRow] = []
        try withStatement(
            "SELECT id, title, payload, title_provenance, title_generation_attempted, recovery_anchor FROM chats"
        ) { statement in
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                guard let id = columnText(statement, index: 0),
                      let title = columnText(statement, index: 1),
                      let payload = columnData(statement, index: 2)
                else { continue }
                rows.append(ChatProjectionMigrationRow(
                    id: id,
                    title: title,
                    payload: payload,
                    titleProvenance: columnText(statement, index: 3),
                    titleGenerationAttempted: optionalInt64(statement, index: 4),
                    recoveryAnchor: optionalData(statement, index: 5)))
            }
        }
        return rows
    }

    func chatIDs() throws -> Set<String> {
        var result: Set<String> = []
        try withStatement("SELECT id FROM chats") { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                if let value = columnText(statement, index: 0) { result.insert(value) }
            }
        }
        return result
    }

    func chatPayload(id: String) throws -> Data? {
        try withStatement("SELECT payload FROM chats WHERE id = ?1") { statement in
            try bind(id, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return columnData(statement, index: 0)
        }
    }

    private func upsertChat(_ chat: AppChat, payload: Data) throws {
        try withStatement(
            """
            INSERT INTO chats(
                id, project_id, title, updated_at, payload,
                title_provenance, title_generation_attempted, recovery_anchor)
            VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(id) DO UPDATE SET
                project_id=excluded.project_id,
                title=excluded.title,
                updated_at=excluded.updated_at,
                payload=excluded.payload,
                title_provenance=excluded.title_provenance,
                title_generation_attempted=excluded.title_generation_attempted,
                recovery_anchor=excluded.recovery_anchor
            """) { statement in
            try bind(chat.id.uuidString, at: 1, to: statement)
            try bind(chat.projectID?.uuidString, at: 2, to: statement)
            try bind(chat.title, at: 3, to: statement)
            try bind(chat.updatedAt.timeIntervalSince1970, at: 4, to: statement)
            try bind(payload, at: 5, to: statement)
            try bind(chat.titleProvenance.rawValue, at: 6, to: statement)
            try bind(chat.titleGenerationAttempted ? Int64(1) : Int64(0), at: 7, to: statement)
            let anchorData = try chat.recoveryAnchor.map { try JSONEncoder().encode($0) }
            try bind(anchorData, at: 8, to: statement)
            try stepDone(statement)
        }
    }

    private func updateSearchIndex(for chat: AppChat) throws {
        try withStatement("DELETE FROM chat_fts WHERE chat_id = ?1") { statement in
            try bind(chat.id.uuidString, at: 1, to: statement)
            try stepDone(statement)
        }
        func searchableText(_ message: AppChatMessage) -> [String] {
            [message.content, message.reasoning]
                + message.alternates.flatMap(searchableText)
        }
        let messageText = chat.messages.flatMap(searchableText)
        let attachmentText = chat.draftAttachments.map(\.extractedText)
        let body = ([chat.draft] + messageText + attachmentText).joined(separator: "\n")
        try withStatement(
            "INSERT INTO chat_fts(chat_id, title, body) VALUES(?1, ?2, ?3)"
        ) { statement in
            try bind(chat.id.uuidString, at: 1, to: statement)
            try bind(chat.title, at: 2, to: statement)
            try bind(body, at: 3, to: statement)
            try stepDone(statement)
        }
    }

    private func replaceNormalizedChildren(for chat: AppChat, encoder: JSONEncoder) throws {
        let chatID = chat.id.uuidString
        for table in ["alternates", "messages", "drafts", "attachments", "artifacts", "todos"] {
            try withStatement("DELETE FROM \(table) WHERE chat_id = ?1") { statement in
                try bind(chatID, at: 1, to: statement)
                try stepDone(statement)
            }
        }
        try withStatement(
            "INSERT INTO drafts(chat_id, text, payload_version, payload) VALUES(?1, ?2, 1, ?3)"
        ) { statement in
            try bind(chatID, at: 1, to: statement)
            try bind(chat.draft, at: 2, to: statement)
            try bind(try encoder.encode(chat.draftAttachments), at: 3, to: statement)
            try stepDone(statement)
        }
        for (ordinal, message) in chat.messages.enumerated() {
            try withStatement(
                "INSERT INTO messages(id, chat_id, ordinal, role, created_at, payload_version, payload) VALUES(?1, ?2, ?3, ?4, ?5, 1, ?6)"
            ) { statement in
                try bind(message.id.uuidString, at: 1, to: statement)
                try bind(chatID, at: 2, to: statement)
                try bind(Int64(ordinal), at: 3, to: statement)
                try bind(message.role.rawValue, at: 4, to: statement)
                try bind(message.createdAt.timeIntervalSince1970, at: 5, to: statement)
                try bind(try encoder.encode(message), at: 6, to: statement)
                try stepDone(statement)
            }
            for (alternateOrdinal, alternate) in message.alternates.enumerated() {
                try withStatement(
                    "INSERT INTO alternates(id, chat_id, message_id, ordinal, payload_version, payload) VALUES(?1, ?2, ?3, ?4, 1, ?5)"
                ) { statement in
                    try bind(alternate.id.uuidString, at: 1, to: statement)
                    try bind(chatID, at: 2, to: statement)
                    try bind(message.id.uuidString, at: 3, to: statement)
                    try bind(Int64(alternateOrdinal), at: 4, to: statement)
                    try bind(try encoder.encode(alternate), at: 5, to: statement)
                    try stepDone(statement)
                }
            }
        }
        for attachment in chat.draftAttachments {
            try withStatement(
                "INSERT INTO attachments(id, chat_id, asset_id, file_name, extracted_text, payload_version, payload) VALUES(?1, ?2, ?3, ?4, ?5, 1, ?6)"
            ) { statement in
                try bind(attachment.id.uuidString, at: 1, to: statement)
                try bind(chatID, at: 2, to: statement)
                try bind(attachment.sourcePath.flatMap(ManagedAssetStore.assetID(from:)), at: 3, to: statement)
                try bind(attachment.fileName, at: 4, to: statement)
                try bind(attachment.extractedText, at: 5, to: statement)
                try bind(try encoder.encode(attachment), at: 6, to: statement)
                try stepDone(statement)
            }
        }
        for artifact in chat.artifacts {
            try withStatement(
                "INSERT INTO artifacts(id, chat_id, asset_id, title, payload_version, payload) VALUES(?1, ?2, ?3, ?4, 1, ?5)"
            ) { statement in
                try bind(artifact.id.uuidString, at: 1, to: statement)
                try bind(chatID, at: 2, to: statement)
                try bind(artifact.path.flatMap(ManagedAssetStore.assetID(from:)), at: 3, to: statement)
                try bind(artifact.title, at: 4, to: statement)
                try bind(try encoder.encode(artifact), at: 5, to: statement)
                try stepDone(statement)
            }
        }
        for (ordinal, todo) in chat.todos.enumerated() {
            try withStatement(
                "INSERT INTO todos(id, chat_id, ordinal, status, payload_version, payload) VALUES(?1, ?2, ?3, ?4, 1, ?5)"
            ) { statement in
                try bind(todo.id, at: 1, to: statement)
                try bind(chatID, at: 2, to: statement)
                try bind(Int64(ordinal), at: 3, to: statement)
                try bind(todo.status, at: 4, to: statement)
                try bind(try encoder.encode(todo), at: 5, to: statement)
                try stepDone(statement)
            }
        }
    }

    private func deleteChat(id: String) throws {
        try withStatement("DELETE FROM chat_fts WHERE chat_id = ?1") { statement in
            try bind(id, at: 1, to: statement)
            try stepDone(statement)
        }
        try withStatement("DELETE FROM chats WHERE id = ?1") { statement in
            try bind(id, at: 1, to: statement)
            try stepDone(statement)
        }
    }

    private func loadMetadata(key: String) throws -> String? {
        try withStatement("SELECT value FROM vault_metadata WHERE key = ?1") { statement in
            try bind(key, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return columnText(statement, index: 0)
        }
    }

    private func saveMetadata(key: String, value: String) throws {
        try withStatement(
            """
            INSERT INTO vault_metadata(key, value) VALUES(?1, ?2)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """) { statement in
            try bind(key, at: 1, to: statement)
            try bind(value, at: 2, to: statement)
            try stepDone(statement)
        }
    }

    private func deleteMetadata(key: String) throws {
        try withStatement("DELETE FROM vault_metadata WHERE key = ?1") { statement in
            try bind(key, at: 1, to: statement)
            try stepDone(statement)
        }
    }

    private func transaction(_ body: () throws -> Void) throws {
        lock.lock()
        defer { lock.unlock() }
        try execute("BEGIN IMMEDIATE;")
        do {
            try body()
            try execute("COMMIT;")
        } catch {
            try? execute("ROLLBACK;")
            throw error
        }
    }

    func loadMemoryLedger() throws -> Data? {
        try withStatement("SELECT payload FROM memory_system WHERE id = 1") { statement in
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return columnData(statement, index: 0)
        }
    }

    /// The JSON snapshot is the recovery point. SQL rows and Markdown are
    /// rebuilt in this same transaction, so search never sees half a change.
    func saveMemoryLedger(_ payload: Data, projections: [String: Data]) throws {
        let state = try JSONDecoder().decode(MemoryLedgerState.self, from: payload)
        try transaction {
            try withStatement(
                "INSERT INTO memory_system(id,payload,updated_at) VALUES(1,?1,?2) "
                    + "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at"
            ) { statement in
                try bind(payload, at: 1, to: statement)
                try bind(Date().timeIntervalSince1970, at: 2, to: statement)
                try stepDone(statement)
            }
            try execute("DELETE FROM memory_claims_fts;")
            try execute("DELETE FROM memory_evidence;")
            for claim in state.claims {
                try withStatement(
                    "INSERT INTO memory_claims(id,scope,kind,status,confidence,text,supersedes_claim_id,created_at,updated_at) "
                        + "VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9) "
                        + "ON CONFLICT(id) DO UPDATE SET scope=excluded.scope,kind=excluded.kind,"
                        + "status=excluded.status,confidence=excluded.confidence,text=excluded.text,"
                        + "supersedes_claim_id=excluded.supersedes_claim_id,updated_at=excluded.updated_at"
                ) { statement in
                    try bind(claim.id.uuidString, at: 1, to: statement)
                    try bind(claim.scope, at: 2, to: statement)
                    try bind(claim.kind, at: 3, to: statement)
                    try bind(claim.status.rawValue, at: 4, to: statement)
                    try bind(claim.confidence, at: 5, to: statement)
                    try bind(claim.text, at: 6, to: statement)
                    try bind(claim.supersedesClaimID?.uuidString, at: 7, to: statement)
                    try bind(claim.createdAt.timeIntervalSince1970, at: 8, to: statement)
                    try bind(claim.updatedAt.timeIntervalSince1970, at: 9, to: statement)
                    try stepDone(statement)
                }
                for evidence in claim.evidence {
                    try withStatement(
                        "INSERT INTO memory_evidence(id,claim_id,chat_id,message_id,quote,source_hash) "
                            + "VALUES(?1,?2,?3,?4,?5,?6)"
                    ) { statement in
                        try bind(evidence.id.uuidString, at: 1, to: statement)
                        try bind(claim.id.uuidString, at: 2, to: statement)
                        try bind(evidence.chatID.uuidString, at: 3, to: statement)
                        try bind(evidence.messageID.uuidString, at: 4, to: statement)
                        try bind(evidence.quote, at: 5, to: statement)
                        try bind(evidence.sourceHash, at: 6, to: statement)
                        try stepDone(statement)
                    }
                }
                if claim.status == .active {
                    try withStatement("INSERT INTO memory_claims_fts(claim_id,text) VALUES(?1,?2)") { statement in
                        try bind(claim.id.uuidString, at: 1, to: statement)
                        try bind(claim.text, at: 2, to: statement)
                        try stepDone(statement)
                    }
                }
            }
            let retainedIDs = Set(state.claims.map { $0.id.uuidString })
            let storedIDs = try withStatement("SELECT id FROM memory_claims") { statement -> [String] in
                var ids: [String] = []
                while sqlite3_step(statement) == SQLITE_ROW {
                    if let id = columnText(statement, index: 0) { ids.append(id) }
                }
                return ids
            }
            for id in storedIDs where !retainedIDs.contains(id) {
                try withStatement("DELETE FROM memory_claims WHERE id = ?1") { statement in
                    try bind(id, at: 1, to: statement)
                    try stepDone(statement)
                }
            }
            try execute("DELETE FROM memory_claim_embeddings WHERE claim_id IN "
                + "(SELECT id FROM memory_claims WHERE status != 'active');")
            let existing = try loadRecords(prefix: "memory:file:profile/")
            for (key, _) in existing where
                key == "memory:file:profile/MEMORY.md" ||
                key == "memory:file:profile/ALIGNMENT_SYNTHESIS.md" ||
                key.hasPrefix("memory:file:profile/daily/") ||
                key.hasPrefix("memory:file:profile/bank/") ||
                key.hasPrefix("memory:file:profile/dreams/") ||
                key.hasPrefix("memory:file:profile/legacy/") {
                if projections[key] == nil { try deleteRecord(key: key) }
            }
            for (key, data) in projections { try saveRecord(key: key, payload: data) }

            let projectClaims = state.claims.filter { $0.scope.hasPrefix("project:") }
            let groups = Dictionary(grouping: projectClaims, by: { String($0.scope.dropFirst("project:".count)) })
            for (projectKey, claims) in groups {
                let base = "memory:file:projects/\(projectKey)/memory/"
                let indexKey = base + "MEMORY.md"
                var index = ""
                let activeNames = Set(claims.filter { $0.status == .active }.compactMap(\.projectTopicName))
                let inactive = claims.filter { $0.status != .active }
                for claim in inactive {
                    let name = claim.projectTopicName ?? "claim-\(claim.id.uuidString.prefix(12).lowercased())"
                    guard MemoryStore.isValidTopicName(name), !activeNames.contains(name) else { continue }
                    try deleteRecord(key: base + name + ".md")
                }
                let archive = claims.filter { $0.legacySourceText != nil && $0.status != .retracted }
                    .compactMap { claim -> String? in
                        guard let original = claim.legacySourceText else { return nil }
                        return "<!-- claim \(claim.id.uuidString) -->\n\(original)"
                    }.joined(separator: "\n\n")
                if !archive.isEmpty {
                    try saveRecord(key: "memory:file:projects/\(projectKey)/legacy/IMPORTED_TOPICS.md",
                                   payload: Data(archive.utf8))
                } else {
                    try deleteRecord(key: "memory:file:projects/\(projectKey)/legacy/IMPORTED_TOPICS.md")
                }
                for claim in claims where claim.status == .active {
                    let name = claim.projectTopicName ?? "claim-\(claim.id.uuidString.prefix(12).lowercased())"
                    guard MemoryStore.isValidTopicName(name) else { continue }
                    let description = String(claim.text.components(separatedBy: .newlines).first?.prefix(120) ?? "")
                        .replacingOccurrences(of: "\r", with: " ")
                    let document = "---\nname: \(name)\ndescription: \(description)\ntype: project\n---\n\n\(claim.text)\n"
                    try saveRecord(key: base + name + ".md", payload: Data(document.utf8))
                    index = MemoryStore.upsertingIndexLine(
                        index, fileName: name + ".md", title: name, hook: description)
                }
                try saveRecord(key: indexKey, payload: Data(index.utf8))
            }
        }
    }

    func saveMemoryEmbeddings(_ rows: [MemoryClaimEmbedding]) throws {
        try transaction {
            for model in Set(rows.map(\.model)) {
                try withStatement("DELETE FROM memory_claim_embeddings WHERE model = ?1") { statement in
                    try bind(model, at: 1, to: statement)
                    try stepDone(statement)
                }
            }
            for row in rows {
                guard !row.vector.isEmpty else { continue }
                let stillActive = try withStatement(
                    "SELECT status,text FROM memory_claims WHERE id = ?1"
                ) { statement -> Bool in
                    try bind(row.claimID.uuidString, at: 1, to: statement)
                    guard sqlite3_step(statement) == SQLITE_ROW else { return false }
                    let status = columnText(statement, index: 0) ?? ""
                    let text = columnText(statement, index: 1) ?? ""
                    return status == "active" && MemoryLedgerStore.hash(text) == row.sourceHash
                }
                guard stillActive else { continue }
                try withStatement(
                    "INSERT INTO memory_claim_embeddings(claim_id,model,source_hash,dimensions,vector) "
                        + "VALUES(?1,?2,?3,?4,?5)"
                ) { statement in
                    try bind(row.claimID.uuidString, at: 1, to: statement)
                    try bind(row.model, at: 2, to: statement)
                    try bind(row.sourceHash, at: 3, to: statement)
                    try bind(Int64(row.vector.count), at: 4, to: statement)
                    try bind(try JSONEncoder().encode(row.vector), at: 5, to: statement)
                    try stepDone(statement)
                }
            }
        }
    }

    func loadMemoryEmbeddings(model: String) throws -> [MemoryClaimEmbedding] {
        try withStatement(
            "SELECT claim_id,source_hash,dimensions,vector FROM memory_claim_embeddings WHERE model = ?1"
        ) { statement in
            try bind(model, at: 1, to: statement)
            var rows: [MemoryClaimEmbedding] = []
            while sqlite3_step(statement) == SQLITE_ROW {
                guard let idText = columnText(statement, index: 0),
                      let id = UUID(uuidString: idText),
                      let sourceHash = columnText(statement, index: 1),
                      let data = columnData(statement, index: 3),
                      let vector = try? JSONDecoder().decode([Float].self, from: data),
                      vector.count == Int(sqlite3_column_int(statement, 2))
                else { continue }
                rows.append(MemoryClaimEmbedding(
                    claimID: id, model: model, sourceHash: sourceHash, vector: vector))
            }
            return rows
        }
    }

    func clearMemoryEmbeddings() throws {
        try execute("DELETE FROM memory_claim_embeddings;")
    }

    func memoryClaimSearchIDs(query: String, limit: Int) throws -> [UUID] {
        let words = query.lowercased().split { !$0.isLetter && !$0.isNumber }
            .prefix(12).map(String.init)
        guard !words.isEmpty else { return [] }
        let expression = words.map { "\"\($0)\"" }.joined(separator: " OR ")
        return try withStatement(
            "SELECT claim_id FROM memory_claims_fts WHERE memory_claims_fts MATCH ?1 "
                + "ORDER BY rank LIMIT ?2"
        ) { statement in
            try bind(expression, at: 1, to: statement)
            try bind(Int64(max(0, limit)), at: 2, to: statement)
            var ids: [UUID] = []
            while sqlite3_step(statement) == SQLITE_ROW {
                if let raw = columnText(statement, index: 0), let id = UUID(uuidString: raw) {
                    ids.append(id)
                }
            }
            return ids
        }
    }

    func scalarText(_ sql: String) throws -> String? {
        try withStatement(sql) { statement in
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return columnText(statement, index: 0)
        }
    }

    func execute(_ sql: String) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else { throw DatabaseError.closed }
        var message: UnsafeMutablePointer<Int8>?
        let result = sqlite3_exec(handle, sql, nil, nil, &message)
        guard result == SQLITE_OK else {
            let detail = message.map { String(cString: $0) } ?? String(cString: sqlite3_errmsg(handle))
            sqlite3_free(message)
            throw DatabaseError.statement(detail)
        }
    }

    private func withStatement<T>(_ sql: String, _ body: (OpaquePointer) throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else { throw DatabaseError.closed }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK,
              let statement
        else { throw DatabaseError.statement(String(cString: sqlite3_errmsg(handle))) }
        defer { sqlite3_finalize(statement) }
        return try body(statement)
    }

    private func stepDone(_ statement: OpaquePointer) throws {
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw DatabaseError.step(errorMessage)
        }
    }

    private var errorMessage: String {
        guard let handle else { return "database closed" }
        return String(cString: sqlite3_errmsg(handle))
    }

    private func bind(_ value: String?, at index: Int32, to statement: OpaquePointer) throws {
        let result: Int32
        if let value {
            result = sqlite3_bind_text(statement, index, value, -1, Self.transient)
        } else {
            result = sqlite3_bind_null(statement, index)
        }
        guard result == SQLITE_OK else { throw DatabaseError.bind(errorMessage) }
    }

    private func bind(_ value: Data, at index: Int32, to statement: OpaquePointer) throws {
        let result = value.withUnsafeBytes { bytes in
            sqlite3_bind_blob(statement, index, bytes.baseAddress, Int32(value.count), Self.transient)
        }
        guard result == SQLITE_OK else { throw DatabaseError.bind(errorMessage) }
    }

    private func bind(_ value: Data?, at index: Int32, to statement: OpaquePointer) throws {
        guard let value else {
            guard sqlite3_bind_null(statement, index) == SQLITE_OK else {
                throw DatabaseError.bind(errorMessage)
            }
            return
        }
        try bind(value, at: index, to: statement)
    }

    private func bind(_ value: Double, at index: Int32, to statement: OpaquePointer) throws {
        guard sqlite3_bind_double(statement, index, value) == SQLITE_OK else {
            throw DatabaseError.bind(errorMessage)
        }
    }

    private func bind(_ value: Int64, at index: Int32, to statement: OpaquePointer) throws {
        guard sqlite3_bind_int64(statement, index, value) == SQLITE_OK else {
            throw DatabaseError.bind(errorMessage)
        }
    }

    private func bind(_ value: Int64?, at index: Int32, to statement: OpaquePointer) throws {
        guard let value else {
            guard sqlite3_bind_null(statement, index) == SQLITE_OK else {
                throw DatabaseError.bind(errorMessage)
            }
            return
        }
        try bind(value, at: index, to: statement)
    }

    private func columnText(_ statement: OpaquePointer, index: Int32) -> String? {
        guard let text = sqlite3_column_text(statement, index) else { return nil }
        return String(cString: text)
    }

    private func columnData(_ statement: OpaquePointer, index: Int32) -> Data? {
        let count = Int(sqlite3_column_bytes(statement, index))
        guard count > 0, let bytes = sqlite3_column_blob(statement, index) else { return Data() }
        return Data(bytes: bytes, count: count)
    }

    private func optionalData(_ statement: OpaquePointer, index: Int32) -> Data? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else { return nil }
        return columnData(statement, index: index)
    }

    private func optionalInt64(_ statement: OpaquePointer, index: Int32) -> Int64? {
        guard sqlite3_column_type(statement, index) != SQLITE_NULL else { return nil }
        return sqlite3_column_int64(statement, index)
    }

    static func applyFilePermissions(at databaseURL: URL) throws {
        let manager = FileManager.default
        for url in [
            databaseURL,
            URL(fileURLWithPath: databaseURL.path + "-wal"),
            URL(fileURLWithPath: databaseURL.path + "-shm"),
        ] where manager.fileExists(atPath: url.path) {
            try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }
}

extension ProfileDatabase: WorkflowSavedDefinitionStore {
    func save(_ definition: WorkflowSavedDefinition) async throws {
        try requireWorkflowSchemaAvailable()

        let timestamp = definition.updatedAt.timeIntervalSince1970
        guard timestamp.isFinite else {
            throw WorkflowSavedDefinitionStorageError.invalidUpdatedAt
        }
        let declarationData = try JSONEncoder().encode(definition.declarations)
        let declarationJSON = String(decoding: declarationData, as: UTF8.self)

        try transaction {
            try withStatement(
                """
                INSERT INTO saved_workflows(
                    workflow_id, name, scope, script_source, args_declaration_json,
                    description, updated_at
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    name = excluded.name,
                    scope = excluded.scope,
                    script_source = excluded.script_source,
                    args_declaration_json = excluded.args_declaration_json,
                    description = excluded.description,
                    updated_at = excluded.updated_at
                """) { statement in
                try bind(definition.id.uuidString, at: 1, to: statement)
                try bind(definition.name, at: 2, to: statement)
                try bind(definition.scope.rawValue, at: 3, to: statement)
                try bind(definition.source, at: 4, to: statement)
                try bind(declarationJSON, at: 5, to: statement)
                try bind(definition.description, at: 6, to: statement)
                try bind(String(timestamp), at: 7, to: statement)
                try stepDone(statement)
            }
        }
    }

    func definition(id: UUID) async throws -> WorkflowSavedDefinition? {
        try requireWorkflowSchemaAvailable()
        return try withStatement(
            """
            SELECT workflow_id, name, scope, script_source, args_declaration_json,
                   description, updated_at
            FROM saved_workflows
            WHERE workflow_id = ?1
            """) { statement in
            try bind(id.uuidString, at: 1, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return nil }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return try decodeSavedDefinition(statement)
        }
    }

    func listDefinitions() async throws -> [WorkflowSavedDefinition] {
        try requireWorkflowSchemaAvailable()
        var definitions: [WorkflowSavedDefinition] = []
        try withStatement(
            """
            SELECT workflow_id, name, scope, script_source, args_declaration_json,
                   description, updated_at
            FROM saved_workflows
            ORDER BY workflow_id COLLATE BINARY ASC
            """) { statement in
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                definitions.append(try decodeSavedDefinition(statement))
            }
        }
        return definitions
    }

    private func requireWorkflowSchemaAvailable() throws {
        let status = workflowSchemaStatus
        guard status == .available(version: 1) else {
            throw WorkflowSavedDefinitionStorageError.schemaUnavailable(status)
        }
    }

    private func decodeSavedDefinition(_ statement: OpaquePointer) throws -> WorkflowSavedDefinition {
        let storedID = columnText(statement, index: 0) ?? "<null>"
        guard let id = UUID(uuidString: storedID) else {
            throw WorkflowSavedDefinitionStorageError.malformedRow(
                id: storedID,
                reason: "workflow_id is not a UUID")
        }
        guard let name = columnText(statement, index: 1),
              let scopeText = columnText(statement, index: 2),
              let scope = WorkflowDefinitionScope(rawValue: scopeText),
              let source = columnText(statement, index: 3),
              let declarationJSON = columnText(statement, index: 4),
              let description = columnText(statement, index: 5),
              let timestampText = columnText(statement, index: 6),
              let timestamp = Double(timestampText),
              timestamp.isFinite
        else {
            throw WorkflowSavedDefinitionStorageError.malformedRow(
                id: storedID,
                reason: "required field is missing or invalid")
        }

        let declarations: [WorkflowArgumentDeclaration]
        do {
            declarations = try JSONDecoder().decode(
                [WorkflowArgumentDeclaration].self,
                from: Data(declarationJSON.utf8))
        } catch {
            throw WorkflowSavedDefinitionStorageError.malformedRow(
                id: storedID,
                reason: "argument declarations could not be decoded: \(error.localizedDescription)")
        }

        return WorkflowSavedDefinition(
            id: id,
            name: name,
            scope: scope,
            source: source,
            declarations: declarations,
            description: description,
            updatedAt: Date(timeIntervalSince1970: timestamp))
    }
}

extension ProfileDatabase: WorkflowJournalStore {
    func createRun(
        _ descriptor: WorkflowRunDescriptor,
        state: WorkflowRunState,
        at date: Date
    ) async throws {
        try requireWorkflowJournalSchemaAvailable()
        let timestamp = try journalTimestamp(date)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let argsJSON = String(decoding: try encoder.encode(descriptor.args.allValues), as: UTF8.self)

        try withWorkflowTransaction {
            guard try !workflowRunExists(descriptor.id) else {
                throw WorkflowJournalError.runAlreadyExists(descriptor.id)
            }
            try withStatement(
                """
                INSERT INTO workflow_runs(
                    run_id, name, state, script_source, script_hash, facade_version,
                    args_json, parent_run_id, launch_source, canonicalization_version,
                    usage_json, created_at, updated_at
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, NULL, ?8, NULL, ?9, ?9)
                """) { statement in
                try bind(descriptor.id.uuidString, at: 1, to: statement)
                try bind(descriptor.name, at: 2, to: statement)
                try bind(state.rawValue, at: 3, to: statement)
                try bind(descriptor.source, at: 4, to: statement)
                try bind(descriptor.sourceHash, at: 5, to: statement)
                try bind(Int64(descriptor.facadeVersion), at: 6, to: statement)
                try bind(argsJSON, at: 7, to: statement)
                try bind(Int64(WorkflowCanonicalSerialization.currentVersion), at: 8, to: statement)
                try bind(timestamp, at: 9, to: statement)
                try stepDone(statement)
            }
            _ = try insertWorkflowEvent(
                runID: descriptor.id,
                identity: nil,
                payload: .runStarted,
                at: date)
        }
    }

    func append(
        runID: UUID,
        identity: WorkflowRequestIdentity?,
        payload: WorkflowJournalEventPayload,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        _ = try journalTimestamp(date)
        switch payload {
        case .runStarted, .requestResolved, .requestResolvedWithActorTranscript:
            throw WorkflowJournalError.resolutionMustBeAtomic
        case .questionOpened, .questionResolved:
            throw WorkflowJournalError.malformedHistory("Question events must update the question table atomically.")
        case .requestStarted:
            guard identity != nil else { throw WorkflowJournalError.invalidEventIdentity }
        case .phaseEntered:
            if let identity {
                guard identity.site.lane == "main",
                      identity.site.siteIndex == WorkflowJournalSystemSites.phase else {
                    throw WorkflowJournalError.invalidEventIdentity
                }
            }
        default:
            guard identity == nil else { throw WorkflowJournalError.invalidEventIdentity }
        }

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            if case .stateChanged(let state) = payload {
                let timestamp = try journalTimestamp(date)
                try withStatement("UPDATE workflow_runs SET state = ?2, updated_at = ?3 WHERE run_id = ?1") {
                    statement in
                    try bind(runID.uuidString, at: 1, to: statement)
                    try bind(state.rawValue, at: 2, to: statement)
                    try bind(timestamp, at: 3, to: statement)
                    try stepDone(statement)
                }
            }
            let event = try insertWorkflowEvent(runID: runID, identity: identity, payload: payload, at: date)
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func resolveRequest(
        runID: UUID,
        identity: WorkflowRequestIdentity,
        outcome: WorkflowJournalOutcome,
        actorName: String?,
        transcript: WorkflowActorTranscriptSnapshot?,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        _ = try journalTimestamp(date)
        guard (actorName == nil) == (transcript == nil) else {
            throw WorkflowJournalError.malformedHistory("Actor transcript updates require both an actor name and a snapshot.")
        }
        if identity.site.lane == "main" {
            guard actorName == nil else {
                throw WorkflowJournalError.malformedHistory("The entry lane cannot commit an actor transcript.")
            }
        } else {
            guard let actorName, transcript != nil else {
                throw WorkflowJournalError.actorTranscriptRequired(identity.site.lane)
            }
            guard actorName == identity.site.lane else {
                throw WorkflowJournalError.actorTranscriptLaneMismatch(
                    expected: identity.site.lane,
                    actual: actorName)
            }
        }
        if let actorName, actorName.isEmpty {
            throw WorkflowJournalError.malformedHistory("Actor name cannot be empty.")
        }
        if let transcript, transcript.version != WorkflowActorTranscriptSnapshot.currentVersion {
            throw WorkflowJournalError.unsupportedTranscriptVersion(actor: actorName ?? "", version: transcript.version)
        }

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            if let existing = try findResolvedWorkflowEvent(runID: runID, identity: identity) {
                return existing
            }

            let payload: WorkflowJournalEventPayload
            if let actorName, let transcript {
                payload = .requestResolvedWithActorTranscript(
                    outcome: outcome,
                    actorName: actorName,
                    transcript: transcript,
                    transcriptHash: try transcript.sha256())
            } else {
                payload = .requestResolved(outcome)
            }
            let event = try insertWorkflowEvent(
                runID: runID,
                identity: identity,
                payload: payload,
                at: date)
            if let actorName, let transcript {
                try saveActorTranscript(runID: runID, actorName: actorName, snapshot: transcript)
            }
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func recordedOutcome(
        runID: UUID,
        identity: WorkflowRequestIdentity
    ) async throws -> WorkflowJournalEvent? {
        try requireWorkflowJournalSchemaAvailable()
        try requireWorkflowRun(runID)
        return try findResolvedWorkflowEvent(runID: runID, identity: identity)
    }

    func openQuestion(
        runID: UUID,
        questionID: UUID,
        prompt: String,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        let timestamp = try journalTimestamp(date)
        guard !prompt.isEmpty else {
            throw WorkflowJournalError.malformedHistory("Question prompt cannot be empty.")
        }

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            guard try !workflowQuestionExists(runID: runID, questionID: questionID) else {
                throw WorkflowJournalError.questionAlreadyExists(questionID)
            }
            let question = WorkflowJournalQuestion(
                id: questionID,
                prompt: prompt,
                state: .pending,
                answer: nil,
                createdAt: date,
                resolvedAt: nil)
            try withStatement(
                """
                INSERT INTO workflow_questions(
                    run_id, question_id, prompt, answered, answer, created_at, resolved_at
                ) VALUES(?1, ?2, ?3, 0, NULL, ?4, NULL)
                """) { statement in
                try bind(runID.uuidString, at: 1, to: statement)
                try bind(questionID.uuidString, at: 2, to: statement)
                try bind(prompt, at: 3, to: statement)
                try bind(timestamp, at: 4, to: statement)
                try stepDone(statement)
            }
            let event = try insertWorkflowEvent(
                runID: runID,
                identity: nil,
                payload: .questionOpened(question),
                at: date)
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func resolveQuestion(
        runID: UUID,
        questionID: UUID,
        answer: String?,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        let timestamp = try journalTimestamp(date)

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            guard let question = try loadWorkflowQuestion(runID: runID, questionID: questionID),
                  question.state == .pending
            else { throw WorkflowJournalError.questionNotPending(questionID) }
            let resolved = WorkflowJournalQuestion(
                id: question.id,
                prompt: question.prompt,
                state: answer == nil ? .unanswered : .answered,
                answer: answer,
                createdAt: question.createdAt,
                resolvedAt: date)
            try withStatement(
                """
                UPDATE workflow_questions
                SET answered = ?3, answer = ?4, resolved_at = ?5
                WHERE run_id = ?1 AND question_id = ?2
                """) { statement in
                try bind(runID.uuidString, at: 1, to: statement)
                try bind(questionID.uuidString, at: 2, to: statement)
                try bind(Int64(answer == nil ? 0 : 1), at: 3, to: statement)
                try bind(answer, at: 4, to: statement)
                try bind(timestamp, at: 5, to: statement)
                try stepDone(statement)
            }
            let event = try insertWorkflowEvent(
                runID: runID,
                identity: nil,
                payload: .questionResolved(resolved),
                at: date)
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func history(runID: UUID) async throws -> WorkflowJournalHistory {
        try requireWorkflowJournalSchemaAvailable()
        return try withWorkflowReadTransaction {
            let run = try loadWorkflowRunHeader(runID)
            return WorkflowJournalHistory(
                runID: runID,
                state: run.state,
                canonicalizationVersion: run.canonicalizationVersion,
                source: run.source,
                sourceHash: run.sourceHash,
                facadeVersion: run.facadeVersion,
                args: run.args,
                events: try loadWorkflowEvents(runID),
                questions: try loadWorkflowQuestions(runID),
                actorTranscripts: try loadActorTranscripts(runID))
        }
    }

    private func requireWorkflowJournalSchemaAvailable() throws {
        let status = workflowSchemaStatus
        guard status == .available(version: 1) else {
            throw WorkflowJournalError.schemaUnavailable(String(describing: status))
        }
    }

    private func journalTimestamp(_ date: Date) throws -> String {
        let value = date.timeIntervalSince1970
        guard value.isFinite else { throw WorkflowJournalError.invalidTimestamp }
        return String(value)
    }

    private func withWorkflowTransaction<T>(_ body: () throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        try execute("BEGIN IMMEDIATE;")
        do {
            let value = try body()
            try execute("COMMIT;")
            return value
        } catch {
            try? execute("ROLLBACK;")
            throw error
        }
    }

    private func withWorkflowReadTransaction<T>(_ body: () throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        try execute("BEGIN;")
        do {
            let value = try body()
            try execute("COMMIT;")
            return value
        } catch {
            try? execute("ROLLBACK;")
            throw error
        }
    }

    private func workflowRunExists(_ runID: UUID) throws -> Bool {
        try withStatement("SELECT 1 FROM workflow_runs WHERE run_id = ?1") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return false }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return true
        }
    }

    private func requireWorkflowRun(_ runID: UUID) throws {
        guard try workflowRunExists(runID) else { throw WorkflowJournalError.runNotFound(runID) }
    }

    private func workflowQuestionExists(runID: UUID, questionID: UUID) throws -> Bool {
        try withStatement(
            "SELECT 1 FROM workflow_questions WHERE run_id = ?1 AND question_id = ?2") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(questionID.uuidString, at: 2, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return false }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return true
        }
    }

    private func nextWorkflowSequence(runID: UUID) throws -> Int64 {
        try withStatement("SELECT MAX(seq) FROM workflow_events WHERE run_id = ?1") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            guard sqlite3_column_type(statement, 0) != SQLITE_NULL else { return 1 }
            let last = sqlite3_column_int64(statement, 0)
            guard last >= 0, last < Int64.max else {
                throw WorkflowJournalError.sequenceExhausted(runID)
            }
            return last + 1
        }
    }

    private func insertWorkflowEvent(
        runID: UUID,
        identity: WorkflowRequestIdentity?,
        payload: WorkflowJournalEventPayload,
        at date: Date
    ) throws -> WorkflowJournalEvent {
        if let identity, !identity.site.isValidJournalKey {
            throw WorkflowJournalError.invalidEventIdentity
        }
        let sequence = try nextWorkflowSequence(runID: runID)
        let timestamp = try journalTimestamp(date)
        let payloadData = try Self.workflowJournalEncoder().encode(payload)
        let payloadJSON = String(decoding: payloadData, as: UTF8.self)
        try withStatement(
            """
            INSERT INTO workflow_events(
                run_id, seq, kind, lane, site_index, site_ordinal, input_hash,
                payload_json, created_at
            ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(sequence, at: 2, to: statement)
            try bind(payload.kind.rawValue, at: 3, to: statement)
            try bind(identity?.site.lane, at: 4, to: statement)
            try bind(identity.map { Int64($0.site.siteIndex) }, at: 5, to: statement)
            try bind(identity.map { Int64($0.site.ordinal) }, at: 6, to: statement)
            try bind(identity?.inputHash, at: 7, to: statement)
            try bind(payloadJSON, at: 8, to: statement)
            try bind(timestamp, at: 9, to: statement)
            try stepDone(statement)
        }
        return WorkflowJournalEvent(
            runID: runID,
            sequence: sequence,
            identity: identity,
            payload: payload,
            createdAt: date)
    }

    private func findResolvedWorkflowEvent(
        runID: UUID,
        identity: WorkflowRequestIdentity
    ) throws -> WorkflowJournalEvent? {
        try withStatement(
            """
            SELECT run_id, seq, kind, lane, site_index, site_ordinal, input_hash,
                   payload_json, created_at
            FROM workflow_events
            WHERE run_id = ?1 AND kind = ?2 AND lane = ?3 AND site_index = ?4
              AND site_ordinal = ?5 AND input_hash = ?6
            ORDER BY seq ASC LIMIT 1
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(WorkflowJournalEventKind.requestResolved.rawValue, at: 2, to: statement)
            try bind(identity.site.lane, at: 3, to: statement)
            try bind(Int64(identity.site.siteIndex), at: 4, to: statement)
            try bind(Int64(identity.site.ordinal), at: 5, to: statement)
            try bind(identity.inputHash, at: 6, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return nil }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return try decodeWorkflowEvent(statement)
        }
    }

    private func saveActorTranscript(
        runID: UUID,
        actorName: String,
        snapshot: WorkflowActorTranscriptSnapshot
    ) throws {
        let snapshotJSON = try snapshot.encodedJSON()
        let digest = try snapshot.sha256()
        try withStatement(
            """
            INSERT INTO workflow_actors(run_id, actor_name, usage_json, transcript_json, transcript_sha256)
            VALUES(?1, ?2, NULL, ?3, ?4)
            ON CONFLICT(run_id, actor_name) DO UPDATE SET
                transcript_json = excluded.transcript_json,
                transcript_sha256 = excluded.transcript_sha256
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(actorName, at: 2, to: statement)
            try bind(snapshotJSON, at: 3, to: statement)
            try bind(digest, at: 4, to: statement)
            try stepDone(statement)
        }
    }

    private func touchWorkflowRun(_ runID: UUID, at date: Date) throws {
        let timestamp = try journalTimestamp(date)
        try withStatement("UPDATE workflow_runs SET updated_at = ?2 WHERE run_id = ?1") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(timestamp, at: 2, to: statement)
            try stepDone(statement)
        }
    }

    private struct WorkflowRunHeader {
        var state: String
        var canonicalizationVersion: Int64
        var source: String
        var sourceHash: String
        var facadeVersion: Int
        var args: [String: String]
    }

    private func loadWorkflowRunHeader(_ runID: UUID) throws -> WorkflowRunHeader {
        try withStatement(
            """
            SELECT state, canonicalization_version, script_source, script_hash,
                   facade_version, args_json
            FROM workflow_runs WHERE run_id = ?1
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { throw WorkflowJournalError.runNotFound(runID) }
            guard result == SQLITE_ROW,
                  let state = columnText(statement, index: 0),
                  sqlite3_column_type(statement, 1) != SQLITE_NULL,
                  let source = columnText(statement, index: 2),
                  let sourceHash = columnText(statement, index: 3),
                  sqlite3_column_type(statement, 4) != SQLITE_NULL,
                  let argsJSON = columnText(statement, index: 5)
            else {
                throw WorkflowJournalError.malformedHistory("Run header is missing required fields.")
            }
            let args: [String: String]
            do {
                args = try JSONDecoder().decode([String: String].self, from: Data(argsJSON.utf8))
            } catch {
                throw WorkflowJournalError.malformedHistory("Run arguments could not be decoded.")
            }
            let facadeVersion = sqlite3_column_int64(statement, 4)
            guard facadeVersion >= 0, facadeVersion <= Int64(Int.max) else {
                throw WorkflowJournalError.malformedHistory("Facade version is outside the supported integer range.")
            }
            return WorkflowRunHeader(
                state: state,
                canonicalizationVersion: sqlite3_column_int64(statement, 1),
                source: source,
                sourceHash: sourceHash,
                facadeVersion: Int(facadeVersion),
                args: args)
        }
    }

    private func loadWorkflowEvents(_ runID: UUID) throws -> [WorkflowJournalEvent] {
        var events: [WorkflowJournalEvent] = []
        try withStatement(
            """
            SELECT run_id, seq, kind, lane, site_index, site_ordinal, input_hash,
                   payload_json, created_at
            FROM workflow_events WHERE run_id = ?1 ORDER BY seq ASC
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                events.append(try decodeWorkflowEvent(statement))
            }
        }
        return events
    }

    private func decodeWorkflowEvent(_ statement: OpaquePointer) throws -> WorkflowJournalEvent {
        let storedRunID = columnText(statement, index: 0) ?? "<null>"
        guard let runID = UUID(uuidString: storedRunID),
              sqlite3_column_type(statement, 1) != SQLITE_NULL,
              let kindText = columnText(statement, index: 2),
              let kind = WorkflowJournalEventKind(rawValue: kindText),
              let payloadJSON = columnText(statement, index: 7),
              let timestampText = columnText(statement, index: 8),
              let timestamp = Double(timestampText),
              timestamp.isFinite
        else {
            throw WorkflowJournalError.malformedHistory("Event row has a missing or invalid field.")
        }
        let sequence = sqlite3_column_int64(statement, 1)
        guard sequence > 0 else {
            throw WorkflowJournalError.malformedHistory("Event sequence must be positive.")
        }
        let payload: WorkflowJournalEventPayload
        do {
            payload = try JSONDecoder().decode(
                WorkflowJournalEventPayload.self,
                from: Data(payloadJSON.utf8))
        } catch {
            throw WorkflowJournalError.malformedHistory("Event payload could not be decoded.")
        }
        guard payload.kind == kind else {
            throw WorkflowJournalError.malformedHistory("Event kind does not match its payload.")
        }

        let lane = columnText(statement, index: 3)
        let siteIndex = optionalInt64(statement, index: 4)
        let ordinal = optionalInt64(statement, index: 5)
        let inputHash = columnText(statement, index: 6)
        let identity: WorkflowRequestIdentity?
        if lane == nil, siteIndex == nil, ordinal == nil, inputHash == nil {
            identity = nil
        } else {
            guard let lane, !lane.isEmpty,
                  let siteIndex, let decodedSiteIndex = Int(exactly: siteIndex),
                  let ordinal, ordinal >= 0, ordinal <= Int64(Int.max),
                  let inputHash
            else {
                throw WorkflowJournalError.malformedHistory("Event request identity is incomplete.")
            }
            let site = WorkflowSiteKey(lane: lane, siteIndex: decodedSiteIndex, ordinal: Int(ordinal))
            guard site.isValidJournalKey else { throw WorkflowJournalError.invalidEventIdentity }
            identity = WorkflowRequestIdentity(
                site: site,
                inputHash: inputHash)
        }
        switch payload {
        case .requestStarted, .requestResolved, .requestResolvedWithActorTranscript:
            guard identity != nil else { throw WorkflowJournalError.invalidEventIdentity }
            if case .requestResolvedWithActorTranscript(_, let actorName, _, _) = payload {
                guard let identity, identity.site.lane == actorName, actorName != "main" else {
                    throw WorkflowJournalError.malformedHistory(
                        "Resolved actor transcript does not match its request lane.")
                }
            }
        case .phaseEntered:
            if let identity {
                guard identity.site.lane == "main",
                      identity.site.siteIndex == WorkflowJournalSystemSites.phase else {
                    throw WorkflowJournalError.invalidEventIdentity
                }
            }
        default:
            guard identity == nil else { throw WorkflowJournalError.invalidEventIdentity }
        }
        return WorkflowJournalEvent(
            runID: runID,
            sequence: sequence,
            identity: identity,
            payload: payload,
            createdAt: Date(timeIntervalSince1970: timestamp))
    }

    private func loadWorkflowQuestions(_ runID: UUID) throws -> [WorkflowJournalQuestion] {
        var questions: [WorkflowJournalQuestion] = []
        try withStatement(
            """
            SELECT question_id, prompt, answered, answer, created_at, resolved_at
            FROM workflow_questions WHERE run_id = ?1
            ORDER BY CAST(created_at AS REAL) ASC, question_id COLLATE BINARY ASC
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                questions.append(try decodeWorkflowQuestion(statement))
            }
        }
        return questions
    }

    private func loadWorkflowQuestion(runID: UUID, questionID: UUID) throws -> WorkflowJournalQuestion? {
        try withStatement(
            """
            SELECT question_id, prompt, answered, answer, created_at, resolved_at
            FROM workflow_questions WHERE run_id = ?1 AND question_id = ?2
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(questionID.uuidString, at: 2, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return nil }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return try decodeWorkflowQuestion(statement)
        }
    }

    private func decodeWorkflowQuestion(_ statement: OpaquePointer) throws -> WorkflowJournalQuestion {
        let storedID = columnText(statement, index: 0) ?? "<null>"
        guard let id = UUID(uuidString: storedID),
              let prompt = columnText(statement, index: 1),
              sqlite3_column_type(statement, 2) != SQLITE_NULL,
              let createdAtText = columnText(statement, index: 4),
              let createdAtValue = Double(createdAtText),
              createdAtValue.isFinite
        else {
            throw WorkflowJournalError.malformedHistory("Question row has a missing or invalid field.")
        }
        let answer = columnText(statement, index: 3)
        let resolvedAtText = columnText(statement, index: 5)
        let resolvedAt: Date?
        if let resolvedAtText {
            guard let value = Double(resolvedAtText), value.isFinite else {
                throw WorkflowJournalError.malformedHistory("Question resolution time is invalid.")
            }
            resolvedAt = Date(timeIntervalSince1970: value)
        } else {
            resolvedAt = nil
        }
        let wasAnswered = sqlite3_column_int64(statement, 2) != 0
        let state: WorkflowJournalQuestionState = resolvedAt == nil
            ? .pending
            : (wasAnswered ? .answered : .unanswered)
        return WorkflowJournalQuestion(
            id: id,
            prompt: prompt,
            state: state,
            answer: answer,
            createdAt: Date(timeIntervalSince1970: createdAtValue),
            resolvedAt: resolvedAt)
    }

    private func loadActorTranscripts(_ runID: UUID) throws -> [WorkflowJournalActorTranscript] {
        var transcripts: [WorkflowJournalActorTranscript] = []
        try withStatement(
            """
            SELECT actor_name, transcript_json, transcript_sha256
            FROM workflow_actors WHERE run_id = ?1
            ORDER BY actor_name COLLATE BINARY ASC
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                let actorName = columnText(statement, index: 0) ?? "<null>"
                let encodedSnapshot = columnText(statement, index: 1)
                let digest = columnText(statement, index: 2)
                if encodedSnapshot == nil, digest == nil { continue }
                guard let encodedSnapshot, let digest else {
                    throw WorkflowJournalError.malformedHistory(
                        "Actor '\(actorName)' has an incomplete transcript row.")
                }
                let snapshot = try? JSONDecoder().decode(
                    WorkflowActorTranscriptSnapshot.self,
                    from: Data(encodedSnapshot.utf8))
                transcripts.append(WorkflowJournalActorTranscript(
                    actorName: actorName,
                    sha256: digest,
                    encodedSnapshot: encodedSnapshot,
                    snapshot: snapshot))
            }
        }
        return transcripts
    }

    private static func workflowJournalEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}


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
        guard key.hasPrefix("audio:item:") else { throw CocoaError(.coderInvalidValue) }
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
