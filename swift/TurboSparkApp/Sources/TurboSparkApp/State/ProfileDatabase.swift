import Foundation
import SQLCipher

final class ProfileDatabase: @unchecked Sendable {
    struct AssetMetadata: Equatable, Sendable {
        var id: String
        var fileName: String
        var mimeType: String?
        var byteCount: Int64
        var referenceCount: Int
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

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    let url: URL
    private let lock = NSRecursiveLock()
    private var handle: OpaquePointer?

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
        } else if key.contains("settings") {
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
        let decoder = JSONDecoder()
        try withStatement("SELECT payload FROM chats ORDER BY updated_at DESC, id ASC") { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                guard let payload = columnData(statement, index: 0) else { continue }
                if let chat = try? decoder.decode(AppChat.self, from: payload) {
                    chats.append(chat)
                }
            }
        }
        guard selected != nil || !chats.isEmpty else { return nil }
        let selectedID = selected.flatMap(UUID.init(uuidString:)) ?? chats.first?.id ?? UUID()
        return AppChatArchive(selectedChatID: selectedID, chats: chats)
    }

    func saveChatArchive(_ archive: AppChatArchive) throws {
        let encoder = JSONEncoder()
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
            for removed in existing.subtracting(current) {
                try deleteChat(id: removed)
            }
            try saveMetadata(key: "selected-chat-id", value: archive.selectedChatID.uuidString)
        }
        try Self.applyFilePermissions(at: url)
    }

    func assetMetadata(id: String) throws -> AssetMetadata? {
        try withStatement(
            "SELECT file_name, mime_type, byte_count, reference_count FROM assets WHERE id = ?1"
        ) { statement in
            try bind(id, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return AssetMetadata(
                id: id,
                fileName: columnText(statement, index: 0) ?? "asset",
                mimeType: columnText(statement, index: 1),
                byteCount: sqlite3_column_int64(statement, 2),
                referenceCount: Int(sqlite3_column_int64(statement, 3)))
        }
    }

    func allAssetMetadata() throws -> [AssetMetadata] {
        var result: [AssetMetadata] = []
        try withStatement(
            "SELECT id, file_name, mime_type, byte_count, reference_count FROM assets ORDER BY id"
        ) { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                guard let id = columnText(statement, index: 0) else { continue }
                result.append(AssetMetadata(
                    id: id,
                    fileName: columnText(statement, index: 1) ?? "asset",
                    mimeType: columnText(statement, index: 2),
                    byteCount: sqlite3_column_int64(statement, 3),
                    referenceCount: Int(sqlite3_column_int64(statement, 4))))
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
                reference_count=assets.reference_count + 1
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

    func integrityCheck() throws -> Bool {
        try scalarText("PRAGMA integrity_check;") == "ok"
    }

    func checkpoint() throws {
        try execute("PRAGMA wal_checkpoint(FULL);")
        try Self.applyFilePermissions(at: url)
    }

    /// Produces a transactionally consistent encrypted snapshot using
    /// SQLite's online backup API. Copying the database file itself would
    /// race the WAL and can omit the newest committed rows.
    func backup(to destination: URL, key: Data) throws {
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
        guard sqlite3_exec(target, "PRAGMA integrity_check;", nil, nil, nil) == SQLITE_OK else {
            throw DatabaseError.open(String(cString: sqlite3_errmsg(target)))
        }
        try Self.applyFilePermissions(at: destination)
    }

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
            CREATE TABLE IF NOT EXISTS messages(
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                role TEXT NOT NULL,
                created_at REAL NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_chat_ordinal ON messages(chat_id, ordinal);
            CREATE TABLE IF NOT EXISTS alternates(
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                message_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
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
            CREATE TABLE IF NOT EXISTS attachments(
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                asset_id TEXT,
                file_name TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts(
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                asset_id TEXT,
                title TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory(
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                title TEXT,
                body TEXT NOT NULL,
                updated_at REAL NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB
            );
            CREATE TABLE IF NOT EXISTS embeddings(
                id TEXT PRIMARY KEY,
                memory_id TEXT REFERENCES memory(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                vector BLOB NOT NULL,
                payload_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS todos(
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_version INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
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
    }

    private func chatIDs() throws -> Set<String> {
        var result: Set<String> = []
        try withStatement("SELECT id FROM chats") { statement in
            while sqlite3_step(statement) == SQLITE_ROW {
                if let value = columnText(statement, index: 0) { result.insert(value) }
            }
        }
        return result
    }

    private func chatPayload(id: String) throws -> Data? {
        try withStatement("SELECT payload FROM chats WHERE id = ?1") { statement in
            try bind(id, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return columnData(statement, index: 0)
        }
    }

    private func upsertChat(_ chat: AppChat, payload: Data) throws {
        try withStatement(
            """
            INSERT INTO chats(id, project_id, title, updated_at, payload)
            VALUES(?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(id) DO UPDATE SET
                project_id=excluded.project_id,
                title=excluded.title,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """) { statement in
            try bind(chat.id.uuidString, at: 1, to: statement)
            try bind(chat.projectID?.uuidString, at: 2, to: statement)
            try bind(chat.title, at: 3, to: statement)
            try bind(chat.updatedAt.timeIntervalSince1970, at: 4, to: statement)
            try bind(payload, at: 5, to: statement)
            try stepDone(statement)
        }
    }

    private func updateSearchIndex(for chat: AppChat) throws {
        try withStatement("DELETE FROM chat_fts WHERE chat_id = ?1") { statement in
            try bind(chat.id.uuidString, at: 1, to: statement)
            try stepDone(statement)
        }
        let messageText = chat.messages.flatMap { message in
            [message.content, message.reasoning]
        }
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

    private func transaction(_ body: () throws -> Void) throws {
        try execute("BEGIN IMMEDIATE;")
        do {
            try body()
            try execute("COMMIT;")
        } catch {
            try? execute("ROLLBACK;")
            throw error
        }
    }

    private func scalarText(_ sql: String) throws -> String? {
        try withStatement(sql) { statement in
            guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
            return columnText(statement, index: 0)
        }
    }

    private func execute(_ sql: String) throws {
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

    private func columnText(_ statement: OpaquePointer, index: Int32) -> String? {
        guard let text = sqlite3_column_text(statement, index) else { return nil }
        return String(cString: text)
    }

    private func columnData(_ statement: OpaquePointer, index: Int32) -> Data? {
        let count = Int(sqlite3_column_bytes(statement, index))
        guard count > 0, let bytes = sqlite3_column_blob(statement, index) else { return Data() }
        return Data(bytes: bytes, count: count)
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
