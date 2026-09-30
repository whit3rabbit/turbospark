import Foundation
import SQLCipher
import XCTest

@testable import TurboSparkApp

final class ProfileDatabaseChatProjectionMigrationTests: XCTestCase {
    private var root: URL!
    private let key = Data(repeating: 0x41, count: 32)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ChatProjectionMigrationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testLegacyChatRowsBackfillAndRepeatedMigrationPreservesData() throws {
        let databaseURL = root.appendingPathComponent("legacy.sqlite3")
        let customID = "C4DBA8A3-FE70-4A97-89A9-574FA2036B12"
        let defaultID = "1D96CC24-57EF-44B4-95CA-01C8669691AE"
        let sqlTitleID = "30934207-4B67-41E8-84AA-BB8CE0D6634F"
        try createLegacyDatabase(at: databaseURL, chats: [
            (customID, "Imported title", Data(#"{"id":"C4DBA8A3-FE70-4A97-89A9-574FA2036B12","title":"Imported title","messages":[{"role":"user","content":"keep this row"}]}"#.utf8)),
            (defaultID, "New Chat", Data(#"{"id":"1D96CC24-57EF-44B4-95CA-01C8669691AE","title":"New Chat","messages":[]}"#.utf8)),
            (sqlTitleID, "SQL-only legacy title", Data(#"{"id":"30934207-4B67-41E8-84AA-BB8CE0D6634F","messages":[]}"#.utf8)),
        ])

        let firstOpen = try ProfileDatabase(url: databaseURL, key: key)
        let firstArchive = try XCTUnwrap(firstOpen.loadChatArchive())
        firstOpen.close()

        let customProjection = try readProjection(at: databaseURL, chatID: customID)
        XCTAssertEqual(customProjection.titleProvenance, "user")
        XCTAssertEqual(customProjection.titleGenerationAttempted, 1)
        XCTAssertNil(customProjection.recoveryAnchor)
        XCTAssertTrue(try hasMigrationVersionTwo(at: databaseURL))

        let secondOpen = try ProfileDatabase(url: databaseURL, key: key)
        let secondArchive = try XCTUnwrap(secondOpen.loadChatArchive())
        secondOpen.close()

        let custom = try XCTUnwrap(secondArchive.chats.first { $0.id.uuidString == customID })
        let defaultTitle = try XCTUnwrap(secondArchive.chats.first { $0.id.uuidString == defaultID })
        let sqlTitle = try XCTUnwrap(secondArchive.chats.first { $0.id.uuidString == sqlTitleID })
        XCTAssertEqual(custom.titleProvenance, .user)
        XCTAssertTrue(custom.titleGenerationAttempted)
        XCTAssertEqual(custom.messages.map(\.content), ["keep this row"])
        XCTAssertEqual(defaultTitle.titleProvenance, .unclaimed)
        XCTAssertFalse(defaultTitle.titleGenerationAttempted)
        XCTAssertEqual(sqlTitle.title, "SQL-only legacy title")
        XCTAssertEqual(sqlTitle.titleProvenance, .user)
        XCTAssertTrue(sqlTitle.titleGenerationAttempted)
        XCTAssertEqual(firstArchive.chats.count, 3)
    }

    func testPartiallyAddedProjectionSchemaCompletesMigration() throws {
        let databaseURL = root.appendingPathComponent("partial.sqlite3")
        let chatID = "C4DBA8A3-FE70-4A97-89A9-574FA2036B12"
        try createLegacyDatabase(
            at: databaseURL,
            chats: [(chatID, "Partially migrated", Data(#"{"id":"C4DBA8A3-FE70-4A97-89A9-574FA2036B12","title":"Partially migrated","messages":[]}"#.utf8))],
            existingProjectionColumns: ["title_provenance"])

        let database = try ProfileDatabase(url: databaseURL, key: key)
        let archive = try XCTUnwrap(database.loadChatArchive())
        database.close()

        let projection = try readProjection(at: databaseURL, chatID: chatID)
        XCTAssertEqual(projection.titleProvenance, "user")
        XCTAssertEqual(projection.titleGenerationAttempted, 1)
        XCTAssertEqual(try XCTUnwrap(archive.chats.first).titleProvenance, .user)
    }

    func testPayloadOnlyLegacyWriteRetainsTitleAndRecoveryProjections() throws {
        let databaseURL = root.appendingPathComponent("round-trip.sqlite3")
        let database = try ProfileDatabase(url: databaseURL, key: key)
        let interruptedID = UUID(uuidString: "7C87D175-9CB6-4D39-B35A-3279C2609A2F")!
        let anchor = try XCTUnwrap(RecoveryAnchor(
            retainedMessageCount: 0,
            interruptedMessageID: interruptedID,
            interruptedContentHash: "sha256:partial-row",
            continuationsUsed: 2,
            retriesUsed: 1,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        var chat = AppChat(
            title: "Generated title",
            messages: [AppChatMessage(
                id: interruptedID,
                role: .assistant,
                content: "partial response")],
            titleProvenance: .generated,
            titleGenerationAttempted: true,
            recoveryAnchor: anchor)
        chat.id = UUID(uuidString: "C4DBA8A3-FE70-4A97-89A9-574FA2036B12")!
        _ = try database.saveChatArchive(AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        database.close()

        let legacyPayload = Data(
            #"{"id":"C4DBA8A3-FE70-4A97-89A9-574FA2036B12","title":"Changed by older app","messages":[{"id":"7C87D175-9CB6-4D39-B35A-3279C2609A2F","role":"assistant","content":"partial response"}]}"#.utf8)
        try overwriteLegacyPayload(
            at: databaseURL,
            chatID: chat.id.uuidString,
            title: "Changed by older app",
            payload: legacyPayload)

        let reopened = try ProfileDatabase(url: databaseURL, key: key)
        let archive = try XCTUnwrap(reopened.loadChatArchive())
        reopened.close()

        let restored = try XCTUnwrap(archive.chats.first)
        XCTAssertEqual(restored.title, "Changed by older app")
        XCTAssertEqual(restored.titleProvenance, .generated)
        XCTAssertTrue(restored.titleGenerationAttempted)
        XCTAssertEqual(restored.recoveryAnchor, anchor)

        let projection = try readProjection(at: databaseURL, chatID: chat.id.uuidString)
        XCTAssertEqual(projection.titleProvenance, "generated")
        XCTAssertEqual(projection.titleGenerationAttempted, 1)
        XCTAssertEqual(try JSONDecoder().decode(RecoveryAnchor.self, from: XCTUnwrap(projection.recoveryAnchor)), anchor)
    }

    private func createLegacyDatabase(
        at url: URL,
        chats: [(String, String, Data)],
        existingProjectionColumns: Set<String> = []
    ) throws {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        try execute(
            connection,
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL); INSERT INTO schema_migrations(version, applied_at) VALUES(1, 0); CREATE TABLE chats(id TEXT PRIMARY KEY, project_id TEXT, title TEXT NOT NULL, updated_at REAL NOT NULL, payload BLOB NOT NULL\(existingProjectionColumns.contains("title_provenance") ? ", title_provenance TEXT" : ""));")
        for (id, title, payload) in chats {
            var statement: OpaquePointer?
            guard sqlite3_prepare_v2(
                connection,
                "INSERT INTO chats(id, project_id, title, updated_at, payload) VALUES(?1, NULL, ?2, 0, ?3)",
                -1,
                &statement,
                nil) == SQLITE_OK,
                let statement
            else { throw sqliteError(connection) }
            defer { sqlite3_finalize(statement) }
            try bind(id, at: 1, statement: statement, connection: connection)
            try bind(title, at: 2, statement: statement, connection: connection)
            try bind(payload, at: 3, statement: statement, connection: connection)
            guard sqlite3_step(statement) == SQLITE_DONE else { throw sqliteError(connection) }
        }
    }

    private func overwriteLegacyPayload(at url: URL, chatID: String, title: String, payload: Data) throws {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "UPDATE chats SET title = ?2, payload = ?3 WHERE id = ?1",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        try bind(chatID, at: 1, statement: statement, connection: connection)
        try bind(title, at: 2, statement: statement, connection: connection)
        try bind(payload, at: 3, statement: statement, connection: connection)
        guard sqlite3_step(statement) == SQLITE_DONE else { throw sqliteError(connection) }
    }

    private func readProjection(at url: URL, chatID: String) throws -> (
        titleProvenance: String?,
        titleGenerationAttempted: Int64?,
        recoveryAnchor: Data?
    ) {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "SELECT title_provenance, title_generation_attempted, recovery_anchor FROM chats WHERE id = ?1",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        try bind(chatID, at: 1, statement: statement, connection: connection)
        guard sqlite3_step(statement) == SQLITE_ROW else { throw sqliteError(connection) }
        let provenance = sqlite3_column_text(statement, 0).map { String(cString: $0) }
        let attempted = sqlite3_column_type(statement, 1) == SQLITE_NULL
            ? nil
            : sqlite3_column_int64(statement, 1)
        let anchor: Data?
        if sqlite3_column_type(statement, 2) == SQLITE_NULL {
            anchor = nil
        } else {
            let count = Int(sqlite3_column_bytes(statement, 2))
            if let bytes = sqlite3_column_blob(statement, 2), count > 0 {
                anchor = Data(bytes: bytes, count: count)
            } else {
                anchor = Data()
            }
        }
        return (provenance, attempted, anchor)
    }

    private func hasMigrationVersionTwo(at url: URL) throws -> Bool {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "SELECT 1 FROM schema_migrations WHERE version = 2",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        return sqlite3_step(statement) == SQLITE_ROW
    }

    private func openDatabase(at url: URL) throws -> OpaquePointer {
        var connection: OpaquePointer?
        guard sqlite3_open_v2(
            url.path,
            &connection,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
            nil) == SQLITE_OK,
            let connection
        else { throw connection.map(sqliteError) ?? ProfileDatabase.DatabaseError.open("Could not open test database") }
        let result = key.withUnsafeBytes { bytes in
            sqlite3_key(connection, bytes.baseAddress, Int32(key.count))
        }
        guard result == SQLITE_OK else {
            sqlite3_close(connection)
            throw sqliteError(connection)
        }
        return connection
    }

    private func execute(_ connection: OpaquePointer, _ sql: String) throws {
        var message: UnsafeMutablePointer<Int8>?
        guard sqlite3_exec(connection, sql, nil, nil, &message) == SQLITE_OK else {
            let detail = message.map { String(cString: $0) } ?? String(cString: sqlite3_errmsg(connection))
            sqlite3_free(message)
            throw ProfileDatabase.DatabaseError.statement(detail)
        }
    }

    private func bind(
        _ value: String,
        at index: Int32,
        statement: OpaquePointer,
        connection: OpaquePointer
    ) throws {
        guard sqlite3_bind_text(statement, index, value, -1, Self.transient) == SQLITE_OK else {
            throw sqliteError(connection)
        }
    }

    private func bind(
        _ value: Data,
        at index: Int32,
        statement: OpaquePointer,
        connection: OpaquePointer
    ) throws {
        let result = value.withUnsafeBytes { bytes in
            sqlite3_bind_blob(statement, index, bytes.baseAddress, Int32(value.count), Self.transient)
        }
        guard result == SQLITE_OK else { throw sqliteError(connection) }
    }

    private func sqliteError(_ connection: OpaquePointer) -> Error {
        ProfileDatabase.DatabaseError.statement(String(cString: sqlite3_errmsg(connection)))
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
}
