import Foundation
import SQLCipher
import XCTest

@testable import TurboSparkApp

/// Child projection tables (messages, todos, ...) are keyed per chat, so a
/// branch, a duplicate, or a model that numbers todos "1", "2" in two chats
/// must not make a save fail.
@MainActor
final class ProfileDatabaseChildKeyTests: XCTestCase {
    private var root: URL!
    private let key = Data(repeating: 0x42, count: 32)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ChildKeyTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func chatWithSharedTodo(_ title: String) -> AppChat {
        var chat = AppChat(title: title)
        chat.todos = [TodoItem(id: "1", content: "step in \(title)", status: "pending")]
        chat.messages = [AppChatMessage(role: .user, content: "hello \(title)")]
        return chat
    }

    func testTwoChatsSharingATodoIdAndABranchAllSaveAndReload() throws {
        let database = try ProfileDatabase(url: root.appendingPathComponent("p.sqlite3"), key: key)
        defer { database.close() }
        let first = chatWithSharedTodo("first")
        let second = chatWithSharedTodo("second")
        // A branch built by the app keeps the source prefix; it gets fresh ids.
        let messages = [
            AppChatMessage(role: .user, content: "q0"),
            AppChatMessage(role: .assistant, content: "a0"),
            AppChatMessage(role: .user, content: "q1"),
        ]
        var source = AppChat(title: "source")
        source.messages = messages
        let branch = AppModel.branchedChat(
            from: source, messageIndex: 2, messages: messages, newText: "q1 edited",
            summary: nil, boundary: 0, now: Date())
        XCTAssertEqual(Set(branch.messages.map(\.id)).intersection(messages.map(\.id)), [])
        // Even when ids DO collide across chats (legacy copies), saving works.
        var legacyCopy = source
        legacyCopy.id = UUID()

        let chats = [first, second, source, branch, legacyCopy]
        _ = try database.saveChatArchive(AppChatArchive(selectedChatID: first.id, chats: chats))

        let loaded = try XCTUnwrap(database.loadChatArchive())
        XCTAssertEqual(Set(loaded.chats.map(\.id)), Set(chats.map(\.id)))
        for title in ["first", "second"] {
            let chat = try XCTUnwrap(loaded.chats.first { $0.title == title })
            XCTAssertEqual(chat.todos.map(\.id), ["1"])
            XCTAssertEqual(chat.todos.map(\.content), ["step in \(title)"])
        }
        // A second save (the "every later save fails" symptom) still succeeds.
        _ = try database.saveChatArchive(AppChatArchive(selectedChatID: first.id, chats: chats))
        XCTAssertEqual(try rowCount(table: "todos"), 2)
    }

    func testDuplicatedChatKeepingTodoIdsSaves() throws {
        let database = try ProfileDatabase(url: root.appendingPathComponent("p.sqlite3"), key: key)
        defer { database.close() }
        let original = chatWithSharedTodo("orig")
        let copy = original.duplicated()
        XCTAssertEqual(copy.todos.map(\.id), ["1"])
        _ = try database.saveChatArchive(
            AppChatArchive(selectedChatID: original.id, chats: [original, copy]))
        XCTAssertEqual(try database.loadChatArchive()?.chats.count, 2)
    }

    func testLegacyGlobalKeySchemaMigratesWithoutLosingRows() throws {
        let url = root.appendingPathComponent("legacy.sqlite3")
        let chatA = UUID().uuidString
        let chatB = UUID().uuidString
        let connection = try openConnection(url)
        try exec(connection, """
            CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL);
            INSERT INTO schema_migrations(version, applied_at) VALUES(1, 0);
            CREATE TABLE chats(id TEXT PRIMARY KEY, project_id TEXT, title TEXT NOT NULL, updated_at REAL NOT NULL, payload BLOB NOT NULL);
            CREATE TABLE messages(id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, role TEXT NOT NULL, created_at REAL NOT NULL, payload_version INTEGER NOT NULL, payload BLOB NOT NULL);
            CREATE INDEX messages_chat_ordinal ON messages(chat_id, ordinal);
            CREATE TABLE alternates(id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, message_id TEXT NOT NULL, ordinal INTEGER NOT NULL, payload_version INTEGER NOT NULL, payload BLOB NOT NULL);
            CREATE TABLE attachments(id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, asset_id TEXT, file_name TEXT NOT NULL, extracted_text TEXT NOT NULL, payload_version INTEGER NOT NULL, payload BLOB NOT NULL);
            CREATE TABLE artifacts(id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, asset_id TEXT, title TEXT NOT NULL, payload_version INTEGER NOT NULL, payload BLOB NOT NULL);
            CREATE TABLE todos(id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, status TEXT NOT NULL, payload_version INTEGER NOT NULL, payload BLOB NOT NULL);
            INSERT INTO chats VALUES('\(chatA)', NULL, 'a', 0, x'7b7d');
            INSERT INTO chats VALUES('\(chatB)', NULL, 'b', 0, x'7b7d');
            INSERT INTO messages VALUES('m1', '\(chatA)', 0, 'user', 0, 1, x'7b7d');
            INSERT INTO messages VALUES('m2', '\(chatA)', 1, 'assistant', 0, 1, x'7b7d');
            INSERT INTO todos VALUES('1', '\(chatA)', 0, 'pending', 1, x'7b7d');
            """)
        sqlite3_close(connection)

        let database = try ProfileDatabase(url: url, key: key)
        defer { database.close() }
        XCTAssertEqual(try rowCount(table: "messages", at: url), 2, "existing rows survive the rebuild")
        XCTAssertEqual(try rowCount(table: "todos", at: url), 1)
        // The same id in another chat is now legal at the SQL level.
        let check = try openConnection(url)
        defer { sqlite3_close(check) }
        try exec(check, "INSERT INTO todos VALUES('1', '\(chatB)', 0, 'pending', 1, x'7b7d');")
        try exec(check, "INSERT INTO messages VALUES('m1', '\(chatB)', 0, 'user', 0, 1, x'7b7d');")
        // Reopening is idempotent and the index is back.
        let again = try ProfileDatabase(url: url, key: key)
        again.close()
        XCTAssertEqual(try rowCount(table: "todos", at: url), 2)
    }

    // MARK: - Helpers

    private func rowCount(table: String) throws -> Int {
        try rowCount(table: table, at: root.appendingPathComponent("p.sqlite3"))
    }

    private func rowCount(table: String, at url: URL) throws -> Int {
        let connection = try openConnection(url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(connection, "SELECT COUNT(*) FROM \(table)", -1, &statement, nil) == SQLITE_OK,
              let statement
        else { throw ProfileDatabase.DatabaseError.statement(String(cString: sqlite3_errmsg(connection))) }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_step(statement) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(statement, 0))
    }

    private func openConnection(_ url: URL) throws -> OpaquePointer {
        var connection: OpaquePointer?
        guard sqlite3_open_v2(
            url.path, &connection,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK,
            let connection
        else { throw ProfileDatabase.DatabaseError.open("open failed") }
        _ = key.withUnsafeBytes { sqlite3_key(connection, $0.baseAddress, Int32(key.count)) }
        return connection
    }

    private func exec(_ connection: OpaquePointer, _ sql: String) throws {
        var message: UnsafeMutablePointer<Int8>?
        guard sqlite3_exec(connection, sql, nil, nil, &message) == SQLITE_OK else {
            let detail = message.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(message)
            throw ProfileDatabase.DatabaseError.statement(detail)
        }
    }
}
