import Foundation
import SQLCipher
import XCTest

@testable import TurboSparkApp

final class ProfileDatabaseWorkflowSavedDefinitionTests: XCTestCase {
    private var root: URL!
    private let key = Data(repeating: 0x63, count: 32)
    private let retainedRecord = Data("core-profile-record".utf8)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("WorkflowSavedDefinitionTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testRoundTripsEverySavedDefinitionField() async throws {
        let database = try makeDatabase()
        defer { database.close() }
        let expected = definition(
            id: "00000000-0000-0000-0000-000000000031",
            name: "Research plan",
            source: "async function workflow() { return report({ topic: topic }); }",
            declarations: [
                WorkflowArgumentDeclaration(name: "topic", required: true, maximumUTF8Bytes: 2048),
                WorkflowArgumentDeclaration(name: "format", required: false, maximumUTF8Bytes: 64),
            ],
            description: "Researches a supplied topic and returns a report.",
            timestamp: 1_800_000_000.125)

        try await database.save(expected)

        let actual = try await database.definition(id: expected.id)
        let missing = try await database.definition(id: UUID())
        XCTAssertEqual(actual, expected)
        XCTAssertNil(missing)
    }

    func testReplacementKeepsOneEntryAndListIsDeterministic() async throws {
        let database = try makeDatabase()
        defer { database.close() }
        let replacementID = UUID(uuidString: "00000000-0000-0000-0000-000000000020")!
        let original = definition(id: replacementID, name: "Before", timestamp: 100)
        let replacement = definition(
            id: replacementID,
            name: "After",
            source: "async function workflow() { return report({ result: 'new' }); }",
            declarations: [WorkflowArgumentDeclaration(name: "result", required: true, maximumUTF8Bytes: 512)],
            description: "Replacement definition.",
            timestamp: 200.25)
        let laterID = UUID(uuidString: "00000000-0000-0000-0000-000000000030")!
        let earlierID = UUID(uuidString: "00000000-0000-0000-0000-000000000010")!

        try await database.save(original)
        try await database.save(replacement)
        try await database.save(definition(id: laterID, name: "Later"))
        try await database.save(definition(id: earlierID, name: "Earlier"))

        let fetchedReplacement = try await database.definition(id: replacementID)
        XCTAssertEqual(fetchedReplacement, replacement)
        let expected = [earlierID, replacementID, laterID]
        let firstList = try await database.listDefinitions()
        let secondList = try await database.listDefinitions()
        XCTAssertEqual(firstList.map(\.id), expected)
        XCTAssertEqual(secondList, firstList)
        XCTAssertEqual(firstList.filter { $0.id == replacementID }.count, 1)
    }

    func testFailedReplacementRollsBackAndPreservesPreviousDefinition() async throws {
        let url = root.appendingPathComponent("trigger-failure.sqlite3")
        let original = definition(
            id: "00000000-0000-0000-0000-000000000041",
            name: "Committed definition",
            source: "async function workflow() { return report({ value: 'committed' }); }",
            declarations: [WorkflowArgumentDeclaration(name: "value", required: true, maximumUTF8Bytes: 128)],
            description: "Stored before the failed replacement.",
            timestamp: 300.5)
        let rejected = definition(
            id: original.id,
            name: "Rejected replacement",
            source: "async function workflow() { return report({ value: 'partial' }); }",
            declarations: [WorkflowArgumentDeclaration(name: "value", required: false, maximumUTF8Bytes: 1)],
            description: "Must not be partially stored.",
            timestamp: 301.5)

        let initialDatabase = try ProfileDatabase(url: url, key: key)
        try await initialDatabase.save(original)
        initialDatabase.close()
        try createUpdateFailureTrigger(at: url)

        let database = try ProfileDatabase(url: url, key: key)
        defer { database.close() }
        do {
            try await database.save(rejected)
            XCTFail("Expected the database trigger to reject the replacement")
        } catch let error as ProfileDatabase.DatabaseError {
            guard case .step(let message) = error else {
                XCTFail("Expected a propagated SQLite step error, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("injected saved workflow update failure"), message)
        }

        let persistedDefinition = try await database.definition(id: original.id)
        let persistedList = try await database.listDefinitions()
        XCTAssertEqual(persistedDefinition, original)
        XCTAssertEqual(persistedList, [original])
    }

    func testUnavailableWorkflowSchemaReturnsStorageErrorsBeforeWorkflowQueries() async throws {
        let url = root.appendingPathComponent("workflow-schema-unavailable.sqlite3")
        try createLegacyProfileWithWorkflowConflict(at: url)

        let database = try ProfileDatabase(url: url, key: key)
        defer { database.close() }
        guard case .unavailable = database.workflowSchemaStatus else {
            XCTFail("Expected workflow migration to be unavailable")
            return
        }
        XCTAssertEqual(try database.loadRecord(key: "preserved"), retainedRecord)

        let value = definition(id: "00000000-0000-0000-0000-000000000051", name: "Unavailable")
        do {
            try await database.save(value)
            XCTFail("Expected save to reject an unavailable workflow schema")
        } catch {
            assertSchemaUnavailable(error)
        }
        do {
            _ = try await database.definition(id: value.id)
            XCTFail("Expected lookup to reject an unavailable workflow schema")
        } catch {
            assertSchemaUnavailable(error)
        }
        do {
            _ = try await database.listDefinitions()
            XCTFail("Expected listing to reject an unavailable workflow schema")
        } catch {
            assertSchemaUnavailable(error)
        }
    }

    private func makeDatabase() throws -> ProfileDatabase {
        try ProfileDatabase(url: root.appendingPathComponent("profile.sqlite3"), key: key)
    }

    private func definition(
        id: UUID,
        name: String,
        source: String = "async function workflow() { return report({ ok: true }); }",
        declarations: [WorkflowArgumentDeclaration] = [],
        description: String = "Saved workflow definition.",
        timestamp: TimeInterval = 1234.5
    ) -> WorkflowSavedDefinition {
        WorkflowSavedDefinition(
            id: id,
            name: name,
            scope: .profile,
            source: source,
            declarations: declarations,
            description: description,
            updatedAt: Date(timeIntervalSince1970: timestamp))
    }

    private func definition(
        id: String,
        name: String,
        source: String = "async function workflow() { return report({ ok: true }); }",
        declarations: [WorkflowArgumentDeclaration] = [],
        description: String = "Saved workflow definition.",
        timestamp: TimeInterval = 1234.5
    ) -> WorkflowSavedDefinition {
        definition(
            id: UUID(uuidString: id)!,
            name: name,
            source: source,
            declarations: declarations,
            description: description,
            timestamp: timestamp)
    }

    private func assertSchemaUnavailable(_ error: Error, file: StaticString = #filePath, line: UInt = #line) {
        guard let storageError = error as? ProfileDatabase.WorkflowSavedDefinitionStorageError,
              case .schemaUnavailable(let status) = storageError,
              case .unavailable = status
        else {
            XCTFail("Expected a workflow schema availability error, got \(error)", file: file, line: line)
            return
        }
    }

    private func createUpdateFailureTrigger(at url: URL) throws {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        try execute(
            connection,
            """
            CREATE TRIGGER fail_saved_workflow_update
            BEFORE UPDATE ON saved_workflows
            WHEN NEW.name = 'Rejected replacement'
            BEGIN
                SELECT RAISE(ABORT, 'injected saved workflow update failure');
            END;
            """)
    }

    private func createLegacyProfileWithWorkflowConflict(at url: URL) throws {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        try execute(
            connection,
            """
            CREATE TABLE schema_migrations(
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            );
            INSERT INTO schema_migrations(version, applied_at) VALUES(1, 0);
            CREATE TABLE private_records(
                key TEXT PRIMARY KEY,
                payload BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE chats(
                id TEXT PRIMARY KEY,
                project_id TEXT,
                title TEXT NOT NULL,
                updated_at REAL NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE VIEW workflow_runs AS SELECT 'legacy' AS state, '0' AS updated_at;
            """)

        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "INSERT INTO private_records(key, payload, updated_at) VALUES('preserved', ?1, 0)",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        let bindResult = retainedRecord.withUnsafeBytes { bytes in
            sqlite3_bind_blob(statement, 1, bytes.baseAddress, Int32(retainedRecord.count), Self.transient)
        }
        guard bindResult == SQLITE_OK, sqlite3_step(statement) == SQLITE_DONE else {
            throw sqliteError(connection)
        }
    }

    private func openDatabase(at url: URL) throws -> OpaquePointer {
        var connection: OpaquePointer?
        guard sqlite3_open_v2(
            url.path,
            &connection,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
            nil) == SQLITE_OK,
            let connection
        else {
            throw connection.map(sqliteError)
                ?? ProfileDatabase.DatabaseError.open("Could not open test database")
        }

        let result = key.withUnsafeBytes { bytes in
            sqlite3_key(connection, bytes.baseAddress, Int32(key.count))
        }
        guard result == SQLITE_OK else {
            let error = sqliteError(connection)
            sqlite3_close(connection)
            throw error
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

    private func sqliteError(_ connection: OpaquePointer) -> Error {
        ProfileDatabase.DatabaseError.statement(String(cString: sqlite3_errmsg(connection)))
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
}
