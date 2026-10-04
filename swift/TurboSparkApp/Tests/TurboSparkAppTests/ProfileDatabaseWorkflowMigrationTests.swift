import Foundation
import SQLCipher
import XCTest

@testable import TurboSparkApp

final class ProfileDatabaseWorkflowMigrationTests: XCTestCase {
    private var root: URL!
    private let key = Data(repeating: 0x52, count: 32)
    private let retainedRecord = Data("existing-profile-data".utf8)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("WorkflowMigrationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testOpeningCopiedPreWorkflowProfileAddsSchemaAndRepeatedOpenIsNoOp() throws {
        let legacyURL = root.appendingPathComponent("legacy.sqlite3")
        let upgradedURL = root.appendingPathComponent("upgraded-copy.sqlite3")
        try createLegacyProfile(at: legacyURL)
        try FileManager.default.copyItem(at: legacyURL, to: upgradedURL)

        let firstOpen = try ProfileDatabase(url: upgradedURL, key: key)
        XCTAssertEqual(firstOpen.workflowSchemaStatus, .available(version: 1))
        XCTAssertEqual(try firstOpen.loadRecord(key: "preserved"), retainedRecord)
        firstOpen.close()

        XCTAssertEqual(try workflowSchemaObjects(at: upgradedURL), Set([
            "table:workflow_schema_migrations",
            "table:workflow_runs",
            "table:workflow_actors",
            "table:workflow_events",
            "table:workflow_artifacts",
            "table:workflow_questions",
            "table:saved_workflows",
            "index:workflow_runs_state_updated",
        ]))
        XCTAssertEqual(try workflowMigrationVersion(at: upgradedURL), 1)

        try insertRunWithUnknownCanonicalizationVersion(at: upgradedURL, version: 904)
        let schemaVersionBeforeSecondOpen = try schemaVersion(at: upgradedURL)

        let secondOpen = try ProfileDatabase(url: upgradedURL, key: key)
        XCTAssertEqual(secondOpen.workflowSchemaStatus, .available(version: 1))
        XCTAssertEqual(try secondOpen.loadRecord(key: "preserved"), retainedRecord)
        secondOpen.close()

        XCTAssertEqual(try schemaVersion(at: upgradedURL), schemaVersionBeforeSecondOpen)
        XCTAssertEqual(try runCanonicalizationVersion(at: upgradedURL), 904)
        XCTAssertEqual(try workflowMigrationVersion(at: upgradedURL), 1)
    }

    func testWorkflowDDLFailureRollsBackAndLeavesCoreProfileAvailable() throws {
        let legacyURL = root.appendingPathComponent("legacy-with-conflict.sqlite3")
        try createLegacyProfile(at: legacyURL, conflictingWorkflowRunsView: true)

        let database = try ProfileDatabase(url: legacyURL, key: key)
        if case .unavailable(let message) = database.workflowSchemaStatus {
            XCTAssertTrue(message.localizedCaseInsensitiveContains("view"), message)
            XCTAssertTrue(message.localizedCaseInsensitiveContains("index"), message)
        } else {
            XCTFail("Expected the workflow schema migration to report its DDL failure")
        }
        XCTAssertEqual(try database.loadRecord(key: "preserved"), retainedRecord)
        database.close()

        // The pre-existing view caused the index DDL error and must survive rollback.
        XCTAssertEqual(try workflowSchemaObjects(at: legacyURL), Set(["view:workflow_runs"]))
        XCTAssertNil(try workflowMigrationVersion(at: legacyURL))

        // The failed optional workflow migration must not prevent normal profile access.
        let reopened = try ProfileDatabase(url: legacyURL, key: key)
        XCTAssertEqual(try reopened.loadRecord(key: "preserved"), retainedRecord)
        if case .unavailable = reopened.workflowSchemaStatus {
            // The conflicting legacy object continues to block workflows only.
        } else {
            XCTFail("Expected only workflow schema status to remain unavailable")
        }
        reopened.close()
    }

    func testFutureWorkflowSchemaVersionFailClosesWorkflowQueriesButKeepsCoreProfile() async throws {
        let url = root.appendingPathComponent("future-version.sqlite3")
        try createLegacyProfile(at: url)
        try insertFutureWorkflowSchemaMarker(at: url)

        let database = try ProfileDatabase(url: url, key: key)
        XCTAssertEqual(database.workflowSchemaStatus, .unsupportedVersion(2))
        XCTAssertEqual(try database.loadRecord(key: "preserved"), retainedRecord)

        // This build neither migrates nor creates its workflow objects under
        // a newer version marker; the newer build's objects survive untouched.
        XCTAssertEqual(try workflowSchemaObjects(at: url), Set([
            "table:workflow_schema_migrations",
            "table:workflow_runs",
        ]))
        XCTAssertEqual(try workflowMigrationVersion(at: url), 2)

        do {
            _ = try await database.listDefinitions()
            XCTFail("saved-workflow queries must fail closed under a future schema version")
        } catch let error as ProfileDatabase.WorkflowSavedDefinitionStorageError {
            guard case .schemaUnavailable(.unsupportedVersion(2)) = error else {
                return XCTFail("expected schemaUnavailable(unsupportedVersion(2)), got \(error)")
            }
        }

        let identity = WorkflowRequestIdentity(
            site: WorkflowSiteKey(lane: "main", siteIndex: 0, ordinal: 0),
            inputHash: "hash")
        do {
            _ = try await database.recordedOutcome(runID: UUID(), identity: identity)
            XCTFail("journal queries must fail closed under a future schema version")
        } catch let error as WorkflowJournalError {
            guard case .schemaUnavailable = error else {
                return XCTFail("expected journal schemaUnavailable, got \(error)")
            }
        }
        database.close()
    }

    /// Simulates a newer app build: it wrote workflow schema version 2 with
    /// its own `workflow_runs` shape before this (older) build opens the copy.
    private func insertFutureWorkflowSchemaMarker(at url: URL) throws {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        try execute(
            connection,
            """
            CREATE TABLE workflow_schema_migrations(
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            );
            CREATE TABLE workflow_runs(
                run_id TEXT PRIMARY KEY,
                future_only_column TEXT
            );
            INSERT INTO workflow_schema_migrations(version, applied_at) VALUES(2, 0);
            INSERT INTO workflow_runs(run_id, future_only_column) VALUES('future-run', 'kept');
            """)
    }

    private func createLegacyProfile(
        at url: URL,
        conflictingWorkflowRunsView: Bool = false
    ) throws {
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
        try bind(retainedRecord, at: 1, to: statement, connection: connection)
        guard sqlite3_step(statement) == SQLITE_DONE else { throw sqliteError(connection) }

        if conflictingWorkflowRunsView {
            try execute(
                connection,
                "CREATE VIEW workflow_runs AS SELECT 'legacy' AS state, '0' AS updated_at;")
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

    private func bind(
        _ value: Data,
        at index: Int32,
        to statement: OpaquePointer,
        connection: OpaquePointer
    ) throws {
        let result = value.withUnsafeBytes { bytes in
            sqlite3_bind_blob(statement, index, bytes.baseAddress, Int32(value.count), Self.transient)
        }
        guard result == SQLITE_OK else { throw sqliteError(connection) }
    }

    private func workflowSchemaObjects(at url: URL) throws -> Set<String> {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "SELECT type, name FROM sqlite_master WHERE name LIKE 'workflow_%' OR name = 'saved_workflows' ORDER BY type, name",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }

        var result = Set<String>()
        while true {
            let step = sqlite3_step(statement)
            if step == SQLITE_DONE { return result }
            guard step == SQLITE_ROW,
                  let type = sqlite3_column_text(statement, 0),
                  let name = sqlite3_column_text(statement, 1)
            else { throw sqliteError(connection) }
            result.insert("\(String(cString: type)):\(String(cString: name))")
        }
    }

    private func workflowMigrationVersion(at url: URL) throws -> Int64? {
        let objects = try workflowSchemaObjects(at: url)
        guard objects.contains("table:workflow_schema_migrations") else { return nil }

        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "SELECT MAX(version) FROM workflow_schema_migrations",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_step(statement) == SQLITE_ROW,
              sqlite3_column_type(statement, 0) != SQLITE_NULL
        else { return nil }
        return sqlite3_column_int64(statement, 0)
    }

    private func insertRunWithUnknownCanonicalizationVersion(at url: URL, version: Int64) throws {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "INSERT INTO workflow_runs(run_id, state, canonicalization_version, updated_at) VALUES('future-canonicalization', 'interrupted', ?1, 'now')",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_bind_int64(statement, 1, version) == SQLITE_OK,
              sqlite3_step(statement) == SQLITE_DONE
        else { throw sqliteError(connection) }
    }

    private func runCanonicalizationVersion(at url: URL) throws -> Int64? {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(
            connection,
            "SELECT canonicalization_version FROM workflow_runs WHERE run_id = 'future-canonicalization'",
            -1,
            &statement,
            nil) == SQLITE_OK,
            let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
        return sqlite3_column_int64(statement, 0)
    }

    private func schemaVersion(at url: URL) throws -> Int64 {
        let connection = try openDatabase(at: url)
        defer { sqlite3_close(connection) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(connection, "PRAGMA schema_version;", -1, &statement, nil) == SQLITE_OK,
              let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_step(statement) == SQLITE_ROW else { throw sqliteError(connection) }
        return sqlite3_column_int64(statement, 0)
    }

    private func sqliteError(_ connection: OpaquePointer) -> Error {
        ProfileDatabase.DatabaseError.statement(String(cString: sqlite3_errmsg(connection)))
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
}
