import Foundation
import SQLCipher
import XCTest

@testable import TurboSparkApp

final class WorkflowProfileCrossVersionCompatibilityTests: XCTestCase {
    private let key = Data(repeating: 0x42, count: 32)

    func testCreateFixtureForHistoricalReader() async throws {
        let root = try compatibilityRoot()
        let profileURL = root.appendingPathComponent("workflow-bearing-profile.sqlite3")
        let database = try ProfileDatabase(url: profileURL, key: key)
        XCTAssertEqual(database.workflowSchemaStatus, .available(version: 1))
        try database.saveRecord(key: "compatibility:sentinel", payload: Data("core-record-before-old-reader".utf8))
        try await database.save(WorkflowSavedDefinition(
            id: UUID(uuidString: "80000000-0000-0000-0000-000000000001")!,
            name: "Compatibility fixture",
            scope: .profile,
            source: "async function workflow() { return report({ status: 'preserved' }); }",
            declarations: [WorkflowArgumentDeclaration(
                name: "topic",
                required: true,
                maximumUTF8Bytes: 1024)],
            description: "A deterministic saved workflow for reader compatibility.",
            updatedAt: Date(timeIntervalSince1970: 2_000_000_001.125)))
        database.close()

        try withConnection(at: profileURL) { connection in
            try execute(
                connection,
                """
                BEGIN IMMEDIATE;
                INSERT INTO workflow_runs(
                    run_id, name, state, script_source, script_hash, facade_version,
                    args_json, parent_run_id, launch_source, canonicalization_version,
                    usage_json, created_at, updated_at
                ) VALUES(
                    'compat-run-1', 'compatibility', 'completed',
                    'async function workflow() { return report({ status: ''preserved'' }); }',
                    'script-hash-v1', 1, '{"topic":"history"}', NULL,
                    'saved-definition', 1, '{"input_tokens":3,"output_tokens":5}',
                    '2033-05-18T03:33:21Z', '2033-05-18T03:33:22Z'
                );
                INSERT INTO workflow_actors(
                    run_id, actor_name, usage_json, transcript_json, transcript_sha256
                ) VALUES(
                    'compat-run-1', 'researcher', '{"input_tokens":2}',
                    '{"turns":[{"text":"evidence"}]}', 'transcript-hash-v1'
                );
                INSERT INTO workflow_events(
                    run_id, seq, kind, lane, site_index, site_ordinal, input_hash,
                    payload_json, created_at
                ) VALUES(
                    'compat-run-1', 1, 'artifact', 'researcher', 4, 0,
                    'input-hash-v1', CAST(X'7B226172746966616374223A22707265736572766564227D' AS BLOB),
                    '2033-05-18T03:33:23Z'
                );
                INSERT INTO workflow_artifacts(
                    run_id, artifact_id, version, kind, is_primary, storage_path, created_at
                ) VALUES(
                    'compat-run-1', 'report-1', 1, 'report', 1,
                    'artifacts/compatibility/report-1.json', '2033-05-18T03:33:24Z'
                );
                INSERT INTO workflow_questions(
                    run_id, question_id, prompt, answered, answer, created_at, resolved_at
                ) VALUES(
                    'compat-run-1', 'approval-1', 'Continue with the report?', 1,
                    'yes', '2033-05-18T03:33:25Z', '2033-05-18T03:33:26Z'
                );
                COMMIT;
                """)
        }

        let snapshot = try snapshotWorkflowState(at: profileURL)
        XCTAssertEqual(snapshot.migrationVersion, 1)
        try JSONEncoder.compatibility.encode(snapshot)
            .write(to: root.appendingPathComponent("workflow-state-before.json"), options: .atomic)
    }

    func testVerifyHistoricalReaderCompatibility() throws {
        let root = try compatibilityRoot()
        let profileURL = root.appendingPathComponent("workflow-bearing-profile.sqlite3")
        let snapshotURL = root.appendingPathComponent("workflow-state-before.json")
        let expected = try JSONDecoder().decode(
            WorkflowProfileStateSnapshot.self,
            from: Data(contentsOf: snapshotURL))

        // Inspect raw workflow state before the current initializer can run its own migrations.
        let afterHistoricalReader = try snapshotWorkflowState(at: profileURL)
        XCTAssertEqual(afterHistoricalReader, expected, "The historical initializer changed workflow schema or stored values")
        XCTAssertEqual(afterHistoricalReader.migrationVersion, 1, "The workflow-owned migration marker changed")

        let database = try ProfileDatabase(url: profileURL, key: key)
        XCTAssertEqual(database.workflowSchemaStatus, .available(version: 1))
        XCTAssertEqual(
            try database.loadRecord(key: "compatibility:sentinel"),
            Data("core-record-before-old-reader".utf8))
        XCTAssertEqual(
            try database.loadRecord(key: "compatibility:historical-write"),
            Data("written-by-historical-reader".utf8))
        database.close()

        let afterCurrentReopen = try snapshotWorkflowState(at: profileURL)
        XCTAssertEqual(afterCurrentReopen, expected, "Reopening the profile changed workflow-owned state")
    }

    private func compatibilityRoot() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["WORKFLOW_COMPATIBILITY_ROOT"] else {
            throw XCTSkip("Run Scripts/test-workflow-profile-backward-compatibility.sh to exercise the pinned historical reader")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func withConnection<T>(at url: URL, _ body: (OpaquePointer) throws -> T) throws -> T {
        var connection: OpaquePointer?
        guard sqlite3_open_v2(
            url.path,
            &connection,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX,
            nil) == SQLITE_OK,
            let connection
        else {
            throw connection.map(sqliteError)
                ?? ProfileDatabase.DatabaseError.open("Could not open compatibility fixture")
        }
        defer { sqlite3_close(connection) }

        let keyResult = key.withUnsafeBytes { bytes in
            sqlite3_key(connection, bytes.baseAddress, Int32(key.count))
        }
        guard keyResult == SQLITE_OK else { throw sqliteError(connection) }
        return try body(connection)
    }

    private func execute(_ connection: OpaquePointer, _ sql: String) throws {
        var message: UnsafeMutablePointer<Int8>?
        guard sqlite3_exec(connection, sql, nil, nil, &message) == SQLITE_OK else {
            let detail = message.map { String(cString: $0) } ?? String(cString: sqlite3_errmsg(connection))
            sqlite3_free(message)
            throw ProfileDatabase.DatabaseError.statement(detail)
        }
    }

    private func snapshotWorkflowState(at url: URL) throws -> WorkflowProfileStateSnapshot {
        try withConnection(at: url) { connection in
            let schemaRows = try queryRows(
                connection,
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE name GLOB 'workflow_*' OR name = 'saved_workflows'
                ORDER BY type, name
                """)
            let tableNames = try queryRows(
                connection,
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND (name GLOB 'workflow_*' OR name = 'saved_workflows')
                ORDER BY name
                """).compactMap { row -> String? in
                    guard case .text(let bytes)? = row.first else { return nil }
                    return String(data: bytes, encoding: .utf8)
                }

            let tables = try tableNames.map { name -> WorkflowProfileTableSnapshot in
                let quotedName = "\"\(name.replacingOccurrences(of: "\"", with: "\"\""))\""
                return WorkflowProfileTableSnapshot(
                    name: name,
                    rows: try queryRows(connection, "SELECT * FROM \(quotedName) ORDER BY rowid"))
            }
            let markerRows = try queryRows(
                connection,
                "SELECT MAX(version) FROM workflow_schema_migrations")
            let version: Int64?
            if case .integer(let value)? = markerRows.first?.first {
                version = value
            } else {
                version = nil
            }
            return WorkflowProfileStateSnapshot(schema: schemaRows, tables: tables, migrationVersion: version)
        }
    }

    private func queryRows(_ connection: OpaquePointer, _ sql: String) throws -> [[WorkflowProfileSQLiteValue]] {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(connection, sql, -1, &statement, nil) == SQLITE_OK,
              let statement
        else { throw sqliteError(connection) }
        defer { sqlite3_finalize(statement) }

        var rows: [[WorkflowProfileSQLiteValue]] = []
        while true {
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return rows }
            guard result == SQLITE_ROW else { throw sqliteError(connection) }
            rows.append((0..<sqlite3_column_count(statement)).map { index in
                let column = Int32(index)
                switch sqlite3_column_type(statement, column) {
                case SQLITE_INTEGER:
                    return .integer(sqlite3_column_int64(statement, column))
                case SQLITE_FLOAT:
                    return .realBits(sqlite3_column_double(statement, column).bitPattern)
                case SQLITE_TEXT:
                    return .text(columnBytes(statement, column))
                case SQLITE_BLOB:
                    return .blob(columnBytes(statement, column))
                default:
                    return .null
                }
            })
        }
    }

    private func columnBytes(_ statement: OpaquePointer, _ column: Int32) -> Data {
        let count = Int(sqlite3_column_bytes(statement, column))
        guard count > 0, let bytes = sqlite3_column_blob(statement, column) else { return Data() }
        return Data(bytes: bytes, count: count)
    }

    private func sqliteError(_ connection: OpaquePointer) -> Error {
        ProfileDatabase.DatabaseError.statement(String(cString: sqlite3_errmsg(connection)))
    }
}

private struct WorkflowProfileStateSnapshot: Codable, Equatable {
    var schema: [[WorkflowProfileSQLiteValue]]
    var tables: [WorkflowProfileTableSnapshot]
    var migrationVersion: Int64?
}

private struct WorkflowProfileTableSnapshot: Codable, Equatable {
    var name: String
    var rows: [[WorkflowProfileSQLiteValue]]
}

private enum WorkflowProfileSQLiteValue: Codable, Equatable {
    case null
    case integer(Int64)
    case realBits(UInt64)
    case text(Data)
    case blob(Data)
}

private extension JSONEncoder {
    static var compatibility: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}
