import Foundation
import XCTest

@testable import TurboSparkApp

/// A resume the engine refuses (a journal written with a newer canonicalization
/// version) must not turn into a persisted terminal failure of the run.
final class WorkflowResumeRefusalTests: XCTestCase {
    private var root: URL!
    private let key = Data(repeating: 0x4B, count: 32)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("WorkflowResumeRefusal-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private struct NoopCapabilities: WorkflowCapabilities {
        func perform(
            _ operation: WorkflowInterpreterOperation,
            at site: WorkflowSiteKey?,
            context: WorkflowAttemptContext
        ) async throws -> WorkflowCapabilityResult {
            WorkflowCapabilityResult(value: .null)
        }
    }

    func testRefusedResumeLeavesTheRecordedRunUntouched() async throws {
        let database = try ProfileDatabase(url: root.appendingPathComponent("p.sqlite3"), key: key)
        defer { database.close() }
        let journal = WorkflowJournal(store: database, now: { Date(timeIntervalSince1970: 1_800_000_000) })
        let checked = try XCTUnwrap(WorkflowScriptChecker.check(
            source: "async function workflow() { await report(\"done\"); }",
            name: "Refusal", args: [:]).checked)
        try await journal.createRun(checked.descriptor)
        try database.execute("UPDATE workflow_runs SET canonicalization_version = 777")
        let before = try await database.history(runID: checked.descriptor.id)

        let engine = WorkflowRunEngine(
            program: checked, journal: journal, capabilities: NoopCapabilities())
        await engine.start()

        let after = try await database.history(runID: checked.descriptor.id)
        XCTAssertEqual(after.events.count, before.events.count, "no failure events may be appended")
        XCTAssertEqual(after.state, before.state, "the persisted state must not flip to failed")
        XCTAssertNotEqual(after.state, WorkflowRunState.failed.rawValue)
    }
}
