import XCTest

@testable import TurboSparkApp

/// Which `memory` tool calls the transcript marks with a "Proposed for
/// memory" note. The note must appear only for a save the executor actually
/// staged, so the fixtures run the real executor against scratch stores and
/// feed its real output to the predicate.
final class MemoryProposalNoteTests: XCTestCase {
    private func scratch(_ label: String) -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("MemoryNoteTests-\(label)-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.resolvingSymlinksInPath()
    }

    /// Runs one real `memory` call and returns it with its result.
    private func run(
        _ arguments: [String: String], name: String = "memory"
    ) throws -> (AppToolCall, AppToolResult) {
        let store = MemoryStore(base: scratch("store"))
        store.isModelEnabled = true
        let ledger = MemoryLedgerStore(base: scratch("ledger"))
        let project = AppProject(name: "Demo", rootDirectoryPath: scratch("project").path)
        let output = try MemoryToolExecutor.execute(
            arguments: arguments, project: project, store: store, ledger: ledger)
        let call = AppToolCall(name: name, arguments: arguments, status: .completed)
        return (call, AppToolResult(callID: call.id, output: output))
    }

    func testAProjectSaveIsNotedWithItsDescription() throws {
        let (call, result) = try run([
            "action": "save", "name": "Deploy Flow",
            "description": "Ship on Thursdays", "content": "Body text."])
        let note = try XCTUnwrap(MemoryProposalNote(call: call, result: result))
        XCTAssertEqual(note.scope, .project)
        XCTAssertEqual(note.text, "Ship on Thursdays")
        XCTAssertEqual(note.callID, call.id)
    }

    func testAProjectSaveWithoutDescriptionUsesTheFirstBodyLine() throws {
        let (call, result) = try run([
            "action": "remember", "name": "x", "content": "\n  First line  \nsecond"])
        XCTAssertEqual(MemoryProposalNote(call: call, result: result)?.text, "First line")
    }

    func testAProfileSaveIsNoted() throws {
        let (call, result) = try run([
            "action": "save", "scope": "Profile", "content": "Prefers metric units."])
        let note = try XCTUnwrap(MemoryProposalNote(call: call, result: result))
        XCTAssertEqual(note.scope, .profile)
        XCTAssertEqual(note.text, "Prefers metric units.")
    }

    func testTheRememberAliasToolNameIsNoted() throws {
        let (call, result) = try run(
            ["action": "save", "scope": "profile", "content": "Tea"], name: "remember")
        XCTAssertNotNil(MemoryProposalNote(call: call, result: result))
    }

    func testLongTextIsCutToOneShortLine() throws {
        let long = String(repeating: "word ", count: 60)
        let (call, result) = try run(["action": "save", "scope": "profile", "content": long])
        let text = try XCTUnwrap(MemoryProposalNote(call: call, result: result)?.text)
        XCTAssertLessThanOrEqual(text.count, MemoryProposalNote.maxTextLength + 3)
        XCTAssertTrue(text.hasSuffix("..."))
        XCTAssertFalse(text.contains("\n"))
    }

    func testReadsAndForgetsAreNotNoted() throws {
        let (readCall, readResult) = try run(["action": "read"])
        XCTAssertNil(MemoryProposalNote(call: readCall, result: readResult))
        let (forgetCall, forgetResult) = try run(["action": "forget", "name": "missing"])
        XCTAssertNil(MemoryProposalNote(call: forgetCall, result: forgetResult))
    }

    func testAnErrorOrMissingResultIsNotNoted() throws {
        let call = AppToolCall(
            name: "memory", arguments: ["action": "save", "scope": "profile", "content": "x"],
            status: .completed)
        XCTAssertNil(MemoryProposalNote(call: call, result: nil))
        let failed = AppToolResult(
            callID: call.id, output: "Proposed a profile memory for review in Settings.",
            isError: true)
        XCTAssertNil(MemoryProposalNote(call: call, result: failed))
        let other = AppToolResult(callID: UUID(), output: "Proposed a profile memory for review in Settings.")
        XCTAssertNil(MemoryProposalNote(call: call, result: other), "a result for another call")
    }

    func testAnOutputTheExecutorDidNotWriteIsNotNoted() {
        // A hook or a future wording change must hide the note, not draw a
        // claim that was never staged.
        let call = AppToolCall(
            name: "memory", arguments: ["action": "save", "scope": "profile", "content": "x"],
            status: .completed)
        let result = AppToolResult(callID: call.id, output: "Saved.")
        XCTAssertNil(MemoryProposalNote(call: call, result: result))
    }

    func testOtherToolsAreNotNoted() {
        let call = AppToolCall(name: "memory_search", arguments: ["query": "x"], status: .completed)
        let result = AppToolResult(callID: call.id, output: "Proposed a profile memory for review in Settings.")
        XCTAssertNil(MemoryProposalNote(call: call, result: result))
    }

    func testNotesKeepCallOrderAcrossAGroup() throws {
        let (first, firstResult) = try run(["action": "save", "scope": "profile", "content": "A"])
        let (read, readResult) = try run(["action": "read"])
        let (second, secondResult) = try run(["action": "save", "scope": "profile", "content": "B"])
        let notes = MemoryProposalNote.notes(
            calls: [first, read, second], results: [secondResult, readResult, firstResult])
        XCTAssertEqual(notes.map(\.text), ["A", "B"])
    }

    // MARK: - Addressing the claim by tool call id

    private func claim(_ status: MemoryClaimStatus, callID: UUID?) -> MemoryClaim {
        var claim = MemoryClaim(scope: "profile", kind: "preference", text: "x", status: status)
        claim.sourceToolCallID = callID
        return claim
    }

    /// A ledger written before the field existed must still load: the key
    /// is absent, the claim decodes, and no note can address it.
    func testAClaimWrittenBeforeTheCallIDFieldStillDecodes() throws {
        let id = UUID()
        let json = """
        {"version":1,"claims":[{"id":"\(id.uuidString)","scope":"profile","kind":"preference",
        "text":"Tea","status":"pending","confidence":"agent-proposed","createdAt":0,
        "updatedAt":0,"evidence":[]}],"processedSources":[],"suppressedSources":[],
        "receipts":[],"reflections":[],"reviewBatches":[],"activeGuidance":"",
        "legacyImported":false,"revision":"\(UUID().uuidString)"}
        """
        let state = try JSONDecoder().decode(MemoryLedgerState.self, from: Data(json.utf8))
        XCTAssertEqual(state.claims.map(\.id), [id])
        XCTAssertNil(state.claims[0].sourceToolCallID)
        XCTAssertNil(MemoryProposalNote.resolution(callID: UUID(), claims: state.claims))
    }

    /// The id goes on the ledger only: the model-visible output is unchanged.
    func testTheExecutorRecordsTheCallIDOnAProjectAndAProfileSave() throws {
        let store = MemoryStore(base: scratch("store"))
        store.isModelEnabled = true
        let ledger = MemoryLedgerStore(base: scratch("ledger"))
        let project = AppProject(name: "Demo", rootDirectoryPath: scratch("project").path)
        let projectCall = UUID()
        let profileCall = UUID()
        let projectOutput = try MemoryToolExecutor.execute(
            arguments: ["action": "save", "name": "deploy", "content": "Thursdays"],
            project: project, store: store, ledger: ledger, toolCallID: projectCall)
        let profileOutput = try MemoryToolExecutor.execute(
            arguments: ["action": "save", "scope": "profile", "content": "Tea"],
            project: project, store: store, ledger: ledger, toolCallID: profileCall)
        XCTAssertEqual(projectOutput, "Proposed memory 'deploy' for review in Settings. It is not active yet.")
        XCTAssertEqual(profileOutput, "Proposed a profile memory for review in Settings. It is not active yet.")
        let claims = ledger.snapshot().claims
        XCTAssertEqual(claims.map(\.sourceToolCallID), [projectCall, profileCall])
    }

    func testResolutionFindsThePendingClaimByCallID() {
        let callID = UUID()
        let target = claim(.pending, callID: callID)
        let claims = [claim(.pending, callID: UUID()), target, claim(.pending, callID: nil)]
        XCTAssertEqual(MemoryProposalNote.resolution(callID: callID, claims: claims), .pending(target.id))
    }

    func testResolutionReportsAClaimSettledElsewhere() {
        let callID = UUID()
        for status in [MemoryClaimStatus.active, .dismissed, .retracted, .superseded] {
            XCTAssertEqual(
                MemoryProposalNote.resolution(callID: callID, claims: [claim(status, callID: callID)]),
                .settled(status))
        }
    }

    func testResolutionIsNilWhenUnsure() {
        let callID = UUID()
        XCTAssertNil(MemoryProposalNote.resolution(callID: callID, claims: []))
        XCTAssertNil(MemoryProposalNote.resolution(
            callID: callID, claims: [claim(.pending, callID: nil)]), "never matched by anything else")
        XCTAssertNil(MemoryProposalNote.resolution(
            callID: callID, claims: [claim(.pending, callID: callID), claim(.pending, callID: callID)]),
            "two claims for one call cannot be told apart")
    }
}

/// The row's Undo and Approve, through the model, against a claim the real
/// executor staged.
@MainActor
final class MemoryProposalActionTests: XCTestCase {
    private var savedEnabled = false

    override func setUp() {
        super.setUp()
        savedEnabled = MemoryStore.shared.isModelEnabled
    }

    override func tearDown() {
        MemoryStore.shared.isModelEnabled = savedEnabled
        super.tearDown()
    }

    private func stage(_ ledger: MemoryLedgerStore) throws -> AppToolCall {
        let store = MemoryStore(base: scratchDirectory("store"))
        store.isModelEnabled = true
        let arguments = ["action": "save", "scope": "profile", "content": "Prefers tea."]
        let call = AppToolCall(name: "memory", arguments: arguments, status: .completed)
        _ = try MemoryToolExecutor.execute(
            arguments: arguments, project: nil, store: store, ledger: ledger, toolCallID: call.id)
        return call
    }

    private func scratchDirectory(_ label: String) -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("MemoryActionTests-\(label)-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testUndoDismissesTheProposalItNamed() throws {
        let ledger = MemoryLedgerStore(base: scratchDirectory("ledger"))
        let call = try stage(ledger)
        let other = try stage(ledger)
        let model = AppModel()
        model.memoryEnabled = true

        let after = model.actOnMemoryProposal(callID: call.id, .undo, ledger: ledger)

        XCTAssertEqual(after, .settled(.dismissed))
        let claims = ledger.snapshot().claims
        XCTAssertEqual(claims.first { $0.sourceToolCallID == call.id }?.status, .dismissed)
        XCTAssertEqual(claims.first { $0.sourceToolCallID == other.id }?.status, .pending,
                       "only the named proposal changes")
    }

    func testApproveActivatesTheProposal() throws {
        let ledger = MemoryLedgerStore(base: scratchDirectory("ledger"))
        let call = try stage(ledger)
        let model = AppModel()
        model.memoryEnabled = true

        XCTAssertEqual(model.actOnMemoryProposal(callID: call.id, .approve, ledger: ledger), .settled(.active))
    }

    func testAProposalSettledElsewhereIsLeftAlone() throws {
        let ledger = MemoryLedgerStore(base: scratchDirectory("ledger"))
        let call = try stage(ledger)
        let id = try XCTUnwrap(ledger.snapshot().claims.first?.id)
        try ledger.approve(id)
        let model = AppModel()
        model.memoryEnabled = true

        XCTAssertEqual(model.actOnMemoryProposal(callID: call.id, .undo, ledger: ledger), .settled(.active))
        XCTAssertEqual(ledger.snapshot().claims.first?.status, .active)
    }

    func testNothingChangesWhileMemoryIsOff() throws {
        let ledger = MemoryLedgerStore(base: scratchDirectory("ledger"))
        let call = try stage(ledger)
        let model = AppModel()
        model.memoryEnabled = false

        XCTAssertNil(model.actOnMemoryProposal(callID: call.id, .undo, ledger: ledger))
        XCTAssertEqual(ledger.snapshot().claims.first?.status, .pending)
    }
}
