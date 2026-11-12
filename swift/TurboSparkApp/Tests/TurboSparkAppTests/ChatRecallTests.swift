import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Conversation recall: the pure planner, then the real syntext store.
final class ChatRecallPlannerTests: XCTestCase {
    private typealias Planner = ChatRecallPlanner

    func testRecordableKeepsPersonAndProseOnly() {
        let rows = [
            AppChatMessage(role: .system, content: "system text"),
            AppChatMessage(role: .user, content: "real question"),
            AppChatMessage(role: .user, content: "hook nudge", isSynthetic: true),
            AppChatMessage(
                role: .assistant, content: "calling",
                toolCalls: [AppToolCall(name: "grep", arguments: [:], category: .fileRead)]),
            AppChatMessage(role: .assistant, content: "retry", stopReason: "guardrail_retry"),
            AppChatMessage(role: .assistant, content: "   \n"),
            AppChatMessage(role: .assistant, content: "the answer"),
        ]
        let turns = Planner.recordable(rows)
        XCTAssertEqual(turns.map(\.text), ["real question", "the answer"])
        XCTAssertEqual(turns.map(\.rowIndex), [1, 6])
        XCTAssertEqual(turns.map(\.role), [.user, .assistant])
    }

    func testNormalizeStripsNULAndCRLFAndCaps() {
        XCTAssertEqual(Planner.normalize("a\r\nb\u{0}c"), "a\nbc")
        let long = String(repeating: "x", count: Planner.maxTurnBytes + 100)
        XCTAssertEqual(Planner.normalize(long).utf8.count, Planner.maxTurnBytes)
        // Deterministic for a cut inside a multi-byte character.
        let wide = String(repeating: "\u{00E9}", count: Planner.maxTurnBytes)
        XCTAssertEqual(Planner.normalize(wide), Planner.normalize(wide))
    }

    func testReconcileAppendsSuffixAndRebuildsOnDivergence() {
        let turns = [
            Planner.Turn(rowIndex: 0, role: .user, text: "q1"),
            Planner.Turn(rowIndex: 1, role: .assistant, text: "a1"),
            Planner.Turn(rowIndex: 2, role: .user, text: "q2"),
        ]
        XCTAssertEqual(
            Planner.reconcilePlan(recorded: [("user", "q1")], recordable: turns),
            .append(Array(turns[1...])))
        XCTAssertEqual(
            Planner.reconcilePlan(
                recorded: [("user", "q1"), ("assistant", "a1"), ("user", "q2")], recordable: turns),
            .none)
        // An edited earlier turn.
        XCTAssertEqual(
            Planner.reconcilePlan(recorded: [("user", "q1 edited")], recordable: turns), .rebuild)
        // A rewind: the journal is longer than the transcript.
        XCTAssertEqual(
            Planner.reconcilePlan(
                recorded: [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")],
                recordable: turns),
            .rebuild)
    }

    /// system, summary (user), pin (user), row0 user, row1 assistant, row2 user.
    private func projection() -> (messages: [ChatMessage], rows: [Int?], pin: Int) {
        let messages = [
            ChatMessage(role: .system, content: "sys"),
            ChatMessage(role: .user, content: "<context_summary>s</context_summary>"),
            ChatMessage(role: .user, content: "<pinned>p</pinned>"),
            ChatMessage(role: .user, content: "q1"),
            ChatMessage(role: .assistant, content: "a1"),
            ChatMessage(role: .user, content: "q2"),
        ]
        return (messages, [nil, nil, nil, 0, 1, 2], 2)
    }

    func testEarliestRetainedRowSkipsLeadingSystemSummaryAndPin() {
        let p = projection()
        // The fitter dropped the summary and q1: system, pin, a1, q2 remain.
        let retained = [p.messages[0], p.messages[2], p.messages[4], p.messages[5]]
        let earliest = Planner.earliestRetainedRow(
            projectionMessages: p.messages, sourceRows: p.rows, pinIndex: p.pin,
            boundary: 10, retained: retained)
        XCTAssertEqual(earliest, 11, "rebased row 1 plus the compaction boundary")

        let turns = [
            Planner.Turn(rowIndex: 4, role: .user, text: "old"),
            Planner.Turn(rowIndex: 10, role: .user, text: "q1"),
            Planner.Turn(rowIndex: 11, role: .assistant, text: "a1"),
            Planner.Turn(rowIndex: 12, role: .user, text: "q2"),
        ]
        XCTAssertEqual(Planner.keptTurns(turns, earliestRetainedRow: earliest), 2)
    }

    private func inputs(question: ChatMessage?, budget: Int = 4_000) -> Planner.Inputs {
        let p = projection()
        return Planner.Inputs(
            projectionMessages: p.messages, sourceRows: p.rows, pinIndex: p.pin, boundary: 1,
            turns: [
                Planner.Turn(rowIndex: 0, role: .user, text: "compacted away"),
                Planner.Turn(rowIndex: 1, role: .user, text: "q1"),
                Planner.Turn(rowIndex: 2, role: .assistant, text: "a1"),
                Planner.Turn(rowIndex: 3, role: .user, text: "q2"),
            ],
            question: question, promptBudget: budget)
    }

    func testNothingOutsideThePromptLeavesItByteIdentical() async {
        let p = projection()
        var inputs = inputs(question: p.messages[5])
        inputs.boundary = 0
        inputs.turns = [
            Planner.Turn(rowIndex: 0, role: .user, text: "q1"),
            Planner.Turn(rowIndex: 1, role: .assistant, text: "a1"),
            Planner.Turn(rowIndex: 2, role: .user, text: "q2"),
        ]
        let fitted = Planner.Fit(
            retained: p.messages, measuredTokens: 100, removedTurnCount: 0, hasRoomForGeneration: true)
        let result = await Planner.applyConversationRecall(
            fitted: fitted, inputs: inputs,
            refit: { _ in XCTFail("no refit when nothing is outside"); return nil },
            countTokens: { _ in XCTFail("no count when nothing is outside"); return nil },
            evidence: { _, _ in XCTFail("no evidence request when nothing is outside"); return nil },
            policy: "POLICY")
        XCTAssertEqual(result.retained, fitted.retained)
        XCTAssertEqual(result.measuredTokens, 100)
    }

    func testEvidenceGoesOnThePersonsMessageNotTheSummaryOrPin() async {
        let p = projection()
        let fitted = Planner.Fit(
            retained: p.messages, measuredTokens: 100, removedTurnCount: 0, hasRoomForGeneration: true)
        var askedKept = -1
        let result = await Planner.applyConversationRecall(
            fitted: fitted, inputs: inputs(question: p.messages[5]),
            refit: { _ in XCTFail("headroom is enough"); return nil },
            countTokens: { _ in 300 },
            evidence: { kept, budget in
                askedKept = kept
                XCTAssertLessThanOrEqual(budget, Planner.maxEvidenceTokens)
                return "<evidence id=\"E1\">compacted away</evidence>"
            },
            policy: "POLICY")
        XCTAssertEqual(askedKept, 3, "rows 1...3 are in the prompt; row 0 is behind the boundary")
        XCTAssertEqual(result.retained.count, fitted.retained.count)
        XCTAssertEqual(result.retained[1], p.messages[1], "summary untouched")
        XCTAssertEqual(result.retained[2], p.messages[2], "pin untouched")
        XCTAssertTrue(result.retained[5].content.hasPrefix("POLICY\n\n<evidence"))
        XCTAssertTrue(result.retained[5].content.hasSuffix("\n\nQuestion: q2"))
        XCTAssertEqual(result.measuredTokens, 300)
    }

    func testRecallFallsBackWhenTheFinalPromptWouldNotFit() async {
        let p = projection()
        let fitted = Planner.Fit(
            retained: p.messages, measuredTokens: 100, removedTurnCount: 0, hasRoomForGeneration: true)
        let result = await Planner.applyConversationRecall(
            fitted: fitted, inputs: inputs(question: p.messages[5], budget: 4_000),
            refit: { _ in nil },
            countTokens: { _ in 4_000 },
            evidence: { _, _ in "<evidence>x</evidence>" },
            policy: "POLICY")
        XCTAssertEqual(result.retained, fitted.retained)
    }

    func testRecallRefitsWithTheReserveWhenHeadroomIsShort() async {
        let p = projection()
        // 3,990 of 4,000 used: below the 500-token reserve.
        let fitted = Planner.Fit(
            retained: p.messages, measuredTokens: 3_990, removedTurnCount: 0, hasRoomForGeneration: true)
        var refitBound = 0
        let tighter = Planner.Fit(
            retained: [p.messages[0], p.messages[2], p.messages[4], p.messages[5]],
            measuredTokens: 2_000, removedTurnCount: 2, hasRoomForGeneration: true)
        let result = await Planner.applyConversationRecall(
            fitted: fitted, inputs: inputs(question: p.messages[5], budget: 4_000),
            refit: { bound in refitBound = bound; return tighter },
            countTokens: { _ in 2_400 },
            evidence: { _, _ in "<evidence>x</evidence>" },
            policy: "POLICY")
        XCTAssertEqual(refitBound, 4_000 - 500)
        XCTAssertEqual(result.removedTurnCount, 2)
        XCTAssertTrue(result.retained.last?.content.contains("<evidence>x</evidence>") == true)
    }

    func testNoRecallWhenThePersonsMessageWasDropped() async {
        let p = projection()
        let fitted = Planner.Fit(
            retained: [p.messages[0], p.messages[4]], measuredTokens: 100,
            removedTurnCount: 3, hasRoomForGeneration: true)
        let result = await Planner.applyConversationRecall(
            fitted: fitted, inputs: inputs(question: p.messages[5]),
            refit: { _ in nil }, countTokens: { _ in 1 },
            evidence: { _, _ in XCTFail("no target for the evidence"); return nil },
            policy: "POLICY")
        XCTAssertEqual(result.retained, fitted.retained)
    }

    func testRecallSettingDefaultsOnAndDecodesLeniently() throws {
        XCTAssertTrue(MacAppSettings().conversationRecallEnabled)
        let off = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(MacAppSettings(conversationRecallEnabled: false)))
        XCTAssertFalse(off.conversationRecallEnabled)
        let legacy = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertTrue(legacy.conversationRecallEnabled)
        let wrongType = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(#"{"conversationRecallEnabled":"yes"}"#.utf8))
        XCTAssertTrue(wrongType.conversationRecallEnabled)
    }
}

/// The real syntext 2.7.0 store, in a temporary folder.
final class ChatRecallStoreTests: XCTestCase {
    private var directory: URL!

    override func setUp() {
        super.setUp()
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-recall-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
        super.tearDown()
    }

    private func store(protected: Bool = false) -> ChatRecallStore {
        ChatRecallStore(directory: directory, isProtected: { protected })
    }

    private func turns(_ texts: [String]) -> [ChatRecallPlanner.Turn] {
        texts.enumerated().map { index, text in
            ChatRecallPlanner.Turn(rowIndex: index, role: index.isMultiple(of: 2) ? .user : .assistant, text: text)
        }
    }

    /// Every byte the store wrote, minus the always-empty lock file.
    private func storeBytes() throws -> Data {
        var all = Data()
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            guard url.lastPathComponent != "write.lock",
                  (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
            else { continue }
            all.append(try Data(contentsOf: url))
        }
        return all
    }

    func testReconcileRecordsAndRoundTripsAfterReopen() async {
        let chat = UUID()
        let texts = ["first question", "first answer\nwith two lines", "second question"]
        let recall = store()
        recall.reconcile(chatID: chat, turns: turns(texts))
        await recall.drain()
        var recorded = await recall.recordedTurns(chatID: chat)
        XCTAssertEqual(recorded.map(\.text), texts)

        recall.close()
        await recall.drain()
        let reopened = store()
        recorded = await reopened.recordedTurns(chatID: chat)
        XCTAssertEqual(recorded.map(\.text), texts, "history must match after a relaunch or every launch rebuilds")
        XCTAssertEqual(
            ChatRecallPlanner.reconcilePlan(recorded: recorded, recordable: turns(texts)), .none)
        reopened.close()
        await reopened.drain()
    }

    func testFolderIsOwnerOnly() async throws {
        let recall = store()
        recall.reconcile(chatID: UUID(), turns: turns(["hello there"]))
        await recall.drain()
        let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o700)
        recall.close()
        await recall.drain()
    }

    func testProtectedProfileNeverRecords() async {
        let recall = store(protected: true)
        recall.reconcile(chatID: UUID(), turns: turns(["secret question"]))
        await recall.drain()
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    func testTurningRecallOffDeletesTheStore() async {
        let recall = store()
        recall.reconcile(chatID: UUID(), turns: turns(["something said"]))
        await recall.drain()
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        recall.setEnabled(false)
        await recall.drain()
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        recall.reconcile(chatID: UUID(), turns: turns(["said while off"]))
        await recall.drain()
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    func testEachChatRecallsOnlyItsOwnTurns() async {
        let recall = store()
        let secretChat = UUID()
        let otherChat = UUID()
        let filler = (0..<6).flatMap { ["filler question \($0) about weather", "filler answer \($0) sunny"] }
        recall.reconcile(
            chatID: secretChat,
            turns: turns(["remember the vault codeword", "The vault codeword is PELICANORANGE."] + filler))
        recall.reconcile(chatID: otherChat, turns: turns(filler))
        await recall.drain()

        let path = "\(directory.path)/no-such-model"
        let own = await recall.evidence(
            chatID: secretChat, question: "what is the vault codeword", keptTurns: 0,
            budget: 512, modelPath: path)
        XCTAssertTrue(own?.contains("PELICANORANGE") == true, own ?? "nil")
        let other = await recall.evidence(
            chatID: otherChat, question: "what is the vault codeword", keptTurns: 0,
            budget: 512, modelPath: path)
        XCTAssertFalse(other?.contains("PELICANORANGE") == true, other ?? "nil")
        recall.close()
        await recall.drain()
    }

    func testKeptTurnsAreNotRecalled() async {
        let recall = store()
        let chat = UUID()
        recall.reconcile(chatID: chat, turns: turns(["the launch code is ZEBRAQUARTZ", "noted"]))
        await recall.drain()
        let path = "\(directory.path)/no-such-model"
        let kept = await recall.evidence(
            chatID: chat, question: "what is the launch code", keptTurns: 2, budget: 512, modelPath: path)
        XCTAssertFalse(kept?.contains("ZEBRAQUARTZ") == true, "turns in the prompt are not evidence")
        let outside = await recall.evidence(
            chatID: chat, question: "what is the launch code", keptTurns: 0, budget: 512, modelPath: path)
        XCTAssertTrue(outside?.contains("ZEBRAQUARTZ") == true, outside ?? "nil")
        recall.close()
        await recall.drain()
    }

    func testForgetRemovesTurnsFromRecallAndFromDisk() async throws {
        let recall = store()
        let chat = UUID()
        // Another chat's live turns keep the dead share under syntext's 0.3
        // threshold, so an unforced compaction would leave the bytes; with
        // only the forgotten chat in the store every compaction is due.
        recall.reconcile(
            chatID: UUID(), turns: turns((0..<200).map { "other chat turn \($0) about the weather" }))
        // Past the 32-turn flush, so some turns sit in compacted segments
        // and only a forced compaction takes their bytes off the disk.
        let texts = (0..<40).map { "turn \($0) mentions MARKERWALRUS\($0)" }
        recall.reconcile(chatID: chat, turns: turns(texts))
        await recall.drain()
        XCTAssertNotNil(try storeBytes().range(of: Data("MARKERWALRUS".utf8)))

        recall.forget(chatID: chat)
        await recall.drain()
        await recall.compactNow()
        let recorded = await recall.recordedTurns(chatID: chat)
        XCTAssertTrue(recorded.isEmpty)
        XCTAssertNil(try storeBytes().range(of: Data("MARKERWALRUS".utf8)), "forgotten text left on disk")
        recall.close()
        await recall.drain()
    }

    func testRewriteRebuildsTheJournal() async {
        let recall = store()
        let chat = UUID()
        recall.reconcile(chatID: chat, turns: turns(["original question", "answer"]))
        recall.reconcile(chatID: chat, turns: turns(["edited question"]))
        await recall.drain()
        let recorded = await recall.recordedTurns(chatID: chat)
        XCTAssertEqual(recorded.map(\.text), ["edited question"])
        recall.close()
        await recall.drain()
    }
}

/// The app-level gate: what `executeGenerationTurn` and every reconcile check.
@MainActor
final class ChatRecallEligibilityTests: XCTestCase {
    func testGhostChatsAndTheSettingGateRecall() {
        let model = AppModel()
        model.conversationRecallEnabled = true
        defer { model.conversationRecallEnabled = true }
        // A real row: with an empty test archive, `createChat()` can hand back
        // the placeholder selection, which is not in `chats`.
        let row = AppChat(projectID: nil)
        model.chats.insert(row, at: 0)
        let normal = row.id
        XCTAssertTrue(model.conversationRecallEligible(chatID: normal))

        let ghost = model.enterGhostChat()
        XCTAssertFalse(model.conversationRecallEligible(chatID: ghost), "a Ghost chat must never be journaled")

        model.conversationRecallEnabled = false
        XCTAssertFalse(model.conversationRecallEligible(chatID: normal))
    }
}
