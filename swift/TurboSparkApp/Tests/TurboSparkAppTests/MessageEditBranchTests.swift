import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Message edit / regenerate / branch (`AppModel+MessageEditing.swift`).
///
/// Everything here runs WITHOUT a session, through the static cores the
/// public methods are guards around -- the same split CompactionTests uses.
/// A session would be needed to drive `regenerateResponse` end to end
/// (it re-enters `executeGenerationTurn`, which refuses without one), and
/// what that turn does after the surgery is already covered by the
/// generation lifecycle tests; what is NOT covered elsewhere is the
/// transcript surgery, the variant math, and the archive/vault round trips
/// the new `alternates` field rides on.
@MainActor
final class MessageEditBranchTests: XCTestCase {
    // MARK: - Fixtures

    /// A prompt / response pair with explicit timestamps, so version
    /// ordering is deterministic.
    private func prompt(_ text: String, at time: Date) -> AppChatMessage {
        AppChatMessage(role: .user, content: text, createdAt: time)
    }

    private func response(_ text: String, at time: Date) -> AppChatMessage {
        AppChatMessage(role: .assistant, content: text, createdAt: time)
    }

    private func date(_ seconds: Double) -> Date {
        Date(timeIntervalSinceReferenceDate: 700_000_000 + seconds)
    }

    // MARK: - Archive tolerance (swift/CLAUDE.md Gotcha 13)

    func testAnArchiveWithoutAlternatesStillDecodes() throws {
        // A chats_archive.json written before `alternates` existed carries
        // no such key; the tolerant decoder must read the rows anyway, not
        // quarantine the archive.
        let json = """
        {
          "selectedChatID": "11111111-1111-1111-1111-111111111111",
          "chats": [
            {
              "id": "22222222-2222-2222-2222-222222222222",
              "title": "old chat",
              "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"}
              ]
            }
          ]
        }
        """
        let data = try XCTUnwrap(json.data(using: .utf8))
        let archive = try JSONDecoder().decode(AppChatArchive.self, from: data)
        XCTAssertEqual(archive.chats.count, 1)
        XCTAssertEqual(archive.chats[0].messages.count, 2)
        XCTAssertTrue(archive.chats[0].messages[0].alternates.isEmpty)
        XCTAssertEqual(archive.chats[0].messages[1].content, "hi there")
    }

    // MARK: - Variant math

    func testVersionsSortOldestFirstAndReportPosition() {
        let older = response("v1", at: date(1))
        var active = response("v2", at: date(2))
        active.alternates = [older]

        let versions = AppModel.sortedVersions(of: active)
        XCTAssertEqual(versions.map(\.content), ["v1", "v2"])

        let position = AppModel().variantPosition(of: active)
        XCTAssertEqual(position.position, 2)
        XCTAssertEqual(position.count, 2)
    }

    func testSteppingBackActivatesTheOlderVersionInPlace() {
        let older = response("v1", at: date(1))
        var active = response("v2", at: date(2))
        let rowID = active.id
        active.alternates = [older]

        AppModel.applyVariantStep(&active, delta: -1)

        // The row keeps its identity; only the fields swap.
        XCTAssertEqual(active.id, rowID)
        XCTAssertEqual(active.content, "v1")
        XCTAssertEqual(active.createdAt, date(1))
        XCTAssertEqual(active.alternates.count, 1)
        XCTAssertEqual(active.alternates.first?.content, "v2")
        // The archive stays one level deep: an activated version never
        // carries its own alternates back in.
        XCTAssertEqual(active.alternates.first?.alternates ?? [], [])
        // And the switcher now reads 1 of 2.
        let position = AppModel().variantPosition(of: active)
        XCTAssertEqual(position.position, 1)
    }

    func testSteppingPastEitherEndIsANoOp() {
        let only = response("only", at: date(1))
        var single = only
        AppModel.applyVariantStep(&single, delta: -1)
        AppModel.applyVariantStep(&single, delta: 1)
        XCTAssertEqual(single.content, "only")
        XCTAssertTrue(single.alternates.isEmpty)

        let older = response("v1", at: date(1))
        var newest = response("v2", at: date(2))
        newest.alternates = [older]
        AppModel.applyVariantStep(&newest, delta: 1)
        XCTAssertEqual(newest.content, "v2")
        AppModel.applyVariantStep(&newest, delta: -1)
        AppModel.applyVariantStep(&newest, delta: -1)
        XCTAssertEqual(newest.content, "v1", "a second step back stops at the oldest version")
    }

    // MARK: - Anchor detection

    func testAToolResultRowIsNotAPromptAnchor() {
        let messages = [
            prompt("first", at: date(1)),
            response("ok", at: date(2)),
            prompt("second", at: date(3)),
            // A tool execution turn is a .user row carrying results, and it
            // is the LAST .user row here on purpose: a real prompt after it
            // would make lastIndex pick the prompt with the filter deleted
            // too, and the test would survive its own mutation.
            AppChatMessage(
                role: .user, content: "", toolResults: [
                    AppToolResult(callID: UUID(), output: "out", isError: false, durationSeconds: 0),
                ], createdAt: date(4)),
        ]
        XCTAssertEqual(AppModel.lastPromptAnchorIndex(in: messages), 2)
    }

    func testAQuickSaveRowIsNotAPromptAnchor() {
        let messages = [
            prompt("real question", at: date(1)),
            prompt(UserMemoryInputMessage.wrapping("note to self"), at: date(2)),
        ]
        // The quick-save rides the transcript as a user row, but retrying
        // against it would ask the model to answer a memory note.
        XCTAssertEqual(AppModel.lastPromptAnchorIndex(in: messages), 0)
    }

    // MARK: - Retry

    func testRetryTruncatesToThePromptAndKeepsTheOldReplyAsAVariant() throws {
        let messages = [
            prompt("q1", at: date(1)),
            response("a1", at: date(2)),
            prompt("q2", at: date(3)),
            response("a2", at: date(4)),
        ]
        let applied = try XCTUnwrap(AppModel.retryApplied(to: messages))
        XCTAssertEqual(applied.messages.map(\.content), ["q1", "a1", "q2"])
        XCTAssertTrue(
            applied.messages.last?.alternates.isEmpty ?? false,
            "the old reply does not hang off the prompt; it is seeded onto the regenerated one")
        XCTAssertEqual(applied.variants.map(\.content), ["a2"])
    }

    func testRetryRefusesAChainThatIsNotSingleProse() {
        // No anchor at all.
        XCTAssertNil(AppModel.retryApplied(to: [response("orphan", at: date(1))]))

        // A response that proposed a tool call: re-rolling agent work is
        // not Retry v1.
        let withCall = AppChatMessage(
            role: .assistant, content: "", toolCalls: [
                AppToolCall(name: "list_directory", arguments: [:], category: .fileRead),
            ], createdAt: date(2))
        XCTAssertNil(AppModel.retryApplied(to: [prompt("q", at: date(1)), withCall]))

        // A chain that already carries the executed tool turn.
        let toolTurn = AppChatMessage(
            role: .user, content: "", toolResults: [
                AppToolResult(callID: UUID(), output: "out", isError: false, durationSeconds: 0),
            ], createdAt: date(3))
        XCTAssertNil(
            AppModel.retryApplied(to: [prompt("q", at: date(1)), withCall, toolTurn]))

        XCTAssertFalse(
            AppModel.isSingleProseResponse([response("a", at: date(1)), response("b", at: date(2))]))
    }

    // MARK: - Edit

    func testAnEditPushesTheOldPromptAndTruncatesTheChain() {
        let messages = [
            prompt("q1", at: date(1)),
            response("a1", at: date(2)),
            prompt("q2 with a typo", at: date(3)),
            response("a2", at: date(4)),
        ]
        let now = date(50)
        let applied = AppModel.editApplied(
            to: messages, anchorIndex: 2, newText: "q2 corrected", now: now)

        XCTAssertEqual(applied.messages.count, 3, "the response chain after the prompt goes")
        let edited = applied.messages[2]
        XCTAssertEqual(edited.content, "q2 corrected")
        XCTAssertEqual(edited.createdAt, now, "the edit sorts newest")
        XCTAssertEqual(edited.alternates.count, 1)
        XCTAssertEqual(edited.alternates.first?.content, "q2 with a typo")
        XCTAssertEqual(edited.alternates.first?.createdAt, date(3), "the old prompt keeps its own timestamp")
        XCTAssertEqual(applied.responseVariants.map(\.content), ["a2"])
    }

    func testAnEditKeepsNoResponseVariantBehindAToolChain() {
        let messages = [
            prompt("q", at: date(1)),
            AppChatMessage(
                role: .assistant, content: "", toolCalls: [
                    AppToolCall(name: "list_directory", arguments: [:], category: .fileRead),
                ], createdAt: date(2)),
        ]
        let applied = AppModel.editApplied(
            to: messages, anchorIndex: 0, newText: "q2", now: date(50))
        XCTAssertTrue(applied.responseVariants.isEmpty, "a replaced tool chain is not a prose variant")
        XCTAssertEqual(applied.messages.count, 1)
    }

    // MARK: - Branch

    func testABranchCarriesThePrefixClampsTheBoundaryAndLeavesTheSourceUntouched() {
        var source = AppChat(projectID: nil, title: "long talk")
        source.systemPrompt = "be brief"
        source.contextSummary = "summary of early rows"
        source.compactedMessageCount = 2
        let messages = [
            prompt("q0", at: date(1)),
            response("a0", at: date(2)),
            prompt("q1", at: date(3)),
            response("a1", at: date(4)),
            prompt("q2", at: date(5)),
        ]
        source.messages = messages

        let branch = AppModel.branchedChat(
            from: source, messageIndex: 4, messages: messages, newText: "q2 edited",
            summary: source.contextSummary, boundary: 2, now: date(60))

        XCTAssertEqual(branch.title, "long talk (branch)")
        XCTAssertEqual(branch.systemPrompt, "be brief")
        XCTAssertEqual(branch.messages.count, 5)
        XCTAssertEqual(branch.messages[4].content, "q2 edited")
        XCTAssertTrue(branch.messages[4].alternates.isEmpty, "the branch starts with a clean history")
        XCTAssertEqual(branch.compactedMessageCount, 2, "the summary still covers the same rows")
        XCTAssertNotNil(branch.contextSummary)

        // And the source row is untouched by construction: the branch is a
        // new AppChat, so assert the copy is not the source.
        XCTAssertNotEqual(branch.id, source.id)
        XCTAssertEqual(source.messages[4].content, "q2")

        // A boundary past the edited row comes down to it, or the branch's
        // own re-run would send a prompt the summary already replaced.
        let clamped = AppModel.branchedChat(
            from: source, messageIndex: 1, messages: messages, newText: "q0 rewritten",
            summary: source.contextSummary, boundary: 2, now: date(60))
        XCTAssertEqual(clamped.compactedMessageCount, 1)
    }

    // MARK: - Compaction clamp

    func testCompactionBoundaryClampsToThePromptRow() throws {
        let model = AppModel()
        var chat = AppChat(title: "talk")
        chat.messages = (0..<5).map { prompt("q\($0)", at: date(Double($0))) }
        chat.contextSummary = "early rows"
        chat.compactedMessageCount = 4
        model.chats = [chat]

        model.clampStoredCompaction(chatID: chat.id, toRow: 2)

        let clamped = try XCTUnwrap(model.chats.first)
        XCTAssertEqual(clamped.compactedMessageCount, 2)
        XCTAssertEqual(clamped.contextSummary, "early rows", "the summary text is kept for the rows it still covers")

        // A boundary already at or below the row is a no-op, and one that
        // never existed stays absent.
        model.clampStoredCompaction(chatID: chat.id, toRow: 3)
        XCTAssertEqual(model.chats.first?.compactedMessageCount, 2)
    }

    // MARK: - Vault round trip

    func testAlternatesSurviveTheGhostVaultRoundTrip() {
        let model = AppModel()
        let ghostID = model.enterGhostChat()
        defer {
            model.ghostVault.wipe(for: ghostID)
        }
        let withVariants = AppChatMessage(
            role: .assistant, content: "current", alternates: [response("earlier", at: date(1))],
            createdAt: date(2))

        model.mutateTurnMessages(for: ghostID) { $0.append(withVariants) }

        let roundTripped = model.turnMessages(for: ghostID)
        XCTAssertEqual(roundTripped.count, 1)
        XCTAssertEqual(roundTripped[0].content, "current")
        XCTAssertEqual(roundTripped[0].alternates.map(\.content), ["earlier"])
    }

    // MARK: - Lifecycle guards

    func testEditingAffordancesRefuseWhileATurnRuns() {
        let model = AppModel()
        let messages = [
            prompt("q", at: date(1)),
            response("a", at: date(2)),
        ]
        var chat = AppChat(title: "talk")
        chat.messages = messages
        model.chats = [chat]

        model.generating = true
        XCTAssertFalse(model.canRetry(response: messages[1]))
        XCTAssertFalse(model.canEditInPlace(messages[0]))
        XCTAssertFalse(model.beginEdit(messageID: messages[0].id))
        XCTAssertNil(model.editingMessageID)

        // Stepping a variant mid-turn is refused too, and leaves the row.
        let varianted = AppChatMessage(
            role: .assistant, content: "v2", alternates: [response("v1", at: date(3))],
            createdAt: date(4))
        model.mutateTurnMessages(for: chat.id) { $0 = [messages[0], varianted] }
        model.stepVariant(of: varianted.id, delta: -1)
        XCTAssertEqual(model.turnMessages(for: chat.id)[1].content, "v2")
    }

    func testEditingAffordancesRefuseWithoutASession() {
        let model = AppModel()
        let userMessage = prompt("q", at: date(1))
        var chat = AppChat(title: "talk")
        chat.messages = [userMessage, response("a", at: date(2))]
        model.chats = [chat]

        // No model is loaded in this test; every affordance that ends in a
        // re-run hides, and the mutating entry points refuse.
        XCTAssertFalse(model.canRetry(response: chat.messages[1]))
        XCTAssertFalse(model.canEditInPlace(userMessage))
        XCTAssertFalse(model.canBranch(userMessage))
        XCTAssertFalse(model.beginEdit(messageID: userMessage.id))
        XCTAssertFalse(model.regenerateResponse())
        XCTAssertFalse(model.commitEdit(messageID: userMessage.id, newText: "rewritten"))
        XCTAssertNil(model.branchFrom(messageID: userMessage.id, editedText: "rewritten"))
        XCTAssertEqual(model.turnMessages(for: chat.id).count, 2, "refusals leave the transcript alone")
    }
}
