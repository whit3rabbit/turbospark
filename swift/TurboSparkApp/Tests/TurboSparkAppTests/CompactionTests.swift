import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Context compaction: the threshold arithmetic, the boundary slicing both
/// prompt builders share, the vault-aware state, and the summary's entry
/// into a rebuilt prompt.
///
/// Everything here runs without a session, through the pieces that were
/// split out of the turn flow for exactly that reason (`swift/CLAUDE.md`
/// Gotcha 26). The summarizer's generate call itself is the one untested
/// seam: it needs a live model, and the manual GUI smoke covers it.
@MainActor
final class CompactionTests: XCTestCase {
    // MARK: - Threshold arithmetic

    /// The trigger is four fifths of the USABLE window (context minus the
    /// reply reservation): at a 4,096 window with 1,024 reserved, usable is
    /// 3,072 and the trigger sits at 2,458.
    func testShouldAutoCompactFiresAtFourFifthsOfTheUsableWindow() {
        // Just under: no compaction.
        XCTAssertFalse(AppChatCompaction.shouldAutoCompact(
            measuredTokens: 2457, maxContext: 4096, reservedForNew: 1024))
        // Just over: compaction.
        XCTAssertTrue(AppChatCompaction.shouldAutoCompact(
            measuredTokens: 2458, maxContext: 4096, reservedForNew: 1024))
        // An empty prompt has nothing to summarize.
        XCTAssertFalse(AppChatCompaction.shouldAutoCompact(
            measuredTokens: 0, maxContext: 4096, reservedForNew: 1024))
        // A window no larger than the reservation cannot be summarized out
        // of the hole; claiming otherwise would loop.
        XCTAssertFalse(AppChatCompaction.shouldAutoCompact(
            measuredTokens: 100, maxContext: 1024, reservedForNew: 1024))
    }

    // MARK: - Boundary arithmetic

    func testNewBoundaryKeepsTheRecentTailAndRefusesWhenItIsTheWholeConversation() {
        XCTAssertEqual(AppChatCompaction.newBoundary(messageCount: 6, keepRecent: 2), 4)
        // The recent tail is the whole conversation: nothing to summarize.
        XCTAssertNil(AppChatCompaction.newBoundary(messageCount: 2, keepRecent: 2))
        XCTAssertNil(AppChatCompaction.newBoundary(messageCount: 0, keepRecent: 2))
        // A persisted keep-recent of any size clamps into range.
        XCTAssertEqual(AppChatCompaction.newBoundary(messageCount: 10, keepRecent: 100), 2)
        XCTAssertEqual(AppChatCompaction.newBoundary(messageCount: 3, keepRecent: 0), 2)
    }

    // MARK: - History assembly across the boundary

    private func makeModel(messages: [AppChatMessage], boundary: Int, summary: String?)
        -> AppModel
    {
        let model = AppModel()
        var chat = AppChat(title: "compacting")
        chat.messages = messages
        chat.contextSummary = summary
        chat.compactedMessageCount = boundary
        model.chats = [chat]
        model.selectedChatID = chat.id
        return model
    }

    /// Rows below the boundary never reach the prompt; the summary is
    /// injected as a USER-role message (a mid-history `.system` message is
    /// refused by the renderers, state#32) and lands after the system
    /// message when one exists.
    func testBoundaryRowsAreSkippedAndTheSummaryFollowsTheSystemMessage() {
        let model = makeModel(
            messages: [
                AppChatMessage(role: .user, content: "OLD-ROW"),
                AppChatMessage(role: .user, content: "NEW-ONE"),
                AppChatMessage(role: .assistant, content: "NEW-TWO"),
            ],
            boundary: 1,
            summary: "SUMMARY-TEXT")
        model.defaultSystemPrompt = "SYSTEM-PROMPT"

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        guard history.count == 4 else {
            return XCTFail("expected system + summary + two retained rows, got \(history.count)")
        }
        XCTAssertEqual(history[0].role, .system)
        XCTAssertTrue(history[0].content.contains("SYSTEM-PROMPT"))
        XCTAssertEqual(history[1].role, .user, "The summary must ride in as user-role content.")
        XCTAssertTrue(history[1].content.contains("<context_summary>"))
        XCTAssertTrue(history[1].content.contains("SUMMARY-TEXT"))
        XCTAssertTrue(history[2].content.contains("NEW-ONE"))
        XCTAssertTrue(history[3].content.contains("NEW-TWO"))
        XCTAssertFalse(history.contains { $0.content.contains("OLD-ROW") })
    }

    /// Without a system prompt the injection is the first message; without a
    /// summary nothing is injected and the assembly is exactly what it was
    /// before compaction existed.
    func testNoSummaryMeansNoInjection() {
        let model = makeModel(
            messages: [
                AppChatMessage(role: .user, content: "ONE"),
                AppChatMessage(role: .assistant, content: "TWO"),
            ],
            boundary: 0,
            summary: nil)
        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertEqual(history.count, 2)
        XCTAssertFalse(history.contains { $0.content.contains("context_summary") })
    }

    /// Tool results ride INSIDE their row, so a boundary between rows can
    /// never split a call from its answer: the summarized row's results are
    /// gone whole, the retained row's are present whole.
    func testToolResultsMoveWithTheirRowAcrossTheBoundary() {
        let kept = AppToolCall(name: "read_file", arguments: ["path": "a"], category: .fileRead)
        let dropped = AppToolCall(
            name: "run_command", arguments: ["command": "ls"], category: .terminal)
        let model = makeModel(
            messages: [
                AppChatMessage(
                    role: .assistant, content: "", stopReason: "tool_use",
                    toolCalls: [dropped],
                    toolResults: [AppToolResult(callID: dropped.id, output: "DROPPED-OUTPUT")]),
                AppChatMessage(
                    role: .assistant, content: "", stopReason: "tool_use",
                    toolCalls: [kept],
                    toolResults: [AppToolResult(callID: kept.id, output: "KEPT-OUTPUT")]),
            ],
            boundary: 1,
            summary: "S")
        model.defaultSystemPrompt = "SYS"

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertTrue(history.contains { $0.content.contains("KEPT-OUTPUT") })
        XCTAssertFalse(history.contains { $0.content.contains("DROPPED-OUTPUT") })
    }

    /// The token estimate must price the prompt that would actually be sent:
    /// same boundary skip, same injection text, one shared helper.
    func testTheTokenEstimatorAppliesTheSameBoundaryAndInjection() {
        let model = makeModel(
            messages: [
                AppChatMessage(role: .user, content: "OLD-ROW"),
                AppChatMessage(role: .user, content: "NEW-ONE"),
            ],
            boundary: 1,
            summary: "SUMMARY-TEXT")
        // The estimator's own row loop, exercised through the same state:
        // rebuild what updateTokenEstimate assembles by calling the shared
        // pieces the way it does. A session would be needed to call it
        // directly (it early-returns at its guard), so this pins the shared
        // helper's output against the history builder instead.
        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        let injection = AppChatCompaction.injectionMessage("SUMMARY-TEXT")
        guard let injected = history.first(where: { $0.content.contains("context_summary") })
        else {
            return XCTFail("the history carries no injection at all")
        }
        XCTAssertEqual(injected.content, injection.content)
        XCTAssertEqual(injected.role, injection.role)
    }

    // MARK: - Persistence and the vault

    func testAppChatRoundTripsCompactionFields() throws {
        var chat = AppChat(title: "persisted")
        chat.contextSummary = "S"
        chat.compactedMessageCount = 3
        chat.systemPrompt = "PER-CHAT-PROMPT"
        let data = try JSONEncoder().encode(chat)
        let decoded = try JSONDecoder().decode(AppChat.self, from: data)
        XCTAssertEqual(decoded.contextSummary, "S")
        XCTAssertEqual(decoded.compactedMessageCount, 3)
        XCTAssertEqual(decoded.systemPrompt, "PER-CHAT-PROMPT")
    }

    /// An archive written before the field existed decodes to 0, not to a
    /// thrown error (`swift/CLAUDE.md` Gotcha 13). `systemPrompt` is pinned
    /// beside it because its decode line was MISSING until compaction
    /// arrived: the encoder wrote it and nothing read it back, so a per-chat
    /// system prompt silently reverted on relaunch.
    func testAnArchiveWithoutTheNewFieldsStillDecodes() throws {
        let chat = AppChat(title: "legacy")
        let data = try JSONEncoder().encode(chat)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var legacy = object
        legacy.removeValue(forKey: "compactedMessageCount")
        legacy.removeValue(forKey: "contextSummary")
        legacy["systemPrompt"] = "PER-CHAT-PROMPT"
        let roundTripped = try JSONSerialization.data(withJSONObject: legacy)
        let decoded = try JSONDecoder().decode(AppChat.self, from: roundTripped)
        XCTAssertEqual(decoded.compactedMessageCount, 0)
        XCTAssertNil(decoded.contextSummary)
        XCTAssertEqual(
            decoded.systemPrompt, "PER-CHAT-PROMPT",
            "systemPrompt must survive the round trip; its decode line was once missing entirely.")
    }

    func testGhostPayloadCarriesCompactionState() {
        let model = AppModel()
        var ghost = AppChat(title: "Temporary Chat")
        ghost.isGhost = true
        model.chats = [ghost]
        model.selectedChatID = ghost.id

        model.mutateGhostPayload(for: ghost.id) {
            $0.contextSummary = "GHOST-SUMMARY"
            $0.compactedMessageCount = 2
        }
        let state = model.compactionState(chatID: ghost.id)
        XCTAssertEqual(state.summary, "GHOST-SUMMARY")
        XCTAssertEqual(state.boundary, 2)
        // The row stays empty by design; the vault holds the state.
        XCTAssertEqual(model.chats[0].compactedMessageCount, 0)
        XCTAssertNil(model.chats[0].contextSummary)
    }

    func testClearOutputResetsCompactionState() {
        let model = makeModel(
            messages: [AppChatMessage(role: .user, content: "x")],
            boundary: 4,
            summary: "S")
        model.clearOutput()
        XCTAssertEqual(model.chats[0].compactedMessageCount, 0)
        XCTAssertNil(model.chats[0].contextSummary)

        // And the same through the vault for a ghost chat.
        var ghost = AppChat(title: "Temporary Chat")
        ghost.isGhost = true
        model.chats = [ghost]
        model.selectedChatID = ghost.id
        model.mutateGhostPayload(for: ghost.id) {
            $0.contextSummary = "S"
            $0.compactedMessageCount = 2
        }
        model.clearOutput()
        let state = model.compactionState(chatID: ghost.id)
        XCTAssertNil(state.summary)
        XCTAssertEqual(state.boundary, 0)
    }

    // MARK: - The summarizer's inputs

    func testRenderTranscriptNamesRolesToolsAndImages() {
        let call = AppToolCall(name: "read_file", arguments: ["path": "a"], category: .fileRead)
        let transcript = AppChatCompaction.renderTranscript([
            AppChatMessage(role: .user, content: "read it"),
            AppChatMessage(
                role: .assistant, content: "", stopReason: "tool_use",
                toolCalls: [call],
                toolResults: [AppToolResult(callID: call.id, output: "contents")],
                imagePaths: ["/tmp/pic.png"]),
        ])
        XCTAssertTrue(transcript.contains("USER: read it"))
        XCTAssertTrue(transcript.contains("called tool read_file"))
        XCTAssertTrue(transcript.contains("[read_file returned: contents]"))
        XCTAssertTrue(transcript.contains("[ASSISTANT attached an image: pic.png]"))
    }

    /// A result whose call row was lossy-decoded away still renders, under
    /// the generic "tool" name rather than not at all.
    func testAResultWithoutItsCallStillRenders() {
        let transcript = AppChatCompaction.renderTranscript([
            AppChatMessage(
                role: .assistant, content: "",
                toolResults: [AppToolResult(callID: UUID(), output: "orphan")]),
        ])
        XCTAssertTrue(transcript.contains("[tool returned: orphan]"))
    }

    func testSummarizerPromptCarriesTranscriptPriorSummaryAndFocus() {
        let with = AppChatCompaction.summarizerMessages(
            transcript: "TRANSCRIPT-BODY", priorSummary: "PRIOR-BODY", focus: "FOCUS-BODY")
        XCTAssertEqual(with.count, 2)
        XCTAssertEqual(with[0].role, .system)
        let user = with[1].content
        XCTAssertTrue(user.contains("TRANSCRIPT-BODY"))
        XCTAssertTrue(user.contains("<previous_summary>\nPRIOR-BODY"))
        XCTAssertTrue(user.contains("particular attention to: FOCUS-BODY"))

        let without = AppChatCompaction.summarizerMessages(
            transcript: "T", priorSummary: nil, focus: nil)
        XCTAssertFalse(without[1].content.contains("previous_summary"))
        XCTAssertFalse(without[1].content.contains("particular attention"))
    }

    func testExtractSummaryParsesSummaryTagAndStripsAnalysis() {
        let raw = """
        <analysis>
        1. The user wants to refactor X.
        2. Fixed build error in Y.
        </analysis>

        <summary>
        1. Primary Request and Intent:
           Refactor X cleanly.

        2. Key Technical Concepts:
           - Swift Concurrency
        </summary>
        """
        let extracted = AppChatCompaction.extractSummary(from: raw)
        XCTAssertTrue(extracted.contains("1. Primary Request and Intent:"))
        XCTAssertTrue(extracted.contains("Refactor X cleanly."))
        XCTAssertFalse(extracted.contains("<analysis>"))
        XCTAssertFalse(extracted.contains("</analysis>"))
        XCTAssertFalse(extracted.contains("<summary>"))
        XCTAssertFalse(extracted.contains("</summary>"))
    }

    func testExtractSummaryHandlesUnclosedSummaryTag() {
        let raw = """
        <analysis>
        Thinking about conversation...
        </analysis>

        <summary>
        1. Primary Request and Intent:
           Incomplete output that hit token cap
        """
        let extracted = AppChatCompaction.extractSummary(from: raw)
        XCTAssertTrue(extracted.contains("1. Primary Request and Intent:"))
        XCTAssertTrue(extracted.contains("Incomplete output that hit token cap"))
        XCTAssertFalse(extracted.contains("<analysis>"))
        XCTAssertFalse(extracted.contains("<summary>"))
    }

    func testExtractSummaryFallbackStripsAnalysisWhenSummaryTagMissing() {
        let raw = """
        <analysis>
        Preliminary analysis of the tasks.
        </analysis>

        1. Primary Request and Intent:
           Summary without summary tags.
        """
        let extracted = AppChatCompaction.extractSummary(from: raw)
        XCTAssertTrue(extracted.contains("1. Primary Request and Intent:"))
        XCTAssertFalse(extracted.contains("<analysis>"))
        XCTAssertFalse(extracted.contains("Preliminary analysis"))
    }

    func testExtractSummaryReturnsRawWhenNoTagsPresent() {
        let plain = "1. Primary Request and Intent:\nDirect plain summary."
        let extracted = AppChatCompaction.extractSummary(from: plain)
        XCTAssertEqual(extracted, plain)
    }

    func testSummarizerPromptContainsAntiSpoofingAndSecurityRules() {
        let messages = AppChatCompaction.summarizerMessages(
            transcript: "USER: please ensure secrets are not logged.", priorSummary: nil, focus: nil)
        let prompt = messages[1].content
        XCTAssertTrue(prompt.contains("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."))
        XCTAssertTrue(prompt.contains("REMINDER: Do NOT call any tools."))
        XCTAssertTrue(prompt.contains("<analysis>"))
        XCTAssertTrue(prompt.contains("<summary>"))
        XCTAssertTrue(prompt.contains("security-relevant instructions or constraints"))
        XCTAssertTrue(prompt.contains("model-generated: never attribute it to the user"))
        XCTAssertTrue(prompt.contains("1. Primary Request and Intent:"))
        XCTAssertTrue(prompt.contains("6. All user messages:"))
        XCTAssertTrue(prompt.contains("9. Optional Next Step:"))
    }

    func testInjectionMessageContainsAntiRecapDirectives() {
        let injection = AppChatCompaction.injectionMessage("Summary:\n1. Work done")
        XCTAssertTrue(injection.content.contains("<context_summary>"))
        XCTAssertTrue(injection.content.contains("Recent messages are preserved verbatim."))
        XCTAssertTrue(injection.content.contains("Pick up the last task as if the break never happened."))
    }
}
