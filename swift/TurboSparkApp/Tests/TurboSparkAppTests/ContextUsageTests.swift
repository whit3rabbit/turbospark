import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// The context ring's model layer: the tint tiers, the estimate arithmetic
/// the ring reads, the sectioned system-prompt builder the breakdown
/// slices, the piece assembly shared with the exact estimate, and the
/// compaction trigger the popover names.
///
/// Everything here runs without a session, through the pieces that were
/// split out of the turn flow for exactly that reason. The raw-string
/// counting itself is the one untested seam: it needs a live tokenizer, and
/// the manual GUI smoke covers it.
@MainActor
final class ContextUsageTests: XCTestCase {
    // MARK: - Tint tiers

    private func tintName(_ fraction: Double) -> String {
        let tint = ContextUsageSummary.tint(forFraction: fraction)
        if tint == .red { return "red" }
        if tint == .yellow { return "yellow" }
        return "green"
    }

    /// Green while comfortable, yellow from 70 percent, red from 90.
    func testTintTiersFlipAtSeventyAndNinetyPercent() {
        XCTAssertEqual(tintName(0.0), "green")
        XCTAssertEqual(tintName(0.69), "green")
        XCTAssertEqual(tintName(0.7), "yellow")
        XCTAssertEqual(tintName(0.89), "yellow")
        XCTAssertEqual(tintName(0.9), "red")
        XCTAssertEqual(tintName(1.0), "red")
    }

    // MARK: - Headline arithmetic

    /// The exact count wins when present; the rows' sum is only the
    /// fallback before the first estimate lands.
    func testUsedTokensPrefersTheExactMeasurement() {
        let rows = [
            ContextUsageComponent(kind: .system, label: "System prompt", tokens: 300),
            ContextUsageComponent(kind: .conversation, label: "Conversation", tokens: 600),
        ]
        let exact = ContextUsageSummary(
            components: rows, measuredTokens: 500,
            windowTokens: 4096, reservedForNewTokens: 1024, autoCompactEnabled: true)
        XCTAssertEqual(exact.usedTokens, 500)
        let approximated = ContextUsageSummary(
            components: rows, measuredTokens: nil,
            windowTokens: 4096, reservedForNewTokens: 1024, autoCompactEnabled: true)
        XCTAssertEqual(approximated.usedTokens, 900)
    }

    /// Free space and the fill fraction clamp instead of going negative or
    /// past full: an over-budget prompt must read 100 percent, not 130.
    func testFreeSpaceAndFractionClampAtTheWindow() {
        let over = ContextUsageSummary(
            components: [], measuredTokens: 5000,
            windowTokens: 4096, reservedForNewTokens: 1024, autoCompactEnabled: true)
        XCTAssertEqual(over.freeTokens, 0)
        XCTAssertEqual(over.fraction, 1.0)
    }

    // MARK: - Compaction trigger

    /// The popover's trigger is the same threshold the engine acts on:
    /// ceiling division, so a prompt AT the trigger satisfies
    /// `shouldAutoCompact` and one token under it does not.
    func testCompactionTriggerMatchesTheEngineThreshold() {
        let summary = ContextUsageSummary(
            components: [], measuredTokens: nil,
            windowTokens: 4096, reservedForNewTokens: 1024, autoCompactEnabled: true)
        XCTAssertEqual(summary.compactionTriggerTokens, 2458)
        XCTAssertTrue(AppChatCompaction.shouldAutoCompact(
            measuredTokens: summary.compactionTriggerTokens!,
            maxContext: 4096, reservedForNew: 1024))
        XCTAssertFalse(AppChatCompaction.shouldAutoCompact(
            measuredTokens: summary.compactionTriggerTokens! - 1,
            maxContext: 4096, reservedForNew: 1024))
    }

    /// A window no larger than the reply reservation has no trigger, and
    /// the flag rides through for the popover's footer line.
    func testTriggerIsNilWhenTheReservationLeavesNothingUsable() {
        let summary = ContextUsageSummary(
            components: [], measuredTokens: nil,
            windowTokens: 1024, reservedForNewTokens: 1024, autoCompactEnabled: true)
        XCTAssertNil(summary.compactionTriggerTokens)
        XCTAssertTrue(summary.autoCompactEnabled)
        let disabled = ContextUsageSummary(
            components: [], measuredTokens: nil,
            windowTokens: 4096, reservedForNewTokens: 1024, autoCompactEnabled: false)
        XCTAssertFalse(disabled.autoCompactEnabled)
    }

    // MARK: - Sectioned system prompt

    /// The full prompt IS the join of the sections: one builder, so the
    /// breakdown's slices can never describe different text than what is
    /// sent. Both the projectless and project arms.
    func testSystemPromptIsTheJoinOfItsSections() {
        let model = AppModel()

        let bare = model.buildSystemPrompt(for: nil, userPrompt: "USER-PROMPT")
        let bareSections = model.buildSystemPromptSections(for: nil, userPrompt: "USER-PROMPT")
        XCTAssertEqual(bareSections.map(\.content), ["USER-PROMPT"])
        XCTAssertEqual(bare, bareSections.map(\.content).joined(separator: "\n\n"))

        let project = AppProject(
            name: "Demo",
            rootDirectoryPath: "/path/to/demo",
            customInstructions: "RULE-TEXT")
        let full = model.buildSystemPrompt(for: project, userPrompt: "USER-PROMPT")
        let fullSections = model.buildSystemPromptSections(for: project, userPrompt: "USER-PROMPT")
        XCTAssertTrue(fullSections.contains { $0.section == .userPrompt })
        XCTAssertTrue(fullSections.contains { $0.section == .agentPrompt })
        XCTAssertTrue(fullSections.contains { $0.section == .workspace })
        XCTAssertTrue(fullSections.contains { $0.section == .projectRules })
        XCTAssertTrue(fullSections.contains { $0.section == .tools })
        XCTAssertEqual(full, fullSections.map(\.content).joined(separator: "\n\n"))
    }

    // MARK: - Piece assembly

    private func makeModel(
        messages: [AppChatMessage], boundary: Int, summary: String?
    ) -> AppModel {
        let model = AppModel()
        var chat = AppChat(title: "usage")
        chat.messages = messages
        chat.contextSummary = summary
        chat.compactedMessageCount = boundary
        model.chats = [chat]
        model.selectedChatID = chat.id
        return model
    }

    /// The pieces are the same conversation the exact count sees: system
    /// prompt (workspace included), boundary rows skipped, the raw summary,
    /// and the draft. `buildEstimateParts` is ONE builder with two
    /// consumers, so a piece that drifts from the history is a bug here.
    func testEstimatePartsCarriesTheSameConversationAsTheHistory() {
        let model = makeModel(
            messages: [
                AppChatMessage(role: .user, content: "OLD-ROW"),
                AppChatMessage(role: .user, content: "NEW-ONE"),
                AppChatMessage(role: .assistant, content: "NEW-TWO"),
            ],
            boundary: 1,
            summary: "SUMMARY-TEXT")
        model.defaultSystemPrompt = "SYS-PROMPT"
        model.promptText = "DRAFT-TEXT"

        let parts = model.buildEstimateParts()

        // History: system, summary injection, two retained rows, draft.
        XCTAssertEqual(parts.history.count, 5)
        XCTAssertEqual(parts.history[0].role, .system)
        XCTAssertTrue(parts.history[0].content.contains("SYS-PROMPT"))
        XCTAssertTrue(parts.history[1].content.contains("SUMMARY-TEXT"))
        XCTAssertFalse(parts.history.contains { $0.content.contains("OLD-ROW") })
        XCTAssertTrue(parts.history.last!.content.contains("DRAFT-TEXT"))

        // Pieces: every labeled slice traces to its own content, and the
        // summary piece carries the RAW summary (the wrapper is the
        // history's business).
        let byLabel = Dictionary(uniqueKeysWithValues: parts.pieces.map { ($0.label, $0) })
        XCTAssertEqual(byLabel["System prompt"]?.content, "SYS-PROMPT")
        XCTAssertEqual(byLabel["System prompt"]?.kind, .system)
        XCTAssertTrue(byLabel["Conversation"]!.content.contains("NEW-ONE"))
        XCTAssertTrue(byLabel["Conversation"]!.content.contains("NEW-TWO"))
        XCTAssertFalse(byLabel["Conversation"]!.content.contains("OLD-ROW"))
        XCTAssertEqual(byLabel["Compaction summary"]?.content, "SUMMARY-TEXT")
        XCTAssertEqual(byLabel["Compaction summary"]?.kind, .summary)
        XCTAssertEqual(byLabel["Draft"]?.content, "DRAFT-TEXT")
        XCTAssertEqual(byLabel["Draft"]?.kind, .draft)
    }

    /// The project-derived sections land in the piece grouping under their
    /// own labels, so a section dropped from a group reddens here rather
    /// than silently vanishing from the popover.
    func testProjectSectionsAreGroupedIntoLabeledPieces() {
        let model = AppModel()
        model.interactionMode = .projects
        let project = AppProject(
            name: "Demo",
            rootDirectoryPath: "/path/to/demo",
            customInstructions: "RULE-TEXT")
        var chat = AppChat(title: "usage")
        chat.projectID = project.id
        model.projects = [project]
        model.chats = [chat]
        model.selectedChatID = chat.id

        let parts = model.buildEstimateParts()
        let byLabel = Dictionary(uniqueKeysWithValues: parts.pieces.map { ($0.label, $0) })
        XCTAssertTrue(
            byLabel["System prompt"]!.content.contains("## Workspace Environment"),
            "the workspace section must ride in the system-prompt piece")
        XCTAssertTrue(byLabel["System prompt"]!.content.contains("RULE-TEXT"))
        XCTAssertFalse(byLabel["Tools & skills"]!.content.contains("RULE-TEXT"))
        XCTAssertEqual(byLabel["Tools & skills"]?.kind, .tools)
    }

    // MARK: - Headline estimate

    /// The transcript's chars/4 term is gone: `estimatedPromptTokens` is
    /// already the exact count of the transcript, and adding the
    /// approximation on top was overstating the fill by roughly a quarter
    /// of the conversation. With no attachments the headline is exactly the
    /// exact count.
    func testEstimatedContextTokensDoesNotDoubleCountTheTranscript() {
        let transcript = String(repeating: "A", count: 400)
        let model = makeModel(
            messages: [AppChatMessage(role: .user, content: transcript)],
            boundary: 0,
            summary: nil)
        model.estimatedPromptTokens = 100
        XCTAssertEqual(model.estimatedContextTokens, 100)
    }
}
