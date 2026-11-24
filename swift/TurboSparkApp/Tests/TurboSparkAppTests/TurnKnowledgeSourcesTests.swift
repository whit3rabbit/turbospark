import XCTest

@testable import TurboSparkApp

/// The composer's "What feeds this turn" list: which knowledge sources the
/// next prompt draws on, their state, and where an estimate exists.
final class TurnKnowledgeSourcesTests: XCTestCase {

    private typealias Sources = TurnKnowledgeSources

    private func inputs(
        hasProject: Bool = true,
        projectRulesSet: Bool = true,
        memoryEnabled: Bool = true,
        recallEnabled: Bool = true,
        isTemporaryChat: Bool = false,
        isProtectedProfile: Bool = false,
        skillStateEnabled: Bool = false,
        memoryTokens: Int? = nil,
        pinnedInstructionTokens: Int? = nil,
        attachmentCount: Int = 0,
        attachmentTokens: Int? = nil,
        mcpServerCount: Int = 0,
        mcpServerTokens: Int? = nil
    ) -> Sources.Inputs {
        Sources.Inputs(
            hasProject: hasProject, projectRulesSet: projectRulesSet,
            memoryEnabled: memoryEnabled, recallEnabled: recallEnabled,
            isTemporaryChat: isTemporaryChat, isProtectedProfile: isProtectedProfile,
            skillStateEnabled: skillStateEnabled, memoryTokens: memoryTokens,
            pinnedInstructionTokens: pinnedInstructionTokens,
            attachmentCount: attachmentCount, attachmentTokens: attachmentTokens,
            mcpServerCount: mcpServerCount, mcpServerTokens: mcpServerTokens)
    }

    private func row(_ source: Sources.Source, in rows: [Sources.Row]) -> Sources.Row? {
        rows.first { $0.source == source }
    }

    // MARK: Which rows exist

    /// The always-present rows in a project with nothing optional attached,
    /// in the fixed display order.
    func testBaselineProjectRowsAndOrder() {
        let rows = Sources.rows(inputs())
        XCTAssertEqual(rows.map(\.source), [.projectRules, .memory, .conversationRecall])
    }

    /// Project rules and MCP servers are project-derived: without a project
    /// the prompt builder emits neither, so the list must not claim them.
    func testNoProjectDropsProjectDerivedRows() {
        let rows = Sources.rows(inputs(hasProject: false, mcpServerCount: 2))
        XCTAssertNil(row(.projectRules, in: rows))
        XCTAssertNil(row(.mcpServers, in: rows))
        XCTAssertNotNil(row(.memory, in: rows))
    }

    func testOptionalRowsAppearOnlyWhenPresent() {
        let rows = Sources.rows(inputs(
            pinnedInstructionTokens: 40, attachmentCount: 2, attachmentTokens: 300,
            mcpServerCount: 1, mcpServerTokens: 25))
        XCTAssertEqual(
            rows.map(\.source),
            [.projectRules, .memory, .conversationRecall, .pinnedInstructions, .attachments, .mcpServers])
        XCTAssertEqual(row(.attachments, in: rows)?.count, 2)
        XCTAssertEqual(row(.mcpServers, in: rows)?.count, 1)

        let none = Sources.rows(inputs(pinnedInstructionTokens: 0, attachmentCount: 0, mcpServerCount: 0))
        XCTAssertNil(row(.pinnedInstructions, in: none))
        XCTAssertNil(row(.attachments, in: none))
        XCTAssertNil(row(.mcpServers, in: none))
    }

    // MARK: Status

    func testProjectRulesNotSetIsNotIncluded() {
        XCTAssertEqual(row(.projectRules, in: Sources.rows(inputs(projectRulesSet: false)))?.status, .notSet)
        XCTAssertEqual(row(.projectRules, in: Sources.rows(inputs()))?.status, .included)
    }

    func testMemoryFollowsTheGlobalSwitch() {
        XCTAssertEqual(row(.memory, in: Sources.rows(inputs(memoryEnabled: true)))?.status, .included)
        XCTAssertEqual(row(.memory, in: Sources.rows(inputs(memoryEnabled: false)))?.status, .off)
    }

    /// Recall adds evidence only when older turns fall out of the window,
    /// so "on" is reported as when-needed, never as included.
    func testRecallOnIsWhenNeeded() {
        XCTAssertEqual(row(.conversationRecall, in: Sources.rows(inputs()))?.status, .whenNeeded)
    }

    /// The same exclusions `conversationRecallEligible` applies, each with
    /// its reason, so the list never says recall runs where it cannot.
    func testRecallExclusionsMatchEligibility() {
        XCTAssertEqual(
            row(.conversationRecall, in: Sources.rows(inputs(isTemporaryChat: true)))?.status,
            .unavailable(.temporaryChat))
        XCTAssertEqual(
            row(.conversationRecall, in: Sources.rows(inputs(isProtectedProfile: true)))?.status,
            .unavailable(.protectedProfile))
        XCTAssertEqual(
            row(.conversationRecall, in: Sources.rows(inputs(skillStateEnabled: true)))?.status,
            .unavailable(.skillState))
        XCTAssertEqual(
            row(.conversationRecall, in: Sources.rows(inputs(recallEnabled: false)))?.status, .off)
    }

    /// A protected profile forces the settings toggle off, so its reason
    /// wins over the switch's own value.
    func testProtectedProfileWinsOverRecallSwitch() {
        XCTAssertEqual(
            row(.conversationRecall, in: Sources.rows(inputs(
                recallEnabled: false, isTemporaryChat: true, isProtectedProfile: true)))?.status,
            .unavailable(.protectedProfile))
    }

    /// SKILL.state is a project setting; outside a project it cannot apply.
    func testSkillStateIgnoredWithoutProject() {
        XCTAssertEqual(
            row(.conversationRecall, in: Sources.rows(inputs(hasProject: false, skillStateEnabled: true)))?.status,
            .whenNeeded)
    }

    // MARK: Scope

    func testScopes() {
        let rows = Sources.rows(inputs(
            pinnedInstructionTokens: 1, attachmentCount: 1, mcpServerCount: 1))
        XCTAssertEqual(row(.projectRules, in: rows)?.scope, .thisProject)
        XCTAssertEqual(row(.memory, in: rows)?.scope, .allChats)
        XCTAssertEqual(row(.conversationRecall, in: rows)?.scope, .allChats)
        XCTAssertEqual(row(.pinnedInstructions, in: rows)?.scope, .thisChat)
        XCTAssertEqual(row(.attachments, in: rows)?.scope, .nextMessage)
        XCTAssertEqual(row(.mcpServers, in: rows)?.scope, .thisProject)
    }

    // MARK: Token estimates

    func testTokensOnlyWherePositiveAndIncluded() {
        let rows = Sources.rows(inputs(
            memoryTokens: 120, pinnedInstructionTokens: 40, attachmentCount: 1,
            attachmentTokens: 300, mcpServerCount: 1, mcpServerTokens: 25))
        XCTAssertEqual(row(.memory, in: rows)?.estimatedTokens, 120)
        XCTAssertEqual(row(.pinnedInstructions, in: rows)?.estimatedTokens, 40)
        XCTAssertEqual(row(.attachments, in: rows)?.estimatedTokens, 300)
        XCTAssertEqual(row(.mcpServers, in: rows)?.estimatedTokens, 25)
        // Rules are folded into the system prompt slice and recall is fitted
        // at send time: neither has a figure the app can stand behind.
        XCTAssertNil(row(.projectRules, in: rows)?.estimatedTokens)
        XCTAssertNil(row(.conversationRecall, in: rows)?.estimatedTokens)
    }

    /// A stale count must not survive the switch going off, and a zero or
    /// negative count is no estimate at all.
    func testTokensDroppedWhenOffOrNonPositive() {
        XCTAssertNil(row(.memory, in: Sources.rows(inputs(memoryEnabled: false, memoryTokens: 120)))?.estimatedTokens)
        XCTAssertNil(row(.memory, in: Sources.rows(inputs(memoryTokens: 0)))?.estimatedTokens)
        XCTAssertNil(row(.memory, in: Sources.rows(inputs(memoryTokens: -5)))?.estimatedTokens)
        // An image-only attachment set has a count but no estimate.
        let images = Sources.rows(inputs(attachmentCount: 2, attachmentTokens: nil))
        XCTAssertEqual(row(.attachments, in: images)?.count, 2)
        XCTAssertNil(row(.attachments, in: images)?.estimatedTokens)
    }
}
