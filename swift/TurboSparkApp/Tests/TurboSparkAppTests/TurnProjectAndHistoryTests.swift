import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// The Tier 1 agent-loop cluster: a turn's PROJECT (state#30), the two
/// silent history-assembly defects (state#31, state#32), and the lifecycle
/// flags an approve or a deny leaves behind (state#29).
///
/// `executeGenerationTurn` returns at its own `session` guard with no model
/// loaded, so everything here is reached through the pieces that were split
/// out of it for exactly that reason (`swift/CLAUDE.md` Gotcha 26).
@MainActor
final class TurnProjectAndHistoryTests: XCTestCase {
    private func makeProject(_ model: AppModel, name: String, agentType: AppAgentType = .coder)
        -> AppProject
    {
        let project = AppProject(
            name: name,
            rootDirectoryPath: "/tmp/\(name)",
            agentType: agentType,
            customInstructions: "RULES-FOR-\(name)",
            permissions: .standard,
            maxAutonomousSteps: 5
        )
        model.projects.append(project)
        return project
    }

    // MARK: - state#30: the turn's project comes from its chat

    /// The review's own case: a chat under project A while project B is
    /// selected must build its prompt from A. `selectProject` guards only on
    /// `!generating`, which is false for the whole time a call sits at an
    /// approval card, so this switch is not merely possible -- it is legal
    /// exactly when a turn is mid-flight.
    func testATurnResolvesItsProjectFromItsChatNotFromTheSelection() {
        let model = AppModel()
        model.interactionMode = .projects
        let projectA = makeProject(model, name: "alpha")
        let projectB = makeProject(model, name: "beta")

        var chat = AppChat(title: "under alpha")
        chat.projectID = projectA.id
        model.chats = [chat]
        model.selectedChatID = chat.id

        // The user switches to project B while the turn is in flight.
        model.selectedProjectID = projectB.id

        XCTAssertEqual(
            model.turnProject(chatID: chat.id)?.id, projectA.id,
            "The turn belongs to the chat's project, not to whatever is selected now.")

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: model.turnProject(chatID: chat.id))
        let system = history.first { $0.role == .system }?.content ?? ""
        XCTAssertTrue(system.contains("RULES-FOR-alpha"), "System prompt must come from project A.")
        XCTAssertFalse(system.contains("RULES-FOR-beta"), "Project B's rules must not leak in.")
        XCTAssertTrue(system.contains("/tmp/alpha"), "Workspace root must be project A's.")
    }

    /// The step cap, the guardrails mode and the agent profile are read off
    /// the same argument, so they cannot disagree with the system prompt.
    func testTheStepCapAndAgentProfileFollowTheTurnsProject() {
        let model = AppModel()
        model.interactionMode = .projects
        var slowProject = makeProject(model, name: "capped", agentType: .researcher)
        slowProject.maxAutonomousSteps = 2
        model.projects[0] = slowProject
        let other = makeProject(model, name: "wide")

        var chat = AppChat(title: "capped chat")
        chat.projectID = slowProject.id
        model.chats = [chat]
        model.selectedProjectID = other.id

        let resolved = model.turnProject(chatID: chat.id)
        XCTAssertEqual(resolved?.maxAutonomousSteps, 2)
        XCTAssertEqual(model.agentType(for: resolved), .researcher)
        XCTAssertEqual(model.agentType(for: other), .coder)
    }

    /// Conversational Chat mode sends no project, and therefore no tool
    /// definitions. Preserved from the call site this replaced: widening it
    /// would offer tools that `extractToolCalls` then refuses to parse.
    func testChatModeStillResolvesNoProject() {
        let model = AppModel()
        model.interactionMode = .chat
        let project = makeProject(model, name: "gamma")
        var chat = AppChat(title: "chat mode")
        chat.projectID = project.id
        model.chats = [chat]
        model.selectedProjectID = project.id

        XCTAssertNil(model.turnProject(chatID: chat.id))
    }

    // MARK: - state#31: a tool turn with no prose survives history assembly

    /// `ForgeGuardrailsEngine.sanitizeProse` returns "" when the reply was
    /// nothing but the call, so a RESCUED call arrives with empty content and
    /// a real result. The emptiness guard skipped the whole message before
    /// the result loop ran, so the model was never told the call had been
    /// answered and reissued it until the step cap.
    func testAToolResultSurvivesAMessageWithNoProse() {
        let model = AppModel()
        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        let result = AppToolResult(callID: call.id, output: "a.txt\nb.txt")
        var chat = AppChat(title: "rescued")
        chat.messages = [
            AppChatMessage(role: .user, content: "what is in this directory"),
            AppChatMessage(
                role: .assistant,
                content: "",
                stopReason: "tool_use",
                toolCalls: [call],
                toolResults: [result]),
        ]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertTrue(
            history.contains { $0.content.contains("a.txt\nb.txt") },
            "The tool result must reach the model even when its message carries no prose.")
    }

    /// The guard still drops a genuinely empty turn: a message with no text,
    /// no image and no tool activity is not a turn.
    func testATrulyEmptyMessageIsStillDropped() {
        let model = AppModel()
        model.defaultSystemPrompt = ""
        model.selectedSystemPromptID = nil
        model.selectedPersonalityID = nil
        var chat = AppChat(title: "empty")
        chat.messages = [
            AppChatMessage(role: .user, content: "hello"),
            AppChatMessage(role: .assistant, content: ""),
        ]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertEqual(history.count, 1)
        XCTAssertEqual(history[0].role, .user)
    }

    // MARK: - state#32: tool results go back as `.tool`

    /// A mid-history `.system` message is refused by three of the five
    /// fallback renderers and by real templates that require alternating
    /// roles; `fit_window` prices a failing render at `u64::MAX` and drops
    /// turns until it stops failing, so the wrong role loses history rather
    /// than reporting anything.
    func testToolResultsAreSentUnderTheToolRole() {
        let model = AppModel()
        let ok = AppToolCall(name: "read_file", arguments: ["path": "a"], category: .fileRead)
        let bad = AppToolCall(name: "run_command", arguments: ["command": "false"], category: .terminal)
        var chat = AppChat(title: "roles")
        chat.messages = [
            AppChatMessage(role: .user, content: "go"),
            AppChatMessage(
                role: .assistant, content: "reading", stopReason: "tool_use",
                toolCalls: [ok], toolResults: [AppToolResult(callID: ok.id, output: "contents")]),
            AppChatMessage(
                role: .assistant, content: "running", stopReason: "tool_use",
                toolCalls: [bad],
                toolResults: [AppToolResult(callID: bad.id, output: "boom", isError: true)]),
        ]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        let toolTurns = history.filter { $0.role == .tool }
        // `guard` rather than a bare `XCTAssertEqual`: the subscripts below
        // TRAP on an empty array, and a trap kills the xctest process without
        // printing a `failed (` line at all -- which reads as the mutation
        // SURVIVING (`swift/CLAUDE.md` Gotcha 41).
        guard toolTurns.count == 2 else {
            return XCTFail("expected two tool turns, got \(toolTurns.count)")
        }
        XCTAssertTrue(toolTurns[0].content.hasPrefix("<tool_response>"))
        XCTAssertTrue(toolTurns[1].content.hasPrefix("<tool_error>"))
        XCTAssertFalse(
            history.dropFirst().contains { $0.role == .system },
            "No system message may appear after the first position.")
    }

    // MARK: - state#29: approve and deny both own a turn

    /// Approving used to lower `generating` on top of the continuation it had
    /// just started, because the `defer` was unguarded and `continueOrStop`
    /// runs synchronously. The epoch bump is what a newer turn outranks.
    func testApprovingACallBumpsTheEpochSoItsContinuationCannotBeUnwound() {
        let model = AppModel()
        let chat = AppChat(title: "approve")
        model.chats = [chat]
        model.selectedChatID = chat.id

        let call = AppToolCall(name: "read_file", arguments: ["path": "a"], category: .fileRead)
        model.pendingToolCall = call
        model.pendingToolCallChatID = chat.id
        let before = model.generationEpoch

        model.approvePendingToolCall(id: call.id)

        XCTAssertTrue(model.generating, "The execution window belongs to the turn.")
        XCTAssertGreaterThan(
            model.generationEpoch, before,
            "Without a bump the unguarded defer lowers `generating` under the continuation turn.")
    }

    /// Denying raised nothing at all, so `canRun` stayed true across the
    /// awaited notification hook (up to 120 s) and the whole continuation --
    /// a Send there started a second `generate()` on the serial session.
    func testDenyingACallRaisesGeneratingForItsContinuation() {
        let model = AppModel()
        let chat = AppChat(title: "deny")
        model.chats = [chat]
        model.selectedChatID = chat.id

        let call = AppToolCall(
            name: "run_command", arguments: ["command": "rm -rf /"], category: .terminal)
        model.pendingToolCall = call
        model.pendingToolCallChatID = chat.id
        let before = model.generationEpoch

        model.denyPendingToolCall(id: call.id)

        XCTAssertTrue(model.generating, "Deny owns its continuation the way approve does.")
        XCTAssertFalse(model.canRun, "Send must be refused while the deny path continues.")
        XCTAssertGreaterThan(model.generationEpoch, before)
    }

    // MARK: - state#33 / state#34: the two windows Stop could not reach

    /// While `run()` awaits a `UserPromptSubmit` hook, Send is refused by
    /// `canRun` -- and Stop was refused too, leaving a wedged hook with no
    /// exit at all.
    func testStopIsOfferedWhileASubmissionIsAwaitingItsHook() {
        let model = AppModel()
        model.submitting = true
        XCTAssertTrue(model.canCancel)
        XCTAssertFalse(model.canUnloadModel)
        XCTAssertFalse(model.canLoadModel)
    }

    /// A pending call lowers `generating` on purpose (state#9), so the only
    /// ways out of an approval card were Approve, Deny or deleting the chat.
    func testStopIsOfferedWhileACallAwaitsApproval() {
        let model = AppModel()
        let chat = AppChat(title: "pending")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let call = AppToolCall(name: "read_file", arguments: ["path": "a"], category: .fileRead)
        model.pendingToolCall = call
        model.pendingToolCallChatID = chat.id

        XCTAssertTrue(model.canCancel, "A pending approval must be stoppable.")

        model.cancel()

        XCTAssertNil(model.pendingToolCall)
        let recorded = model.chats[0].messages.last
        XCTAssertEqual(
            recorded?.toolCalls.first?.status, .denied,
            "A stopped call is recorded as denied, not left at pendingApproval forever.")
        XCTAssertFalse(
            model.isCancellationPending,
            "Nothing else runs a tail to clear this, and a latched flag refuses every later turn.")
    }
}
