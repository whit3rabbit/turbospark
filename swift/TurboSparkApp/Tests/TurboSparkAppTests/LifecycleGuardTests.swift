import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Tier 2 of the third state-layer review: the guards that were spelled on
/// one side of a pair and not the other.
///
/// `canRun` grew `submitting` and `opening` and never grew `pendingToolCall`,
/// even though `canCancel` carries it (state#76). The chat-side lifecycle
/// calls grew `pendingToolCall` (state#49) while the PROJECT-side ones stayed
/// on `!generating` alone (state#77). And a project switch is the bigger
/// move of the two: it resets the hashes a pending `edit_file` was evaluated
/// against.
@MainActor
final class LifecycleGuardTests: XCTestCase {
    private func makeProject(name: String) throws -> (AppProject, () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("guards_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return (AppProject(name: name, rootDirectoryPath: dir.path), {
            try? FileManager.default.removeItem(at: dir)
        })
    }

    private func pendingCall(on appModel: AppModel, chatID: UUID) -> AppToolCall {
        let call = AppToolCall(
            name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chatID
        appModel.pendingToolCallStep = 0
        return call
    }

    // MARK: - state#76

    func testSendIsRefusedWhileACallWaitsForApproval() {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        appModel.promptText = "another question"

        // No session, so `canRun` is false for that reason alone; the term
        // under test is the pending one, asserted directly.
        _ = pendingCall(on: appModel, chatID: chat.id)
        XCTAssertNil(appModel.session, "Precondition: this suite has no model.")
        XCTAssertTrue(
            appModel.canCancel,
            "Stop is live while a card is up, which is what makes Send being live a bug and not "
                + "a symmetry.")

        // The discriminating form: with a session the only remaining false
        // term would be the pending call, so assert the predicate reads it.
        appModel.pendingToolCall = nil
        let withoutPending = AppModel.canRunTerms(
            generating: false, submitting: false, opening: false, hasPendingCall: false,
            hasSession: true, hasInput: true)
        let withPending = AppModel.canRunTerms(
            generating: false, submitting: false, opening: false, hasPendingCall: true,
            hasSession: true, hasInput: true)
        XCTAssertTrue(withoutPending, "An idle model with input must still be runnable.")
        XCTAssertFalse(withPending, "A call awaiting approval must refuse a second turn.")
    }

    func testApprovingIsARefusedNoOpWhileATurnIsRunning() {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        let call = pendingCall(on: appModel, chatID: chat.id)

        // Something else raised `generating` -- a Send the old `canRun`
        // permitted, or the other button on this card.
        appModel.generating = true
        appModel.approvePendingToolCall(id: call.id)

        XCTAssertNotNil(
            appModel.pendingToolCall,
            "The card must still be there: approving mid-turn spawns a second toolExecutionTask "
                + "and the second assignment drops the first where cancel() cannot reach it.")
        appModel.denyPendingToolCall(id: call.id)
        XCTAssertNotNil(appModel.pendingToolCall, "Deny carries the same guard.")
    }

    // MARK: - state#77

    func testSwitchingProjectIsRefusedWhileACallWaitsForApproval() throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        let (projectA, cleanA) = try makeProject(name: "A")
        let (projectB, cleanB) = try makeProject(name: "B")
        defer {
            cleanA()
            cleanB()
        }
        appModel.projects = [projectA, projectB]
        appModel.selectProject(id: projectA.id)

        _ = pendingCall(on: appModel, chatID: chat.id)
        appModel.selectProject(id: projectB.id)

        XCTAssertEqual(
            appModel.selectedProjectID, projectA.id,
            "A project switch resets FileSnapshotStore, which holds the hashes the waiting call "
                + "was evaluated against.")
    }

    func testDeletingTheProjectIsRefusedWhileACallWaitsForApproval() throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        let (project, cleanup) = try makeProject(name: "A")
        defer { cleanup() }
        appModel.projects = [project]
        appModel.selectProject(id: project.id)

        _ = pendingCall(on: appModel, chatID: chat.id)
        appModel.deleteProject(id: project.id)

        XCTAssertEqual(
            appModel.projects.count, 1,
            "The waiting call captured this project at proposal time; deleting it leaves the card "
                + "naming a workspace that is gone.")
    }

    // MARK: - state#79

    func testADraftChatCarriesTheSelectedProject() throws {
        let appModel = AppModel()
        let (project, cleanup) = try makeProject(name: "A")
        defer { cleanup() }
        appModel.projects = [project]
        appModel.chats = []
        appModel.selectedProjectID = project.id
        // The `didSet` is what builds the draft.
        appModel.selectedChatID = UUID()

        appModel.promptText = "hello"

        let materialized = try XCTUnwrap(appModel.chats.first)
        XCTAssertEqual(
            materialized.projectID, project.id,
            "A chat with no project is filtered out of the sidebar and gets no system prompt, no "
                + "tools and no workspace root.")
    }

    // MARK: - state#80

    func testDeletingTheLastChatOfAProjectDoesNotSelectAnotherProjectsChat() throws {
        let appModel = AppModel()
        let (projectA, cleanA) = try makeProject(name: "A")
        let (projectB, cleanB) = try makeProject(name: "B")
        defer {
            cleanA()
            cleanB()
        }
        appModel.projects = [projectA, projectB]
        appModel.selectedProjectID = projectA.id

        let inA = AppChat(projectID: projectA.id, title: "in A")
        let inB = AppChat(projectID: projectB.id, title: "in B")
        appModel.chats = [inA, inB]
        appModel.selectedChatID = inA.id

        appModel.deleteChat(id: inA.id)

        XCTAssertNotEqual(
            appModel.selectedChatID, inB.id,
            "The replacement came out of the unfiltered list, so deleting project A's last chat "
                + "selected a project-B conversation the sidebar does not even show.")
    }

    // MARK: - state#82

    func testNoProjectMeansNoToolCallsAreParsed() {
        let appModel = AppModel()
        appModel.interactionMode = .projects
        let reply = """
        <tool_call>
        <name>websearch</name>
        <arguments>{"query": "anything"}</arguments>
        </tool_call>
        """
        XCTAssertTrue(
            appModel.extractToolCalls(from: reply, project: nil).isEmpty,
            "`buildSystemPrompt(for: nil)` offers no tools, so text that looks like a call was "
                + "never offered one to invoke. `websearch` is not workspace-rooted, so nothing "
                + "downstream refuses it either.")
        let (project, _) = (AppProject(name: "P", rootDirectoryPath: "/tmp"), ())
        XCTAssertEqual(
            appModel.extractToolCalls(from: reply, project: project).count, 1,
            "And it must still parse when a project IS attached, or the gate is a blanket refusal.")
    }

    // MARK: - state#84

    func testLoadingIsPossibleAgainAfterAnUnload() async throws {
        let appModel = AppModel()
        let model = InstalledModel(
            alias: "probe-\(UUID().uuidString)", repo: "x/y",
            path: FileManager.default.temporaryDirectory
                .appendingPathComponent("\(UUID().uuidString).gturbo").path,
            family: "gemma4")
        // Selected but NOT loaded: the state after an unload, or after an
        // open that failed.
        appModel.selected = model
        appModel.error = nil
        XCTAssertNil(appModel.session, "Precondition: nothing is loaded.")
        XCTAssertNotEqual(
            appModel.modelPathText, model.path, "Precondition: nothing has pointed at it yet.")

        appModel.selectModel(model)

        // The open runs in a `Task`, so `opening` is not observable
        // synchronously; what IS synchronous is that `selectModel` got past
        // its guard at all. The path is the tell, and the failed open behind
        // it is the confirmation.
        XCTAssertEqual(
            appModel.modelPathText, model.path,
            "\"Already selected\" is not \"already loaded\": after an unload or a failed open, "
                + "Load Model was dead for the very model on screen.")
        let deadline = Date().addingTimeInterval(10)
        while appModel.error == nil && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertNotNil(
            appModel.error, "And the open was actually attempted, not merely permitted.")
    }

    // MARK: - state#88

    func testARemovedScannedModelStaysRemovedAndCanBeRestored() {
        let store = ModelOrganizationStore.shared
        let path = "/tmp/scan-probe-\(UUID().uuidString)/model.gturbo"
        defer { store.restoreToScan(path: path) }

        XCTAssertFalse(store.isExcludedFromScan(path: path))
        store.excludeFromScan(path: path)
        XCTAssertTrue(
            store.isExcludedFromScan(path: path),
            "The scan finds the row again on every refresh, so the exclusion is the only thing "
                + "that can make \"Remove from TurboSpark\" mean anything.")
        store.restoreToScan(path: path)
        XCTAssertFalse(store.isExcludedFromScan(path: path), "And it has to be reversible.")
    }

    // MARK: - state#93

    func testASwitchedOffAgentIsNotRunnableByName() {
        let appModel = AppModel()
        let name = AgentManager.shared.builtInAgents[0].name
        AgentManager.shared.setAgentEnabled(false, name: name, scope: .builtIn)
        defer {
            AgentManager.shared.setAgentEnabled(true, name: name, scope: .builtIn)
            appModel.reloadAgents()
        }
        appModel.reloadAgents()

        XCTAssertNil(
            appModel.findAgent(named: name),
            "The `agent` TOOL refuses a disabled agent; the slash path resolved against the "
                + "inspection list, which deliberately includes the switched-off ones.")
        XCTAssertNotNil(
            appModel.findAnyAgent(named: name),
            "And the unfiltered lookup must still find it, or the refusal cannot say why.")
    }

    // MARK: - state#95

    func testAnAgentFileCannotAskForAnUnboundedTurnBudget() {
        let greedy = AppAgentDefinition(
            name: "greedy", agentDescription: "", systemPrompt: "", maxTurns: 100_000)
        XCTAssertEqual(
            greedy.maxTurns, AppAgentDefinition.maxTurnsCeiling,
            "An agent file out of a cloned repository is a request, not a setting.")

        let modest = AppAgentDefinition(
            name: "modest", agentDescription: "", systemPrompt: "", maxTurns: 3)
        XCTAssertEqual(modest.maxTurns, 3, "And a reasonable value is left alone.")
    }

    func testAProjectStepCapIsClampedToThePickersOwnRange() throws {
        // Round-tripped through the real encoder and then hand-edited, the
        // way a user editing `projects_archive.json` would: a literal fixture
        // would be asserting this test's idea of the shape rather than the
        // app's.
        let encoded = try JSONEncoder().encode(AppProject(name: "P", maxAutonomousSteps: 5))
        let text = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        let edited = text.replacingOccurrences(
            of: "\"maxAutonomousSteps\":5", with: "\"maxAutonomousSteps\":100000")
        XCTAssertNotEqual(edited, text, "The fixture must actually carry the edited value.")
        let decoded = try JSONDecoder().decode(AppProject.self, from: Data(edited.utf8))
        XCTAssertEqual(
            decoded.maxAutonomousSteps, AppProject.autonomousStepRange.upperBound,
            "`projects_archive.json` is a plain file a user can edit, and an unclamped step cap "
                + "is an agent loop that runs tools until the model stops proposing them.")
    }
}
