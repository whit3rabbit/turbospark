import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Regression tests for the agent-loop lifecycle cluster found in the
/// 2026-09-03 state review: A2/A3 (`generating` dropped while a tool ran, so
/// Send was live and Stop was dead for exactly that window), A4 (a subagent
/// executed model-proposed tools with no permission gate at all), A6 (an
/// approved call appended a SECOND assistant turn and left the proposal stuck
/// at `.pendingApproval` forever), A9 (approve/deny resumed without the
/// `maxAutonomousSteps` check), A13 (`cancel()` cleared one of four pending
/// fields) and D1 (an approved call ran under whichever project was selected
/// at approval time, not the one its `.allow` was computed against).
///
/// `AgentLoopRoutingTests` next door covers the earlier state#6/#7/#9 routing
/// fixes; these are the ones that needed the turn's LIFECYCLE rather than its
/// chat id.
///
/// **ONE PART OF A3 IS DELIBERATELY NOT HERE.** `continueAgentLoop`'s
/// `!isCancellationPending` guard has no observable effect without a live
/// session: `executeGenerationTurn` returns at its own `guard let session`
/// before touching any state, so a test asserting "nothing happened" passes
/// identically with the guard deleted. What IS testable -- that `cancel()`
/// reaches the task the loop spawned outside `runTask` -- is below.
@MainActor
final class AgentLoopLifecycleTests: XCTestCase {
    /// A directory holding one uniquely-named file, so a `list_directory`
    /// result names which project it actually ran in.
    private func makeProject(
        name: String,
        marker: String,
        permissions: AppProjectPermissions = AppProjectPermissions(),
        maxAutonomousSteps: Int = 5
    ) throws -> (project: AppProject, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: dir.appendingPathComponent(marker).path, contents: Data())
        let project = AppProject(
            name: name,
            rootDirectoryPath: dir.path,
            permissions: permissions,
            maxAutonomousSteps: maxAutonomousSteps
        )
        return (project, { try? FileManager.default.removeItem(at: dir) })
    }

    private func waitUntil(
        _ condition: () -> Bool, timeout: TimeInterval = 5
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    // MARK: - D1: the project is captured at proposal, not read at approval

    func testAnApprovedCallRunsInTheProjectItWasProposedUnder() async throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let (projectA, cleanA) = try makeProject(name: "A", marker: "only-in-a.txt")
        let (projectB, cleanB) = try makeProject(name: "B", marker: "only-in-b.txt")
        defer {
            cleanA()
            cleanB()
        }
        appModel.projects = [projectA, projectB]
        appModel.selectedProjectID = projectA.id

        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallStep = 0
        appModel.pendingToolCallProject = projectA

        // The user switches projects while the call sits waiting. `selectProject`
        // guards only on `!generating`, which is false at this point.
        appModel.selectedProjectID = projectB.id

        appModel.approvePendingToolCall(id: call.id)
        try await waitUntil { !(appModel.chats[0].messages.isEmpty) }

        let output = appModel.chats[0].messages.first?.toolResults.first?.output ?? ""
        XCTAssertTrue(
            output.contains("only-in-a.txt"),
            "The call must run in the project its permission decision was made against. Got: \(output)")
        XCTAssertFalse(
            output.contains("only-in-b.txt"),
            "It must not run in whatever project the user selected while it waited.")
    }

    // MARK: - state#67: a hook runs in the TURN's project, not the selection

    func testAPostToolUseHookRunsInTheProjectTheCallWasProposedUnder() async throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let (projectA, cleanA) = try makeProject(
            name: "A", marker: "only-in-a.txt",
            permissions: AppProjectPermissions(mode: .auto, fileRead: .allow))
        let (projectB, cleanB) = try makeProject(
            name: "B", marker: "only-in-b.txt",
            permissions: AppProjectPermissions(mode: .auto, fileRead: .allow))
        defer {
            cleanA()
            cleanB()
        }
        appModel.projects = [projectA, projectB]
        appModel.selectProject(id: projectA.id)

        // The hook records the workspace it was told about. `TURBOSPARK_PROJECT_DIR`
        // is set from the dispatch's `workingDirectory` and by nothing else,
        // so the file it writes IS the answer to "whose project ran this".
        let witness = FileManager.default.temporaryDirectory
            .appendingPathComponent("hook_cwd_\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: witness) }
        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Record Project",
            event: .postToolUse,
            type: .command,
            command: "printf '%s' \"$TURBOSPARK_PROJECT_DIR\" > '\(witness.path)'",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallStep = 0
        appModel.pendingToolCallProject = projectA

        // **THROUGH `selectProject`, NOT BY ASSIGNING THE ID.** The sibling
        // case above sets `selectedProjectID` directly, which is exactly what
        // this one cannot do: `selectProject` is what REBINDS `AppHookStore`
        // to the new root, and the rebind is half of the defect.
        appModel.selectProject(id: projectB.id)

        appModel.approvePendingToolCall(id: call.id)
        try await waitUntil { FileManager.default.fileExists(atPath: witness.path) }

        let recorded = (try? String(contentsOf: witness, encoding: .utf8)) ?? ""
        XCTAssertEqual(
            recorded, projectA.rootDirectoryPath,
            "A PostToolUse hook describes the call that just ran, so it belongs to the project "
                + "that call ran in -- not to whatever the user selected while it waited.")
    }

    // MARK: - A2: `generating` covers tool execution, not just token streaming

    func testGeneratingStaysTrueWhileAnApprovedToolRuns() async throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let (project, cleanup) = try makeProject(name: "P", marker: "marker.txt")
        defer { cleanup() }
        appModel.projects = [project]
        appModel.selectedProjectID = project.id

        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallProject = project

        XCTAssertFalse(appModel.generating, "Nothing is running while a call waits for a human.")
        appModel.approvePendingToolCall(id: call.id)

        // Synchronously after approval, before the tool has had a chance to
        // finish: the turn owns the app again. This is what makes `canRun`
        // false (no second turn beside the tool) and `canCancel` true (Stop
        // works for exactly the window a shell command runs in).
        XCTAssertTrue(appModel.generating, "Execution is part of the turn, so `generating` must cover it.")
        XCTAssertFalse(appModel.canRun, "Send must be refused while a tool runs.")

        try await waitUntil { !appModel.generating }
        XCTAssertFalse(appModel.generating, "It must come back down when the tool is done.")
    }

    // MARK: - A6: the proposal is updated in place, never duplicated

    func testDenyingUpdatesTheProposalRatherThanAppendingASecondTurn() {
        let appModel = AppModel()
        var chat = AppChat(title: "one")

        var proposed = AppToolCall(
            name: "run_command", arguments: ["command": "ls"], category: .terminal)
        proposed.status = .pendingApproval
        chat.messages = [
            AppChatMessage(
                role: .assistant, content: "I will list the directory.", stopReason: "tool_use",
                toolCalls: [proposed])
        ]
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        appModel.pendingToolCall = proposed
        appModel.pendingToolCallChatID = chat.id

        appModel.denyPendingToolCall(id: proposed.id)

        XCTAssertEqual(
            appModel.chats[0].messages.count, 1,
            "One call is one assistant turn. A second append feeds the next prompt two.")
        XCTAssertEqual(
            appModel.chats[0].messages[0].toolCalls.first?.status, .denied,
            "The proposal must not stay at `.pendingApproval` forever.")
        XCTAssertEqual(appModel.chats[0].messages[0].toolResults.count, 1)
    }

    func testAToolTurnWithNoProposalToUpdateIsStillAppended() {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        // The `.allow` path never appends a proposal, so there is nothing to
        // find and appending remains correct.
        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.appendToolExecutionTurn(
            call: call, result: AppToolResult(callID: call.id, output: "ok"), chatID: chat.id)

        XCTAssertEqual(appModel.chats[0].messages.count, 1)
    }

    // MARK: - A9: approve/deny resume through the step cap

    func testDenyingAtTheLastPermittedStepDoesNotRunAnotherTurn() async throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let (project, cleanup) = try makeProject(name: "P", marker: "m.txt", maxAutonomousSteps: 1)
        defer { cleanup() }
        appModel.projects = [project]
        appModel.selectedProjectID = project.id

        // A Stop hook that BLOCKS is the observable: `continueOrStop` consults
        // it once the cap is reached and folds its reason in as the next user
        // turn, where `continueAgentLoop` called directly would have started a
        // turn at step 1 and never asked. Without a live session the turn
        // itself cannot run either way, so the hook's message is what
        // distinguishes the two paths.
        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Cap Probe",
            event: .stop,
            type: .command,
            command:
                "echo '{\"decision\":\"block\",\"reason\":\"step cap reached and Stop was asked\"}'",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let call = AppToolCall(name: "run_command", arguments: ["command": "ls"], category: .terminal)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallStep = 0  // maxAutonomousSteps is 1, so 0 is the last one
        appModel.pendingToolCallProject = project

        appModel.denyPendingToolCall(id: call.id)
        try await waitUntil {
            appModel.chats[0].messages.contains(where: { $0.role == .user })
        }

        XCTAssertTrue(
            appModel.chats[0].messages.contains {
                $0.role == .user && $0.content.contains("step cap reached")
            },
            "Resuming must go through `continueOrStop`, which is the only place the step cap is tested.")
    }

    func testApprovingAtTheLastPermittedStepDoesNotRunAnotherTurn() async throws {
        // The approve path is a separate call site from deny and skipped the
        // cap independently; mutating one leaves the other's test green, so
        // both are pinned.
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let (project, cleanup) = try makeProject(name: "P", marker: "m.txt", maxAutonomousSteps: 1)
        defer { cleanup() }
        appModel.projects = [project]
        appModel.selectedProjectID = project.id

        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Cap Probe Approve",
            event: .stop,
            type: .command,
            command:
                "echo '{\"decision\":\"block\",\"reason\":\"approve reached the step cap\"}'",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallStep = 0
        appModel.pendingToolCallProject = project

        appModel.approvePendingToolCall(id: call.id)
        try await waitUntil {
            appModel.chats[0].messages.contains(where: { $0.role == .user })
        }

        XCTAssertTrue(
            appModel.chats[0].messages.contains {
                $0.role == .user && $0.content.contains("approve reached the step cap")
            },
            "Approving must resume through `continueOrStop`, the one place the cap is tested.")
    }

    // MARK: - A8: a checklist belongs to the calling chat, not the selected one

    func testTodoWriteLandsInTheCallingChatNotTheSelectedOne() async throws {
        let appModel = AppModel()
        let chatA = AppChat(title: "A")
        let chatB = AppChat(title: "B")
        appModel.chats = [chatA, chatB]
        appModel.selectedChatID = chatA.id

        let (project, cleanup) = try makeProject(name: "P", marker: "m.txt")
        defer { cleanup() }
        appModel.projects = [project]
        appModel.selectedProjectID = project.id

        // `TodoWriteExecutor.execute` takes a `chatID` and its one call site
        // never passed it, so the callback fell back to `selectedChatID`
        // asynchronously -- one chat's checklist overwriting another's.
        let call = AppToolCall(
            name: "todowrite",
            arguments: ["todos": #"[{"content":"ship it","status":"pending"}]"#],
            category: .automation)
        appModel.selectedChatID = chatB.id  // the user switches mid-execution
        _ = await AppToolRegistry.execute(call: call, in: project, chatID: chatA.id)

        try await waitUntil { !(appModel.chats.first(where: { $0.id == chatA.id })?.todos.isEmpty ?? true) }
        XCTAssertEqual(
            appModel.chats.first(where: { $0.id == chatA.id })?.todos.count, 1,
            "The checklist must land in the chat whose agent wrote it.")
        XCTAssertEqual(
            appModel.chats.first(where: { $0.id == chatB.id })?.todos.count, 0,
            "The chat the user switched to must not have its checklist replaced.")
    }

    // MARK: - A14: an error mid-turn is not recorded as a cancellation

    func testAnErroredTurnIsNotRecordedAsCancelled() {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        appModel.outputText = "partial answer"

        appModel.finishCancelled(chatID: chat.id, reason: "error")

        XCTAssertEqual(
            appModel.chats[0].messages.first?.stopReason, "error",
            "A transcript must distinguish the user pressing Stop from the engine failing.")
    }

    // MARK: - A13: cancel clears every field describing the pending call

    func testCancelClearsTheWholePendingCallAndNotJustTheCall() throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        let (project, cleanup) = try makeProject(name: "P", marker: "m.txt")
        defer { cleanup() }

        appModel.generating = true  // so `canCancel` admits the cancel
        appModel.pendingToolCall = AppToolCall(
            name: "run_command", arguments: ["command": "ls"], category: .terminal)
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallStep = 4
        appModel.pendingToolCallProject = project

        appModel.cancel()

        XCTAssertNil(appModel.pendingToolCall)
        XCTAssertNil(appModel.pendingToolCallChatID, "A stale chat id is read by the NEXT pending call.")
        XCTAssertEqual(appModel.pendingToolCallStep, 0, "A stale step resumes the next call at the wrong position.")
        XCTAssertNil(appModel.pendingToolCallProject, "A stale project runs the next call in the wrong root.")
        XCTAssertTrue(appModel.isCancellationPending)
    }

    // MARK: - A3: Stop reaches the tool-execution task, not just `runTask`

    func testCancelCancelsTheToolExecutionTask() async throws {
        let appModel = AppModel()
        appModel.generating = true

        // An approved call runs HERE rather than in `runTask`: the proposing
        // turn's stream ended when the call was parked for approval, so
        // `runTask?.cancel()` alone cancels a task with nothing left to do.
        let observed = Observed()
        appModel.toolExecutionTask = Task {
            do {
                try await Task.sleep(nanoseconds: 5_000_000_000)
            } catch {
                await observed.markCancelled()
            }
        }

        appModel.cancel()
        let deadline = Date().addingTimeInterval(5)
        while await !observed.wasCancelled, Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        let wasCancelled = await observed.wasCancelled
        XCTAssertTrue(wasCancelled, "Stop must reach work the agent loop spawned outside `runTask`.")
    }

    /// Bridges the cancelled-ness of a detached task back to the test.
    private actor Observed {
        var wasCancelled = false
        func markCancelled() { wasCancelled = true }
    }
}
