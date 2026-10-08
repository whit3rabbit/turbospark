import XCTest

@testable import TurboSparkApp

/// Regression tests for the agent-loop review items: scheduled prompt
/// delivery, deferred /skill expansion, turn-tail cleanup, /rewind over a
/// compacted chat, duplicate todo ids, approval-mode rule preservation and
/// the image producer slot.
@MainActor
final class AgentLoopReviewFixesTests: XCTestCase {
    var model: AppModel!

    override func setUp() {
        super.setUp()
        model = AppModel()
        model.stopCronScheduler()
    }

    override func tearDown() {
        model = nil
        super.tearDown()
    }

    private func makeChat(messages: [AppChatMessage] = []) -> AppChat {
        let chat = AppChat(title: "T", messages: messages)
        model.chats = [chat]
        model.selectedChatID = chat.id
        return chat
    }

    // MARK: - #22 / #7 scheduled prompts

    func testScheduledBangPromptReachesTheModelAsPlainTextAndRunsNoShell() {
        let chat = makeChat()
        model.scheduledDeliveryModelReadyOverride = true
        let job = AppCronJob(cron: "* * * * *", prompt: "!echo x", chatID: chat.id)

        XCTAssertTrue(model.fireCronJob(job))

        let messages = model.chats[0].messages
        // A bang command would have appended shell output rows and cleared
        // the prompt; plain delivery is exactly one user row.
        XCTAssertEqual(messages.map(\.role), [.user])
        XCTAssertEqual(messages.first?.content, "!echo x")
        XCTAssertTrue((model.pendingUserMessages[chat.id] ?? []).isEmpty)
    }

    func testIdleChatWithEmptyComposerRunsTheScheduledJob() {
        let chat = makeChat()
        model.scheduledDeliveryModelReadyOverride = true
        model.promptText = ""

        XCTAssertTrue(model.fireCronJob(AppCronJob(cron: "* * * * *", prompt: "daily summary", chatID: chat.id)))

        XCTAssertEqual(model.chats[0].messages.map(\.content), ["daily summary"])
    }

    func testScheduledDeliveryKeepsTheComposerDraft() {
        let chat = makeChat()
        model.scheduledDeliveryModelReadyOverride = true
        model.promptText = "half typed thought"

        XCTAssertTrue(model.fireCronJob(AppCronJob(cron: "* * * * *", prompt: "tick", chatID: chat.id)))

        XCTAssertEqual(model.promptText, "half typed thought")
        XCTAssertEqual(model.chats[0].messages.map(\.content), ["tick"])
    }

    func testScheduledPromptWaitsInTheQueueWhileTheChatIsBusy() {
        let chat = makeChat()
        model.scheduledDeliveryModelReadyOverride = true
        model.generating = true

        XCTAssertTrue(model.fireCronJob(AppCronJob(cron: "* * * * *", prompt: "!rm -rf x", chatID: chat.id)))

        XCTAssertTrue(model.chats[0].messages.isEmpty)
        let parked = model.pendingUserMessages[chat.id] ?? []
        XCTAssertEqual(parked.map(\.text), ["!rm -rf x"])
        XCTAssertEqual(parked.map(\.isScheduled), [true])

        // The turn tail's drain delivers it, still as plain text.
        model.generating = false
        model.drainPendingUserMessagesIfIdle(chatID: chat.id)
        XCTAssertEqual(model.chats[0].messages.map(\.content), ["!rm -rf x"])
    }

    func testScheduledPromptStaysParkedWithoutAModel() {
        let chat = makeChat()
        model.scheduledDeliveryModelReadyOverride = false
        XCTAssertTrue(model.fireCronJob(AppCronJob(cron: "* * * * *", prompt: "tick", chatID: chat.id)))
        XCTAssertTrue(model.chats[0].messages.isEmpty)
        XCTAssertEqual(model.pendingUserMessages[chat.id]?.count, 1)
    }

    func testCronJobForADeletedChatIsStillAMissedRun() {
        _ = makeChat()
        XCTAssertFalse(model.fireCronJob(AppCronJob(cron: "* * * * *", prompt: "x", chatID: UUID())))
    }

    // MARK: - #9 deferred skill expansion

    func testSkillExpansionInsideASubmissionIsDeferredNotParked() {
        let chat = makeChat()
        model.submitting = true
        model.promptText = "/some-skill"

        model.submitExpandedSkillPrompt("Execute the following skill instructions:\n\nbody", presentation: nil)

        XCTAssertNil(model.pendingUserMessages[chat.id], "must not be parked in the queue")
        XCTAssertEqual(model.deferredSkillSubmission?.text.hasPrefix("Execute the following"), true)
    }

    func testFlushReplaysTheDeferredExpansionThroughTheComposer() {
        let chat = makeChat()
        model.promptText = "/some-skill"
        model.deferredSkillSubmission = AppModel.DeferredSkillSubmission(
            chatID: chat.id, text: "Execute the following skill instructions:\n\nbody", presentation: nil)

        model.flushDeferredSkillSubmission(cancelled: false)

        XCTAssertNil(model.deferredSkillSubmission)
        XCTAssertEqual(model.promptText, "Execute the following skill instructions:\n\nbody")
        XCTAssertNil(model.pendingUserMessages[chat.id])
    }

    func testFlushDropsADeferredExpansionOnCancel() {
        let chat = makeChat()
        model.promptText = "/some-skill"
        model.deferredSkillSubmission = AppModel.DeferredSkillSubmission(
            chatID: chat.id, text: "x", presentation: nil)
        model.flushDeferredSkillSubmission(cancelled: true)
        XCTAssertNil(model.deferredSkillSubmission)
        XCTAssertEqual(model.promptText, "/some-skill")
    }

    func testFlushQueuesForAChatTheUserMovedAwayFrom() {
        let origin = AppChat(title: "A")
        let other = AppChat(title: "B")
        model.chats = [origin, other]
        model.selectedChatID = other.id
        model.promptText = "typing in B"
        model.deferredSkillSubmission = AppModel.DeferredSkillSubmission(
            chatID: origin.id, text: "expanded", presentation: nil)

        model.flushDeferredSkillSubmission(cancelled: false)

        XCTAssertEqual(model.promptText, "typing in B")
        XCTAssertEqual(model.pendingUserMessages[origin.id]?.map(\.text), ["expanded"])
    }

    // MARK: - #10 turn tail

    func testTurnTailResetsLatchedStateWhenNoNewTurnStarted() {
        let chat = makeChat()
        model.generating = true
        model.isCancellationPending = true
        model.generationEpoch = 7

        model.finishTurnTail(myEpoch: 7, turnChatID: chat.id, session: nil)

        XCTAssertFalse(model.generating)
        XCTAssertFalse(model.isCancellationPending)
        XCTAssertNil(model.runTask)
    }

    func testStaleTurnTailDoesNotClobberANewerTurn() {
        let chat = makeChat()
        model.generating = true
        model.generationEpoch = 8
        model.finishTurnTail(myEpoch: 7, turnChatID: chat.id, session: nil)
        XCTAssertTrue(model.generating)
    }

    // MARK: - #11 rewind over a compacted chat

    func testRewindOnACompactedChatKeepsNewPromptsInModelHistory() {
        let u1 = AppChatMessage(role: .user, content: "first prompt")
        let a1 = AppChatMessage(role: .assistant, content: "first answer")
        let u2 = AppChatMessage(role: .user, content: "second prompt")
        let a2 = AppChatMessage(role: .assistant, content: "second answer")
        _ = makeChat(messages: [u1, a1, u2, a2])
        model.setStoredCompaction(chatIndex: 0, summary: "SUMMARY", boundary: 4)

        model.rewindTo(anchorMessageID: u1.id)
        XCTAssertEqual(model.compactionState(chatID: model.chats[0].id).boundary, 0)

        model.chats[0].messages.append(AppChatMessage(role: .user, content: "brand new prompt"))
        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        let texts = history.map(\.content)
        XCTAssertTrue(texts.contains("first prompt"))
        XCTAssertTrue(texts.contains("brand new prompt"), "the post-rewind prompt must reach the model")
    }

    // MARK: - #12 duplicate todo ids

    func testDuplicateTodoIdsDoNotTrapTheCompactionPolicy() {
        let previous = [
            TodoItem(id: "1", content: "a", status: "pending"),
            TodoItem(id: "1", content: "b", status: "pending")
        ]
        let current = [
            TodoItem(id: "1", content: "a", status: "completed"),
            TodoItem(id: "1", content: "b", status: "completed")
        ]
        XCTAssertTrue(
            TodoBoundaryCompactionPolicy.completedTransition(previous: previous, current: current))
        XCTAssertFalse(
            TodoBoundaryCompactionPolicy.completedTransition(previous: current, current: current))
    }

    // MARK: - #14 approval mode keeps rules

    func testChangingApprovalModeKeepsRulesAndGrants() throws {
        var project = AppProject(name: "P", rootDirectoryPath: NSTemporaryDirectory())
        project.permissions.mcpDenyRules = ["mcp__evil__*"]
        project.permissions.mcpAllowRules = ["mcp__fs__read"]
        project.permissions.browserOriginAllowlist = ["https://example.com"]
        project.permissions.replFileAccessRoots = ["/tmp/work"]
        model.projects = [project]
        model.selectedProjectID = project.id

        for mode in [AppPermissionMode.agentAuto, .auto, .agentAuto] {
            model.setEffectivePermissionMode(mode)
            let live = try XCTUnwrap(model.projects.first).permissions
            XCTAssertEqual(live.mode, mode)
            XCTAssertEqual(live.mcpDenyRules, ["mcp__evil__*"])
            XCTAssertEqual(live.mcpAllowRules, ["mcp__fs__read"])
            XCTAssertEqual(live.browserOriginAllowlist, ["https://example.com"])
            XCTAssertEqual(live.replFileAccessRoots, ["/tmp/work"])
        }
    }

    // MARK: - #3 image producer slot

    func testStaleProducerClearDoesNotNullTheNewerTask() async {
        let slot = ImageProducerSlot()
        let oldToken = UUID()
        let newToken = UUID()
        let old = Task<Void, Never> {}
        let new = Task<Void, Never> { try? await Task.sleep(nanoseconds: 5_000_000_000) }
        slot.store(old, token: oldToken)
        slot.store(new, token: newToken)

        slot.clear(token: oldToken)  // the previous run finishing late
        slot.cancel()

        XCTAssertTrue(new.isCancelled, "the newer run must still be cancellable")
    }

    func testWaitUntilIdleBlocksUntilACancelledProducerReallyStops() async {
        let slot = ImageProducerSlot()
        let finished = FlagBox()
        // Ignores cancellation for a while, like model load / VAE decode.
        let task = Task<Void, Never> {
            try? await Task.sleep(nanoseconds: 300_000_000)
            await finished.set()
        }
        slot.store(task, token: UUID())
        slot.cancel()

        await slot.waitUntilIdle()
        let done = await finished.value
        XCTAssertTrue(done, "idle must mean the producer finished, not merely cancelled")
    }
}

private actor FlagBox {
    private(set) var value = false
    func set() { value = true }
}
