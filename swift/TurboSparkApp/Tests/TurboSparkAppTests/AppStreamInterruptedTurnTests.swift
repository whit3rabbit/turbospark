import XCTest

import TurboSpark
@testable import TurboSparkApp

@MainActor
final class AppStreamInterruptedTurnTests: XCTestCase {
    func testInterruptionClassificationPreservesUserCancellationAcrossEngineThrows() {
        XCTAssertEqual(
            AppStreamRecovery.interruptionReason(taskIsCancelled: false, cancellationPending: false),
            .error)
        XCTAssertEqual(
            AppStreamRecovery.interruptionReason(taskIsCancelled: true, cancellationPending: false),
            .cancelled)
        XCTAssertEqual(
            AppStreamRecovery.interruptionReason(taskIsCancelled: false, cancellationPending: true),
            .cancelled)
    }

    func testEngineFailurePersistsPartialOutputOnceWithAMatchingAnchor() throws {
        let model = AppModel()
        model.stopCronScheduler()
        var chat = AppChat(title: "failure")
        chat.messages = [AppChatMessage(role: .user, content: "prompt")]
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.outputText = "partial answer"
        model.outputReasoningText = "partial reasoning"

        model.finishCancelled(chatID: chat.id, reason: "error")

        let saved = try XCTUnwrap(model.chats.first(where: { $0.id == chat.id }))
        XCTAssertEqual(saved.messages.count, 2)
        XCTAssertEqual(saved.messages[1].content, "partial answer")
        XCTAssertEqual(saved.messages[1].reasoning, "partial reasoning")
        XCTAssertEqual(saved.messages[1].stopReason, "error")
        let anchor = try XCTUnwrap(saved.recoveryAnchor)
        XCTAssertEqual(anchor.retainedMessageCount, 1)
        XCTAssertEqual(anchor.interruptedMessageID, saved.messages[1].id)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: saved.messages))
        let restoredChat = try JSONDecoder().decode(AppChat.self, from: JSONEncoder().encode(saved))
        XCTAssertTrue(AppStreamRecovery.validateAnchor(try XCTUnwrap(restoredChat.recoveryAnchor), messages: restoredChat.messages))
        XCTAssertEqual(model.streamRecoveryEvents[chat.id], [
            .partialPreserved(rows: 1),
            .anchorRecorded(anchor),
        ])

        model.finishCancelled(chatID: chat.id, reason: "error")
        XCTAssertEqual(model.turnMessages(for: chat.id).count, 2, "A drained partial buffer must not append twice.")
    }

    func testUserCancellationPersistsPartialOutputWithCancellationOutcome() throws {
        let model = AppModel()
        model.stopCronScheduler()
        var chat = AppChat(title: "cancelled")
        chat.messages = [AppChatMessage(role: .user, content: "prompt")]
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.outputText = "answer before stop"

        model.finishCancelled(chatID: chat.id, reason: "cancelled")

        let saved = try XCTUnwrap(model.chats.first(where: { $0.id == chat.id }))
        XCTAssertEqual(saved.messages.count, 2)
        XCTAssertEqual(saved.messages[1].content, "answer before stop")
        XCTAssertEqual(saved.messages[1].stopReason, "cancelled")
        let anchor = try XCTUnwrap(saved.recoveryAnchor)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: saved.messages))
        XCTAssertEqual(model.streamRecoveryEvents[chat.id], [
            .partialPreserved(rows: 1),
            .anchorRecorded(anchor),
        ])
    }

    func testNewInterruptedTurnDoesNotInheritEarlierExhaustedRetryBudget() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let earlierPartial = AppChatMessage(role: .assistant, content: "earlier partial", stopReason: "error")
        var chat = AppChat(title: "new turn", messages: [
            AppChatMessage(role: .user, content: "earlier prompt"),
            earlierPartial,
            AppChatMessage(role: .user, content: "new prompt"),
        ])
        chat.recoveryAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: earlierPartial.id,
            persistedRowContent: earlierPartial.content,
            continuationsUsed: 0,
            retriesUsed: AppStreamRecovery.maxRecoveryRetries))
        model.chats = [chat]
        model.outputText = "new partial"

        model.finishCancelled(chatID: chat.id, reason: "error")

        let saved = try XCTUnwrap(model.chats.first)
        let anchor = try XCTUnwrap(saved.recoveryAnchor)
        XCTAssertEqual(anchor.retriesUsed, 0)
        XCTAssertEqual(anchor.retainedMessageCount, 3)
        XCTAssertEqual(
            AppStreamRecovery.retryOutcome(anchor: anchor, messages: saved.messages), .retry(attempt: 1))
    }

    func testRecoveryAnchorHashesThePersistedIncompleteCallLabel() throws {
        let model = AppModel()
        model.stopCronScheduler()
        var chat = AppChat(title: "labeled recovery")
        chat.messages = [AppChatMessage(role: .user, content: "prompt")]
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.outputText = "Before <tool_call><name>read_file</name>"

        model.finishCancelled(chatID: chat.id, reason: "error")

        let saved = try XCTUnwrap(model.chats.first(where: { $0.id == chat.id }))
        let partial = try XCTUnwrap(saved.messages.last)
        let anchor = try XCTUnwrap(saved.recoveryAnchor)
        let selectedLanguage = AppLanguage.resolve(
            UserDefaults.standard.string(forKey: AppLanguage.storageKey)
                ?? AppLanguage.system.rawValue)
        let localizedLabel = String(
            localized: "Incomplete tool call, not executed",
            bundle: .module,
            locale: selectedLanguage.locale)
        XCTAssertTrue(partial.content.contains(localizedLabel))
        XCTAssertEqual(anchor.interruptedContentHash, AppStreamRecovery.contentHash(
            forPersistedRowContent: partial.content))
        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: saved.messages))
    }

    func testInterruptedRecoveryRetryPersistsItsAttemptCountAndOriginalVariant() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let interrupted = AppChatMessage(role: .assistant, content: "first partial", stopReason: "error")
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: interrupted.id,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: 1,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        var chat = AppChat(title: "recovery retry")
        chat.messages = [prompt, interrupted]
        chat.recoveryAnchor = anchor
        model.chats = [chat]
        model.selectedChatID = chat.id

        guard case .retry(let plan) = AppStreamRecovery.prepareAnchoredRetry(
            anchor: anchor, messages: chat.messages)
        else { return XCTFail("Expected a retry plan for the current anchor.") }
        model.pendingResponseVariants[chat.id] = plan.alternates
        model.pendingRecoveryRetryCounts[chat.id] = plan.attempt
        model.mutateTurnMessages(for: chat.id) { $0 = plan.transcript }
        model.outputText = "second partial"
        model.finishCancelled(chatID: chat.id, reason: "error")

        let saved = try XCTUnwrap(model.chats.first(where: { $0.id == chat.id }))
        XCTAssertEqual(saved.messages.count, 2)
        XCTAssertEqual(saved.messages[1].content, "second partial")
        XCTAssertEqual(saved.messages[1].alternates.map(\.content), ["first partial"])
        let nextAnchor = try XCTUnwrap(saved.recoveryAnchor)
        XCTAssertEqual(nextAnchor.retriesUsed, 2)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(nextAnchor, messages: saved.messages))
    }

    func testInFlightRecoveryRetryReloadRestoresThePartialAndRemainingBound() throws {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let interrupted = AppChatMessage(
            role: .assistant,
            content: "first partial",
            reasoning: "partial reasoning",
            stopReason: "error")
        let initialAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: interrupted.id,
            persistedRowContent: interrupted.content,
            continuationsUsed: 1,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        var inFlight = AppChat(title: "retry reload", messages: [prompt, interrupted])
        inFlight.recoveryAnchor = initialAnchor
        guard case .retry(let retryPlan) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: initialAnchor,
            messages: inFlight.messages)
        else { return XCTFail("Expected the matching interrupted row to prepare a retry.") }

        // Model the archive written after Retry moved the partial out of the
        // active transcript. The interrupted row and consumed attempt belong
        // to the in-flight anchor until the retry commits or is interrupted.
        inFlight.messages = retryPlan.transcript
        inFlight.recoveryAnchor = try XCTUnwrap(retryPlan.stagedRecoveryAnchor)

        let restored = try JSONDecoder().decode(
            AppChat.self,
            from: JSONEncoder().encode(inFlight))

        XCTAssertEqual(restored.messages, [prompt, interrupted])
        let restoredAnchor = try XCTUnwrap(restored.recoveryAnchor)
        XCTAssertEqual(restoredAnchor.retriesUsed, 1)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(restoredAnchor, messages: restored.messages))
        guard case .retry(let nextRetry) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: restoredAnchor,
            messages: restored.messages)
        else { return XCTFail("The reloaded partial should allow the next bounded anchored retry.") }
        XCTAssertEqual(nextRetry.recoveryAttempt, 2)
    }

    func testSuccessfulRecoveryCommitAtomicallyClearsPersistedAnchor() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let interrupted = AppChatMessage(role: .assistant, content: "old partial", stopReason: "error")
        var chat = AppChat(title: "successful recovery")
        chat.messages = [prompt]
        chat.recoveryAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: interrupted.id,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        model.chats = [chat]
        model.selectedChatID = chat.id
        let resultJSON = """
        {"promptTokens":0,"newTokens":4,"prefillSeconds":0,"decodeSeconds":0,"stopReason":"endOfTurn","content":"new answer"}
        """
        let result = try JSONDecoder().decode(GenerationResult.self, from: Data(resultJSON.utf8))

        model.mutateRecoveryState(for: chat.id) { messages, anchor in
            AppStreamRecovery.commitSuccessfulRecoveryRetry(
                into: &messages,
                recoveryAnchor: &anchor,
                content: "new answer",
                reasoning: "new reasoning",
                result: result,
                alternates: [interrupted])
        }

        let saved = try XCTUnwrap(model.chats.first(where: { $0.id == chat.id }))
        XCTAssertNil(saved.recoveryAnchor)
        XCTAssertEqual(saved.messages.map(\.content), ["prompt", "new answer"])
        XCTAssertEqual(saved.messages[1].alternates.map(\.content), ["old partial"])
    }

    func testRecoveryRetryWithNoOutputRestoresTheAnchoredPartial() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let interrupted = AppChatMessage(role: .assistant, content: "preserved partial", stopReason: "cancelled")
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: interrupted.id,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        var chat = AppChat(title: "empty retry")
        chat.messages = [prompt]
        chat.recoveryAnchor = anchor
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.pendingResponseVariants[chat.id] = [interrupted]
        model.pendingRecoveryRetryCounts[chat.id] = 1

        model.finishCancelled(chatID: chat.id, reason: "error")

        let saved = try XCTUnwrap(model.chats.first(where: { $0.id == chat.id }))
        XCTAssertEqual(saved.messages, [prompt, interrupted])
        let updatedAnchor = try XCTUnwrap(saved.recoveryAnchor)
        XCTAssertEqual(updatedAnchor.retriesUsed, 1)
        XCTAssertEqual(updatedAnchor.retainedMessageCount, anchor.retainedMessageCount)
        XCTAssertEqual(updatedAnchor.interruptedMessageID, anchor.interruptedMessageID)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(updatedAnchor, messages: saved.messages))
        XCTAssertNil(model.pendingRecoveryRetryCounts[chat.id])
    }

    func testOrdinaryRetryWithNoOutputPreservesThePreviousReplyAsAVariant() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let response = AppChatMessage(role: .assistant, content: "completed answer", stopReason: "endOfTurn")
        guard case .retry(let plan) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: nil, messages: [prompt, response])
        else { return XCTFail("Expected an ordinary retry plan.") }
        let chat = AppChat(title: "empty ordinary retry", messages: plan.transcript)
        model.chats = [chat]
        model.pendingResponseVariants[chat.id] = plan.alternates

        model.finishCancelled(chatID: chat.id, reason: "error")

        let saved = try XCTUnwrap(model.chats.first)
        let interrupted = try XCTUnwrap(saved.messages.last(where: { $0.role == .assistant }))
        XCTAssertEqual(saved.messages.count, 2)
        XCTAssertEqual(interrupted.content, "")
        XCTAssertEqual(interrupted.stopReason, "error")
        XCTAssertEqual(interrupted.alternates, [response])
        let anchor = try XCTUnwrap(saved.recoveryAnchor)
        XCTAssertEqual(anchor.retriesUsed, 0)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: saved.messages))
        guard case .retry(let next) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: anchor, messages: saved.messages)
        else { return XCTFail("An empty failed replacement must remain recoverable.") }
        XCTAssertEqual(next.recoveryAttempt, 1)
        XCTAssertNil(model.pendingResponseVariants[chat.id])
        model.finishCancelled(chatID: chat.id, reason: "error")
        XCTAssertEqual(model.turnMessages(for: chat.id).count, 2)
        let restored = try JSONDecoder().decode(AppChat.self, from: JSONEncoder().encode(saved))
        XCTAssertEqual(restored.messages.last?.alternates, [response])
    }

    func testEditedPromptWithNoOutputRetainsTheEditedPromptAndPreviousAnswerVariant() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let prompt = AppChatMessage(role: .user, content: "original prompt")
        let response = AppChatMessage(role: .assistant, content: "original answer", stopReason: "endOfTurn")
        let applied = AppModel.editApplied(
            to: [prompt, response], anchorIndex: 0, newText: "edited prompt", now: Date())
        let chat = AppChat(title: "empty edit", messages: applied.messages)
        model.chats = [chat]
        model.pendingResponseVariants[chat.id] = applied.responseVariants

        model.finishCancelled(chatID: chat.id, reason: "cancelled")

        let saved = try XCTUnwrap(model.chats.first)
        XCTAssertEqual(saved.messages.count, 2)
        let savedPrompt = try XCTUnwrap(saved.messages.first(where: { $0.role == .user }))
        let interrupted = try XCTUnwrap(saved.messages.last(where: { $0.role == .assistant }))
        XCTAssertEqual(savedPrompt.content, "edited prompt")
        XCTAssertEqual(savedPrompt.alternates, [prompt])
        XCTAssertEqual(interrupted.content, "")
        XCTAssertEqual(interrupted.stopReason, "cancelled")
        XCTAssertEqual(interrupted.alternates, [response])
        let anchor = try XCTUnwrap(saved.recoveryAnchor)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: saved.messages))
        XCTAssertNil(model.pendingResponseVariants[chat.id])
    }

    func testGhostPartialAndAnchorRemainInsideTheVault() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let chatID = model.enterGhostChat()
        model.outputText = "temporary partial"

        model.finishCancelled(chatID: chatID, reason: "cancelled")

        let ghostRow = try XCTUnwrap(model.chats.first(where: { $0.id == chatID }))
        XCTAssertTrue(ghostRow.messages.isEmpty)
        XCTAssertNil(ghostRow.recoveryAnchor)
        let payload = model.ghostPayload(for: chatID)
        XCTAssertEqual(payload.messages.count, 1)
        let anchor = try XCTUnwrap(payload.recoveryAnchor)
        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: payload.messages))
    }

    func testUnclosedAndMalformedRecognizedCallWrappersReceiveAnInertLabel() {
        let xml = "Before prose. <tool_call><name>read_file</name><arguments>{\"path\":\"a\"}"
        let labeledXML = AppStreamRecovery.labelIncompleteToolCallFragments(in: xml)
        XCTAssertTrue(labeledXML.hasPrefix("Before prose. \n[Incomplete tool call, not executed]\n&lt;tool_call&gt;"))
        XCTAssertTrue(labeledXML.hasSuffix("&lt;tool_call&gt;<name>read_file</name><arguments>{\"path\":\"a\"}"))
        XCTAssertEqual(labeledXML.components(separatedBy: "[Incomplete tool call, not executed]").count - 1, 1)
        XCTAssertTrue(ToolCallParser.parse(from: labeledXML).isEmpty)

        let markdown = "Ordinary prose.\n```tool_call\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}"
        let labeledMarkdown = AppStreamRecovery.labelIncompleteToolCallFragments(in: markdown)
        XCTAssertTrue(labeledMarkdown.contains(
            "[Incomplete tool call, not executed]\n&#96;&#96;&#96;tool_call\n"))
        XCTAssertTrue(ToolCallParser.parse(from: labeledMarkdown).isEmpty)

        let jsonMarkdown = "```json_tool_call\n{\"name\":\"read_file\"}"
        XCTAssertTrue(AppStreamRecovery.labelIncompleteToolCallFragments(in: jsonMarkdown)
            .contains("[Incomplete tool call, not executed]\n&#96;&#96;&#96;json_tool_call"))

        let complete = "<tool_call><name>read_file</name><arguments>{}</arguments></tool_call>"
        XCTAssertEqual(AppStreamRecovery.labelIncompleteToolCallFragments(in: complete), complete)
        let lookalike = "A literal <tool_call fragment and a ```tool_calls block are ordinary text."
        XCTAssertEqual(AppStreamRecovery.labelIncompleteToolCallFragments(in: lookalike), lookalike)
    }

    func testClosedMalformedCallWrappersAreLabeledOnceAndValidWrappersStayUnchanged() {
        let malformedXML = [
            "<tool_call><name>read_file</name></tool_call>",
            "<tool_call><name>read_file</name><arguments>{not-json}</arguments></tool_call>",
            "<tool_call><name>read_file</name><arguments>[1,2]</arguments></tool_call>",
            "<tool_call><name>read_file</name><arguments>{\"path\":\"a\"}</arguments><broken></tool_call>",
        ]
        let malformedMarkdown = [
            "```tool_call\n{\"arguments\":{\"path\":\"a\"}}\n```",
            "```tool_call  \r\n{\"arguments\":{\"path\":\"a\"}}\r\n```",
            "```tool_call\n{not-json}\n```",
            "```json_tool_call\n{\"name\":\"read_file\",\"arguments\":[1,2]}\n```",
        ]
        let validXML = "<tool_call><name>read_file</name><arguments>{\"path\":\"a\"}</arguments></tool_call>"
        let validMarkdown = "```tool_call\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}\n```"

        for fragment in malformedXML + malformedMarkdown {
            let source = "Before \(fragment) after."
            let labeled = AppStreamRecovery.labelIncompleteToolCallFragments(in: source)
            let inertFragment: String
            if fragment.hasPrefix("<tool_call>") {
                inertFragment = fragment.replacingOccurrences(
                    of: "<tool_call>", with: "&lt;tool_call&gt;")
            } else {
                inertFragment = fragment.replacingOccurrences(of: "```", with: "&#96;&#96;&#96;")
            }
            let expected = "Before \n[Incomplete tool call, not executed]\n\(inertFragment) after."
            XCTAssertEqual(
                labeled.components(separatedBy: "[Incomplete tool call, not executed]").count - 1,
                1,
                "Each recognized malformed wrapper gets one label: \(fragment)")
            XCTAssertEqual(labeled, expected, "Only the label and opening marker change: \(fragment)")
            XCTAssertEqual(
                AppStreamRecovery.labelIncompleteToolCallFragments(in: labeled),
                labeled,
                "Relabeling persisted transcript text must not duplicate its inert marker.")
        }

        let surrounding = "Keep this prose.\n\(validXML)\nThen this prose.\n\(validMarkdown)"
        XCTAssertEqual(
            AppStreamRecovery.labelIncompleteToolCallFragments(in: surrounding),
            surrounding,
            "Complete wrappers and their surrounding prose stay byte-for-byte unchanged.")

        let multiple = malformedXML[0] + " between " + malformedMarkdown[0]
        let labeledMultiple = AppStreamRecovery.labelIncompleteToolCallFragments(in: multiple)
        XCTAssertEqual(
            labeledMultiple.components(separatedBy: "[Incomplete tool call, not executed]").count - 1,
            2,
            "Every malformed wrapper is labeled once.")
    }

    func testLabeledClosedMalformedWrapperRemainsInertAtCompletedDispatchGate() {
        let malformed = "<tool_call><name>read_file</name><arguments>{\"path\":\"a\"}</arguments><broken></tool_call>"
        let content = AppStreamRecovery.labelIncompleteToolCallFragments(in: "Run this: \(malformed)")
        let available = AppToolCatalog.captureTurnAvailableTools(
            for: nil,
            globalMcpServers: [],
            contextTokens: nil,
            webToolsEnabled: false)

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertTrue(result.preservedContent.contains("[Incomplete tool call, not executed]"))
        XCTAssertTrue(result.preservedContent.contains(
            malformed.replacingOccurrences(of: "<tool_call>", with: "&lt;tool_call&gt;")))
    }

    func testPersistedInertLabelSurvivesAppLanguageChange() {
        let defaults = UserDefaults.standard
        let previousLanguage = defaults.object(forKey: AppLanguage.storageKey)
        defer {
            if let previousLanguage {
                defaults.set(previousLanguage, forKey: AppLanguage.storageKey)
            } else {
                defaults.removeObject(forKey: AppLanguage.storageKey)
            }
        }

        defaults.set(AppLanguage.french.rawValue, forKey: AppLanguage.storageKey)
        let previousLanguageLabel = "Previous language: incomplete tool call"
        let malformedXML = "<tool_call><name>read_file</name><arguments>{\"path\":\"a\"}</arguments><broken></tool_call>"
        let malformedMarkdown = "```tool_call  \r\n{\"arguments\":{\"path\":\"a\"}}\r\n```"
        let persisted = AppStreamRecovery.labelIncompleteToolCallFragments(
            in: "Before. \(malformedXML) between. \(malformedMarkdown) after.",
            label: previousLanguageLabel)
        let inertXML = malformedXML.replacingOccurrences(
            of: "<tool_call>", with: "&lt;tool_call&gt;")
        let inertMarkdown = malformedMarkdown.replacingOccurrences(
            of: "```", with: "&#96;&#96;&#96;")
        XCTAssertEqual(
            persisted.components(separatedBy: previousLanguageLabel).count - 1,
            2)
        XCTAssertTrue(persisted.contains("\n\(previousLanguageLabel)\n\(inertXML)"))
        XCTAssertTrue(persisted.contains("\n\(previousLanguageLabel)\n\(inertMarkdown)"))

        defaults.set(AppLanguage.english.rawValue, forKey: AppLanguage.storageKey)
        XCTAssertEqual(defaults.string(forKey: AppLanguage.storageKey), AppLanguage.english.rawValue)
        XCTAssertFalse(persisted.contains("<tool_call>"))
        XCTAssertFalse(persisted.contains("```tool_call"))
        XCTAssertTrue(ToolCallParser.parseCandidates(from: persisted).isEmpty)
        let available = AppToolCatalog.captureTurnAvailableTools(
            for: nil,
            globalMcpServers: [],
            contextTokens: nil,
            webToolsEnabled: false)
        let result = ToolCallDispatchGate.evaluate(
            content: persisted,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertTrue(result.preservedContent.contains(previousLanguageLabel))
        XCTAssertTrue(result.preservedContent.contains(inertXML))
        XCTAssertTrue(result.preservedContent.contains(inertMarkdown))
    }

    func testNestedCandidateInsideMalformedWrapperIsNeutralized() {
        let nested = "<tool_call><broken><tool_call><name>read_file</name><arguments>{\"path\":\"a\"}</arguments></tool_call>"

        let labeled = AppStreamRecovery.labelIncompleteToolCallFragments(in: nested)

        XCTAssertEqual(labeled.components(separatedBy: "[Incomplete tool call, not executed]").count - 1, 1)
        XCTAssertFalse(labeled.contains("<tool_call>"))
        XCTAssertTrue(labeled.contains("&lt;tool_call&gt;<broken>&lt;tool_call&gt;"))
        XCTAssertTrue(ToolCallParser.parseCandidates(from: labeled).isEmpty)
    }

    func testMixedNestedWrapperFamiliesRemainInertAtCompletedDispatchGate() {
        let callJSON = #"{"name":"read_file","arguments":{"path":"a.txt"}}"#
        let xmlCall = #"<tool_call><name>read_file</name><arguments>{"path":"a.txt"}</arguments></tool_call>"#
        let markdownCall = "```tool_call\n\(callJSON)\n```"
        let invalidWrappers = [
            "<tool_call><broken>\n\(markdownCall)",
            "<tool_call><broken>\n\(markdownCall)\n</tool_call>",
            "```tool_call\n\(xmlCall)",
            "```tool_call\n\(xmlCall)\n```",
        ]
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "read_file",
                description: "Read a file",
                parameters: .object(
                    properties: ["path": .string()],
                    required: ["path"],
                    additionalProperties: false)),
        ])

        for fragment in invalidWrappers {
            let labeled = AppStreamRecovery.labelIncompleteToolCallFragments(in: fragment)
            XCTAssertTrue(
                ToolCallParser.parseCandidates(from: labeled).isEmpty,
                "A nested recognized wrapper must not survive recovery: \(fragment)")
            XCTAssertFalse(labeled.contains("<tool_call>"))
            XCTAssertFalse(labeled.contains("```tool_call"))

            let result = ToolCallDispatchGate.evaluate(
                content: labeled,
                streamState: .completed,
                availableTools: available,
                forgeGuardrailsEnabled: false)
            XCTAssertTrue(result.dispatchableCalls.isEmpty, fragment)
            XCTAssertTrue(result.refusals.isEmpty, fragment)
        }
    }

    func testLabeledFragmentCannotBeRescuedByForgeGuardrailsAlternateDialect() {
        let fragment = "<tool_call><broken>\n<function=CronList>{}</function>\ncall:CronList{}\n</tool_call>"
        let labeled = AppStreamRecovery.labelIncompleteToolCallFragments(in: fragment)
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        XCTAssertTrue(labeled.contains("&lt;function=CronList>"))
        XCTAssertTrue(labeled.contains("call&#58;CronList"))
        XCTAssertTrue(ToolCallParser.parseCandidates(from: labeled).isEmpty)
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(
            from: labeled,
            availableToolNames: ["CronList"]
        ).isEmpty)

        let result = ToolCallDispatchGate.evaluate(
            content: labeled,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: true)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertTrue(result.preservedContent.contains("[Incomplete tool call, not executed]"))
    }

    func testLabeledFragmentCannotBeRescuedFromAnyForgeGuardrailsMarkdownFence() {
        let callJSON = #"{"name":"CronList","arguments":{}}"#
        let fences = [
            "```\n\(callJSON)\n```",
            "```json\(callJSON)```",
            "```tool_call\(callJSON)```",
        ]
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        for fence in fences {
            XCTAssertFalse(
                ForgeGuardrailsEngine.rescueToolCalls(
                    from: fence,
                    availableToolNames: ["CronList"]
                ).isEmpty,
                "The fixture must exercise a currently supported rescue form: \(fence)")

            let fragment = "<tool_call><broken>\n\(fence)\n</tool_call>"
            let labeled = AppStreamRecovery.labelIncompleteToolCallFragments(in: fragment)
            XCTAssertTrue(labeled.contains("&#96;&#96;&#96;"), fence)
            XCTAssertTrue(ToolCallParser.parseCandidates(from: labeled).isEmpty, fence)
            XCTAssertTrue(
                ForgeGuardrailsEngine.rescueToolCalls(
                    from: labeled,
                    availableToolNames: ["CronList"]
                ).isEmpty,
                fence)

            let result = ToolCallDispatchGate.evaluate(
                content: labeled,
                streamState: .completed,
                availableTools: available,
                forgeGuardrailsEnabled: true)
            XCTAssertTrue(result.dispatchableCalls.isEmpty, fence)
            XCTAssertTrue(result.refusals.isEmpty, fence)
        }
    }

    func testAlreadyLabeledCompleteMarkdownCallStaysInertThroughGuardrailRescue() {
        let callJSON = #"{"name":"CronList","arguments":{}}"#
        let persisted = "[Incomplete tool call, not executed]\n```tool_call\n\(callJSON)\n```"
        let recovered = AppStreamRecovery.labelIncompleteToolCallFragments(in: persisted)
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        XCTAssertTrue(recovered.contains("[Incomplete tool call, not executed]"))
        XCTAssertTrue(ToolCallParser.parseCandidates(from: recovered).isEmpty)
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(
            from: recovered,
            availableToolNames: ["CronList"]
        ).isEmpty)

        let result = ToolCallDispatchGate.evaluate(
            content: recovered,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: true)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertTrue(result.preservedContent.contains("[Incomplete tool call, not executed]"))
        XCTAssertTrue(result.preservedContent.contains(callJSON))
    }

    func testAlreadyLabeledGemmaCallStaysInertThroughGuardrailRescue() {
        let persisted = "[Incomplete tool call, not executed]\ncall:CronList{}"
        let recovered = AppStreamRecovery.labelIncompleteToolCallFragments(in: persisted)
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        XCTAssertEqual(recovered, persisted)
        XCTAssertTrue(ToolCallParser.parseCandidates(from: recovered).isEmpty)
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(
            from: recovered,
            availableToolNames: ["CronList"]
        ).isEmpty)

        let result = ToolCallDispatchGate.evaluate(
            content: recovered,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: true)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertTrue(result.preservedContent.contains("[Incomplete tool call, not executed]"))
        XCTAssertTrue(result.preservedContent.contains("call:CronList{}"))
    }

    func testPersistedCRLFInertLabelPreventsMarkdownCallDispatch() {
        let callJSON = #"{"name":"CronList","arguments":{}}"#
        let persisted = "[Incomplete tool call, not executed]\r\n```tool_call\r\n\(callJSON)\r\n```"
        let recovered = AppStreamRecovery.labelIncompleteToolCallFragments(in: persisted)
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        XCTAssertEqual(recovered, persisted)
        XCTAssertTrue(ToolCallParser.parseCandidates(from: recovered).isEmpty)
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(
            from: recovered,
            availableToolNames: ["CronList"]
        ).isEmpty)

        let result = ToolCallDispatchGate.evaluate(
            content: recovered,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: true)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertTrue(result.preservedContent.contains("[Incomplete tool call, not executed]"))
        XCTAssertTrue(result.preservedContent.contains(callJSON))
    }

    func testIndentedInertLabelBlocksGemmaAndMarkdownCalls() {
        let callJSON = #"{"name":"CronList","arguments":{}}"#
        let gemma = "[Incomplete tool call, not executed]\n  call:CronList{}"
        let markdown = "[Incomplete tool call, not executed]\n  ```tool_call\n\(callJSON)\n  ```"
        let indentedLabelGemma = "  [Incomplete tool call, not executed]  \r\n  call:CronList{}"
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(
            from: gemma,
            availableToolNames: ["CronList"]
        ).isEmpty)
        XCTAssertTrue(ToolCallParser.parseCandidates(from: markdown).isEmpty)

        for persisted in [gemma, markdown, indentedLabelGemma] {
            XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(
                from: persisted,
                availableToolNames: ["CronList"]
            ).isEmpty, persisted)
            let result = ToolCallDispatchGate.evaluate(
                content: persisted,
                streamState: .completed,
                availableTools: available,
                forgeGuardrailsEnabled: true)
            XCTAssertTrue(result.dispatchableCalls.isEmpty, persisted)
        }
    }

    func testLabeledMalformedFragmentDoesNotHideSeparateForgeGuardrailsCall() {
        let malformed = "<tool_call><broken>\n</tool_call>"
        let separateCall = "<function=CronList>{}</function>"
        let recovered = AppStreamRecovery.labelIncompleteToolCallFragments(
            in: malformed + "\n" + separateCall)
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        XCTAssertTrue(recovered.contains("[Incomplete tool call, not executed]"))
        XCTAssertTrue(recovered.hasSuffix(separateCall))
        XCTAssertEqual(
            ForgeGuardrailsEngine.rescueToolCalls(
                from: recovered,
                availableToolNames: ["CronList"]
            ).map(\.name),
            ["CronList"])

        let result = ToolCallDispatchGate.evaluate(
            content: recovered,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: true)

        XCTAssertEqual(result.dispatchableCalls.map(\.name), ["CronList"])
        XCTAssertTrue(result.preservedContent.contains("[Incomplete tool call, not executed]"))
    }

    func testXMLCloseMarkerInsideJSONStringDoesNotExposeNestedCandidate() {
        let nestedCall = #"<tool_call><name>read_file</name><arguments>{"path":"inside.txt"}</arguments></tool_call>"#
        let escapedNestedCall = nestedCall.replacingOccurrences(of: "\"", with: "\\\"")
        let malformedOuter = #"<tool_call><name>read_file</name><arguments>{"path":"literal </tool_call>\#(escapedNestedCall)"}</arguments><broken></tool_call>"#

        let labeled = AppStreamRecovery.labelIncompleteToolCallFragments(in: malformedOuter)

        XCTAssertEqual(
            labeled.components(separatedBy: "[Incomplete tool call, not executed]").count - 1,
            1,
            "A delimiter-shaped string value belongs to the outer arguments payload.")
        XCTAssertTrue(labeled.contains("&lt;tool_call&gt;<name>read_file</name>"))
        XCTAssertTrue(ToolCallParser.parseCandidates(from: labeled).isEmpty)

        let available = TurnAvailableTools(definitions: [
            .function(
                name: "read_file",
                description: "Read a file",
                parameters: .object(
                    properties: ["path": .string()],
                    required: ["path"],
                    additionalProperties: false)),
        ])
        let result = ToolCallDispatchGate.evaluate(
            content: labeled,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: true)
        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
    }

    func testFailureAndCancellationPersistClosedMalformedFragmentsOnce() throws {
        for reason in ["error", "cancelled"] {
            let model = AppModel()
            model.stopCronScheduler()
            var chat = AppChat(title: "malformed \(reason)")
            chat.messages = [AppChatMessage(role: .user, content: "prompt")]
            model.chats = [chat]
            model.selectedChatID = chat.id
            let fragment = "<tool_call><name>read_file</name><arguments>{not-json}</arguments></tool_call>"
            model.outputText = "Before \(fragment) after."

            model.finishCancelled(chatID: chat.id, reason: reason)

            let saved = try XCTUnwrap(model.chats.first(where: { $0.id == chat.id }))
            let partial = try XCTUnwrap(saved.messages.last)
            let selectedLanguage = AppLanguage.resolve(
                UserDefaults.standard.string(forKey: AppLanguage.storageKey)
                    ?? AppLanguage.system.rawValue)
            let label = String(
                localized: "Incomplete tool call, not executed",
                bundle: .module,
                locale: selectedLanguage.locale)
            let inertFragment = fragment.replacingOccurrences(
                of: "<tool_call>", with: "&lt;tool_call&gt;")
            XCTAssertEqual(partial.content.components(separatedBy: label).count - 1, 1)
            XCTAssertTrue(partial.content.contains("Before \n\(label)\n\(inertFragment) after."))
            XCTAssertEqual(
                AppStreamRecovery.labelIncompleteToolCallFragments(in: partial.content, label: label),
                partial.content)
            XCTAssertEqual(partial.stopReason, reason)
            let anchor = try XCTUnwrap(saved.recoveryAnchor)
            XCTAssertEqual(anchor.interruptedContentHash, AppStreamRecovery.contentHash(
                forPersistedRowContent: partial.content))
            XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: saved.messages))
        }
    }

    func testLabeledIncompleteFragmentCannotReachToolDispatch() {
        let content = AppStreamRecovery.labelIncompleteToolCallFragments(
            in: "Run this: <tool_call><name>read_file</name><arguments>{\"path\":\"secret\"}")
        let available = AppToolCatalog.captureTurnAvailableTools(
            for: nil,
            globalMcpServers: [],
            contextTokens: nil,
            webToolsEnabled: false)

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertTrue(result.preservedContent.contains("[Incomplete tool call, not executed]"))
    }

    func testAlreadyLabeledKimiMistralAndLongcatCallsStayInertThroughGuardrailRescue() {
        let callJSON = #"{"name":"CronList","arguments":{}}"#
        let dialects = [
            "kimi": "<|tool_call_begin|>functions.CronList:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
            "mistral": "[TOOL_CALLS] [\(callJSON)]",
            "longcat": "<longcat_tool_call>\(callJSON)</longcat_tool_call>",
        ]
        let label = "[Incomplete tool call, not executed]"
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "CronList",
                description: "List scheduled workflows",
                parameters: .object(properties: [:], required: [], additionalProperties: false)),
        ])

        for (dialect, fixture) in dialects {
            XCTAssertFalse(
                ForgeGuardrailsEngine.rescueToolCalls(
                    from: fixture,
                    availableToolNames: ["CronList"]
                ).isEmpty,
                "The fixture must exercise a currently supported rescue form: \(dialect)")

            let persisted = "\(label)\n\(fixture)"
            XCTAssertTrue(
                ForgeGuardrailsEngine.rescueToolCalls(
                    from: persisted,
                    availableToolNames: ["CronList"]
                ).isEmpty,
                "A labeled fragment must stay inert in the \(dialect) dialect")

            let result = ToolCallDispatchGate.evaluate(
                content: persisted,
                streamState: .completed,
                availableTools: available,
                forgeGuardrailsEnabled: true)
            XCTAssertTrue(result.dispatchableCalls.isEmpty, dialect)
            XCTAssertTrue(result.refusals.isEmpty, dialect)
            XCTAssertTrue(result.preservedContent.contains(label), dialect)
        }
    }
}
