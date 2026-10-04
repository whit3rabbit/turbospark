import XCTest

@testable import TurboSparkApp
import TurboSpark

final class AppChatInstructionPinSelectionTests: XCTestCase {
    private final class SessionMarker {}

    func testCandidatesComeOnlyFromExplicitlyPinnedEligibleRowsBeforeBoundary() {
        let messages = [
            AppChatMessage(role: .user, content: "Old but not pinned."),
            AppChatMessage(role: .assistant, content: "Assistant text.", isStandingInstruction: true),
            AppChatMessage(
                role: .user,
                content: "--- Attachment: notes.txt (Text) ---\nprivate file text\n--- End of notes.txt ---",
                isStandingInstruction: true),
            AppChatMessage(
                role: .user,
                content: "  Keep this wording exactly.\n",
                isStandingInstruction: true),
            AppChatMessage(
                role: .user,
                content: "Image message.",
                imagePaths: ["/image.png"],
                isStandingInstruction: true),
            AppChatMessage(
                role: .user,
                content: "Beyond the summary boundary.",
                isStandingInstruction: true),
        ]

        let candidates = AppChatInstructionPin.pinnedUserRows(messages, before: 5)

        XCTAssertEqual(candidates.count, 1)
        XCTAssertEqual(candidates[0].index, 3)
        XCTAssertEqual(candidates[0].text, "  Keep this wording exactly.\n")
        XCTAssertEqual(candidates[0].messageID, messages[3].id)
    }

    func testSelectionSortsTranscriptOrderAndSkipsAnOversizedMessage() async {
        let smallBlock = AppChatInstructionPin.injectionBlock(["later small pin"]).content
        let outcome = await AppChatInstructionPin.select(
            userRows: [
                .init(index: 4, text: "later small pin"),
                .init(index: 1, text: "first oversized pin"),
                .init(index: 5, text: "outside boundary"),
            ],
            boundary: 5,
            ceilingTokens: smallBlock.count,
            countTokens: { block in
                block.contains("first oversized pin") ? smallBlock.count + 1 : smallBlock.count
            })

        XCTAssertEqual(outcome.pinnedGroups, ["later small pin"])
        XCTAssertEqual(outcome.estimatedTokens, smallBlock.count)
    }

    func testSelectionCountsTheWholeRenderedBlockAndZeroCeilingDisablesIt() async {
        let group = "Keep every word."
        let rendered = AppChatInstructionPin.injectionBlock([group]).content
        let row = AppChatInstructionPin.CandidateRow(index: 0, text: group)
        let atLimit = await AppChatInstructionPin.select(
            userRows: [row],
            boundary: 1,
            ceilingTokens: rendered.count,
            countTokens: { $0.count })
        let belowLimit = await AppChatInstructionPin.select(
            userRows: [row],
            boundary: 1,
            ceilingTokens: rendered.count - 1,
            countTokens: { $0.count })
        var countCalls = 0
        let zeroCeiling = await AppChatInstructionPin.select(
            userRows: [row],
            boundary: 1,
            ceilingTokens: 0,
            countTokens: { _ in
                countCalls += 1
                return 1
            })

        XCTAssertEqual(atLimit.pinnedGroups, [group])
        XCTAssertEqual(atLimit.estimatedTokens, rendered.count)
        XCTAssertTrue(belowLimit.pinnedGroups.isEmpty)
        XCTAssertTrue(zeroCeiling.pinnedGroups.isEmpty)
        XCTAssertEqual(countCalls, 0)
    }

    func testWindowFitKeepsThePinnedBlockAndReinsertsItBeforeTheSummary() async {
        let pin = AppChatInstructionPin.injectionBlock(["Keep this instruction."])
        let system = ChatMessage(role: .system, content: "System")
        let summary = ChatMessage(role: .user, content: "Summary")
        let oldTurn = ChatMessage(role: .assistant, content: String(repeating: "old ", count: 20))
        let recent = ChatMessage(role: .user, content: "Recent question")
        let messages = [system, pin, summary, oldTurn, recent]
        let fullTokenCount = messages.reduce(0) { $0 + $1.content.utf8.count }

        let outcome = await AppChatInstructionPin.fitPreservingBlock(
            messages,
            injectedBlockIndex: 1,
            pinnedBlock: pin,
            maximumTokens: 105,
            countTokens: { request in
                request.reduce(0) { $0 + $1.content.utf8.count }
            },
            fitWindow: { request, limit in
                XCTAssertFalse(request.contains(pin))
                var retained = request
                var count = retained.reduce(0) { $0 + $1.content.utf8.count }
                while count >= limit {
                    let firstRemovable = retained.firstIndex { $0.role != .system } ?? 0
                    guard firstRemovable < retained.count - 1 else { break }
                    retained.remove(at: firstRemovable)
                    count = retained.reduce(0) { $0 + $1.content.utf8.count }
                }
                return AppChatInstructionPin.FitCandidate(
                    retained: retained,
                    measuredTokens: count,
                    removedTurnCount: request.count - retained.count,
                    hasRoomForGeneration: count < limit)
            })

        let pinIndex = outcome.retained.firstIndex(of: pin)
        let systemIndex = outcome.retained.firstIndex(of: system)
        let summaryIndex = outcome.retained.firstIndex(of: summary)
        XCTAssertNotNil(pinIndex)
        XCTAssertNotNil(systemIndex)
        XCTAssertEqual(pinIndex, systemIndex! + 1)
        if let summaryIndex {
            XCTAssertLessThan(pinIndex!, summaryIndex)
        }
        XCTAssertEqual(
            outcome.measuredTokens,
            outcome.retained.reduce(0) { $0 + $1.content.utf8.count })
        XCTAssertLessThan(outcome.measuredTokens, 105)
        XCTAssertLessThan(outcome.retained.count, messages.count)
        XCTAssertGreaterThan(fullTokenCount, 105)
        XCTAssertTrue(outcome.hasRoomForGeneration)
    }

    func testContinuationWindowFitKeepsPinsUnderContextPressure() async {
        let system = ChatMessage(role: .system, content: "System")
        let pin = AppChatInstructionPin.injectionBlock(["Keep this instruction verbatim."])
        let oldTurn = ChatMessage(role: .assistant, content: String(repeating: "old ", count: 20))
        let recent = ChatMessage(role: .user, content: "Recent question")
        let assistantPrefix = ChatMessage(role: .assistant, content: "Partial answer")
        let continuationMessages = [system, pin, oldTurn, recent, assistantPrefix]
        let maximumTokens = 110

        let outcome = await AppChatInstructionPin.fitPreservingBlock(
            continuationMessages,
            injectedBlockIndex: 1,
            pinnedBlock: pin,
            maximumTokens: maximumTokens,
            countTokens: { messages in
                messages.reduce(0) { $0 + $1.content.utf8.count }
            },
            fitWindow: { messages, limit in
                XCTAssertFalse(messages.contains(pin))
                var retained = messages
                var measured = retained.reduce(0) { $0 + $1.content.utf8.count }
                while measured >= limit {
                    guard let firstRemovable = retained.firstIndex(where: { $0.role != .system }),
                        firstRemovable < retained.count - 1
                    else { break }
                    retained.remove(at: firstRemovable)
                    measured = retained.reduce(0) { $0 + $1.content.utf8.count }
                }
                return AppChatInstructionPin.FitCandidate(
                    retained: retained,
                    measuredTokens: measured,
                    removedTurnCount: messages.count - retained.count,
                    hasRoomForGeneration: measured < limit)
            })

        XCTAssertTrue(outcome.retained.contains(pin))
        XCTAssertEqual(outcome.retained.firstIndex(of: pin), 1)
        XCTAssertTrue(outcome.retained.contains(assistantPrefix))
        XCTAssertFalse(outcome.retained.contains(oldTurn))
        XCTAssertLessThan(outcome.measuredTokens, maximumTokens)
        XCTAssertTrue(outcome.hasRoomForGeneration)
    }

    func testContinuationWindowFitDoesNotReinsertPinsWhenDisabled() async {
        let system = ChatMessage(role: .system, content: "System")
        let pin = AppChatInstructionPin.injectionBlock(["Keep this instruction."])
        let user = ChatMessage(role: .user, content: "Recent question")
        let assistantPrefix = ChatMessage(role: .assistant, content: "Partial answer")
        let continuationMessages = [system, pin, user, assistantPrefix]

        let outcome = await AppChatInstructionPin.fitPreservingBlock(
            continuationMessages,
            injectedBlockIndex: 1,
            pinnedBlock: nil,
            maximumTokens: 128,
            countTokens: { messages in
                messages.reduce(0) { $0 + $1.content.utf8.count }
            },
            fitWindow: { messages, limit in
                let measured = messages.reduce(0) { $0 + $1.content.utf8.count }
                return AppChatInstructionPin.FitCandidate(
                    retained: messages,
                    measuredTokens: measured,
                    removedTurnCount: continuationMessages.count - messages.count,
                    hasRoomForGeneration: measured < limit)
            })

        XCTAssertFalse(outcome.retained.contains(pin))
        XCTAssertTrue(outcome.retained.contains(user))
        XCTAssertTrue(outcome.retained.contains(assistantPrefix))
    }

    func testWindowFitTightensByPinnedOverheadWithoutFixedRetryLimit() async {
        let pin = AppChatInstructionPin.injectionBlock(["Keep this instruction."])
        let system = ChatMessage(role: .system, content: "System")
        let oldTurn = ChatMessage(role: .assistant, content: "Old turn")
        let recent = ChatMessage(role: .user, content: "Recent turn")
        let messages = [system, pin, oldTurn, recent]

        let outcome = await AppChatInstructionPin.fitPreservingBlock(
            messages,
            injectedBlockIndex: 1,
            pinnedBlock: pin,
            maximumTokens: 200,
            countTokens: { request in
                request.contains(oldTurn) ? 201 : 40
            },
            fitWindow: { request, limit in
                var retained = request
                if limit <= 99, let oldIndex = retained.firstIndex(of: oldTurn) {
                    retained.remove(at: oldIndex)
                }
                let measured = retained.contains(oldTurn) ? 100 : 40
                return AppChatInstructionPin.FitCandidate(
                    retained: retained,
                    measuredTokens: measured,
                    removedTurnCount: request.count - retained.count,
                    hasRoomForGeneration: measured < limit)
            })

        XCTAssertTrue(outcome.hasRoomForGeneration)
        XCTAssertFalse(outcome.retained.contains(oldTurn))
        XCTAssertTrue(outcome.retained.contains(pin))
        XCTAssertEqual(outcome.measuredTokens, 40)
    }

    func testWindowFitRemovesOnlyTheSyntheticBlockByProjectionIndex() async {
        let pin = AppChatInstructionPin.injectionBlock(["Preserved user instructions:"])
        let system = ChatMessage(role: .system, content: "System")
        let transcriptRow = pin
        let recent = ChatMessage(role: .user, content: "Recent turn")
        let messages = [system, pin, transcriptRow, recent]

        let outcome = await AppChatInstructionPin.fitPreservingBlock(
            messages,
            injectedBlockIndex: 1,
            pinnedBlock: pin,
            maximumTokens: 100,
            countTokens: { _ in 10 },
            fitWindow: { request, limit in
                AppChatInstructionPin.FitCandidate(
                    retained: request,
                    measuredTokens: 10,
                    removedTurnCount: 0,
                    hasRoomForGeneration: 10 < limit)
            })

        XCTAssertTrue(outcome.hasRoomForGeneration)
        XCTAssertEqual(outcome.retained.filter { $0 == pin }.count, 2)
    }

    func testWindowFitDropsStaleInjectedBlockWhenPinCacheIsInvalid() async {
        let stalePin = AppChatInstructionPin.injectionBlock(["No longer enabled."])
        let system = ChatMessage(role: .system, content: "System")
        let summary = ChatMessage(role: .user, content: "Summary")
        let recent = ChatMessage(role: .user, content: "Recent turn")
        let messages = [system, stalePin, summary, recent]
        var fitInput: [ChatMessage] = []

        let outcome = await AppChatInstructionPin.fitPreservingBlock(
            messages,
            injectedBlockIndex: 1,
            pinnedBlock: nil,
            maximumTokens: 100,
            countTokens: { _ in 20 },
            fitWindow: { request, limit in
                fitInput = request
                return AppChatInstructionPin.FitCandidate(
                    retained: request,
                    measuredTokens: 20,
                    removedTurnCount: 0,
                    hasRoomForGeneration: 20 < limit)
            })

        XCTAssertFalse(fitInput.contains(stalePin))
        XCTAssertFalse(outcome.retained.contains(stalePin))
        XCTAssertTrue(outcome.retained.contains(summary))
        XCTAssertTrue(outcome.hasRoomForGeneration)
    }

    func testCachedSelectionIsRejectedForDifferentLoadedSession() {
        let cachedSession = SessionMarker()
        let activeSession = SessionMarker()
        let entry = AppChatInstructionPin.CacheEntry(
            boundary: 1,
            outcome: AppChatInstructionPin.PinningOutcome(
                pinnedGroups: ["Keep this instruction."], estimatedTokens: 12),
            sessionIdentity: ObjectIdentifier(cachedSession))

        XCTAssertTrue(
            AppChatInstructionPin.cacheMatches(
                entry,
                activeSessionIdentity: ObjectIdentifier(cachedSession),
                ceilingTokens: 512,
                reasoning: .off,
                sourceRows: []))
        XCTAssertFalse(
            AppChatInstructionPin.cacheMatches(
                entry,
                activeSessionIdentity: ObjectIdentifier(activeSession),
                ceilingTokens: 512,
                reasoning: .off,
                sourceRows: []))
        XCTAssertFalse(
            AppChatInstructionPin.cacheMatches(
                entry, activeSessionIdentity: nil, ceilingTokens: 1024, reasoning: .off,
                sourceRows: []))
        XCTAssertFalse(
            AppChatInstructionPin.cacheMatches(
                entry, activeSessionIdentity: nil, ceilingTokens: 512, reasoning: .high,
                sourceRows: []))
        XCTAssertFalse(
            AppChatInstructionPin.cacheMatches(
                entry, activeSessionIdentity: nil, ceilingTokens: 512, reasoning: .off,
                sourceRows: []))
        XCTAssertFalse(
            AppChatInstructionPin.cacheMatches(
                entry,
                activeSessionIdentity: ObjectIdentifier(cachedSession),
                ceilingTokens: 512,
                reasoning: .off,
                sourceRows: [.init(index: 0, messageID: UUID(), text: "Changed pin")]))
    }

    func testCompactionUsesTheReasoningSnapshotCapturedForTheRequest() {
        XCTAssertEqual(
            AppChatCompaction.pinningReasoning(requestSnapshot: .off, currentReasoning: .high),
            .off)
        XCTAssertEqual(
            AppChatCompaction.pinningReasoning(requestSnapshot: nil, currentReasoning: .high),
            .high)
    }

    func testInjectionBlockUsesUserRoleAndKeepsPinnedTextVerbatim() {
        let text = "  Keep this wording exactly.\n"

        let block = AppChatInstructionPin.injectionBlock([text, "Second instruction."])

        XCTAssertEqual(block.role, .user)
        XCTAssertTrue(block.content.contains("Preserved user instructions"))
        XCTAssertTrue(block.content.contains("\n\(text)\n"))
        XCTAssertTrue(
            block.content.range(of: text)!.lowerBound
                < block.content.range(of: "Second instruction.")!.lowerBound)
    }

    @MainActor
    func testHistoryPlacesPinnedBlockAfterSystemAndBeforeSummary() async throws {
        let chat = AppChat(
            messages: [
                AppChatMessage(role: .user, content: "Keep this instruction.", isStandingInstruction: true),
                AppChatMessage(role: .assistant, content: "Older answer."),
                AppChatMessage(role: .user, content: "Recent prompt."),
            ],
            contextSummary: "Earlier turns summarized here.",
            compactedMessageCount: 2)
        let model = AppModel()
        model.stopCronScheduler()
        model.chats = [chat]
        model.selectedChatID = chat.id
        await model.rebuildInstructionPinningCache(
            chatID: chat.id,
            boundary: 2,
            messages: chat.messages,
            sessionIdentity: nil,
            reasoning: .off,
            countTokens: { $0.utf8.count })

        let projection = model.buildAppendOnlyHistoryProjection(chatIndex: 0, project: nil)

        let pinIndex = try XCTUnwrap(projection.messages.firstIndex {
            $0.content.contains("Preserved user instructions")
        })
        let summaryIndex = try XCTUnwrap(projection.messages.firstIndex {
            $0.content.contains("Earlier turns summarized here.")
        })
        let recentIndex = try XCTUnwrap(projection.messages.firstIndex {
            $0.content == "Recent prompt."
        })
        XCTAssertEqual(projection.messages[pinIndex].role, .user)
        XCTAssertLessThan(pinIndex, summaryIndex)
        XCTAssertLessThan(summaryIndex, recentIndex)
        XCTAssertEqual(projection.sourceRowIndexByMessage.count, projection.messages.count)
        XCTAssertNil(projection.sourceRowIndexByMessage[pinIndex])
        XCTAssertNil(projection.sourceRowIndexByMessage[summaryIndex])

        let estimate = model.buildEstimateParts()
        let estimatePinIndex = try XCTUnwrap(estimate.history.firstIndex {
            $0.content.contains("Preserved user instructions")
        })
        let estimateSummaryIndex = try XCTUnwrap(estimate.history.firstIndex {
            $0.content.contains("Earlier turns summarized here.")
        })
        XCTAssertLessThan(estimatePinIndex, estimateSummaryIndex)
        XCTAssertEqual(
            estimate.pieces.first(where: { $0.label == "Pinned instructions" })?.content,
            projection.messages[pinIndex].content)

        model.instructionPinningEnabled = false
        let disabledProjection = model.buildAppendOnlyHistoryProjection(chatIndex: 0, project: nil)
        XCTAssertFalse(disabledProjection.messages.contains {
            $0.content.contains("Preserved user instructions")
        })
    }

    @MainActor
    func testManualCompactCommandCommitsBoundaryAndRefreshesPins() async throws {
        let chat = compactionChat()
        let model = AppModel()
        model.stopCronScheduler()
        model.chats = [chat]
        model.selectedChatID = chat.id
        installCompactionOverrides(on: model, summary: "Manual summary.")

        model.handleCompactCommand("/compact")
        let task = try XCTUnwrap(model.submissionTask)
        await task.value

        try assertCompactionPins(
            on: model, chat: chat, summary: "Manual summary.", expectedBoundary: 4)
    }

    @MainActor
    func testAutomaticCompactionCommitsBoundaryAndRefreshesPins() async throws {
        let chat = compactionChat()
        let model = AppModel()
        model.stopCronScheduler()
        model.chats = [chat]
        model.selectedChatID = chat.id
        installCompactionOverrides(on: model, summary: "Automatic summary.")

        let compacted = await model.runAutoCompactionIfNeeded(
            chatID: chat.id,
            project: nil,
            rawHistory: [
                ChatMessage(role: .system, content: "System"),
                ChatMessage(role: .user, content: "Older context"),
            ],
            maxContext: 100,
            reservedForNew: 10,
            reasoning: .off)

        XCTAssertTrue(compacted)
        try assertCompactionPins(
            on: model, chat: chat, summary: "Automatic summary.", expectedBoundary: 4)
    }

    @MainActor
    private func compactionChat() -> AppChat {
        AppChat(
            messages: [
                AppChatMessage(role: .user, content: "First instruction.", isStandingInstruction: true),
                AppChatMessage(role: .assistant, content: "First answer."),
                AppChatMessage(role: .user, content: "Second instruction.", isStandingInstruction: true),
                AppChatMessage(role: .assistant, content: "Second answer."),
                AppChatMessage(role: .user, content: "Recent instruction.", isStandingInstruction: true),
                AppChatMessage(role: .user, content: "Recent question."),
            ])
    }

    @MainActor
    private func installCompactionOverrides(on model: AppModel, summary: String) {
        model.compactionKeepRecentTurns = 2
        model.compactionSummaryOverride = { _, options in
            XCTAssertEqual(options.reasoning, .off)
            return summary
        }
        model.compactionTokenCountOverride = { messages, _ in
            if messages.count > 1 { return 90 }
            return messages.reduce(0) { $0 + $1.content.utf8.count }
        }
    }

    @MainActor
    private func assertCompactionPins(
        on model: AppModel, chat: AppChat, summary: String, expectedBoundary: Int
    ) throws {
        let state = model.compactionState(chatID: chat.id)
        XCTAssertEqual(state.summary, summary)
        XCTAssertEqual(state.boundary, expectedBoundary)
        let cache = model.instructionPinningCache[chat.id]
        XCTAssertEqual(cache?.boundary, expectedBoundary)
        XCTAssertEqual(
            cache?.outcome.pinnedGroups, ["First instruction.", "Second instruction."])

        let projection = model.buildAppendOnlyHistoryProjection(chatIndex: 0, project: nil)
        let pinBlock = try XCTUnwrap(projection.instructionPinBlockIndex.map {
            projection.messages[$0]
        })
        XCTAssertTrue(pinBlock.content.contains("First instruction."))
        XCTAssertTrue(pinBlock.content.contains("Second instruction."))
        XCTAssertFalse(pinBlock.content.contains("Recent instruction."))
        XCTAssertEqual(projection.sourceRowIndexByMessage.count, projection.messages.count)
    }

    @MainActor
    func testUnpinningACompactedMessageInvalidatesItsCachedSelection() {
        let pinnedMessage = AppChatMessage(
            role: .user, content: "Keep this instruction.", isStandingInstruction: true)
        let chat = AppChat(
            messages: [pinnedMessage, AppChatMessage(role: .assistant, content: "Older answer.")],
            contextSummary: "Summary.",
            compactedMessageCount: 1)
        let model = AppModel()
        model.stopCronScheduler()
        model.chats = [chat]
        model.instructionPinningCache[chat.id] = AppChatInstructionPin.CacheEntry(
            boundary: 1,
            outcome: AppChatInstructionPin.PinningOutcome(
                pinnedGroups: [pinnedMessage.content], estimatedTokens: 12),
            sourceRows: AppChatInstructionPin.pinnedUserRows(chat.messages, before: 1))

        model.setMessageStandingInstruction(id: pinnedMessage.id, pinned: false, chatID: chat.id)

        XCTAssertNil(model.instructionPinningCache[chat.id])
        XCTAssertFalse(model.chats[0].messages[0].isStandingInstruction)
    }
}
