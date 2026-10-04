import Foundation
import XCTest

import TurboSpark
@testable import TurboSparkApp

final class AppStreamRecoveryTests: XCTestCase {
    @MainActor
    func testRecoveryEventsRetainAcrossAgentLoopAndAnchoredRetryButResetForNewTurn() {
        let existing: [StreamRecoveryEvent] = [.continued(attempt: 1, tokensSoFar: 80)]
        let model = AppModel()
        model.stopCronScheduler()
        let chat = AppChat(title: "recovery events")
        model.chats = [chat]
        model.streamRecoveryEvents[chat.id] = existing

        model.beginRecoveryEventHistoryForGenerationStart(step: 1, chatID: chat.id)
        XCTAssertEqual(
            model.streamRecoveryEvents[chat.id],
            existing,
            "Agent-loop re-entry must retain the current logical turn's events.")

        model.pendingRecoveryRetryCounts[chat.id] = 1
        model.beginRecoveryEventHistoryForGenerationStart(step: 0, chatID: chat.id)
        XCTAssertEqual(
            model.streamRecoveryEvents[chat.id],
            existing,
            "An anchored retry also belongs to the interrupted logical turn.")

        model.pendingRecoveryRetryCounts[chat.id] = nil
        model.beginRecoveryEventHistoryForGenerationStart(step: 0, chatID: chat.id)
        XCTAssertEqual(
            model.streamRecoveryEvents[chat.id],
            [],
            "A standalone step-zero generation starts a new recovery-event history.")
    }

    func testSuccessfulRecoveryRetryEventIsTypedAndBounded() throws {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let partial = AppChatMessage(role: .assistant, content: "partial", stopReason: "error")
        var transcript = [prompt]
        var anchor: RecoveryAnchor? = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: partial.id,
            persistedRowContent: partial.content,
            continuationsUsed: 0,
            retriesUsed: 1,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        var events: [StreamRecoveryEvent] = []
        let result = try generationResult(stopReason: "endOfTurn")

        AppStreamRecovery.commitSuccessfulRecoveryRetry(
            into: &transcript,
            recoveryAnchor: &anchor,
            content: "replacement answer",
            reasoning: "",
            result: result,
            alternates: [partial])
        AppStreamRecovery.recordSuccessfulRecoveryRetry(attempt: 2, into: &events)

        XCTAssertEqual(transcript.last?.content, "replacement answer")
        XCTAssertNil(anchor)
        XCTAssertEqual(events, [.retrySucceeded(attempt: 2)])
        AppStreamRecovery.recordSuccessfulRecoveryRetry(attempt: 4, into: &events)
        XCTAssertEqual(events, [.retrySucceeded(attempt: 2)])
    }

    func testContinuationDecisionAllowsOnlyTheConfiguredNumberOfCalls() throws {
        XCTAssertEqual(AppStreamRecovery.maxContinuationsPerTurn, 3)
        XCTAssertEqual(AppStreamRecovery.continuationOutcome(used: 0), .proceed(attempt: 1))
        XCTAssertEqual(AppStreamRecovery.continuationOutcome(used: 2), .proceed(attempt: 3))
        XCTAssertEqual(AppStreamRecovery.continuationOutcome(used: 3), .exhausted(limit: 3))
        XCTAssertEqual(AppStreamRecovery.continuationOutcome(used: 4), .exhausted(limit: 3))
        XCTAssertEqual(AppStreamRecovery.continuationOutcome(used: -1), .exhausted(limit: 3))

        let maxTokens = try generationResult(stopReason: "maxTokens")
        let completed = try generationResult(stopReason: "endOfTurn")
        XCTAssertTrue(AppStreamRecovery.shouldContinue(maxTokens, used: 2))
        XCTAssertFalse(AppStreamRecovery.shouldContinue(maxTokens, used: 3))
        XCTAssertFalse(AppStreamRecovery.shouldContinue(completed, used: 0))
    }

    func testRetryDecisionValidatesAnchorAndStopsAtTheHardLimit() throws {
        XCTAssertEqual(AppStreamRecovery.maxRecoveryRetries, 3)

        let messageID = UUID()
        let row = AppChatMessage(id: messageID, role: .assistant, content: "partial")
        let messages = [AppChatMessage(role: .user, content: "prompt"), row]
        let availableAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: messageID,
            persistedRowContent: row.content,
            continuationsUsed: 3,
            retriesUsed: 2,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let exhaustedAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: messageID,
            persistedRowContent: row.content,
            continuationsUsed: 3,
            retriesUsed: 3,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))

        XCTAssertEqual(
            AppStreamRecovery.retryOutcome(anchor: availableAnchor, messages: messages),
            .retry(attempt: 3))
        XCTAssertEqual(
            AppStreamRecovery.retryOutcome(anchor: exhaustedAnchor, messages: messages),
            .exhausted(limit: 3))

        let changedRow = AppChatMessage(id: messageID, role: .assistant, content: "edited")
        let changedMessages = [messages[0], changedRow]
        XCTAssertEqual(AppStreamRecovery.retryOutcome(anchor: availableAnchor, messages: changedMessages), .conflict)
    }

    func testAnchoredRetryMovesOnlyTheMatchingPartialIntoTheVariantPath() throws {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let partial = AppChatMessage(
            role: .assistant,
            content: "partial answer",
            reasoning: "partial reasoning",
            stopReason: "error")
        let messages = [prompt, partial]
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: partial.id,
            persistedRowContent: partial.content,
            continuationsUsed: 1,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))

        let preparation = AppStreamRecovery.prepareAnchoredRetry(anchor: anchor, messages: messages)

        guard case .retry(let plan) = preparation else {
            return XCTFail("Expected a matching recovery anchor to produce a retry plan.")
        }
        XCTAssertEqual(plan.attempt, 1)
        XCTAssertEqual(plan.transcript, [prompt])
        XCTAssertEqual(plan.alternates.count, 1)
        XCTAssertEqual(plan.alternates[0].content, partial.content)
        XCTAssertEqual(plan.alternates[0].reasoning, partial.reasoning)
        XCTAssertTrue(plan.alternates[0].alternates.isEmpty)
        XCTAssertEqual(messages, [prompt, partial], "Preparing a retry must not mutate its input.")
    }

    func testAnchoredRetryMismatchAndExhaustionReturnTypedOutcomesWithoutMutation() throws {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let partial = AppChatMessage(role: .assistant, content: "partial", stopReason: "cancelled")
        let messages = [prompt, partial]
        let matchingAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: partial.id,
            persistedRowContent: partial.content,
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let exhaustedAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: partial.id,
            persistedRowContent: partial.content,
            continuationsUsed: 0,
            retriesUsed: AppStreamRecovery.maxRecoveryRetries,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))

        var changed = messages
        changed[1].content = "modified"
        let mismatchBefore = changed
        XCTAssertEqual(
            AppStreamRecovery.prepareAnchoredRetry(anchor: matchingAnchor, messages: changed),
            .conflict)
        XCTAssertEqual(changed, mismatchBefore)

        let laterTranscript = messages + [AppChatMessage(role: .user, content: "later prompt")]
        XCTAssertEqual(
            AppStreamRecovery.prepareAnchoredRetry(anchor: matchingAnchor, messages: laterTranscript),
            .conflict)

        let beforeExhaustion = messages
        XCTAssertEqual(
            AppStreamRecovery.prepareAnchoredRetry(anchor: exhaustedAnchor, messages: messages),
            .exhausted(limit: AppStreamRecovery.maxRecoveryRetries))
        XCTAssertEqual(messages, beforeExhaustion)
    }

    func testReloadedInFlightRetrySnapshotMismatchAndExhaustionFailClosed() throws {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let interrupted = AppChatMessage(role: .assistant, content: "partial", stopReason: "error")
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: interrupted.id,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        guard case .retry(let retryPlan) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: anchor,
            messages: [prompt, interrupted]),
            let stagedAnchor = retryPlan.stagedRecoveryAnchor
        else { return XCTFail("Expected a staged in-flight recovery anchor.") }

        var changedSnapshot = interrupted
        changedSnapshot.content = "tampered partial"
        let mismatchAnchor = try XCTUnwrap(RecoveryAnchor(
            retainedMessageCount: stagedAnchor.retainedMessageCount,
            interruptedMessageID: stagedAnchor.interruptedMessageID,
            interruptedContentHash: stagedAnchor.interruptedContentHash,
            continuationsUsed: stagedAnchor.continuationsUsed,
            retriesUsed: stagedAnchor.retriesUsed,
            recordedAt: stagedAnchor.recordedAt,
            interruptedMessage: changedSnapshot))
        let mismatchedChat = AppChat(
            title: "mismatched retry reload",
            messages: [prompt],
            recoveryAnchor: mismatchAnchor)
        let restoredMismatch = try JSONDecoder().decode(
            AppChat.self,
            from: JSONEncoder().encode(mismatchedChat))
        let mismatchBefore = restoredMismatch.messages
        XCTAssertEqual(mismatchBefore, [prompt], "A mismatched snapshot must not be reattached.")
        XCTAssertEqual(
            AppStreamRecovery.prepareResponseRetry(
                recoveryAnchor: restoredMismatch.recoveryAnchor,
                messages: restoredMismatch.messages),
            .conflict)
        XCTAssertEqual(restoredMismatch.messages, mismatchBefore)

        let almostExhaustedAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: interrupted.id,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: AppStreamRecovery.maxRecoveryRetries - 1,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        guard case .retry(let finalRetry) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: almostExhaustedAnchor,
            messages: [prompt, interrupted]),
            let exhaustedInFlightAnchor = finalRetry.stagedRecoveryAnchor
        else { return XCTFail("Expected the last permitted retry to stage its count.") }
        var inFlightChat = AppChat(
            title: "exhausted retry reload",
            messages: [prompt, interrupted],
            recoveryAnchor: almostExhaustedAnchor)
        inFlightChat.messages = finalRetry.transcript
        inFlightChat.recoveryAnchor = exhaustedInFlightAnchor
        let restoredExhaustion = try JSONDecoder().decode(
            AppChat.self,
            from: JSONEncoder().encode(inFlightChat))
        let exhaustionBefore = restoredExhaustion.messages
        XCTAssertEqual(exhaustionBefore, [prompt, interrupted])
        XCTAssertEqual(
            AppStreamRecovery.prepareResponseRetry(
                recoveryAnchor: restoredExhaustion.recoveryAnchor,
                messages: restoredExhaustion.messages),
            .exhausted(limit: AppStreamRecovery.maxRecoveryRetries))
        XCTAssertEqual(restoredExhaustion.messages, exhaustionBefore)
    }

    func testInterruptedResponseWithoutAnchorCannotFallThroughToOrdinaryRetry() {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let interrupted = AppChatMessage(role: .assistant, content: "partial", stopReason: "error")
        let messages = [prompt, interrupted]

        XCTAssertEqual(
            AppStreamRecovery.prepareResponseRetry(recoveryAnchor: nil, messages: messages),
            .conflict)
        XCTAssertEqual(messages, [prompt, interrupted])
    }

    func testCompletedProseStillUsesTheOrdinaryRetryPath() {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let completed = AppChatMessage(role: .assistant, content: "complete", stopReason: "endOfTurn")

        guard case .retry(let plan) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: nil, messages: [prompt, completed])
        else { return XCTFail("A completed prose response should keep the ordinary retry path.") }

        XCTAssertNil(plan.recoveryAttempt)
        XCTAssertEqual(plan.transcript, [prompt])
        XCTAssertEqual(plan.alternates.map(\.content), ["complete"])
    }

    func testEarlierRecoveryAnchorDoesNotBlockRetryOfLaterCompletedReply() throws {
        let firstPrompt = AppChatMessage(role: .user, content: "first prompt")
        let interrupted = AppChatMessage(role: .assistant, content: "partial", stopReason: "error")
        let laterPrompt = AppChatMessage(role: .user, content: "later prompt")
        let completed = AppChatMessage(role: .assistant, content: "complete", stopReason: "endOfTurn")
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: interrupted.id,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: AppStreamRecovery.maxRecoveryRetries))
        let messages = [firstPrompt, interrupted, laterPrompt, completed]

        guard case .retry(let plan) = AppStreamRecovery.prepareResponseRetry(
            recoveryAnchor: anchor, messages: messages)
        else { return XCTFail("A previous interrupted turn must not govern the later reply.") }

        XCTAssertNil(plan.recoveryAttempt)
        XCTAssertNil(plan.stagedRecoveryAnchor)
        XCTAssertEqual(plan.transcript, [firstPrompt, interrupted, laterPrompt])
        XCTAssertEqual(plan.alternates, [completed])
        XCTAssertEqual(
            AppStreamRecovery.prepareAnchoredRetry(anchor: anchor, messages: messages), .conflict)
    }

    func testSuccessfulAnchoredRetryCommitClearsItsRecoveryAnchor() throws {
        let prompt = AppChatMessage(role: .user, content: "prompt")
        let partial = AppChatMessage(role: .assistant, content: "partial", stopReason: "error")
        var transcript = [prompt]
        var anchor: RecoveryAnchor? = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: partial.id,
            persistedRowContent: partial.content,
            continuationsUsed: 0,
            retriesUsed: 1,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let result = try generationResult(stopReason: "endOfTurn")

        AppStreamRecovery.commitSuccessfulRecoveryRetry(
            into: &transcript,
            recoveryAnchor: &anchor,
            content: "replacement answer",
            reasoning: "replacement reasoning",
            result: result,
            alternates: [partial])

        XCTAssertNil(anchor)
        XCTAssertEqual(transcript.count, 2)
        XCTAssertEqual(transcript[1].content, "replacement answer")
        XCTAssertEqual(transcript[1].alternates.count, 1)
        XCTAssertEqual(transcript[1].alternates[0].content, partial.content)
    }

    func testAnchorFactoryRejectsCountsOutsidePersistedBounds() {
        let messageID = UUID()
        let date = Date(timeIntervalSince1970: 1_700_000_000)

        XCTAssertNil(AppStreamRecovery.anchor(
            retainedMessageCount: -1,
            messageID: messageID,
            persistedRowContent: "partial",
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: date))
        XCTAssertNil(AppStreamRecovery.anchor(
            retainedMessageCount: 0,
            messageID: messageID,
            persistedRowContent: "partial",
            continuationsUsed: 4,
            retriesUsed: 0,
            recordedAt: date))
        XCTAssertNil(AppStreamRecovery.anchor(
            retainedMessageCount: 0,
            messageID: messageID,
            persistedRowContent: "partial",
            continuationsUsed: 0,
            retriesUsed: 4,
            recordedAt: date))
    }

    func testAnchorEncodingKeepsPersistedShapeAndRoundTrips() throws {
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 2,
            messageID: UUID(uuidString: "3D71D674-9DDC-41B7-A68A-8F62A849A4ED")!,
            persistedRowContent: "abc",
            continuationsUsed: 1,
            retriesUsed: 2,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))

        let encoded = try JSONEncoder().encode(anchor)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(Set(object.keys), Set([
            "retainedMessageCount",
            "interruptedMessageID",
            "interruptedContentHash",
            "continuationsUsed",
            "retriesUsed",
            "recordedAt",
        ]))
        XCTAssertEqual(
            object["interruptedContentHash"] as? String,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        XCTAssertEqual(try JSONDecoder().decode(RecoveryAnchor.self, from: encoded), anchor)
    }

    func testExactRowValidationRequiresIndexIdentityAndContentHash() throws {
        let messageID = UUID()
        let prefix = AppChatMessage(role: .user, content: "prompt")
        let interrupted = AppChatMessage(id: messageID, role: .assistant, content: "partial")
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: messageID,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let matchingMessages = [prefix, interrupted]

        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: matchingMessages))
        XCTAssertTrue(AppStreamRecovery.validateAnchor(anchor, messages: matchingMessages + [
            AppChatMessage(role: .user, content: "later row"),
        ]))

        let wrongIndexAnchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 0,
            messageID: messageID,
            persistedRowContent: interrupted.content,
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let wrongID = [prefix, AppChatMessage(role: .assistant, content: interrupted.content)]
        let wrongHash = [prefix, AppChatMessage(id: messageID, role: .assistant, content: "changed")]
        let missingRow = [prefix]

        XCTAssertFalse(AppStreamRecovery.validateAnchor(wrongIndexAnchor, messages: matchingMessages))
        for mismatched in [wrongID, wrongHash, missingRow] {
            let before = mismatched
            XCTAssertFalse(AppStreamRecovery.validateAnchor(anchor, messages: mismatched))
            XCTAssertEqual(mismatched, before, "Anchor validation must not mutate transcript rows.")
        }
    }

    func testRecoveryEventsKeepTypedPayloads() throws {
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: UUID(),
            persistedRowContent: "partial",
            continuationsUsed: 1,
            retriesUsed: 2,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let events: [StreamRecoveryEvent] = [
            .continued(attempt: 2, tokensSoFar: 384),
            .partialPreserved(rows: 1),
            .tailDiscarded(description: "Untrusted incomplete tail"),
            .anchorRecorded(anchor),
            .retryConflict,
            .retryExhausted,
        ]

        XCTAssertEqual(events[0], .continued(attempt: 2, tokensSoFar: 384))
        XCTAssertEqual(events[1], .partialPreserved(rows: 1))
        XCTAssertEqual(events[2], .tailDiscarded(description: "Untrusted incomplete tail"))
        XCTAssertEqual(events[3], .anchorRecorded(anchor))
        XCTAssertEqual(events[4], .retryConflict)
        XCTAssertEqual(events[5], .retryExhausted)
    }

    func testRecoveryEventPresentationCoversEveryKindAndMarksTerminalOutcomes() throws {
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 1,
            messageID: UUID(),
            persistedRowContent: "private partial text",
            continuationsUsed: 1,
            retriesUsed: 2,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let events: [StreamRecoveryEvent] = [
            .continued(attempt: 2, tokensSoFar: 384),
            .partialPreserved(rows: 1),
            .tailDiscarded(description: "  incomplete continuation\n tail  "),
            .anchorRecorded(anchor),
            .retrySucceeded(attempt: 2),
            .retryConflict,
            .retryExhausted,
        ]

        let presentations = events.map(StreamRecoveryEventPresentation.init(event:))

        XCTAssertEqual(presentations, [
            .continued(attempt: 2, tokensSoFar: 384),
            .partialPreserved(rows: 1),
            .tailDiscarded(reason: "incomplete continuation tail"),
            .anchorRecorded(retainedMessages: 1, retriesUsed: 2),
            .retrySucceeded(attempt: 2),
            .retryConflict,
            .retryExhausted(limit: AppStreamRecovery.maxRecoveryRetries),
        ])
        XCTAssertEqual(presentations.map(\.accessibilityIdentifier), [
            "stream-recovery-continued",
            "stream-recovery-partial-preserved",
            "stream-recovery-tail-discarded",
            "stream-recovery-anchor-recorded",
            "stream-recovery-retry-succeeded",
            "stream-recovery-retry-conflict",
            "stream-recovery-retry-exhausted",
        ])
        XCTAssertEqual(presentations.map(\.isTerminal), [false, false, false, false, false, true, true])
        XCTAssertEqual(
            StreamRecoveryEventPresentation(event: .tailDiscarded(description: " \n\t")),
            .tailDiscarded(reason: nil))
    }

    private func generationResult(stopReason: String) throws -> GenerationResult {
        let json = """
        {"promptTokens":0,"newTokens":32,"prefillSeconds":0,"decodeSeconds":0,"stopReason":"\(stopReason)","content":"partial"}
        """
        return try JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }
}
