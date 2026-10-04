import Foundation
import XCTest

import TurboSpark
@testable import TurboSparkApp

final class AppStreamContinuationTests: XCTestCase {
    func testEnabledContinuationUsesAccumulatedAssistantPrefixAndRemainingBudget() async throws {
        let history = [ChatMessage(role: .user, content: "prompt")]
        let options = generationOptions(maxNewTokens: 10)
        var requests: [([ChatMessage], UInt32)] = []
        var preparedContinuation: ([ChatMessage], UInt32)?
        var results = [
            try generationResult(content: "first ", newTokens: 10, stopReason: "maxTokens"),
            try generationResult(content: "second", newTokens: 4, stopReason: "endOfTurn"),
        ]

        let completed = try await AppStreamRecovery.generateUntilComplete(
            baseMessages: history,
            options: options,
            enabled: true,
            prepareContinuationMessages: { messages, reservedOutputTokens in
                preparedContinuation = (messages, reservedOutputTokens)
                return messages
            }
        ) { messages, segmentOptions in
            requests.append((messages, segmentOptions.maxNewTokens))
            return results.removeFirst()
        }

        XCTAssertEqual(completed.content, "first second")
        XCTAssertEqual(completed.reasoning, "")
        XCTAssertEqual(completed.continuationsUsed, 1)
        XCTAssertEqual(completed.outputTokensGenerated, 14)
        XCTAssertEqual(completed.recoveryEvents, [.continued(attempt: 1, tokensSoFar: 10)])
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests[0].0, history)
        XCTAssertEqual(requests[0].1, 10)
        XCTAssertEqual(requests[1].0, history + [ChatMessage(role: .assistant, content: "first ")])
        XCTAssertEqual(requests[1].1, 10)
        XCTAssertEqual(
            preparedContinuation?.0,
            history + [ChatMessage(role: .assistant, content: "first ")])
        XCTAssertEqual(preparedContinuation?.1, 10)

        var transcript = [AppChatMessage(role: .user, content: "prompt")]
        AppStreamRecovery.commitAssistantRow(
            into: &transcript,
            content: completed.content,
            reasoning: completed.reasoning,
            result: completed.result)
        XCTAssertEqual(transcript.filter { if case .assistant = $0.role { return true }; return false }.count, 1)
        XCTAssertEqual(transcript.last?.content, "first second")
    }

    func testDisabledContinuationReturnsTheOrdinaryMaxTokenResult() async throws {
        let history = [ChatMessage(role: .user, content: "prompt")]
        var callCount = 0

        let completed = try await AppStreamRecovery.generateUntilComplete(
            baseMessages: history,
            options: generationOptions(maxNewTokens: 8),
            enabled: false
        ) { _, _ in
            callCount += 1
            return try generationResult(content: "partial", newTokens: 8, stopReason: "maxTokens")
        }

        XCTAssertEqual(callCount, 1)
        XCTAssertEqual(completed.content, "partial")
        XCTAssertEqual(completed.continuationsUsed, 0)
        XCTAssertEqual(completed.result.stopReason, .maxTokens)
    }

    func testRepeatedCutoffStopsAfterThreeContinuationRequests() async throws {
        var callCount = 0
        var deliveredEvents: [StreamRecoveryEvent] = []
        do {
            _ = try await AppStreamRecovery.generateUntilComplete(
                baseMessages: [ChatMessage(role: .user, content: "prompt")],
                options: generationOptions(maxNewTokens: 5),
                enabled: true,
                onRecoveryEvent: { deliveredEvents.append($0) }
            ) { _, _ in
                callCount += 1
                return try generationResult(
                    content: "segment\(callCount)", newTokens: 5, stopReason: "maxTokens")
            }
            XCTFail("Expected the fixed continuation bound to use the ordinary error path.")
        } catch let error as ContinuationGenerationError {
            XCTAssertEqual(error, .exhausted(limit: 3, outputTokensGenerated: 20))
        }

        XCTAssertEqual(callCount, 4, "One initial request plus three continuations are allowed.")
        XCTAssertEqual(deliveredEvents, [
            .continued(attempt: 1, tokensSoFar: 5),
            .continued(attempt: 2, tokensSoFar: 10),
            .continued(attempt: 3, tokensSoFar: 15),
        ])
    }

    func testContinuationEventSurvivesLaterThrowAndPrecedesPartialAndAnchorEvents() async throws {
        var deliveredEvents: [StreamRecoveryEvent] = []
        var requestCount = 0
        do {
            _ = try await AppStreamRecovery.generateUntilComplete(
                baseMessages: [ChatMessage(role: .user, content: "prompt")],
                options: generationOptions(maxNewTokens: 8),
                enabled: true,
                onRecoveryEvent: { deliveredEvents.append($0) }
            ) { _, _ in
                requestCount += 1
                if requestCount == 1 {
                    return try generationResult(content: "first segment", newTokens: 8, stopReason: "maxTokens")
                }
                throw CocoaError(.fileReadUnknown)
            }
            XCTFail("Expected the second request to fail after continuation preparation.")
        } catch is CocoaError {
            // The continuation event must remain delivered even though no completed turn returns.
        }

        var transcript = [AppChatMessage(role: .user, content: "prompt")]
        let preservation = try XCTUnwrap(AppStreamRecovery.preserveInterruptedOutput(
            into: &transcript,
            content: "partial after the later request failed",
            reasoning: "",
            stopReason: "error"))
        deliveredEvents.append(contentsOf: preservation.events)

        XCTAssertEqual(deliveredEvents, [
            .continued(attempt: 1, tokensSoFar: 8),
            .partialPreserved(rows: 1),
            .anchorRecorded(preservation.anchor),
        ])
    }

    func testContinuationFitFailureDoesNotDeliverContinuedEvent() async throws {
        var deliveredEvents: [StreamRecoveryEvent] = []
        var requestCount = 0
        do {
            _ = try await AppStreamRecovery.generateUntilComplete(
                baseMessages: [ChatMessage(role: .user, content: "prompt")],
                options: generationOptions(maxNewTokens: 8),
                enabled: true,
                prepareContinuationMessages: { _, _ in
                    throw ContinuationGenerationError.continuationContextDoesNotFit
                },
                onRecoveryEvent: { deliveredEvents.append($0) }
            ) { _, _ in
                requestCount += 1
                return try generationResult(content: "first segment", newTokens: 8, stopReason: "maxTokens")
            }
            XCTFail("Expected continuation context fitting to fail.")
        } catch let error as ContinuationGenerationError {
            XCTAssertEqual(error, .continuationContextDoesNotFit)
        }

        XCTAssertEqual(requestCount, 1)
        XCTAssertTrue(deliveredEvents.isEmpty)
    }

    private func generationOptions(maxNewTokens: UInt32) -> GenerateOptions {
        var options = GenerateOptions()
        options.maxNewTokens = maxNewTokens
        return options
    }

    private func generationResult(
        content: String,
        newTokens: Int,
        stopReason: String,
        reasoning: String = ""
    ) throws -> GenerationResult {
        let json = """
        {"promptTokens":1,"newTokens":\(newTokens),"prefillSeconds":0,"decodeSeconds":0,"stopReason":"\(stopReason)","content":"\(content)","reasoning":"\(reasoning)"}
        """
        return try JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }
}
