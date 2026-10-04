import XCTest
import TurboSpark

@testable import TurboSparkApp

final class SubagentHistoryTests: XCTestCase {
    func testCancelledGenerationIsNotReportedAsCompletedActorWork() async throws {
        let actor = try XCTUnwrap(AgentManager.shared.builtInAgents.first)
        let result = await SubagentGenerationPortContext.$current.withValue(
            ScriptedSubagentGenerationPort(
                probe: SubagentInvocationProbe(), answer: "Partial reply", stopReason: .cancelled))
        {
            await SubagentRunner.run(
                agent: actor, taskPrompt: "Ask", session: nil, project: nil,
                maxTurnsOverride: 1, priorHistory: [])
        }

        XCTAssertEqual(result.status, "cancelled")
        XCTAssertEqual(result.finalResponse, "Partial reply")
        XCTAssertEqual(result.totalToolCalls, 0)
        XCTAssertEqual(result.transcript, [ChatMessage(role: .user, content: "Ask")])
    }

    func testStreamWithoutResultFailsInsteadOfCompletingActorWork() async throws {
        let actor = try XCTUnwrap(AgentManager.shared.builtInAgents.first)
        let result = await SubagentGenerationPortContext.$current.withValue(
            ScriptedSubagentGenerationPort(
                probe: SubagentInvocationProbe(), answer: "Partial reply", stopReason: nil))
        {
            await SubagentRunner.run(
                agent: actor, taskPrompt: "Ask", session: nil, project: nil,
                maxTurnsOverride: 1, priorHistory: [])
        }

        XCTAssertEqual(result.status, "error")
        XCTAssertTrue(result.finalResponse.contains("ended without a result"))
        XCTAssertTrue(result.finalResponse.contains("Partial reply"))
        XCTAssertEqual(result.totalToolCalls, 0)
        XCTAssertEqual(result.transcript, [ChatMessage(role: .user, content: "Ask")])
    }

    func testRunDispatchesExplicitHistoryAndReturnsUpdatedActorTurns() async throws {
        let priorTurns = [
            ChatMessage(role: .user, content: "Earlier ask"),
            ChatMessage(role: .assistant, content: "Earlier answer"),
        ]
        let actor = try XCTUnwrap(AgentManager.shared.builtInAgents.first)

        let continuationProbe = SubagentInvocationProbe()
        let continuationResult = await SubagentGenerationPortContext.$current.withValue(
            ScriptedSubagentGenerationPort(probe: continuationProbe, answer: "Next answer"))
        {
            await SubagentRunner.run(
                agent: actor,
                taskPrompt: "Next ask",
                session: nil,
                project: nil,
                maxTurnsOverride: 1,
                priorHistory: priorTurns)
        }
        let continuationInvocations = await continuationProbe.snapshot()

        XCTAssertEqual(continuationInvocations.count, 1)
        XCTAssertEqual(
            Array(try XCTUnwrap(continuationInvocations.first).dropFirst()),
            priorTurns + [ChatMessage(role: .user, content: "Next ask")])
        XCTAssertEqual(
            continuationResult.transcript,
            priorTurns + [
                ChatMessage(role: .user, content: "Next ask"),
                ChatMessage(role: .assistant, content: "Next answer"),
            ])

        let freshProbe = SubagentInvocationProbe()
        let freshResult = await SubagentGenerationPortContext.$current.withValue(
            ScriptedSubagentGenerationPort(probe: freshProbe, answer: "First answer"))
        {
            await SubagentRunner.run(
                agent: actor,
                taskPrompt: "First ask",
                session: nil,
                project: nil,
                maxTurnsOverride: 1)
        }
        let freshInvocations = await freshProbe.snapshot()

        XCTAssertEqual(freshInvocations.count, 1)
        XCTAssertEqual(
            Array(try XCTUnwrap(freshInvocations.first).dropFirst()),
            [ChatMessage(role: .user, content: "First ask")])
        XCTAssertNil(freshResult.transcript)
    }

    func testContinuationStartsWithCurrentSystemPromptThenPriorTurnsAndNewAsk() {
        let priorTurns = [
            ChatMessage(role: .user, content: "Earlier ask"),
            ChatMessage(role: .assistant, content: "Earlier answer"),
        ]

        let history = SubagentRunner.initialHistory(
            systemPrompt: "Current actor instructions",
            priorHistory: priorTurns,
            taskPrompt: "Next ask")

        XCTAssertEqual(history, [
            ChatMessage(role: .system, content: "Current actor instructions"),
            ChatMessage(role: .user, content: "Earlier ask"),
            ChatMessage(role: .assistant, content: "Earlier answer"),
            ChatMessage(role: .user, content: "Next ask"),
        ])
    }

    func testFreshInvocationStartsWithOnlySystemPromptAndAsk() {
        let history = SubagentRunner.initialHistory(
            systemPrompt: "Current actor instructions",
            priorHistory: nil,
            taskPrompt: "First ask")

        XCTAssertEqual(history, [
            ChatMessage(role: .system, content: "Current actor instructions"),
            ChatMessage(role: .user, content: "First ask"),
        ])
    }

    func testOnlyExplicitContinuationReturnsActorTurnsWithoutSystemPrompt() {
        let history = [
            ChatMessage(role: .system, content: "System"),
            ChatMessage(role: .user, content: "Ask"),
            ChatMessage(role: .assistant, content: "Answer"),
        ]
        let actorTurns = Array(history.dropFirst())

        XCTAssertNil(SubagentRunner.transcriptForResult(history, priorHistory: nil))
        XCTAssertEqual(
            SubagentRunner.transcriptForResult(
                history,
                priorHistory: [ChatMessage(role: .user, content: "Earlier")]),
            actorTurns)
    }
}

private actor SubagentInvocationProbe {
    private var invocations: [[ChatMessage]] = []

    func record(_ messages: [ChatMessage]) {
        invocations.append(messages)
    }

    func snapshot() -> [[ChatMessage]] {
        invocations
    }
}

private struct ScriptedSubagentGenerationPort: SubagentGenerationPort {
    let probe: SubagentInvocationProbe
    let answer: String
    var stopReason: GenerationResult.StopReason? = .endOfTurn
    let maxContext: UInt32 = 4_096

    func fitWindow(
        _ messages: [ChatMessage],
        maxTokens: UInt32,
        reasoning: GenerateOptions.Reasoning
    ) async throws -> SubagentPromptFit {
        SubagentPromptFit(
            retained: messages,
            measuredTokens: 1,
            removedTurnCount: 0,
            hasRoomForGeneration: true)
    }

    func generate(
        _ messages: [ChatMessage], options: GenerateOptions
    ) async -> AsyncThrowingStream<GenerationEvent, Error> {
        await probe.record(messages)
        let answer = answer
        let result = stopReason.map(Self.completedResult)
        return AsyncThrowingStream { continuation in
            continuation.yield(.content(answer))
            if let result {
                continuation.yield(.finished(result))
            }
            continuation.finish()
        }
    }

    private static func completedResult(_ stopReason: GenerationResult.StopReason) -> GenerationResult {
        let json = Data("{\"promptTokens\":1,\"newTokens\":1,\"prefillSeconds\":0,\"decodeSeconds\":0,\"stopReason\":\"\(stopReason.rawValue)\",\"content\":\"answer\"}".utf8)
        return try! JSONDecoder().decode(GenerationResult.self, from: json)
    }
}
