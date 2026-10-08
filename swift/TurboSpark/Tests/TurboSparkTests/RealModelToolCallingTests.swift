import Foundation
import XCTest

@testable import TurboSpark

/// The tool-calling surface against a REAL checkpoint, which the scripted
/// tests in `crates/ffi/tests/c_surface.rs` cannot stand in for: they prove the
/// plumbing, not that an actual model's markup is recognised, nor that it
/// understands a replayed call and result.
///
///     TURBOSPARK_TEST_MODEL=~/.turbospark/models/text/qwen36.gturbo swift test --filter RealModelToolCalling
///
/// Skips when the install's chat markup does not frame tool calls natively.
final class RealModelToolCallingTests: RealModelTestCase {
    private let weather = ToolSpec(
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["city": .object(["type": .string("string")])]),
            "required": .array([.string("city")]),
        ]))

    private func run(
        _ session: TurboSparkSession, _ messages: [ChatMessage], options: GenerateOptions
    ) async throws -> (result: GenerationResult, toolEvents: [GenerationToolCall]) {
        var result: GenerationResult?
        var events: [GenerationToolCall] = []
        for try await event in session.generate(messages, options: options) {
            switch event {
            case .toolCall(let call): events.append(call)
            case .finished(let r): result = r
            default: break
            }
        }
        return (try XCTUnwrap(result, "the stream must end with .finished"), events)
    }

    func testAnOfferedToolIsCalledAndItsResultIsUnderstood() async throws {
        var open = OpenOptions()
        open.maxContext = .fixed(8192)
        let session = try await TurboSparkSession(modelPath: try modelPath(), options: open)
        try XCTSkipUnless(
            session.info.toolCalling.native,
            "this checkpoint does not frame tool calls natively: \(session.info.toolCalling.reason ?? "no reason given")")

        var options = GenerateOptions()
        options.temperature = 0
        options.topK = 0
        options.topP = 1
        options.maxNewTokens = 400
        options.tools = [weather]

        // Turn 1: the model should call the tool rather than guess.
        var history: [ChatMessage] = [
            .system("You are a helpful assistant. Use the provided tools when they help."),
            .user("What is the weather in Oslo right now? Use the get_weather tool."),
        ]
        let first = try await run(session, history, options: options)
        print("TURN1 stop=\(first.result.stopReason) calls=\(first.result.toolCalls.map { "\($0.name) \($0.argumentsJSON)" }) content=\(first.result.content.prefix(200))")

        XCTAssertEqual(first.result.stopReason, .toolCalls, "content: \(first.result.content)")
        let call = try XCTUnwrap(first.result.toolCalls.first, "the model did not call the tool")
        XCTAssertEqual(call.name, "get_weather")
        XCTAssertEqual(first.toolEvents.map(\.name), first.result.toolCalls.map(\.name),
                       "the streamed events and the result must agree")
        let arguments = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8)) as? [String: Any])
        XCTAssertTrue(
            (arguments["city"] as? String)?.localizedCaseInsensitiveContains("oslo") == true,
            "arguments: \(call.argumentsJSON)")
        XCTAssertFalse(call.id.isEmpty)

        // Budgeting must see the definitions, or the app under-reserves.
        let bare = try await session.countTokens(history, reasoning: .off)
        let withTools = try await session.countTokens(history, reasoning: .off, tools: [weather])
        XCTAssertGreaterThan(withTools, bare + 20)

        // Turn 2: answer the call and let the model finish.
        history.append(.assistant(first.result.content, toolCalls: first.result.toolCalls))
        history.append(.tool("Sunny, 14 degrees Celsius, light wind.", toolCallId: call.id, name: call.name))
        let second = try await run(session, history, options: options)
        print("TURN2 stop=\(second.result.stopReason) content=\(second.result.content.prefix(300))")

        XCTAssertNotEqual(second.result.stopReason, .toolCalls, "it should answer, not call again")
        XCTAssertTrue(second.result.toolCalls.isEmpty)
        let answer = second.result.content.lowercased()
        XCTAssertTrue(
            answer.contains("14") || answer.contains("sunny"),
            "the answer should use the tool result, got: \(second.result.content)")
    }

    func testNoToolsOfferedMeansNoToolCallsEvenWhenAskedToCallOne() async throws {
        var open = OpenOptions()
        open.maxContext = .fixed(4096)
        let session = try await TurboSparkSession(modelPath: try modelPath(), options: open)
        var options = GenerateOptions()
        options.temperature = 0
        options.maxNewTokens = 120
        let r = try await run(
            session, [.user("Call the get_weather tool for Oslo.")], options: options)
        XCTAssertTrue(r.result.toolCalls.isEmpty, "nothing was offered, so nothing is a call")
        XCTAssertNotEqual(r.result.stopReason, .toolCalls)
    }
}
