import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The app's native tool lane against a REAL checkpoint, using the app's real
/// tool set, system prompt builder, gate, executor and history replay. Only the
/// SwiftUI loop around them is absent.
///
///     TURBOSPARK_TEST_MODEL=~/.turbospark/models/text/qwen36.gturbo \
///       swift test -Xbuild-tools-swiftc -suppress-warnings --filter RealModelNativeToolLane
///
/// Skips without the variable, and when the install does not frame tool calls
/// natively.
@MainActor
final class RealModelNativeToolLaneTests: XCTestCase {
    func testTheAppsOwnPromptToolsGateExecutorAndReplayWorkEndToEnd() async throws {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_MODEL"], !path.isEmpty
        else { throw XCTSkip("set TURBOSPARK_TEST_MODEL to a .gturbo install to run this") }

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("native-lane-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "The secret word is PINEAPPLE.\nSecond line.\n".write(
            to: root.appendingPathComponent("notes.txt"), atomically: true, encoding: .utf8)
        let project = AppProject(name: "lane", rootDirectoryPath: root.path)

        var open = OpenOptions()
        open.maxContext = .fixed(16384)
        let session = try await TurboSparkSession(modelPath: path, options: open)
        try XCTSkipUnless(
            session.info.toolCalling.native,
            "this checkpoint does not frame tool calls natively")

        let model = AppModel()
        model.session = session
        model.interactionMode = .projects
        model.nativeToolCallingEnabled = true

        // The lane is chosen exactly as a turn chooses it.
        XCTAssertTrue(model.nativeToolLane(project: project))
        let tools = model.captureTurnTools(project: project, session: session)
        let specs = AppModel.nativeToolSpecs(from: tools)
        XCTAssertFalse(specs.isEmpty)
        print("OFFERED \(specs.count) tools: \(specs.map(\.name).prefix(12))")

        let system = model.buildSystemPrompt(
            for: project, availableTools: tools, nativeToolCalling: true)
        XCTAssertFalse(system.contains("To invoke a tool"))
        var messages: [ChatMessage] = [
            .system(system),
            .user("Use a tool to read the file notes.txt in the project directory, then tell me the secret word it contains."),
        ]

        var options = GenerateOptions()
        options.temperature = 0
        options.topK = 0
        options.topP = 1
        options.maxNewTokens = 600
        options.tools = specs

        // The budget the app computes must include the definitions.
        let bare = try await session.countTokens(messages, reasoning: .off)
        let priced = try await session.countTokens(messages, reasoning: .off, tools: specs)
        print("PROMPT tokens bare=\(bare) with tools=\(priced)")
        XCTAssertGreaterThan(priced, bare + 200, "a full tool set must be visible to the budget")

        func run(_ history: [ChatMessage]) async throws -> (GenerationResult, [GenerationToolCall]) {
            var result: GenerationResult?
            var calls: [GenerationToolCall] = []
            for try await event in session.generate(history, options: options) {
                switch event {
                case .toolCall(let call): calls.append(call)
                case .finished(let r): result = r
                default: break
                }
            }
            return (try XCTUnwrap(result), calls)
        }

        // Turn 1: the model calls a tool; the app's gate validates it.
        let (first, firstCalls) = try await run(messages)
        print("TURN1 stop=\(first.stopReason) calls=\(firstCalls.map { "\($0.name) \($0.argumentsJSON)" }) content=\(first.content.prefix(160))")
        XCTAssertEqual(first.stopReason, .toolCalls)
        let gate = ToolCallDispatchGate.evaluate(
            content: first.content, streamState: .completed, availableTools: tools,
            forgeGuardrailsEnabled: true, allowsParsing: true,
            projectURL: project.rootDirectoryURL, nativeCalls: firstCalls)
        print("GATE dispatchable=\(gate.dispatchableCalls.map(\.name)) refusals=\(gate.refusals)")
        let call = try XCTUnwrap(gate.dispatchableCalls.first, "refusals: \(gate.refusals)")
        XCTAssertNotNil(call.nativeCallID)
        XCTAssertTrue(
            call.arguments.values.contains { $0.contains("notes.txt") },
            "arguments: \(call.arguments)")

        // The app's real executor runs it.
        let toolResult = await AppToolRegistry.execute(call: call, in: project)
        print("EXEC error=\(toolResult.isError) output=\(toolResult.output.prefix(120))")
        XCTAssertFalse(toolResult.isError)
        XCTAssertTrue(toolResult.output.contains("PINEAPPLE"))

        // Turn 2: replay through the app's own history helper, as a stored row.
        let row = AppChatMessage(
            role: .assistant, content: gate.preservedContent,
            toolCalls: [call], toolResults: [toolResult])
        messages.append(contentsOf: AppModel.historyMessages(
            for: row, nativeLane: true, mediaCapability: .textOnly))
        let (second, secondCalls) = try await run(messages)
        print("TURN2 stop=\(second.stopReason) calls=\(secondCalls.count) content=\(second.content.prefix(200))")
        XCTAssertNotEqual(second.stopReason, .toolCalls, "it should answer, not loop")
        XCTAssertTrue(
            second.content.uppercased().contains("PINEAPPLE"),
            "the answer should use the tool result, got: \(second.content)")
    }
}
