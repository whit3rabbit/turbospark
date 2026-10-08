import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The native tool-calling lane around the engine. The engine side (offering,
/// parsing, replay) is covered in the package and in `crates/ffi`, and against
/// real checkpoints in `RealModelToolCallingTests`; these cover how the APP
/// joins it: a native call becoming the same `AppToolCall` a text call is,
/// going through the same validation, and being replayed and measured.
@MainActor
final class NativeToolCallingTests: XCTestCase {
    private func available() -> TurnAvailableTools {
        TurnAvailableTools(definitions: [
            .function(
                name: "read_file", description: "Read a file",
                parameters: .object(
                    properties: ["path": .string(), "limit": JSONSchemaProperty(type: "integer")],
                    required: ["path"], additionalProperties: false))
        ])
    }

    private func engineCall(_ name: String, _ arguments: String, id: String = "toolu_0")
        -> GenerationToolCall
    {
        GenerationToolCall(id: id, name: name, argumentsJSON: arguments)
    }

    // MARK: candidates and the gate

    func testANativeCallBecomesTheSameKindOfCallATextCallIs() throws {
        let candidates = ToolCallParser.nativeCandidates(
            from: [engineCall("read_file", #"{"path":"a.txt","limit":5}"#)])
        let call = try XCTUnwrap(candidates.first?.call)
        XCTAssertEqual(call.name, "read_file")
        XCTAssertEqual(call.arguments["path"], "a.txt")
        XCTAssertEqual(call.arguments["limit"], "5")
        XCTAssertEqual(call.status, .pendingApproval, "it still has to pass the permission engine")
        XCTAssertTrue(call.nativeCallID?.hasPrefix("toolu_") == true)
        XCTAssertEqual(call.nativeArgumentsJSON, #"{"path":"a.txt","limit":5}"#, "types are kept")
        XCTAssertTrue(call.rawInvocation.contains("<name>read_file</name>"))
        XCTAssertNil(candidates.first?.refusal)
    }

    func testNativeCallIDsAreUniquePerCallNotPerTurn() throws {
        let a = try XCTUnwrap(ToolCallParser.nativeCandidates(from: [engineCall("read_file", "{}")]).first?.call)
        let b = try XCTUnwrap(ToolCallParser.nativeCandidates(from: [engineCall("read_file", "{}")]).first?.call)
        XCTAssertNotEqual(a.nativeCallID, b.nativeCallID, "the engine restarts at toolu_0 every turn")
    }

    func testNonObjectOrMissingArgumentsAreTreatedLikeATextCall() {
        let candidates = ToolCallParser.nativeCandidates(from: [
            engineCall("read_file", "[1,2]"),
            engineCall("", "{}"),
            engineCall("ping", ""),
        ])
        XCTAssertEqual(candidates[0].refusal, .malformedJSON)
        XCTAssertEqual(candidates[1].refusal, .malformedJSON)
        XCTAssertNil(candidates[2].refusal, "no arguments is a legal call to a no-parameter tool")
        XCTAssertEqual(candidates[2].call?.nativeArgumentsJSON, "{}")
    }

    func testTheGateValidatesNativeCallsAgainstTheOfferedSchema() {
        let good = ToolCallDispatchGate.evaluate(
            content: "Let me look.", streamState: .completed, availableTools: available(),
            forgeGuardrailsEnabled: true,
            nativeCalls: [engineCall("read_file", #"{"path":"a.txt"}"#)])
        XCTAssertEqual(good.dispatchableCalls.map(\.name), ["read_file"])
        XCTAssertTrue(good.refusals.isEmpty)
        XCTAssertEqual(good.preservedContent, "Let me look.")
        XCTAssertNil(good.retryNudge)
        XCTAssertNotNil(good.dispatchableCalls.first?.typedArguments, "typed values reach MCP")

        let unknown = ToolCallDispatchGate.evaluate(
            content: "", streamState: .completed, availableTools: available(),
            forgeGuardrailsEnabled: false,
            nativeCalls: [engineCall("rm_rf", #"{"path":"/"}"#)])
        XCTAssertTrue(unknown.dispatchableCalls.isEmpty)
        XCTAssertEqual(unknown.refusals, [.unavailableTool(name: "rm_rf")])

        let badType = ToolCallDispatchGate.evaluate(
            content: "", streamState: .completed, availableTools: available(),
            forgeGuardrailsEnabled: false,
            nativeCalls: [engineCall("read_file", #"{"path":"a.txt","limit":"many"}"#)])
        XCTAssertTrue(badType.dispatchableCalls.isEmpty, "the schema still gates native calls")
        XCTAssertEqual(badType.refusals.count, 1)

        let extra = ToolCallDispatchGate.evaluate(
            content: "", streamState: .completed, availableTools: available(),
            forgeGuardrailsEnabled: false,
            nativeCalls: [engineCall("read_file", #"{"path":"a.txt","evil":true}"#)])
        XCTAssertTrue(extra.dispatchableCalls.isEmpty, "additionalProperties: false is enforced")
    }

    func testTextInTheReplyIsNotReadAsASecondCopyWhenTheEngineParsedCalls() {
        let text =
            "<tool_call><name>read_file</name><arguments>{\"path\":\"other.txt\"}</arguments></tool_call>"
        let result = ToolCallDispatchGate.evaluate(
            content: text, streamState: .completed, availableTools: available(),
            forgeGuardrailsEnabled: true,
            nativeCalls: [engineCall("read_file", #"{"path":"a.txt"}"#)])
        XCTAssertEqual(result.dispatchableCalls.count, 1)
        XCTAssertEqual(result.dispatchableCalls.first?.arguments["path"], "a.txt")
    }

    func testWithNoNativeCallsTheTextGateIsUnchanged() {
        let text =
            "<tool_call><name>read_file</name><arguments>{\"path\":\"a.txt\"}</arguments></tool_call>"
        let result = ToolCallDispatchGate.evaluate(
            content: text, streamState: .completed, availableTools: available(),
            forgeGuardrailsEnabled: true, nativeCalls: [])
        XCTAssertEqual(result.dispatchableCalls.map(\.name), ["read_file"])
        XCTAssertNil(result.dispatchableCalls.first?.nativeCallID, "a text call carries no native id")
    }

    func testACancelledTurnDispatchesNothingEvenWithNativeCalls() {
        let result = ToolCallDispatchGate.evaluate(
            content: "", streamState: .failed, availableTools: available(),
            forgeGuardrailsEnabled: false,
            nativeCalls: [engineCall("read_file", #"{"path":"a.txt"}"#)])
        XCTAssertTrue(result.dispatchableCalls.isEmpty)
    }

    // MARK: offering

    func testToolSpecsCarryTheNameDescriptionAndFullSchema() throws {
        let specs = AppModel.nativeToolSpecs(from: available())
        XCTAssertEqual(specs.map(\.name), ["read_file"])
        XCTAssertEqual(specs[0].description, "Read a file")
        let json = try JSONEncoder().encode(specs[0])
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: json) as? [String: Any])
        let schema = try XCTUnwrap(object["parameters"] as? [String: Any], "an object, not text")
        XCTAssertEqual(schema["type"] as? String, "object")
        XCTAssertEqual((schema["required"] as? [String]) ?? [], ["path"])
        XCTAssertNotNil((schema["properties"] as? [String: Any])?["path"])
        XCTAssertEqual(AppModel.nativeToolSpecs(from: nil), [])
    }

    func testTheLaneIsOffWithoutASessionRegardlessOfTheSetting() {
        let model = AppModel()
        model.nativeToolCallingEnabled = true
        XCTAssertFalse(model.nativeToolLane(project: nil))
    }

    func testTheSystemPromptStopsDescribingToolsInTheNativeLaneButKeepsItsGuidance() {
        let text = AppToolCatalog.systemPromptAddendum(
            for: .general, availableTools: available(), nativeToolCalling: false)
        XCTAssertTrue(text.contains("## Available Tools"))
        XCTAssertTrue(text.contains("To invoke a tool, output a tool call block"))
        XCTAssertTrue(text.contains("- `read_file`: Read a file"))

        let native = AppToolCatalog.systemPromptAddendum(
            for: .general, availableTools: available(), nativeToolCalling: true)
        XCTAssertFalse(native.contains("## Available Tools"))
        XCTAssertFalse(native.contains("To invoke a tool"))
        XCTAssertFalse(native.contains("<tool_call>"), "a second calling format would confuse the model")
        XCTAssertFalse(native.contains("- `read_file`"))
        XCTAssertTrue(native.contains("## Task & Progress Tracking"))
        XCTAssertTrue(native.contains("## Subagents"))
        XCTAssertFalse(native.contains("one tool call block each"))
    }

    // MARK: replay

    private func nativeRow(
        content: String = "Checking.", isError: Bool = false, output: String = "file body"
    ) -> AppChatMessage {
        let candidate = ToolCallParser.nativeCandidates(
            from: [engineCall("read_file", #"{"path":"a.txt","limit":5}"#)])[0]
        let call = candidate.call!
        let result = AppToolResult(callID: call.id, output: output, isError: isError)
        return AppChatMessage(
            role: .assistant, content: content, toolCalls: [call], toolResults: [result])
    }

    func testANativeRowReplaysAsAnAssistantCallAndAnIdMatchedBareToolMessage() throws {
        let row = nativeRow()
        let messages = AppModel.historyMessages(for: row, nativeLane: true, mediaCapability: .textOnly)
        XCTAssertEqual(messages.map(\.role), [.assistant, .tool])
        let assistant = messages[0]
        XCTAssertEqual(assistant.content, "Checking.")
        XCTAssertEqual(assistant.toolCalls.count, 1)
        XCTAssertEqual(assistant.toolCalls[0].name, "read_file")
        XCTAssertEqual(assistant.toolCalls[0].argumentsJSON, #"{"path":"a.txt","limit":5}"#)
        let tool = messages[1]
        XCTAssertEqual(tool.toolCallId, assistant.toolCalls[0].id, "the pair is matched by id")
        XCTAssertEqual(tool.name, "read_file")
        XCTAssertEqual(tool.content, "file body", "bare: the template wraps a tool result itself")
        XCTAssertFalse(tool.content.contains("<tool_response>"))
    }

    func testAToolErrorIsMarkedWithoutTheTextLanesTags() {
        let messages = AppModel.historyMessages(
            for: nativeRow(isError: true), nativeLane: true, mediaCapability: .textOnly)
        XCTAssertEqual(messages.last?.content, "Error: file body")
    }

    func testACallWithNoProseStillReplays() {
        let messages = AppModel.historyMessages(
            for: nativeRow(content: ""), nativeLane: true, mediaCapability: .textOnly)
        XCTAssertEqual(messages.map(\.role), [.assistant, .tool])
        XCTAssertEqual(messages[0].content, "")
        XCTAssertEqual(messages[0].toolCalls.count, 1, "dropping it would delete a turn")
    }

    func testAnUnfinishedNativeTurnStillCarriesItsCallsAndTheNote() {
        var row = nativeRow()
        row.stopReason = "cancelled"
        let assistant = AppModel.historyMessages(for: row, nativeLane: true, mediaCapability: .textOnly)[0]
        XCTAssertTrue(assistant.content.contains("stopped by the user"))
        XCTAssertEqual(assistant.toolCalls.count, 1)
    }

    func testANativeRowInTheTextLaneIsWrittenBackAsTheTextItNeverHad() {
        // Native calling turned off, or a different checkpoint, mid-chat.
        let messages = AppModel.historyMessages(
            for: nativeRow(), nativeLane: false, mediaCapability: .textOnly)
        XCTAssertEqual(messages.map(\.role), [.assistant, .tool])
        XCTAssertTrue(messages[0].content.contains("Checking."))
        XCTAssertTrue(messages[0].content.contains("<name>read_file</name>"))
        XCTAssertTrue(messages[0].toolCalls.isEmpty, "the text lane sends no structured calls")
        XCTAssertTrue(messages[1].content.hasPrefix("<tool_response>"))
        XCTAssertNil(messages[1].toolCallId)
    }

    func testALegacyTextRowReplaysIdenticallyInBothLanes() {
        let call = AppToolCall(
            name: "read_file", arguments: ["path": "a.txt"],
            rawInvocation: "<tool_call><name>read_file</name><arguments>{}</arguments></tool_call>")
        let row = AppChatMessage(
            role: .assistant, content: "Before <tool_call>…</tool_call>",
            toolCalls: [call],
            toolResults: [AppToolResult(callID: call.id, output: "body")])
        let text = AppModel.historyMessages(for: row, nativeLane: false, mediaCapability: .textOnly)
        let native = AppModel.historyMessages(for: row, nativeLane: true, mediaCapability: .textOnly)
        XCTAssertEqual(text, native, "an archived text row must not change meaning when the lane is on")
        XCTAssertEqual(text[0].content, row.content)
        XCTAssertEqual(text[1].content, "<tool_response>\nbody\n</tool_response>")
    }

    func testMicrocompactClearsOldNativeResultsAndKeepsTheirIdentity() async {
        // Microcompact only acts when replacing a body actually saves tokens,
        // so the result has to be larger than its placeholder.
        let row = nativeRow(output: String(repeating: "line of file content\n", count: 80))
        var messages = AppModel.historyMessages(for: row, nativeLane: true, mediaCapability: .textOnly)
        messages.append(ChatMessage.user("next"))
        let history = AppChatHistoryProjection(
            messages: messages, sourceRowIndexByMessage: [0, 0, 1],
            instructionPinBlockIndex: nil, sourceTranscriptRowCount: 6)
        let outcome = await AppChatMicrocompact.apply(
            history: history, keepRecent: 1, minimumSavingsTokens: 0,
            countTokens: { messages in messages.reduce(0) { $0 + $1.content.count } })
        let tool = outcome.history.messages[1]
        XCTAssertEqual(outcome.rowsCleared, 1)
        XCTAssertEqual(tool.content, "[Tool output omitted from model context.]")
        XCTAssertEqual(tool.toolCallId, messages[0].toolCalls[0].id, "the pairing survives compaction")
        XCTAssertEqual(tool.name, "read_file")
    }

    // MARK: persistence

    func testNativeFieldsRoundTripAndAnOldArchiveDecodesWithoutThem() throws {
        let call = try XCTUnwrap(
            ToolCallParser.nativeCandidates(from: [engineCall("read_file", #"{"path":"a"}"#)])[0].call)
        let again = try JSONDecoder().decode(
            AppToolCall.self, from: JSONEncoder().encode(call))
        XCTAssertEqual(again.nativeCallID, call.nativeCallID)
        XCTAssertEqual(again.nativeArgumentsJSON, #"{"path":"a"}"#)

        let old = #"{"name":"read_file","arguments":{"path":"a"},"rawInvocation":"x"}"#
        let decoded = try JSONDecoder().decode(AppToolCall.self, from: Data(old.utf8))
        XCTAssertNil(decoded.nativeCallID)
        XCTAssertNil(decoded.nativeArgumentsJSON)
        XCTAssertFalse(AppModel.isNativeToolRow(AppChatMessage(role: .assistant, content: "", toolCalls: [decoded])))
    }

    func testTheSettingDefaultsOnAndSurvivesARoundTrip() throws {
        var settings = MacAppSettings()
        XCTAssertTrue(settings.nativeToolCallingEnabled)
        settings.nativeToolCallingEnabled = false
        let restored = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertFalse(restored.nativeToolCallingEnabled)
        let legacy = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertTrue(legacy.nativeToolCallingEnabled, "settings written before the option stay on")
    }
}
