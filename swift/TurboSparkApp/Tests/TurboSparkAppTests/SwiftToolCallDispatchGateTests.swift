import XCTest

@testable import TurboSparkApp

final class SwiftToolCallDispatchGateTests: XCTestCase {
    private func availableTools() -> TurnAvailableTools {
        TurnAvailableTools(definitions: [
            .function(
                name: "read_file",
                description: "Read a file",
                parameters: .object(
                    properties: ["path": .string()],
                    required: ["path"],
                    additionalProperties: false))
        ])
    }

    private func evaluateBothInspectionModes(_ content: String) -> [ToolCallDispatchGateResult] {
        [false, true].map { forgeGuardrailsEnabled in
            ToolCallDispatchGate.evaluate(
                content: content,
                streamState: .completed,
                availableTools: availableTools(),
                forgeGuardrailsEnabled: forgeGuardrailsEnabled)
        }
    }

    func testValidCompletedCallIsDispatchableWithForgeGuardrailsOnOrOff() {
        let content = "Before. <tool_call><name>read_file</name><arguments>{\"path\":\"a.txt\"}</arguments></tool_call> After."

        for forgeGuardrailsEnabled in [false, true] {
            let result = ToolCallDispatchGate.evaluate(
                content: content,
                streamState: .completed,
                availableTools: availableTools(),
                forgeGuardrailsEnabled: forgeGuardrailsEnabled)

            XCTAssertEqual(result.dispatchableCalls.map(\.name), ["read_file"])
            XCTAssertTrue(result.refusals.isEmpty)
            XCTAssertEqual(result.preservedContent, content)
        }
    }

    func testMalformedArgumentsAreRefusedAndSurroundingProseIsPreserved() {
        let content = "Keep this.\n<tool_call><name>read_file</name><arguments>{</arguments></tool_call>\nKeep this too."

        for result in evaluateBothInspectionModes(content) {
            XCTAssertTrue(result.dispatchableCalls.isEmpty)
            XCTAssertEqual(result.refusals, [.malformedJSON])
            XCTAssertEqual(result.preservedContent, "Keep this.\n\nKeep this too.")
            XCTAssertFalse(result.preservedContent.contains("<tool_call>"))
        }
    }

    func testMalformedFencedJSONIsRefusedAndRemovedWithForgeGuardrailsOnOrOff() {
        let content = "Before the request.\n```tool_call\n{\"name\":\"read_file\",\"arguments\":{\"path\":\n```\nAfter the request."

        for result in evaluateBothInspectionModes(content) {
            XCTAssertTrue(result.dispatchableCalls.isEmpty)
            XCTAssertEqual(result.refusals, [.malformedJSON])
            XCTAssertEqual(result.preservedContent, "Before the request.\n\nAfter the request.")
            XCTAssertFalse(result.preservedContent.contains("```tool_call"))
        }
    }

    func testFencedJSONMissingCallFieldsIsRefusedAndRemoved() {
        for body in [#"{"name":"read_file"}"#, #"{"arguments":{"path":"a.txt"}}"#] {
            let content = "Intro.\n```json_tool_call\n\(body)\n```\nOutro."
            for result in evaluateBothInspectionModes(content) {
                XCTAssertTrue(result.dispatchableCalls.isEmpty)
                XCTAssertEqual(result.refusals, [.malformedJSON])
                XCTAssertEqual(result.preservedContent, "Intro.\n\nOutro.")
                XCTAssertFalse(result.preservedContent.contains("json_tool_call"))
            }
        }
    }

    func testXMLWrapperMissingNameIsRefusedAndRemovedWithForgeGuardrailsOnOrOff() {
        let invocation = #"<tool_call><arguments>{"path":"a.txt"}</arguments></tool_call>"#
        let content = "Before.\n\(invocation)\nAfter."

        for result in evaluateBothInspectionModes(content) {
            XCTAssertTrue(result.dispatchableCalls.isEmpty)
            XCTAssertEqual(result.refusals, [.malformedJSON])
            XCTAssertEqual(result.preservedContent, "Before.\n\nAfter.")
            XCTAssertFalse(result.preservedContent.contains(invocation))
        }
    }

    func testXMLWrapperEmptyNameIsRefusedAndRemovedWithForgeGuardrailsOnOrOff() {
        let invocation = #"<tool_call><name>   </name><arguments>{"path":"a.txt"}</arguments></tool_call>"#
        let content = "Before.\n\(invocation)\nAfter."

        for result in evaluateBothInspectionModes(content) {
            XCTAssertTrue(result.dispatchableCalls.isEmpty)
            XCTAssertEqual(result.refusals, [.malformedJSON])
            XCTAssertEqual(result.preservedContent, "Before.\n\nAfter.")
            XCTAssertFalse(result.preservedContent.contains(invocation))
        }
    }

    func testXMLWrapperMissingArgumentsIsRefusedEvenForEmptyArgumentsSchema() {
        let available = TurnAvailableTools(definitions: [
            .function(
                name: "empty_args",
                description: "Accept no arguments",
                parameters: .object(
                    properties: [:],
                    required: [],
                    additionalProperties: false))
        ])
        let invocation = #"<tool_call><name>empty_args</name></tool_call>"#
        let content = "Before.\n\(invocation)\nAfter."

        for forgeGuardrailsEnabled in [false, true] {
            let result = ToolCallDispatchGate.evaluate(
                content: content,
                streamState: .completed,
                availableTools: available,
                forgeGuardrailsEnabled: forgeGuardrailsEnabled)

            XCTAssertTrue(result.dispatchableCalls.isEmpty)
            XCTAssertEqual(result.refusals, [.malformedJSON])
            XCTAssertEqual(result.preservedContent, "Before.\n\nAfter.")
            XCTAssertFalse(result.preservedContent.contains(invocation))
        }
    }

    func testUnavailableToolIsRefusedWithoutDispatch() {
        let content = #"<tool_call><name>delete_everything</name><arguments>{}</arguments></tool_call>"#

        for result in evaluateBothInspectionModes(content) {
            XCTAssertTrue(result.dispatchableCalls.isEmpty)
            XCTAssertEqual(result.refusals, [.unavailableTool(name: "delete_everything")])
            XCTAssertFalse(result.preservedContent.contains("<tool_call>"))
        }
    }

    func testNestedJSONTypesAreValidatedBeforeExecutorProjection() {
        let nestedTool = OpenAITool.function(
            name: "read_file",
            description: "Read a file",
            parameters: .object(
                properties: [
                    "payload": .object(
                        properties: ["count": .integer()],
                        required: ["count"],
                        additionalProperties: false)
                ],
                required: ["payload"],
                additionalProperties: false))
        let available = TurnAvailableTools(definitions: [nestedTool])
        let content = #"<tool_call><name>read_file</name><arguments>{"payload":{"count":"3"}}</arguments></tool_call>"#

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .completed,
            availableTools: available,
            forgeGuardrailsEnabled: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertEqual(result.refusals, [.invalidArguments(path: "$.payload.count", reason: "type")])
    }

    func testValidNestedArgumentsDoNotFailTheLegacyInspectionWhenEnabled() {
        let nestedTool = OpenAITool.function(
            name: "read_file",
            description: "Read a file",
            parameters: .object(
                properties: [
                    "payload": .object(
                        properties: ["count": .integer()],
                        required: ["count"],
                        additionalProperties: false)
                ],
                required: ["payload"],
                additionalProperties: false))
        let available = TurnAvailableTools(definitions: [nestedTool])
        let content = #"<tool_call><name>read_file</name><arguments>{"payload":{"count":3}}</arguments></tool_call>"#

        for forgeGuardrailsEnabled in [false, true] {
            let result = ToolCallDispatchGate.evaluate(
                content: content,
                streamState: .completed,
                availableTools: available,
                forgeGuardrailsEnabled: forgeGuardrailsEnabled)

            XCTAssertEqual(result.dispatchableCalls.count, 1)
            XCTAssertEqual(result.dispatchableCalls.first?.arguments["payload"], #"{"count":3}"#)
            XCTAssertTrue(result.refusals.isEmpty)
        }
    }

    func testIncompleteCallRemainsOrdinaryContentAndIsNotDispatchable() {
        let content = "Before. <tool_call><name>read_file</name><arguments>{\"path\":\"a.txt\"}"

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .completed,
            availableTools: availableTools(),
            forgeGuardrailsEnabled: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertEqual(result.preservedContent, content)
    }

    func testDisabledToolParsingPreservesTheExistingChatModeRestriction() {
        let content = #"<tool_call><name>read_file</name><arguments>{"path":"a.txt"}</arguments></tool_call>"#

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .completed,
            availableTools: availableTools(),
            forgeGuardrailsEnabled: true,
            allowsParsing: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertTrue(result.refusals.isEmpty)
        XCTAssertEqual(result.preservedContent, content)
    }

    func testRepeatedQuotedInvocationSyntaxIsRemovedByItsMatchedSpansOnly() {
        let invocation = #"<tool_call><name>missing</name><arguments>{}</arguments></tool_call>"#
        let content = "Quoted earlier: \(invocation)\nActual answer: \(invocation)\nDone."

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .completed,
            availableTools: availableTools(),
            forgeGuardrailsEnabled: false)

        XCTAssertEqual(result.refusals, [
            .unavailableTool(name: "missing"), .unavailableTool(name: "missing")
        ])
        XCTAssertEqual(result.preservedContent, "Quoted earlier: \nActual answer: \nDone.")
    }

    func testStreamErrorPreventsDispatchEvenWhenTextContainsACompleteCall() {
        let content = #"<tool_call><name>read_file</name><arguments>{"path":"a.txt"}</arguments></tool_call>"#

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .failed,
            availableTools: availableTools(),
            forgeGuardrailsEnabled: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertEqual(result.preservedContent, content)
    }

    func testRefusalSummariesResolveThroughTheCompiledFrenchCatalog() {
        let french = Locale(identifier: "fr")

        XCTAssertEqual(
            ToolCallValidationRefusal.unavailableTool(name: "read_file")
                .localizedUserFacingSummary(locale: french),
            "L'outil `read_file` n'a pas été proposé pour ce tour.")
        XCTAssertEqual(
            ToolCallValidationRefusal.malformedJSON.localizedUserFacingSummary(locale: french),
            "Les arguments de l'outil sont un JSON mal formé.")
        XCTAssertEqual(
            ToolCallValidationRefusal.nonObjectArguments.localizedUserFacingSummary(locale: french),
            "Les arguments de l'outil doivent être un objet JSON.")
        XCTAssertEqual(
            ToolCallValidationRefusal.unsupportedSchema(
                toolName: "read_file", path: "$.path", reason: "type")
                .localizedUserFacingSummary(locale: french),
            "Le schéma d'arguments de l'outil `read_file` n'est pas pris en charge à $.path.")
        XCTAssertEqual(
            ToolCallValidationRefusal.invalidArguments(path: "$.path", reason: "type")
                .localizedUserFacingSummary(locale: french),
            "Les arguments de l'outil ont été refusés à $.path (type).")
    }
}
