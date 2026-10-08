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

    func testPriorLocalizationIncompleteLabelsRemainInertAfterLanguageSwitch() throws {
        let defaults = UserDefaults.standard
        let previousLanguage = defaults.object(forKey: AppLanguage.storageKey)
        defer {
            if let previousLanguage { defaults.set(previousLanguage, forKey: AppLanguage.storageKey) }
            else { defaults.removeObject(forKey: AppLanguage.storageKey) }
        }
        defaults.set(AppLanguage.english.rawValue, forKey: AppLanguage.storageKey)
        let path = try XCTUnwrap(Bundle.module.path(forResource: "es", ofType: "lproj"))
        let spanish = try XCTUnwrap(Bundle(path: path))
        let label = spanish.localizedString(forKey: "Incomplete tool call, not executed", value: nil, table: nil)
        XCTAssertNotEqual(label, "Incomplete tool call, not executed")

        for fragment in [
            "<tool_call><name>read_file</name><arguments>{\"path\":\"a.txt\"}</arguments></tool_call>",
            "```tool_call\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}\n```",
            "call:read_file{path:<quote>a.txt<quote>}",
        ] {
            let content = "[\(label)]\n\(fragment)"
            for result in evaluateBothInspectionModes(content) {
                XCTAssertTrue(result.dispatchableCalls.isEmpty, content)
                XCTAssertTrue(result.refusals.isEmpty, content)
                XCTAssertEqual(result.preservedContent, content)
            }
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

    func testToolAvailableInCatalogButAbsentFromCapturedSnapshotIsRefused() {
        let content = #"<tool_call><name>run_command</name><arguments>{"command":"pwd"}</arguments></tool_call>"#

        let result = ToolCallDispatchGate.evaluate(
            content: content,
            streamState: .completed,
            availableTools: availableTools(),
            forgeGuardrailsEnabled: false)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        XCTAssertEqual(result.refusals, [.unavailableTool(name: "run_command")])
        XCTAssertFalse(result.preservedContent.contains("<tool_call>"))
    }

    func testRescuedCallIsRevalidatedAgainstTheCapturedSnapshotSchema() {
        let integerTool = OpenAITool.function(
            name: "read_count",
            description: "Read a count",
            parameters: .object(
                properties: ["count": .integer()],
                required: ["count"],
                additionalProperties: false))
        let captured = TurnAvailableTools(definitions: [integerTool])
        // A well-typed rescued integer is coerced to the schema type and
        // dispatched; it used to be refused because rescued values were
        // validated as all-string JSON.
        let valid = ToolCallDispatchGate.evaluate(
            content: #"<function=read_count>{"count":3}</function>"#,
            streamState: .completed,
            availableTools: captured,
            forgeGuardrailsEnabled: true)
        XCTAssertEqual(valid.dispatchableCalls.map(\.name), ["read_count"])
        XCTAssertTrue(valid.refusals.isEmpty)

        // A value that cannot be coerced is still re-validated and refused.
        let result = ToolCallDispatchGate.evaluate(
            content: #"<function=read_count>{"count":"abc"}</function>"#,
            streamState: .completed,
            availableTools: captured,
            forgeGuardrailsEnabled: true)

        XCTAssertTrue(result.dispatchableCalls.isEmpty)
        // The guardrail engine turns the type error into a retry nudge.
        XCTAssertTrue(
            result.retryNudge?.contains("must be an integer") == true
                || result.refusals == [.invalidArguments(path: "$.count", reason: "type")])
    }

    func testTerminalRetryDropsValidatedCallsAndPreservesAssistantTextAsProse() {
        let assistantText = #"I could not repair this call: <tool_call><name>read_file</name><arguments>{"path":"a.txt"}</arguments></tool_call>"#
        let validatedCall = AppToolCall(name: "read_file", arguments: ["path": "a.txt"])
        let gateResult = ToolCallDispatchGateResult(
            dispatchableCalls: [validatedCall],
            refusals: [],
            preservedContent: "I could not repair this call: ",
            retryNudge: "Try a corrected call.")

        let resolution = ToolCallDispatchResolution.resolve(
            gateResult: gateResult,
            originalContent: assistantText,
            completedAttempts: 3,
            maximumAttempts: 3)

        XCTAssertEqual(resolution, .finishProse(content: assistantText))
    }

    func testMainChatTerminalRetryDoesNotInvokeExecutorHandoff() async {
        await assertTerminalRetryDoesNotInvokeExecutorHandoff()
    }

    func testSubagentTerminalRetryDoesNotInvokeExecutorHandoff() async {
        await assertTerminalRetryDoesNotInvokeExecutorHandoff()
    }

    private func assertTerminalRetryDoesNotInvokeExecutorHandoff() async {
        let gateResult = ToolCallDispatchGateResult(
            dispatchableCalls: [AppToolCall(name: "read_file", arguments: ["path": "a.txt"])],
            refusals: [],
            preservedContent: "partially repaired prose",
            retryNudge: "Try a corrected call.")
        let resolution = ToolCallDispatchResolution.resolve(
            gateResult: gateResult,
            originalContent: "assistant text with a valid call",
            completedAttempts: 3,
            maximumAttempts: 3)
        var retryCallbackCount = 0
        var proseCallbackCount = 0
        var executorHandoffCount = 0

        let result = await ToolCallDispatchResolution.handle(
            resolution,
            retry: { _, _ in retryCallbackCount += 1 },
            finishProse: { _ in proseCallbackCount += 1 },
            dispatch: { _, _ in executorHandoffCount += 1 })

        XCTAssertEqual(result, .finishedProse)
        XCTAssertEqual(retryCallbackCount, 0)
        XCTAssertEqual(proseCallbackCount, 1)
        XCTAssertEqual(executorHandoffCount, 0)
    }

    func testRetryRequiresBothNonemptyNudgeAndRemainingAttempt() {
        let gateResult = ToolCallDispatchGateResult(
            dispatchableCalls: [], refusals: [], preservedContent: "prose", retryNudge: "repair")
        XCTAssertEqual(
            ToolCallDispatchResolution.resolve(
                gateResult: gateResult,
                originalContent: "original prose",
                completedAttempts: 1,
                maximumAttempts: 2),
            .retry(nudge: "repair", assistantContent: "original prose"))

        let emptyNudge = ToolCallDispatchGateResult(
            dispatchableCalls: [AppToolCall(name: "read_file", arguments: ["path": "a.txt"])],
            refusals: [], preservedContent: "prose", retryNudge: "")
        XCTAssertEqual(
            ToolCallDispatchResolution.resolve(
                gateResult: emptyNudge,
                originalContent: "original prose",
                completedAttempts: 1,
                maximumAttempts: 2),
            .finishProse(content: "original prose"))
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
