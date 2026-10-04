import Foundation

enum ToolCallStreamState {
    case completed
    case failed
}

struct ToolCallDispatchGateResult: Equatable {
    /// These are the only calls the caller may send to the existing executor.
    let dispatchableCalls: [AppToolCall]
    let refusals: [ToolCallValidationRefusal]
    /// Original assistant text with refused invocation blocks removed.
    let preservedContent: String
    /// ForgeGuardrails may request one corrective model turn. The caller owns
    /// the existing step-limit policy; calls remain schema-gated either way.
    let retryNudge: String?
}

/// The only outcomes a generation loop may act on after authorization. A
/// terminal retry is prose, so its parsed calls cannot reach an executor.
enum ToolCallDispatchResolution: Equatable {
    case retry(nudge: String, assistantContent: String)
    case finishProse(content: String)
    case dispatch(calls: [AppToolCall], content: String)

    enum HandlingResult: Equatable {
        case retried
        case finishedProse
        case dispatched
    }

    static func resolve(
        gateResult: ToolCallDispatchGateResult,
        originalContent: String,
        completedAttempts: Int,
        maximumAttempts: Int
    ) -> ToolCallDispatchResolution {
        if let nudge = gateResult.retryNudge {
            guard !nudge.isEmpty, completedAttempts < maximumAttempts else {
                return .finishProse(content: originalContent)
            }
            return .retry(nudge: nudge, assistantContent: originalContent)
        }

        guard !gateResult.dispatchableCalls.isEmpty else {
            return .finishProse(content: gateResult.preservedContent)
        }
        return .dispatch(
            calls: gateResult.dispatchableCalls,
            content: gateResult.preservedContent)
    }

    /// Routes exactly one authorized outcome to its matching loop callback.
    /// In particular, prose completion has no path to the executor callback.
    static func handle(
        _ resolution: ToolCallDispatchResolution,
        retry: (String, String) async -> Void,
        finishProse: (String) async -> Void,
        dispatch: ([AppToolCall], String) async -> Void
    ) async -> HandlingResult {
        switch resolution {
        case .retry(let nudge, let assistantContent):
            await retry(nudge, assistantContent)
            return .retried
        case .finishProse(let content):
            await finishProse(content)
            return .finishedProse
        case .dispatch(let calls, let content):
            await dispatch(calls, content)
            return .dispatched
        }
    }
}

enum ToolCallDispatchGate {
    static func evaluate(
        content: String,
        streamState: ToolCallStreamState,
        availableTools: TurnAvailableTools,
        forgeGuardrailsEnabled: Bool,
        allowsParsing: Bool = true,
        projectURL: URL? = nil
    ) -> ToolCallDispatchGateResult {
        guard allowsParsing else {
            return ToolCallDispatchGateResult(
                dispatchableCalls: [], refusals: [], preservedContent: content, retryNudge: nil)
        }
        let candidates = ToolCallParser.parseCandidates(from: content, projectURL: projectURL)
        guard streamState == .completed else {
            return ToolCallDispatchGateResult(
                dispatchableCalls: [], refusals: [], preservedContent: content, retryNudge: nil)
        }

        var dispatchableCalls: [AppToolCall] = []
        var refusals: [ToolCallValidationRefusal] = []
        var refusedRanges: [NSRange] = []
        for candidate in candidates {
            if let refusal = candidate.refusal {
                refusals.append(refusal)
                refusedRanges.append(candidate.sourceRange)
                continue
            }
            guard let proposedCall = candidate.call else {
                refusals.append(.malformedJSON)
                refusedRanges.append(candidate.sourceRange)
                continue
            }
            switch availableTools.validate(
                toolName: proposedCall.name,
                argumentsJSON: candidate.argumentsJSON)
            {
            case .validated(_, let arguments):
                var call = proposedCall
                call.arguments = executorArguments(from: arguments)
                dispatchableCalls.append(call)
            case .refused(let refusal):
                refusals.append(refusal)
                refusedRanges.append(candidate.sourceRange)
            }
        }

        let preservedContent = removing(refusedRanges, from: content)

        guard forgeGuardrailsEnabled else {
            return ToolCallDispatchGateResult(
                dispatchableCalls: dispatchableCalls,
                refusals: refusals,
                preservedContent: preservedContent,
                retryNudge: nil)
        }

        // The inspector receives only the exact turn snapshot. Standard
        // candidate markup is removed from its text input because those calls
        // have already been validated above; this prevents an invalid call
        // from being rediscovered and turned into a retry or dispatch.
        let inspectionText = removing(
            candidates.map(\.sourceRange), from: content)
        switch ForgeGuardrailsEngine.inspect(
            text: inspectionText,
            // Completed parser matches have already passed the recursive
            // snapshot validator; handing their projected string arguments
            // to the legacy shallow inspector would reject valid containers.
            parsedCalls: [],
            availableTools: availableTools.definitions,
            requiresCall: false)
        {
        case .accept:
            return ToolCallDispatchGateResult(
                dispatchableCalls: dispatchableCalls,
                refusals: refusals,
                preservedContent: preservedContent,
                retryNudge: nil)
        case .retry(let nudge):
            return ToolCallDispatchGateResult(
                dispatchableCalls: dispatchableCalls,
                refusals: refusals,
                preservedContent: preservedContent,
                retryNudge: nudge)
        case .rescued(let rescuedCalls, let sanitizedText):
            var allCalls = dispatchableCalls
            var allRefusals = refusals
            for rescued in rescuedCalls {
                guard let argumentsJSON = jsonArguments(from: rescued.arguments) else {
                    allRefusals.append(.malformedJSON)
                    continue
                }
                switch availableTools.validate(
                    toolName: rescued.name, argumentsJSON: argumentsJSON)
                {
                case .validated(_, let arguments):
                    var call = rescued
                    call.arguments = executorArguments(from: arguments)
                    allCalls.append(call)
                case .refused(let refusal):
                    allRefusals.append(refusal)
                }
            }
            return ToolCallDispatchGateResult(
                dispatchableCalls: allCalls,
                refusals: allRefusals,
                preservedContent: sanitizedText,
                retryNudge: nil)
        }
    }

    private static func removing(_ ranges: [NSRange], from content: String) -> String {
        let originalLength = (content as NSString).length
        return ranges.sorted { $0.location > $1.location }.reduce(content) { text, range in
            guard range.location >= 0, NSMaxRange(range) <= originalLength else { return text }
            return (text as NSString).replacingCharacters(in: range, with: "")
        }
    }

    private static func executorArguments(
        from arguments: [String: ToolCallJSONValue]
    ) -> [String: String] {
        arguments.mapValues { value in
            switch value {
            case .string(let string): return string
            case .number(let number): return NSDecimalNumber(decimal: number).stringValue
            case .boolean(let value): return value ? "1" : "0"
            case .null: return "null"
            case .object, .array:
                return encodedJSON(value) ?? "{}"
            }
        }
    }

    private static func encodedJSON(_ value: ToolCallJSONValue) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func jsonArguments(from arguments: [String: String]) -> String? {
        guard let data = try? JSONSerialization.data(
            withJSONObject: arguments, options: [.sortedKeys]),
            let value = String(data: data, encoding: .utf8) else { return nil }
        return value
    }
}

extension ToolCallValidationRefusal {
    var userFacingSummary: String {
        localizedUserFacingSummary()
    }

    func localizedUserFacingSummary(locale: Locale? = nil) -> String {
        let locale = locale ?? AppLanguage.resolve(
            UserDefaults.standard.string(forKey: AppLanguage.storageKey)
                ?? AppLanguage.system.rawValue).locale
        let bundle = localizationBundle(for: locale)
        switch self {
        case .unavailableTool(let name):
            return String(
                localized: "Tool `\(name)` was not offered for this turn.",
                bundle: bundle,
                locale: locale)
        case .malformedJSON:
            return String(
                localized: "Tool arguments were malformed JSON.",
                bundle: bundle,
                locale: locale)
        case .nonObjectArguments:
            return String(
                localized: "Tool arguments must be a JSON object.",
                bundle: bundle,
                locale: locale)
        case .unsupportedSchema(let toolName, let path, _):
            return String(
                localized: "Tool `\(toolName)` has an unsupported argument schema at \(path).",
                bundle: bundle,
                locale: locale)
        case .invalidArguments(let path, let reason):
            return String(
                localized: "Tool arguments were refused at \(path) (\(reason)).",
                bundle: bundle,
                locale: locale)
        }
    }

    private func localizationBundle(for locale: Locale) -> Bundle {
        guard let localization = Bundle.preferredLocalizations(
            from: Bundle.module.localizations,
            forPreferences: [locale.identifier]).first,
            let path = Bundle.module.path(forResource: localization, ofType: "lproj"),
            let bundle = Bundle(path: path)
        else {
            return .module
        }
        return bundle
    }
}
