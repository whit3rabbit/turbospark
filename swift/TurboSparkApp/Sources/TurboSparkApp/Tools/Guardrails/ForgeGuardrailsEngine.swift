import Foundation

/// Verdict from evaluating generated model output against tool specifications.
public enum ForgeGuardrailVerdict: Equatable, Sendable {
    /// The generated turn is acceptable and requires no intervention.
    case accept
    /// Tool calls were successfully rescued from raw text markup that standard parsing missed.
    case rescued(calls: [AppToolCall], sanitizedText: String)
    /// The tool invocation contains errors that can be repaired with a corrective prompt turn.
    case retry(nudge: String)
}

/// Core engine for tool-call repair, alternate dialect parsing, argument validation, and corrective nudges.
public enum ForgeGuardrailsEngine {

    /// Inspects generated output against available tool specifications.
    public static func inspect(
        text: String,
        parsedCalls: [AppToolCall],
        availableTools: [OpenAITool],
        requiresCall: Bool = false
    ) -> ForgeGuardrailVerdict {
        let toolNames = Set(availableTools.map { $0.function.name })
        if toolNames.isEmpty {
            return .accept
        }

        // 1. If standard parsing did not find calls, attempt rescue from raw text
        var calls = parsedCalls
        var wasRescued = false
        if calls.isEmpty {
            let rescued = rescueToolCalls(from: text, availableToolNames: toolNames)
            if !rescued.isEmpty {
                calls = rescued
                wasRescued = true
            }
        }

        // 2. If still no calls found
        if calls.isEmpty {
            if requiresCall {
                let nudge = "A tool call was required for this request. Please invoke one of the available tools: \(toolNames.sorted().joined(separator: ", "))."
                return .retry(nudge: nudge)
            }
            return .accept
        }

        // 3. Validate arguments against schema for all calls
        let specsByName = Dictionary(uniqueKeysWithValues: availableTools.map { ($0.function.name, $0) })
        var validationErrors: [String] = []

        for call in calls {
            guard let spec = specsByName[call.name] else {
                let namesList = toolNames.sorted().joined(separator: ", ")
                validationErrors.append("Unknown tool `\(call.name)`. Available tools are: \(namesList).")
                continue
            }

            let errors = validateArguments(arguments: call.arguments, schema: spec.function.parameters)
            if !errors.isEmpty {
                validationErrors.append("Your call to `\(call.name)` had invalid arguments: \(errors.joined(separator: "; ")).")
            }
        }

        // 4. Formulate verdict
        if !validationErrors.isEmpty {
            let nudge = "\(validationErrors.joined(separator: " ")) Call the tool again with corrected arguments."
            return .retry(nudge: nudge)
        }

        if wasRescued {
            let sanitized = sanitizeProse(text: text, rescuedCalls: calls)
            return .rescued(calls: calls, sanitizedText: sanitized)
        }

        return .accept
    }

    /// Rescues tool calls from varied markup dialects (XML, Markdown, Mistral, bare JSON).
    public static func rescueToolCalls(from text: String, availableToolNames: Set<String>) -> [AppToolCall] {
        var results: [AppToolCall] = []

        // Pattern 1: Qwen / Generic XML style `<function=name>args</function>` or `<function_call name="name">args</function_call>`
        let xmlFuncPatterns = [
            "<function=([a-zA-Z0-9_\\-]+)>([\\s\\S]*?)</function>",
            "<function_call\\s+name=[\"']([a-zA-Z0-9_\\-]+)[\"']>([\\s\\S]*?)</function_call>",
            "<invoke\\s+name=[\"']([a-zA-Z0-9_\\-]+)[\"']>([\\s\\S]*?)</invoke>"
        ]

        for pattern in xmlFuncPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
                let nsString = text as NSString
                let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))
                for match in matches {
                    guard match.numberOfRanges >= 3 else { continue }
                    let name = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let body = nsString.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let raw = nsString.substring(with: match.range(at: 0))

                    // `|| !name.isEmpty` made this condition true for ANY
                    // non-empty tool name, so a rescued XML-dialect call
                    // could manufacture a real `AppToolCall` for a tool the
                    // current agent type was never granted (T6). The
                    // allowlist is the whole point of this check.
                    if isToolNameAllowed(name, in: availableToolNames) {
                        let args = parseArgumentsString(body)
                        let category = AppToolRegistry.category(for: name)
                        let risk = ToolRiskClassifier.assessRisk(name: name, arguments: args)
                        results.append(AppToolCall(
                            name: name,
                            arguments: args,
                            rawInvocation: raw,
                            status: .pendingApproval,
                            category: category,
                            riskAssessment: risk
                        ))
                    }
                }
            }
        }

        if !results.isEmpty { return results }

        // Pattern 1b: GLM pairs -- a function name directly followed by flat
        // `<arg_key>K</arg_key>` / `<arg_value>V</arg_value>` runs. The
        // `<tool_call>` wrapper the template teaches is deliberately NOT in
        // the pattern: where it is ordinary text it is noise, and where a
        // vocabulary carries it as a special token it never survives to the
        // rescue at all, so the pair markup is the only signature present in
        // both cases. The allowlist is what stops a stray word before
        // `<arg_key>` from becoming a tool name.
        let glmCallPattern = "([a-zA-Z0-9_\\-]{1,64})\\s*((?:<arg_key>[\\s\\S]*?</arg_key>\\s*<arg_value>[\\s\\S]*?</arg_value>\\s*)+)"
        let glmPairPattern = "<arg_key>([\\s\\S]*?)</arg_key>\\s*<arg_value>([\\s\\S]*?)</arg_value>"
        if let callRegex = try? NSRegularExpression(pattern: glmCallPattern, options: []),
           let pairRegex = try? NSRegularExpression(pattern: glmPairPattern, options: []) {
            let nsString = text as NSString
            for match in callRegex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length)) {
                guard match.numberOfRanges >= 3 else { continue }
                let name = nsString.substring(with: match.range(at: 1))
                guard isToolNameAllowed(name, in: availableToolNames) else { continue }
                let pairsText = nsString.substring(with: match.range(at: 2))
                var args: [String: String] = [:]
                let pairsNSString = pairsText as NSString
                for pair in pairRegex.matches(in: pairsText, options: [], range: NSRange(location: 0, length: pairsNSString.length)) {
                    guard pair.numberOfRanges >= 3 else { continue }
                    let key = pairsNSString.substring(with: pair.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let value = pairsNSString.substring(with: pair.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !key.isEmpty { args[key] = value }
                }
                let category = AppToolRegistry.category(for: name)
                let risk = ToolRiskClassifier.assessRisk(name: name, arguments: args)
                results.append(AppToolCall(
                    name: name,
                    arguments: args,
                    rawInvocation: nsString.substring(with: match.range(at: 0)),
                    status: .pendingApproval,
                    category: category,
                    riskAssessment: risk
                ))
            }
        }

        if !results.isEmpty { return results }

        // Pattern 1c: Kimi K2 section tags -- there is NO name tag, only the
        // id `functions.NAME:IDX` Moonshot's own tool_call_guidance.md
        // documents (and vLLM's kimi_tool_parser implements), so the name is
        // recovered from the id. A documented-anomaly id (a bare `call_...`
        // string) yields that prefix as the name, which the allowlist then
        // refuses: an opaque id states no name, and inventing one would be
        // worse than the refusal.
        let kimiPattern = "<\\|tool_call_begin\\|>\\s*([\\w.:\\-]+)\\s*<\\|tool_call_argument_begin\\|>([\\s\\S]*?)<\\|tool_call_end\\|>"
        if let kimiRegex = try? NSRegularExpression(pattern: kimiPattern, options: []) {
            let nsString = text as NSString
            for match in kimiRegex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length)) {
                guard match.numberOfRanges >= 3 else { continue }
                let id = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                let name = id.hasPrefix("functions.")
                    ? String(id.dropFirst("functions.".count).split(separator: ":").first ?? "")
                    : String(id.split(separator: ":").first ?? "")
                guard isToolNameAllowed(name, in: availableToolNames) else { continue }
                let body = nsString.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                let args = body.isEmpty ? [:] : parseArgumentsString(body)
                let category = AppToolRegistry.category(for: name)
                let risk = ToolRiskClassifier.assessRisk(name: name, arguments: args)
                results.append(AppToolCall(
                    name: name,
                    arguments: args,
                    rawInvocation: nsString.substring(with: match.range(at: 0)),
                    status: .pendingApproval,
                    category: category,
                    riskAssessment: risk
                ))
            }
        }

        if !results.isEmpty { return results }

        // Pattern 1d: Gemma 4 DSL -- `call:NAME{k:v,...}` (with unquoted or quoted keys and values).
        // The `<|tool_call>` wrapper the template teaches is stripped on normal decode, but
        // may arrive if emitted as text.
        let gemmaPattern = "(?:<\\|tool_call>)?\\s*call:([a-zA-Z0-9_\\-]+)\\s*(\\{[\\s\\S]*?\\})\\s*(?:<tool_call\\|>|\\n|$)"
        if let gemmaRegex = try? NSRegularExpression(pattern: gemmaPattern, options: []) {
            let nsString = text as NSString
            for match in gemmaRegex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length)) {
                guard match.numberOfRanges >= 3 else { continue }
                let name = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                guard isToolNameAllowed(name, in: availableToolNames) else { continue }
                let body = nsString.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                let args = parseGemmaArguments(body)
                let category = AppToolRegistry.category(for: name)
                let risk = ToolRiskClassifier.assessRisk(name: name, arguments: args)
                results.append(AppToolCall(
                    name: name,
                    arguments: args,
                    rawInvocation: nsString.substring(with: match.range(at: 0)),
                    status: .pendingApproval,
                    category: category,
                    riskAssessment: risk
                ))
            }
        }

        if !results.isEmpty { return results }
        if text.contains("[TOOL_CALLS]") {
            let mistralPattern = "\\[TOOL_CALLS\\]\\s*(\\[[\\s\\S]*?\\]|\\{[\\s\\S]*?\\})"
            if let regex = try? NSRegularExpression(pattern: mistralPattern, options: []),
               let match = regex.firstMatch(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length)) {
                let rawBlock = (text as NSString).substring(with: match.range(at: 1))
                if let parsed = parseJsonCalls(rawBlock, availableToolNames: availableToolNames) {
                    return parsed
                }
            }
        }

        // Pattern 3: Markdown code blocks ````tool_call` or ````json
        let blockPattern = "```(?:tool_call|json_tool_call|json)?\\s*\\n?([\\s\\S]*?)\\n?```"
        if let regex = try? NSRegularExpression(pattern: blockPattern, options: []) {
            let nsString = text as NSString
            let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))
            for match in matches {
                guard match.numberOfRanges > 1 else { continue }
                let inner = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                if let parsed = parseJsonCalls(inner, availableToolNames: availableToolNames) {
                    results.append(contentsOf: parsed)
                }
            }
        }

        if !results.isEmpty { return results }

        // Pattern 3b: Longcat -- a plain `{"name", "arguments"}` JSON object
        // inside `<longcat_tool_call>` tags. The Rust server needs no code
        // for this shape (forge's JSON scan reads balanced braces anywhere in
        // the text); Pattern 4 below requires the WHOLE text to parse as
        // JSON, so here the wrapper has to be stripped before that same
        // parse can see the object.
        let longcatPattern = "<longcat_tool_call>\\s*([\\s\\S]*?)\\s*</longcat_tool_call>"
        if let longcatRegex = try? NSRegularExpression(pattern: longcatPattern, options: []),
           let match = longcatRegex.firstMatch(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length)) {
            let inner = (text as NSString).substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            if let parsed = parseJsonCalls(inner, availableToolNames: availableToolNames) {
                // The raw invocation is the whole tagged block, so
                // `sanitizeProse` removes the wrapper along with the call.
                return parsed.map { call in
                    AppToolCall(
                        name: call.name,
                        arguments: call.arguments,
                        rawInvocation: (text as NSString).substring(with: match.range(at: 0)),
                        status: call.status,
                        category: call.category,
                        riskAssessment: call.riskAssessment
                    )
                }
            }
        }

        // Pattern 4: Bare JSON object anywhere in text
        if let parsed = parseJsonCalls(text.trimmingCharacters(in: .whitespacesAndNewlines), availableToolNames: availableToolNames) {
            results.append(contentsOf: parsed)
        }

        return results
    }

    /// Validates arguments against parameter schema.
    public static func validateArguments(arguments: [String: String], schema: JSONSchema) -> [String] {
        var errors: [String] = []

        // Check required properties
        if let required = schema.required {
            for req in required {
                if arguments[req] == nil || arguments[req]?.isEmpty == true {
                    errors.append("missing required parameter `\(req)`")
                }
            }
        }

        // Check types and enum constraints if properties defined
        if let properties = schema.properties {
            for (key, val) in arguments {
                if let prop = properties[key] {
                    // Type validation
                    switch prop.type.lowercased() {
                    case "integer", "int":
                        if Int(val) == nil {
                            errors.append("parameter `\(key)` must be an integer, received \"\(val)\"")
                        }
                    case "number", "float", "double":
                        if Double(val) == nil {
                            errors.append("parameter `\(key)` must be a number, received \"\(val)\"")
                        }
                    case "boolean", "bool":
                        let low = val.lowercased()
                        if low != "true" && low != "false" {
                            errors.append("parameter `\(key)` must be a boolean (true/false), received \"\(val)\"")
                        }
                    default:
                        break
                    }

                    // Enum validation
                    if let enumVals = prop.enumValues, !enumVals.isEmpty {
                        if !enumVals.contains(val) {
                            let allowed = enumVals.map { "\"\($0)\"" }.joined(separator: ", ")
                            errors.append("parameter `\(key)` must be one of [\(allowed)], received \"\(val)\"")
                        }
                    }
                }
            }
        }

        return errors
    }

    /// Removes or sanitizes raw JSON / XML markup from the text when tool calls were rescued.
    public static func sanitizeProse(text: String, rescuedCalls: [AppToolCall]) -> String {
        var cleaned = text
        for call in rescuedCalls {
            if !call.rawInvocation.isEmpty {
                cleaned = cleaned.replacingOccurrences(of: call.rawInvocation, with: "")
            }
        }
        let extraMarkers = [
            "[TOOL_CALLS]",
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call>",
            "<tool_call|>",
            "<tool_call>",
            "</tool_call>",
        ]
        for marker in extraMarkers {
            cleaned = cleaned.replacingOccurrences(of: marker, with: "")
        }
        let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)

        // If the remaining text is essentially an empty container or bare punctuation, return empty string
        if trimmed.isEmpty || trimmed == "```" || trimmed == "```json" || trimmed == "{}" || trimmed == "[]" {
            return ""
        }
        return trimmed
    }

    // MARK: - Private JSON Parsing Helpers

    private static func parseArgumentsString(_ string: String) -> [String: String] {
        // invoke/parameter XML bodies (the Anthropic/Claude shape MiniMax
        // speaks): `<parameter name="key">value</parameter>` pairs are the
        // arguments, and without this extraction the body reached the
        // line-based fallback below, which mangles a tag at the first colon.
        let paramPattern = "<parameter\\s+name=\"([\\s\\S]*?)\"\\s*>([\\s\\S]*?)</parameter>"
        if let regex = try? NSRegularExpression(pattern: paramPattern, options: []) {
            let nsString = string as NSString
            let matches = regex.matches(in: string, options: [], range: NSRange(location: 0, length: nsString.length))
            if !matches.isEmpty {
                var args: [String: String] = [:]
                for match in matches {
                    guard match.numberOfRanges >= 3 else { continue }
                    let key = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let value = nsString.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !key.isEmpty { args[key] = value }
                }
                return args
            }
        }

        // Qwen ChatML parameter XML bodies: `<parameter=key>value</parameter>`
        let qwenParamPattern = "<parameter=([a-zA-Z0-9_\\-]+)>([\\s\\S]*?)</parameter>"
        if let regex = try? NSRegularExpression(pattern: qwenParamPattern, options: []) {
            let nsString = string as NSString
            let matches = regex.matches(in: string, options: [], range: NSRange(location: 0, length: nsString.length))
            if !matches.isEmpty {
                var args: [String: String] = [:]
                for match in matches {
                    guard match.numberOfRanges >= 3 else { continue }
                    let key = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let value = nsString.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !key.isEmpty { args[key] = value }
                }
                return args
            }
        }

        guard let data = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            // Attempt key-value extraction if not standard JSON
            return extractKeyValuePairs(from: string)
        }
        return flattenJsonToDict(json)
    }

    private static func parseGemmaArguments(_ string: String) -> [String: String] {
        if let data = string.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return flattenJsonToDict(json)
        }
        // Match key: value pairs where key may be unquoted and value may be quoted string or literal
        let pairPattern = "([a-zA-Z0-9_\\-.$]+)\\s*:\\s*(?:\"([^\"]*)\"|'([^']*)'|([^,}]+))"
        guard let regex = try? NSRegularExpression(pattern: pairPattern, options: []) else {
            return extractKeyValuePairs(from: string)
        }
        let nsString = string as NSString
        let matches = regex.matches(in: string, options: [], range: NSRange(location: 0, length: nsString.length))
        var args: [String: String] = [:]
        for match in matches {
            guard match.numberOfRanges >= 5 else { continue }
            let key = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            let value: String
            if match.range(at: 2).location != NSNotFound {
                value = nsString.substring(with: match.range(at: 2))
            } else if match.range(at: 3).location != NSNotFound {
                value = nsString.substring(with: match.range(at: 3))
            } else if match.range(at: 4).location != NSNotFound {
                value = nsString.substring(with: match.range(at: 4)).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                value = ""
            }
            if !key.isEmpty { args[key] = value }
        }
        return args.isEmpty ? extractKeyValuePairs(from: string) : args
    }

    private static func parseJsonCalls(_ string: String, availableToolNames: Set<String>) -> [AppToolCall]? {
        guard let data = string.data(using: .utf8) else { return nil }

        // Try single JSON object: {"name": "...", "arguments": {...}} or {"tool": "...", "parameters": {...}}
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let call = convertJsonDictToToolCall(obj, availableToolNames: availableToolNames, raw: string) {
                return [call]
            }
        }

        // Try JSON array: [{"name": "...", "arguments": {...}}, ...]
        if let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            var calls: [AppToolCall] = []
            for item in array {
                if let call = convertJsonDictToToolCall(item, availableToolNames: availableToolNames, raw: string) {
                    calls.append(call)
                }
            }
            if !calls.isEmpty { return calls }
        }

        return nil
    }

    /// Case-insensitive membership check against the tools actually granted
    /// to the current agent type. Every dialect-rescue path must call this
    /// before manufacturing an `AppToolCall`: a rescued call is API-identical
    /// to one the model produced through normal parsing, so skipping this
    /// check lets rescued text grant a tool the agent was never given (T6).
    static func isToolNameAllowed(_ name: String, in availableToolNames: Set<String>) -> Bool {
        !name.isEmpty && availableToolNames.contains(where: { $0.caseInsensitiveCompare(name) == .orderedSame })
    }

    private static func convertJsonDictToToolCall(_ dict: [String: Any], availableToolNames: Set<String>, raw: String) -> AppToolCall? {
        let name = (dict["name"] as? String) ?? (dict["tool"] as? String) ?? (dict["function"] as? String)
        guard let toolName = name, isToolNameAllowed(toolName, in: availableToolNames) else { return nil }

        var args: [String: String] = [:]
        if let rawArgs = (dict["arguments"] as? [String: Any]) ?? (dict["parameters"] as? [String: Any]) ?? (dict["args"] as? [String: Any]) {
            args = flattenJsonToDict(rawArgs)
        } else if let strArgs = dict["arguments"] as? String {
            args = parseArgumentsString(strArgs)
        }

        let category = AppToolRegistry.category(for: toolName)
        let risk = ToolRiskClassifier.assessRisk(name: toolName, arguments: args)
        return AppToolCall(
            name: toolName,
            arguments: args,
            rawInvocation: raw,
            status: .pendingApproval,
            category: category,
            riskAssessment: risk
        )
    }

    private static func flattenJsonToDict(_ json: [String: Any]) -> [String: String] {
        var result: [String: String] = [:]
        for (k, v) in json {
            if let s = v as? String {
                result[k] = s
            } else if let num = v as? NSNumber {
                result[k] = num.stringValue
            } else if let b = v as? Bool {
                result[k] = b ? "true" : "false"
            } else if let subData = try? JSONSerialization.data(withJSONObject: v, options: []),
                      let subStr = String(data: subData, encoding: .utf8) {
                result[k] = subStr
            } else {
                result[k] = "\(v)"
            }
        }
        return result
    }

    private static func extractKeyValuePairs(from string: String) -> [String: String] {
        var dict: [String: String] = [:]
        let lines = string.components(separatedBy: .newlines)
        for line in lines {
            let parts = line.split(separator: ":", maxSplits: 1).map(String.init)
            if parts.count == 2 {
                let k = parts[0].trimmingCharacters(in: .whitespaces.union(.init(charactersIn: "\"\'<>()")))
                let v = parts[1].trimmingCharacters(in: .whitespaces.union(.init(charactersIn: "\"\'<>()")))
                if !k.isEmpty && !v.isEmpty {
                    dict[k] = v
                }
            }
        }
        return dict
    }
}
