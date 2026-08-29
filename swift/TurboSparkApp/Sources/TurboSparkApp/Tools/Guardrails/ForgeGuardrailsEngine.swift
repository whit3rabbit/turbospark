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

                    if availableToolNames.contains(name) || !name.isEmpty {
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

        // Pattern 2: Mistral style `[TOOL_CALLS] [{...}]`
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
        cleaned = cleaned.replacingOccurrences(of: "[TOOL_CALLS]", with: "")
        let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)

        // If the remaining text is essentially an empty container or bare punctuation, return empty string
        if trimmed.isEmpty || trimmed == "```" || trimmed == "```json" || trimmed == "{}" || trimmed == "[]" {
            return ""
        }
        return trimmed
    }

    // MARK: - Private JSON Parsing Helpers

    private static func parseArgumentsString(_ string: String) -> [String: String] {
        guard let data = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            // Attempt key-value extraction if not standard JSON
            return extractKeyValuePairs(from: string)
        }
        return flattenJsonToDict(json)
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

    private static func convertJsonDictToToolCall(_ dict: [String: Any], availableToolNames: Set<String>, raw: String) -> AppToolCall? {
        let name = (dict["name"] as? String) ?? (dict["tool"] as? String) ?? (dict["function"] as? String)
        guard let toolName = name, !toolName.isEmpty else { return nil }

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
