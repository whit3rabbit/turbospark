import Foundation

/// The one reader of a model's proposed tool calls.
///
/// **THIS EXISTED TWICE, AND THAT IS WHY THE SUBAGENT PATH KEPT NEEDING ITS
/// OWN FIXES.** `AppModel.extractToolCalls` and `SubagentRunner.extractToolCalls`
/// carried byte-identical bodies -- the XML arm, the markdown fallback, and a
/// private `parseJSONArguments` each -- so every change to how a call is read
/// had to be made twice, by someone who knew there was a second copy.
/// state#68, state#74 and state#75 are all the same shape one level up: a fix
/// landed on the main loop and the isolated one was found to be missing it a
/// pass later. Two spellings of one question is the defect; the duplication
/// was its cause.
///
/// The two CALLERS still differ, and that difference is deliberate rather
/// than incidental: the main loop refuses to parse at all outside Projects
/// mode and without a project (state#82), and a subagent is already inside
/// one by construction. Each keeps its own guard and neither keeps its own
/// parser.
public enum ToolCallParser {
    /// Every call `text` contains, in the order they appear.
    ///
    /// - Parameter projectURL: the workspace root, so a PROJECT-scoped custom
    ///   tool is classified under the category it declares rather than under
    ///   `category(for:)`'s default arm (state#71).
    public static func parse(from text: String, projectURL: URL? = nil) -> [AppToolCall] {
        var calls = parseXMLBlocks(in: text, projectURL: projectURL)
        // The markdown form is a FALLBACK, not an alternative: a reply
        // carrying both is one that wrapped its XML in a fence, and reading
        // it twice would run the same call twice.
        if calls.isEmpty {
            calls = parseMarkdownBlocks(in: text, projectURL: projectURL)
        }
        return calls
    }

    /// `<tool_call><name>x</name><arguments>{...}</arguments></tool_call>`.
    private static func parseXMLBlocks(in text: String, projectURL: URL?) -> [AppToolCall] {
        guard
            let xmlRegex = try? NSRegularExpression(
                pattern: "<tool_call>([\\s\\S]*?)</tool_call>", options: [])
        else { return [] }

        var calls: [AppToolCall] = []
        let nsString = text as NSString
        let matches = xmlRegex.matches(
            in: text, options: [], range: NSRange(location: 0, length: nsString.length))
        for match in matches {
            guard match.numberOfRanges > 1 else { continue }
            let inner = nsString.substring(with: match.range(at: 1))
            let raw = nsString.substring(with: match.range(at: 0))

            let toolName = firstCapture(in: inner, pattern: "<name>([\\s\\S]*?)</name>")?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !toolName.isEmpty else { continue }

            var arguments: [String: String] = [:]
            if let argsString = firstCapture(
                in: inner, pattern: "<arguments>([\\s\\S]*?)</arguments>")
            {
                arguments = parseJSONArguments(
                    argsString.trimmingCharacters(in: .whitespacesAndNewlines))
            }
            calls.append(
                makeCall(
                    name: toolName, arguments: arguments, raw: raw, projectURL: projectURL))
        }
        return calls
    }

    /// ```` ```tool_call\n{"name": "x", "arguments": {...}}\n``` ````
    private static func parseMarkdownBlocks(in text: String, projectURL: URL?) -> [AppToolCall] {
        guard
            let mdRegex = try? NSRegularExpression(
                pattern: "```(?:tool_call|json_tool_call)\\s*\\n([\\s\\S]*?)\\n```", options: [])
        else { return [] }

        var calls: [AppToolCall] = []
        let nsString = text as NSString
        let matches = mdRegex.matches(
            in: text, options: [], range: NSRange(location: 0, length: nsString.length))
        for match in matches {
            guard match.numberOfRanges > 1 else { continue }
            let inner = nsString.substring(with: match.range(at: 1))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let raw = nsString.substring(with: match.range(at: 0))

            guard let data = inner.data(using: .utf8),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let toolName = (json["name"] as? String) ?? (json["tool"] as? String)
            else { continue }

            var arguments: [String: String] = [:]
            if let rawArgs = json["arguments"] as? [String: Any] {
                for (key, value) in rawArgs {
                    arguments[key] = "\(value)"
                }
            }
            calls.append(
                makeCall(
                    name: toolName, arguments: arguments, raw: raw, projectURL: projectURL))
        }
        return calls
    }

    /// A parsed call, classified and risk-assessed.
    ///
    /// `.pendingApproval` is the status every parsed call starts at, whatever
    /// the gate later decides: a call that has not been through
    /// `AppToolPermissionEngine` is not an approved one, and this is the
    /// value that says so.
    private static func makeCall(
        name: String, arguments: [String: String], raw: String, projectURL: URL?
    ) -> AppToolCall {
        AppToolCall(
            name: name,
            arguments: arguments,
            rawInvocation: raw,
            status: .pendingApproval,
            category: AppToolRegistry.category(for: name, projectURL: projectURL),
            riskAssessment: ToolRiskClassifier.assessRisk(name: name, arguments: arguments)
        )
    }

    /// A JSON object as flat string arguments.
    ///
    /// **THE XML AND MARKDOWN ARMS FLATTEN DIFFERENTLY AND THAT IS PRESERVED
    /// RATHER THAN UNIFIED.** This one spells `NSNumber` out through
    /// `stringValue`; the markdown arm interpolates every value. Both were
    /// already true of the two copies this replaces, on both of them, so
    /// picking one here would be a behaviour change smuggled into a
    /// deduplication -- and the frozen tool-call cases could not tell the
    /// difference either way. Worth settling deliberately, and not here.
    static func parseJSONArguments(_ string: String) -> [String: String] {
        guard let data = string.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }

        var result: [String: String] = [:]
        for (key, value) in json {
            if let text = value as? String {
                result[key] = text
            } else if let number = value as? NSNumber {
                result[key] = number.stringValue
            } else {
                result[key] = "\(value)"
            }
        }
        return result
    }

    private static func firstCapture(in text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return nil
        }
        let ns = text as NSString
        guard
            let match = regex.firstMatch(
                in: text, options: [], range: NSRange(location: 0, length: ns.length)),
            match.numberOfRanges > 1
        else { return nil }
        return ns.substring(with: match.range(at: 1))
    }
}
