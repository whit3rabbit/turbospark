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
    static let incompleteToolCallLabel = "[Incomplete tool call, not executed]"

    /// A complete parser match with its original JSON argument text retained
    /// until the offered schema has authorized it.
    struct Candidate: Equatable {
        let call: AppToolCall?
        let argumentsJSON: String
        /// UTF-16 source range, matching `NSRegularExpression` offsets.
        let sourceRange: NSRange
        let refusal: ToolCallValidationRefusal?
    }

    /// Every call `text` contains, in the order they appear.
    ///
    /// - Parameter projectURL: the workspace root, so a PROJECT-scoped custom
    ///   tool is classified under the category it declares rather than under
    ///   `category(for:)`'s default arm (state#71).
    public static func parse(from text: String, projectURL: URL? = nil) -> [AppToolCall] {
        parseCandidates(from: text, projectURL: projectURL).compactMap(\.call)
    }

    /// Complete candidates only. A partial wrapper cannot produce a
    /// candidate. Malformed wrappers retain typed refusals unless the
    /// transcript explicitly labels them as interrupted inert fragments.
    static func parseCandidates(from text: String, projectURL: URL? = nil) -> [Candidate] {
        var candidates = parseXMLBlocks(in: text, projectURL: projectURL)
        // The markdown form is a FALLBACK, not an alternative: a reply
        // carrying both is one that wrapped its XML in a fence, and reading
        // it twice would run the same call twice.
        if candidates.isEmpty {
            candidates = parseMarkdownBlocks(in: text, projectURL: projectURL)
        }
        return candidates.filter {
            !isPrefixedByIncompleteToolCallLabel($0.sourceRange, in: text)
        }
    }

    /// A persisted interrupted fragment is transcript text, even if a caller
    /// later sends the whole row through the completed-call gate. The label is
    /// localized by the cancellation path, so retain every bundled translation
    /// even after the user changes the app language.
    static func isPrefixedByIncompleteToolCallLabel(_ range: NSRange, in text: String) -> Bool {
        let source = text as NSString
        guard range.location >= 0, range.length > 0, NSMaxRange(range) <= source.length,
              let firstNonWhitespace = firstNonWhitespaceRange(in: source, within: range)
        else { return false }
        let prefix = source.substring(to: firstNonWhitespace.location)
        let normalizedPrefix = prefix
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lineAlignedPrefix = normalizedPrefix.trimmingCharacters(in: .whitespaces)
        guard lineAlignedPrefix.hasSuffix("\n") else { return false }
        let textBeforeLineBreak = String(lineAlignedPrefix.dropLast())
        let precedingLine = textBeforeLineBreak
            .split(separator: "\n", omittingEmptySubsequences: false)
            .last
        return precedingLine.map {
            incompleteToolCallLabels.contains(String($0).trimmingCharacters(in: .whitespaces))
        } ?? false
    }

    private static func firstNonWhitespaceRange(in source: NSString, within range: NSRange) -> NSRange? {
        let firstContent = source.rangeOfCharacter(
            from: CharacterSet.whitespacesAndNewlines.inverted,
            options: [],
            range: range)
        return firstContent.location == NSNotFound ? nil : firstContent
    }

    private static let incompleteToolCallLabels: Set<String> = {
        let key = "Incomplete tool call, not executed"
        var labels: Set<String> = [incompleteToolCallLabel, key]
        for language in Bundle.module.localizations {
            guard let path = Bundle.module.path(forResource: language, ofType: "lproj"),
                  let bundle = Bundle(path: path) else { continue }
            let localized = bundle.localizedString(forKey: key, value: nil, table: nil)
            labels.insert(localized)
            labels.insert("[\(localized)]")
        }
        return labels
    }()

    /// `<tool_call><name>x</name><arguments>{...}</arguments></tool_call>`.
    private static func parseXMLBlocks(in text: String, projectURL: URL?) -> [Candidate] {
        guard
            let xmlRegex = try? NSRegularExpression(
                pattern: "<tool_call>([\\s\\S]*?)</tool_call>", options: [])
        else { return [] }

        var candidates: [Candidate] = []
        let nsString = text as NSString
        let matches = xmlRegex.matches(
            in: text, options: [], range: NSRange(location: 0, length: nsString.length))
        for match in matches {
            guard match.numberOfRanges > 1 else { continue }
            let raw = nsString.substring(with: match.range(at: 0))
            let sourceRange = match.range(at: 0)

            guard let fields = firstCapturePair(
                in: raw,
                pattern: "^<tool_call>\\s*<name>([\\s\\S]*?)</name>\\s*<arguments>([\\s\\S]*?)</arguments>\\s*</tool_call>$")
            else {
                candidates.append(malformedCandidate(sourceRange: sourceRange))
                continue
            }
            let toolName = fields.0.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !toolName.isEmpty else {
                candidates.append(malformedCandidate(sourceRange: sourceRange))
                continue
            }
            let rawArguments = fields.1

            let argumentsJSON = rawArguments.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let argumentsData = argumentsJSON.data(using: .utf8),
                  (try? JSONSerialization.jsonObject(with: argumentsData)) is [String: Any]
            else {
                candidates.append(malformedCandidate(sourceRange: sourceRange))
                continue
            }
            let arguments = parseJSONArguments(argumentsJSON)
            candidates.append(Candidate(
                call: makeCall(
                    name: toolName, arguments: arguments, raw: raw, projectURL: projectURL),
                argumentsJSON: argumentsJSON,
                sourceRange: sourceRange,
                refusal: nil))
        }
        return candidates
    }

    /// ```` ```tool_call\n{"name": "x", "arguments": {...}}\n``` ````
    private static func parseMarkdownBlocks(in text: String, projectURL: URL?) -> [Candidate] {
        guard
            let mdRegex = try? NSRegularExpression(
                pattern: "```(?:tool_call|json_tool_call)\\s*\\n([\\s\\S]*?)\\n```", options: [])
        else { return [] }

        var candidates: [Candidate] = []
        let nsString = text as NSString
        let matches = mdRegex.matches(
            in: text, options: [], range: NSRange(location: 0, length: nsString.length))
        for match in matches {
            guard match.numberOfRanges > 1 else { continue }
            let inner = nsString.substring(with: match.range(at: 1))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let raw = nsString.substring(with: match.range(at: 0))
            let sourceRange = match.range(at: 0)

            guard let data = inner.data(using: .utf8),
                  let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            else {
                candidates.append(malformedCandidate(sourceRange: sourceRange))
                continue
            }

            let toolName = ((json["name"] as? String) ?? (json["tool"] as? String))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard let toolName, !toolName.isEmpty,
                  let rawArguments = json["arguments"] as? [String: Any]
            else {
                candidates.append(malformedCandidate(sourceRange: sourceRange))
                continue
            }
            guard JSONSerialization.isValidJSONObject(["arguments": rawArguments]),
                  let argumentsData = try? JSONSerialization.data(
                    withJSONObject: rawArguments,
                    options: [.sortedKeys, .fragmentsAllowed]),
                  let argumentsJSON = String(data: argumentsData, encoding: .utf8) else {
                candidates.append(malformedCandidate(sourceRange: sourceRange))
                continue
            }
            let arguments = parseJSONArguments(argumentsJSON)
            candidates.append(Candidate(
                call: makeCall(
                    name: toolName, arguments: arguments, raw: raw, projectURL: projectURL),
                argumentsJSON: argumentsJSON,
                sourceRange: sourceRange,
                refusal: nil))
        }
        return candidates
    }

    private static func malformedCandidate(sourceRange: NSRange) -> Candidate {
        Candidate(
            call: nil,
            argumentsJSON: "",
            sourceRange: sourceRange,
            refusal: .malformedJSON)
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
            riskAssessment: ToolRiskClassifier.assessRisk(
                name: name, arguments: arguments, projectURL: projectURL)
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

    private static func firstCapturePair(in text: String, pattern: String) -> (String, String)? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return nil
        }
        let ns = text as NSString
        guard
            let match = regex.firstMatch(
                in: text, options: [], range: NSRange(location: 0, length: ns.length)),
            match.range.location == 0,
            NSMaxRange(match.range) == ns.length,
            match.numberOfRanges > 2
        else { return nil }
        return (ns.substring(with: match.range(at: 1)), ns.substring(with: match.range(at: 2)))
    }
}
