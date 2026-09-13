import Foundation

/// The verdict a permission classifier returns for one tool call
/// (`swift/docs/SWIFT_AGENT_MODE.md`).
public enum ClassifierVerdict: Equatable, Sendable {
    /// The call may run without a human.
    case allow
    /// The call is refused; the reason is shown to the user and fed to the model.
    case block(reason: String)
    /// No verdict was produced: no model is loaded, the call timed out, or
    /// the response did not parse. The caller falls back to manual approval.
    case unavailable(reason: String)
}

/// Everything the classifier sees about one proposed call.
///
/// The projection is deliberate about what crosses: the shell command in
/// full (that IS the thing being judged), write paths with short content
/// previews, the web URL but not any prompt field, and bounded MCP
/// arguments. Tool RESULTS never appear here.
public struct ClassifierRequest: Equatable, Sendable {
    /// Canonical tool name (e.g. `run_command`, `mcp__fs__read`).
    public var toolName: String
    /// The call's functional category.
    public var category: AppToolCategory
    /// The projected, bounded view of the arguments.
    public var projectedCall: String
    /// The user's recent request text, capped, so the verdict can weigh
    /// intent (a soft deny the user explicitly asked for reads differently
    /// from the same action out of nowhere).
    public var recentUserIntent: String

    public init(
        toolName: String, category: AppToolCategory,
        projectedCall: String, recentUserIntent: String
    ) {
        self.toolName = toolName
        self.category = category
        self.projectedCall = projectedCall
        self.recentUserIntent = recentUserIntent
    }
}

/// Turns a proposed tool call into the bounded text the classifier judges.
public enum ToolCallProjection {
    /// Per-string cap for a forwarded argument value.
    static let maxArgumentCharacters = 2_000
    /// Shared budget across all forwarded arguments, measured on the
    /// projected form. Every cut is marked in place so the classifier never
    /// mistakes an omission for an absence.
    static let totalArgumentBudget = 16_000
    /// Preview length for file content carried by a write or edit.
    static let contentPreviewCharacters = 300
    /// Cap for a subagent prompt, which is forwarded in full shape because
    /// it is the instruction the sub-agent would follow.
    static let agentPromptCharacters = 4_000

    /// Projects `call` for the classifier.
    public static func project(_ call: AppToolCall) -> String {
        let arguments = call.arguments
        switch call.category {
        case .terminal:
            let command = call.shellCommand ?? ""
            return "shell command: \(command)"
        case .fileWrite:
            let path = arguments["path"] ?? arguments["file_path"] ?? "(no path)"
            var lines = ["write to path: \(path)"]
            for (key, value) in arguments.sorted(by: { $0.key < $1.key }) {
                guard key != "path", key != "file_path" else { continue }
                lines.append("\(key): \(truncated(value, to: contentPreviewCharacters))")
            }
            return lines.joined(separator: "\n")
        case .web:
            let target = arguments["url"] ?? arguments["uri"] ?? arguments["query"] ?? ""
            return "web request: \(target)"
        case .mcp:
            var lines: [String] = []
            if let target = McpPermissionRule.targetOfCall(name: call.name, arguments: arguments) {
                lines.append("mcp server: \(target.server)")
                lines.append("mcp tool: \(target.tool ?? "(server-wide)")")
            } else {
                lines.append("mcp call: \(call.name)")
            }
            if let annotations = mcpAnnotations(for: call) {
                lines.append(
                    "server annotations (self-reported, unverified): "
                        + "readOnly=\(annotations.readOnly == true), "
                        + "destructiveHint=\(annotations.destructiveHint == true)")
            }
            lines.append(boundedArguments(arguments))
            return lines.joined(separator: "\n")
        case .automation:
            var lines = ["automation call: \(call.name)"]
            lines.append(boundedArguments(arguments))
            return lines.joined(separator: "\n")
        case .fileRead:
            var lines = ["read call: \(call.name)"]
            lines.append(boundedArguments(arguments))
            return lines.joined(separator: "\n")
        }
    }

    /// The full projection including the tool name header, which is what the
    /// prompt actually embeds.
    public static func projectedCall(_ call: AppToolCall) -> String {
        "tool: \(call.name)\n\(project(call))"
    }

    static func mcpAnnotations(for call: AppToolCall) -> McpToolAnnotations? {
        guard let target = McpPermissionRule.targetOfCall(name: call.name, arguments: call.arguments),
              let tool = target.tool else { return nil }
        return McpToolCatalogCache.shared
            .tools(forServerName: target.server)?
            .first(where: { $0.name == tool })?.annotations
    }

    /// Every argument value cut per string and jointly by budget, with
    /// in-place markers (qwen-code's `arguments_truncated` rule, folded into
    /// the text itself because this classifier reads one string).
    static func boundedArguments(_ arguments: [String: String]) -> String {
        guard !arguments.isEmpty else { return "arguments: (none)" }
        var spent = 0
        var parts: [String] = []
        for (key, value) in arguments.sorted(by: { $0.key < $1.key }) {
            let remaining = totalArgumentBudget - spent
            guard remaining > 0 else {
                parts.append("\(key): [omitted: argument budget exhausted]")
                continue
            }
            let clipped = truncated(value, to: min(maxArgumentCharacters, remaining))
            spent += clipped.utf8.count
            if clipped.utf8.count < value.utf8.count {
                let omitted = value.utf8.count - clipped.utf8.count
                parts.append("\(key): \(clipped)...[truncated \(omitted) chars]")
            } else {
                parts.append("\(key): \(clipped)")
            }
        }
        return "arguments:\n" + parts.joined(separator: "\n")
    }

    static func truncated(_ value: String, to limit: Int) -> String {
        guard value.utf8.count > limit else { return value }
        return String(decoding: value.utf8.prefix(limit), as: UTF8.self)
    }
}

/// User-tunable steering for the classifier (`swift/docs/SWIFT_AGENT_MODE.md`).
///
/// Entries are natural-language sentences, not rule patterns: they are
/// embedded in the classifier's policy text alongside the built-in
/// defaults. Caps keep the policy prefix small; longer entries are
/// truncated and extra entries dropped at normalize time.
public struct AgentModeHints: Codable, Equatable, Sendable {
    /// Actions the classifier should auto-approve.
    public var allow: [String]
    /// Destructive or irreversible actions to block UNLESS the user's recent
    /// request asked for exactly that action and scope.
    public var softDeny: [String]
    /// Security-boundary actions to block regardless of hints or intent.
    public var hardDeny: [String]
    /// Facts about this machine and workspace the classifier should know.
    public var environment: [String]

    /// Per-entry character cap (qwen-code parity).
    public static let maxEntryCharacters = 200
    /// Entry cap for each of the three hint lists.
    public static let maxHintEntries = 50
    /// Entry cap for the environment list.
    public static let maxEnvironmentEntries = 20

    public init(
        allow: [String] = [], softDeny: [String] = [],
        hardDeny: [String] = [], environment: [String] = []
    ) {
        self.allow = allow
        self.softDeny = softDeny
        self.hardDeny = hardDeny
        self.environment = environment
    }

    /// Applies the caps. Called on the way INTO the prompt rather than on
    /// the way into settings, so a hand-edited settings.json keeps its text
    /// on disk and still cannot bloat the policy.
    public var normalized: AgentModeHints {
        func clipped(_ entries: [String], limit: Int) -> [String] {
            entries.prefix(limit).map {
                let trimmed = $0.trimmingCharacters(in: .whitespacesAndNewlines)
                guard trimmed.utf8.count > Self.maxEntryCharacters else { return trimmed }
                return String(decoding: trimmed.utf8.prefix(Self.maxEntryCharacters), as: UTF8.self)
            }
            .filter { !$0.isEmpty }
        }
        return AgentModeHints(
            allow: clipped(allow, limit: Self.maxHintEntries),
            softDeny: clipped(softDeny, limit: Self.maxHintEntries),
            hardDeny: clipped(hardDeny, limit: Self.maxHintEntries),
            environment: clipped(environment, limit: Self.maxEnvironmentEntries))
    }

    /// True when every list is empty: the policy is then the built-in one
    /// alone and the settings section can say so.
    public var isEmpty: Bool {
        allow.isEmpty && softDeny.isEmpty && hardDeny.isEmpty && environment.isEmpty
    }
}

/// The contract every permission classifier speaks. One method, three
/// verdicts, never throws: infrastructure failure is a `.unavailable`
/// verdict, not an error, because the caller's fallback (manual approval)
/// is a decision the same way allow and block are.
public protocol ToolCallClassifying: Sendable {
    func classify(_ request: ClassifierRequest) async -> ClassifierVerdict
}
