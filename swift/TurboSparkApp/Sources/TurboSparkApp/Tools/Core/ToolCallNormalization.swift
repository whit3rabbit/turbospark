import Foundation

/// Mapping a model's spelling of a call onto the definition this turn offered.
///
/// **WHY THIS EXISTS.** Local models are trained on three vocabularies that
/// disagree: Claude Code (`Read`, `file_path`, `old_string`), opencode
/// (`read`, `filePath`, `oldString`, `bash`, `task`) and Pi (`read`, `path`,
/// `bash`). `ToolArgumentResolver` already accepts all of those spellings in the
/// executor, but `TurnAvailableTools.validate` sits in front of it and matched
/// the tool name and every required argument EXACTLY, so `read` against an
/// offered `FileRead`, or `path` where the schema says `file_path`, was refused
/// before the executor ever saw it. A model that did the right thing in its own
/// vocabulary got "tool was not offered" or "required" and had to guess again.
///
/// Normalization only ever resolves TO something this turn offered, and only
/// renames an argument to a property that definition declares, so it cannot
/// reach a tool the profile or agent removed, and the result still goes through
/// the same schema validation, permission engine and approval card.
extension TurnAvailableTools {
    /// The offered definition a call named `requested` means, or nil.
    ///
    /// Exact match first, so nothing that worked before resolves differently.
    /// Then case-insensitive, then the same family (`read`, `read_file`,
    /// `FileRead`; `bash`, `shell`, `run_command`; `task`, `subagent`, `agent`).
    func resolvedDefinition(for requested: String) -> OpenAITool? {
        if let exact = definitions.first(where: { $0.function.name == requested }) {
            return exact
        }
        let lowered = requested.lowercased()
        if let insensitive = definitions.first(where: { $0.function.name.lowercased() == lowered }) {
            return insensitive
        }
        let family = ToolCallNormalization.familyKey(for: requested)
        return definitions.first {
            ToolCallNormalization.familyKey(for: $0.function.name) == family
        }
    }

    /// `arguments` with each key this definition does not declare renamed to
    /// the property it does declare for the same idea.
    ///
    /// A key the schema already declares is never touched, and a rename never
    /// overwrites a value the model also sent under the canonical spelling:
    /// two disagreeing values stay for validation and the executor's conflict
    /// check to refuse, rather than one silently winning.
    func canonicalizedArguments(
        _ arguments: [String: ToolCallJSONValue], for tool: OpenAITool
    ) -> [String: ToolCallJSONValue] {
        let declared = Set((tool.function.parameters.properties ?? [:]).keys)
        guard !declared.isEmpty else { return arguments }
        var result = arguments
        for key in arguments.keys.sorted() where !declared.contains(key) {
            guard let target = ToolCallNormalization.declaredProperty(
                      matching: key, among: declared),
                  result[target] == nil, let value = result.removeValue(forKey: key)
            else { continue }
            result[target] = value
        }
        return result
    }
}

enum ToolCallNormalization {
    /// Names that mean the same tool across the three vocabularies. Only tools
    /// whose schemas are interchangeable belong here: `grep` and `grep_search`
    /// take different arguments, so neither resolves to the other.
    private static let families: [String: Set<String>] = [
        "bash": ["bash", "shell", "run_command", "exec", "terminal"],
        "agent": ["agent", "subagent", "task"],
        "todowrite": ["todowrite", "todo_write"],
        "websearch": ["websearch", "web_search", "search_web"],
        "webfetch": ["webfetch", "web_fetch", "fetch_url"],
        "askuserquestion": ["askuserquestion", "ask_user_question", "question"],
    ]

    /// The family a tool name belongs to; a name outside every family is its
    /// own family (its lowercase spelling).
    static func familyKey(for toolName: String) -> String {
        let key = AppToolCatalog.profileKey(for: toolName)
        for (family, members) in families where members.contains(key) { return family }
        return key
    }

    /// Spellings of one argument, folded (lowercase, no `_` or `-`). Camel and
    /// snake case already fold together (`filePath` is `file_path`); these are
    /// the pairs that differ in the word itself.
    private static let synonymGroups: [Set<String>] = [
        ["path", "filepath", "file", "targetfile", "absolutepath"],
        ["oldstring", "oldstr", "oldtext", "targetcontent"],
        ["newstring", "newstr", "newtext", "replacement", "replacementcontent", "filetext"],
        ["content", "codecontent"],
        ["runinbackground", "background"],
        ["glob", "include"],
        ["i", "ignorecase"],
    ]

    static func fold(_ key: String) -> String {
        key.lowercased().replacingOccurrences(of: "_", with: "").replacingOccurrences(of: "-", with: "")
    }

    /// The property in `declared` that `key` is another spelling of, if any.
    /// Deterministic when a schema declares several candidates (sorted order).
    static func declaredProperty(matching key: String, among declared: Set<String>) -> String? {
        let folded = fold(key)
        let candidates = declared.sorted()
        if let same = candidates.first(where: { fold($0) == folded }) { return same }
        guard let group = synonymGroups.first(where: { $0.contains(folded) }) else { return nil }
        return candidates.first { group.contains(fold($0)) }
    }
}
