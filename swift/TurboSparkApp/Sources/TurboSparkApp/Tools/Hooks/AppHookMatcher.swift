import Foundation

extension AppHookExecutionEngine {
    /// Whether `hook` fires for this call, per Claude Code's matcher
    /// dispatch order (https://code.claude.com/docs/en/hooks): empty/`*`/
    /// omitted matches everything; a pattern made only of letters, digits,
    /// `_`, `-`, spaces, `,` and `|` is an exact name or a comma/pipe
    /// separated list; anything else is an unanchored regex. Every name is
    /// checked through `AppHookToolNameAliases` so a matcher written for
    /// Claude Code's own tool names (`Bash`, `Write`, ...) fires on this
    /// app's tool names too, and vice versa.
    func matchesCondition(hook: AppHookCommand, toolName: String?, toolArguments: [String: String]?) -> Bool {
        if let matcher = hook.matcher, !matcher.isEmpty, let toolName {
            guard AppHookMatcherEvaluator.matches(pattern: matcher, toolName: toolName) else { return false }
        }

        // If 'ifCondition' is set (e.g. "Bash(git *)")
        if let ifCond = hook.ifCondition, !ifCond.isEmpty, let toolName {
            if ifCond.contains("(") && ifCond.hasSuffix(")") {
                let toolPrefix = ifCond.prefix(while: { $0 != "(" })
                let aliases = AppHookToolNameAliases.equivalentNames(for: toolName)
                let prefixMatches = aliases.contains(String(toolPrefix).lowercased())
                    || toolName.localizedCaseInsensitiveContains(toolPrefix)
                if !prefixMatches {
                    return false
                }
                if let innerStart = ifCond.firstIndex(of: "("), let innerEnd = ifCond.lastIndex(of: ")") {
                    let pattern = String(ifCond[ifCond.index(after: innerStart)..<innerEnd]).trimmingCharacters(in: .whitespaces)
                    let cmdArg = toolArguments?["command"] ?? toolArguments?["cmd"] ?? toolArguments?["path"] ?? ""
                    if pattern != "*" && !matchesCommandGlob(pattern, against: cmdArg) {
                        return false
                    }
                }
            }
        }

        return true
    }

    /// Matches `text` against a simple glob whose only wildcard is a
    /// trailing `*` (a prefix match), after normalizing whitespace on both
    /// sides. Replaces a bare "contains the pattern with `*` stripped,
    /// anywhere" check that both UNDER-matched a command whose whitespace
    /// did not line up byte-for-byte with the pattern (`"git  push"` against
    /// `"git *"`) and OVER-matched any command that merely mentioned the
    /// pattern text as a substring (`echo "git push "` against the same
    /// pattern) rather than actually being that command (T17).
    func matchesCommandGlob(_ pattern: String, against text: String) -> Bool {
        func normalize(_ s: String) -> String {
            s.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
                .lowercased()
        }
        let normalizedText = normalize(text)
        if pattern.hasSuffix("*") {
            let prefix = normalize(String(pattern.dropLast()))
            return normalizedText.hasPrefix(prefix)
        }
        return normalizedText == normalize(pattern)
    }
}

/// Evaluates one matcher pattern against a tool name, per the dispatch order
/// documented on `AppHookExecutionEngine.matchesCondition`.
enum AppHookMatcherEvaluator {
    static func matches(pattern: String, toolName: String) -> Bool {
        let trimmed = pattern.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty || trimmed == "*" { return true }

        let aliases = AppHookToolNameAliases.equivalentNames(for: toolName)

        let listCharset = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_- ,|"))
        let isListLike = trimmed.unicodeScalars.allSatisfy { listCharset.contains($0) }

        if isListLike {
            let parts = trimmed.components(separatedBy: CharacterSet(charactersIn: ",|"))
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            if parts.isEmpty || parts.contains("*") { return true }
            return parts.contains { aliases.contains($0.lowercased()) }
        }

        // Regex path: unanchored, matched against every alias so a
        // Claude-Code-style matcher (`^Notebook`, `mcp__.*`) still fires on
        // this app's own tool vocabulary, and against the raw tool name for
        // a pattern (e.g. `mcp__.*`) that has no alias entry at all.
        guard let regex = try? NSRegularExpression(pattern: trimmed) else { return false }
        for candidate in aliases {
            let range = NSRange(candidate.startIndex..., in: candidate)
            if regex.firstMatch(in: candidate, range: range) != nil { return true }
        }
        let rawRange = NSRange(toolName.startIndex..., in: toolName)
        return regex.firstMatch(in: toolName, range: rawRange) != nil
    }
}
