import Foundation

extension AppHookExecutionEngine {
    func matchesCondition(hook: AppHookCommand, toolName: String?, toolArguments: [String: String]?) -> Bool {
        // If matcher is set (e.g. "Write|Edit" or "Bash")
        if let matcher = hook.matcher, !matcher.isEmpty, let toolName {
            let parts = matcher.split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }
            // EXACT (case-insensitive) match on tool name, not a bidirectional
            // substring test (T17): the previous
            // `toolName.contains(part) || part.contains(toolName)` matched
            // whenever either string happened to be a substring of the
            // other, so a matcher for one tool could fire for an unrelated
            // tool whose name happened to share characters with it.
            let matched = parts.contains("*") || parts.contains { part in
                part.caseInsensitiveCompare(toolName) == .orderedSame
            }
            if !matched { return false }
        }

        // If 'ifCondition' is set (e.g. "Bash(git *)")
        if let ifCond = hook.ifCondition, !ifCond.isEmpty, let toolName {
            if ifCond.contains("(") && ifCond.hasSuffix(")") {
                let toolPrefix = ifCond.prefix(while: { $0 != "(" })
                if !toolName.localizedCaseInsensitiveContains(toolPrefix) {
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

    func parsePreToolUseOutput(stdout: String, stderr: String, exitCode: Int32, hookName: String) -> AppHookPreToolUseDecision {
        if exitCode == 2 {
            let reason = stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? stdout.trimmingCharacters(in: .whitespacesAndNewlines)
                : stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return AppHookPreToolUseDecision(
                behavior: .deny,
                reason: reason.isEmpty ? "Blocked by hook `\(hookName)`" : reason,
                blockedByHookName: hookName
            )
        }

        if let data = stdout.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let decision = json["permissionDecision"] as? String {
                let reason = json["permissionDecisionReason"] as? String
                switch decision.lowercased() {
                case "deny", "block":
                    return AppHookPreToolUseDecision(behavior: .deny, reason: reason, blockedByHookName: hookName)
                case "ask":
                    return AppHookPreToolUseDecision(behavior: .ask, reason: reason, blockedByHookName: hookName)
                case "allow":
                    return AppHookPreToolUseDecision(behavior: .allow, reason: reason, blockedByHookName: hookName)
                default:
                    break
                }
            }
        }

        return AppHookPreToolUseDecision(behavior: .allow)
    }
}
