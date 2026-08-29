import Foundation

extension AppHookExecutionEngine {
    func matchesCondition(hook: AppHookCommand, toolName: String?, toolArguments: [String: String]?) -> Bool {
        // If matcher is set (e.g. "Write|Edit" or "Bash")
        if let matcher = hook.matcher, !matcher.isEmpty, let toolName {
            let parts = matcher.split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }
            let matched = parts.contains("*") || parts.contains { part in
                toolName.localizedCaseInsensitiveContains(part) || part.localizedCaseInsensitiveContains(toolName)
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
                    if pattern != "*" && !cmdArg.localizedCaseInsensitiveContains(pattern.replacingOccurrences(of: "*", with: "")) {
                        return false
                    }
                }
            }
        }

        return true
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
