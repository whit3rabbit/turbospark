import Foundation

/// Security risk level associated with a tool call.
public enum ToolRiskLevel: String, Codable, CaseIterable, Identifiable, Sendable {
    case safe
    case low
    case high

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .safe: return "Safe"
        case .low: return "Low Risk"
        case .high: return "High Risk"
        }
    }

    public var systemImage: String {
        switch self {
        case .safe: return "checkmark.shield.fill"
        case .low: return "info.circle.fill"
        case .high: return "exclamationmark.triangle.fill"
        }
    }
}

/// Comprehensive risk assessment outcome for an individual tool invocation.
public struct ToolRiskAssessment: Codable, Equatable, Sendable {
    public var level: ToolRiskLevel
    public var category: AppToolCategory
    public var reasons: [String]
    public var isHighRisk: Bool { level == .high }

    public init(level: ToolRiskLevel, category: AppToolCategory, reasons: [String] = []) {
        self.level = level
        self.category = category
        self.reasons = reasons
    }

    public static var safe: ToolRiskAssessment {
        ToolRiskAssessment(level: .safe, category: .fileRead, reasons: [])
    }

    /// Tolerant decode (state#45).
    ///
    /// This type is nested inside `AppToolCall`, which is nested inside
    /// `AppChatMessage`, which is inside the chat archive -- so the
    /// SYNTHESIZED decoder here is a way for one added field or one renamed
    /// enum case to quarantine every conversation the user has, past the
    /// hand-written tolerance `AppChatMessage` and `AppChat` already carry.
    /// The whole point of a tolerant outer decoder is defeated by a strict
    /// inner one.
    ///
    /// **An unknown risk level reads as `.high`, not as `.safe`.** This value
    /// gates an approval card; a level this build cannot interpret is
    /// precisely the case that should reach a human.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        level = container.decodeTolerant(ToolRiskLevel.self, forKey: .level, fallback: .high)
        category = container.decodeTolerant(
            AppToolCategory.self, forKey: .category, fallback: .automation)
        reasons = try container.decodeLossyArray(String.self, forKey: .reasons)
    }
}

/// Analyzes tool arguments and command lines to classify security risk (Unsloth Studio parity).
public enum ToolRiskClassifier {
    // MARK: - High-Risk Terminal Patterns

    private static let dangerousTerminalPatterns: [String] = [
        // Sudo / privilege escalation
        #"^\s*(sudo|su|doas)\b"#,
        #"\b(sudo|su|doas)\s+"#,
        // Destructive file deletion / shredding / raw disk writing
        #"\b(rm\s+-[a-zA-Z]*[rfRF][a-zA-Z]*|shred|dd(\s+[^|;&\n]+)?\s+if=|unlink|mkfs|wipefs)"#,
        // Git destructive actions
        #"\bgit\s+(clean(\s+-[a-zA-Z]*[fF][a-zA-Z]*|\s+--force)|reset\s+--hard|push(\s+-[a-zA-Z]*[fF][a-zA-Z]*|\s+--force)|branch\s+-[a-zA-Z]*[dD][a-zA-Z]*|stash\s+clear|checkout\s+--\s+\.|restore\s+\.)"#,
        // Secret / credential file inspection
        #"(/(etc/(shadow|passwd|master\.passwd)|proc/1/environ|\.ssh/(id_rsa|id_ed25519|id_dsa|authorized_keys)|\.aws/credentials|\.gnupg/secring|\.netrc))"#,
        // Remote piped script execution
        #"\b(curl|wget|fetch)\b[^\n|;&]*\|\s*(sh|bash|zsh|python|perl|ruby)\b"#,
        // Reverse shells / network listener
        #"\b(nc\s+-[a-zA-Z]*e|ncat\s+-[a-zA-Z]*e|bash\s+-i\s+>&|/dev/tcp/)"#,
        // Chroot / namespace escape
        #"\b(chroot|nsenter|docker\s+run\s+.*-v\s+/:|podman\s+run\s+.*-v\s+/:)"#,
        // System tampering / permission blasting
        #"\b(chmod\s+(-R\s+)?(777|0777|u\+s|g\+s)|chown\s+-R|crontab(\s+-|\s+-r|\s+-e)|useradd|usermod|userdel|groupadd)"#,
        // Process termination of PID 1 or mass kill
        #"\b(kill(\s+-[a-zA-Z0-9]+)*\s+1\b|pkill\s+-[a-zA-Z0-9]+|killall\s+-[a-zA-Z0-9]+)"#,
        // Fork bomb pattern
        #":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"#,
        // `find` invoked with an action flag rather than pure filtering:
        // `-delete`/`-exec`/`-execdir`/`-ok`/`-okdir`/`-fprintf` mutate or
        // execute rather than list, but `find` sits in
        // `TerminalCommandClassifier.searchCommands` and its first-word
        // check has no idea an action flag is present (T9).
        #"\bfind\b[^\n;&|]*\s-(delete|exec|execdir|ok|okdir|fprintf)\b"#,
        // `awk`/`sed`/`perl` are read-oriented text tools, but all three can
        // shell out: awk/perl via `system(`/`exec(`, sed via its `e` command
        // or GNU `-i` in-place edit (T9).
        #"\b(awk|perl)\b[^\n;&|]*\b(system|exec)\s*\("#,
        #"\bsed\b[^\n;&|]*\s-[a-zA-Z]*i"#,
        // `sort`/`tee` writing output to an arbitrary path: `sort` sits in
        // `TerminalCommandClassifier.readCommands`, but `-o`/`--output`
        // overwrites whatever file is named, including outside the sandbox
        // the tool call otherwise believes it is confined to (T9).
        #"\bsort\b[^\n;&|]*\s(-o\b|--output\b)"#,
        #"\|\s*tee\b"#
    ]

    private static let dangerousRegexes: [NSRegularExpression] = {
        dangerousTerminalPatterns.compactMap { try? NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }
    }()

    // MARK: - Sensitive File Extensions and Names

    private static let sensitiveFileNames: Set<String> = [
        ".env", ".env.local", ".env.production", ".env.secret",
        "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa",
        "credentials", "secrets.json", "secret.json", "private_key.pem", "id_rsa.pub"
    ]

    private static let sensitivePathPrefixes: [String] = [
        "/etc", "/private/etc", "/var/root", "/System", "/Library", "/usr/bin", "/usr/sbin", "/bin", "/sbin"
    ]

    // MARK: - MCP Risk Patterns

    private static let mcpDestructiveVerbs: Set<String> = [
        "delete", "drop", "destroy", "nuke", "kill", "wipe", "truncate", "format",
        "remove", "purge", "clear", "terminate", "obliterate", "zap"
    ]

    private static let mcpExecutionVerbs: Set<String> = [
        "exec", "execute", "run", "runcommand", "executecommand", "shellexec",
        "runscript", "executescript", "invokeshell", "eval", "python", "node", "bash", "shell"
    ]

    private static let mcpPrivilegeVerbs: Set<String> = [
        "grant", "revoke", "promote", "impersonate", "wire", "transfer", "charge",
        "add_collaborator", "add_team_member", "create_subscription", "wire_payment"
    ]

    private static let mcpSensitiveNouns: Set<String> = [
        "secret", "token", "password", "key", "credential", "auth", "session",
        "apikey", "privatekey", "certificate", "fund", "funds", "bank", "account"
    ]

    // MARK: - Public Risk Evaluator

    /// Evaluates the security risk of any tool call.
    public static func assessRisk(name: String, arguments: [String: String]) -> ToolRiskAssessment {
        let lowerName = name.lowercased()
        let category = AppToolCatalog.category(for: lowerName)

        // 1. Always Safe Builtin Tools
        switch lowerName {
        case "list_directory", "list_dir", "ls", "glob", "read_file", "view_file", "cat", "fileread", "read",
             "search_code", "grep", "search", "skill", "todowrite", "todo_write", "snip", "extract_snippet",
             "ctxinspect", "ctx_inspect", "listmcpresources", "list_mcp_resources", "list_resources",
             "readmcpresource", "read_mcp_resource", "read_resource", "tasklist", "task_list", "taskget",
             "task_get", "taskoutput", "task_output", "sleep", "delay", "askuserquestion", "ask_user_question",
             "ask_question", "question", "enterplanmode", "enter_plan_mode", "plan_mode", "plan",
             "exitplanmode", "exit_plan_mode", "reportfindings", "report_findings", "findings",
             "proposegoal", "propose_goal", "sendfeedback", "send_feedback", "senduserfile", "send_user_file":
            // Check if read_file or snip is accessing a sensitive credential path
            if let path = arguments["path"] ?? arguments["file_path"] {
                if isSensitivePath(path) {
                    return ToolRiskAssessment(
                        level: .high,
                        category: category,
                        reasons: ["Attempting to read sensitive credential or system path: '\(path)'"]
                    )
                }
            }
            return ToolRiskAssessment(level: .safe, category: category, reasons: [])

        default:
            break
        }

        // 2. Terminal Command Risk
        if category == .terminal {
            let cmd = arguments["command"] ?? arguments["cmd"] ?? ""
            return assessTerminalCommand(cmd)
        }

        // 3. File Modification Risk
        if category == .fileWrite {
            let path = arguments["path"] ?? arguments["file_path"] ?? ""
            var reasons: [String] = []

            if isSensitivePath(path) {
                reasons.append("Modifying sensitive or system file: '\(path)'")
            }

            if reasons.isEmpty {
                return ToolRiskAssessment(level: .low, category: category, reasons: ["File modification in workspace"])
            } else {
                return ToolRiskAssessment(level: .high, category: category, reasons: reasons)
            }
        }

        // 4. Web & Network Risk
        if category == .web {
            let urlString = arguments["url"] ?? arguments["uri"] ?? ""
            if !urlString.isEmpty {
                guard let url = URL(string: urlString), let host = url.host, !host.isEmpty else {
                    // A URL this app cannot parse is one it cannot classify.
                    return ToolRiskAssessment(
                        level: .high,
                        category: category,
                        reasons: ["Malformed or unparseable URL: '\(urlString)'"]
                    )
                }

                if AppToolSandbox.isPrivateOrMetadataHost(host) {
                    return ToolRiskAssessment(
                        level: .high,
                        category: category,
                        reasons: ["Accessing private network or cloud metadata endpoint: '\(urlString)'"]
                    )
                }

                // The sandbox's own allow/deny lists, which had no caller at
                // all until now: `SandboxConfig.allowedDomains`,
                // `deniedDomains` and `networkAllowed` were live fields
                // configuring a function nothing invoked (root Gotcha 8).
                do {
                    try AppToolSandbox.validateDomain(host)
                } catch {
                    return ToolRiskAssessment(
                        level: .high,
                        category: category,
                        reasons: [error.localizedDescription]
                    )
                }

                return ToolRiskAssessment(level: .low, category: category, reasons: ["Outbound web fetch to '\(urlString)'"])
            }
            return ToolRiskAssessment(level: .safe, category: category, reasons: [])
        }

        // 5. MCP Tool Risk
        if category == .mcp || lowerName.contains("__") || lowerName.hasPrefix("mcp_") {
            return assessMcpTool(name: name, arguments: arguments)
        }

        // 6. Automation & Cron Risk
        if category == .automation {
            return ToolRiskAssessment(
                level: .low,
                category: category,
                reasons: ["Scheduled background automation or monitoring job"]
            )
        }

        // Default / Unknown tool: Fail closed (low/high risk depending on arguments)
        return ToolRiskAssessment(
            level: .low,
            category: category,
            reasons: ["Generic tool invocation: '\(name)'"]
        )
    }

    // MARK: - Terminal Command Analysis

    /// Analyzes a shell command string for high-risk operations.
    ///
    /// **THE DENYLIST BELOW NAMES REASONS; THE ALLOWLIST DECIDES.** Under
    /// `.auto`, `AppToolPermissionEngine.evaluate` returns `.allow` for
    /// anything that is not `.high` -- `.safe` and `.low` are the same
    /// decision there and differ only in how the card is drawn. So the
    /// question this function really answers is binary: does this string run
    /// with no human in the loop.
    ///
    /// It used to answer it with regexes over the raw text, which is a losing
    /// shape against `/bin/zsh -c`. `rm -rf ~/Documents` matched;
    /// `r""m -rf ~/Documents` did not, and zsh runs them identically. Neither
    /// did `eval $(printf ...)`, `$'\x72m' -rf ~`, or `IFS=X; cmd=rmXX-rf; $cmd ~`.
    /// The patterns are still here because a match produces a specific,
    /// useful sentence for the approval sheet, but they are no longer the
    /// thing standing between the model and the shell.
    public static func assessTerminalCommand(_ command: String) -> ToolRiskAssessment {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ToolRiskAssessment(level: .safe, category: .terminal, reasons: [])
        }

        var reasons: [String] = []

        // Check against known dangerous regex patterns
        let nsString = trimmed as NSString
        let fullRange = NSRange(location: 0, length: nsString.length)

        for regex in dangerousRegexes {
            if regex.firstMatch(in: trimmed, options: [], range: fullRange) != nil {
                reasons.append("Detected high-risk command pattern matching: '\(regex.pattern)'")
            }
        }

        // Substring / token checks for dangerous constructs
        let lower = trimmed.lowercased()
        if lower.contains("rm -rf") || lower.contains("rm -fr") || lower.contains("rm -r -f") {
            if !reasons.contains(where: { $0.contains("rm") }) {
                reasons.append("Recursive forced file deletion (rm -rf)")
            }
        }

        if lower.contains("> /etc/") || lower.contains(">> /etc/") || lower.contains("> /dev/") {
            reasons.append("Redirection to system or device file")
        }

        if !reasons.isEmpty {
            return ToolRiskAssessment(level: .high, category: .terminal, reasons: reasons)
        }

        // Nothing matched a written-down attack. That is not evidence the
        // command is benign, so the decision passes to the allowlist: one
        // plain invocation of a known read or build command runs, and
        // anything this classifier cannot read asks.
        guard TerminalCommandClassifier.isSingleSimpleInvocation(trimmed) else {
            return ToolRiskAssessment(
                level: .high,
                category: .terminal,
                reasons: withAdvisory(
                    [
                        "Command uses shell control characters (pipes, redirection, "
                            + "substitution, quoting or escapes), so what it runs cannot be "
                            + "determined by inspection"
                    ], for: trimmed))
        }

        guard TerminalCommandClassifier.isAutoApprovable(trimmed) else {
            let head = TerminalCommandClassifier.headCommand(trimmed) ?? trimmed
            return ToolRiskAssessment(
                level: .high,
                category: .terminal,
                reasons: withAdvisory(
                    ["'\(head)' is not a recognized read-only or build command"],
                    for: trimmed))
        }

        // **THE ONLY PLACE THE MODEL MAY SPEAK ABOUT WHAT RUNS, AND IT CAN ONLY
        // SAY NO.** Both guards above have already passed, so this is reached
        // only for a command the allowlist admitted, and a veto can only move
        // it to `.high`. Off by default (`CommandGate.vetoEnabled`): measured
        // against the corpus lists in `TerminalRiskGateTests` the model adds
        // zero true positives and one to three false positives, so enabling it
        // costs prompts on `python3 -m pytest` and buys nothing until the
        // corpus is rebuilt around allowlist evasion.
        if let veto = CommandGate.veto(for: trimmed) {
            return ToolRiskAssessment(
                level: .high, category: .terminal, reasons: [veto.reason])
        }

        // A single simple invocation of an allowlisted command. `.safe` for
        // the read set (collapsed in the transcript), `.low` for builds,
        // which mutate a working tree even when they are expected to.
        if TerminalCommandClassifier.isCollapsible(trimmed) {
            return ToolRiskAssessment(level: .safe, category: .terminal, reasons: [])
        }
        return ToolRiskAssessment(level: .low, category: .terminal, reasons: ["Standard workspace terminal command"])
    }

    /// Appends the local classifier's opinion to reasons for a verdict that is
    /// ALREADY `.high`.
    ///
    /// Safe to run unconditionally, and that is the point: it decorates the
    /// approval sheet and can never change what runs. The generic allowlist
    /// sentence is accurate and unhelpful, so "local classifier: hazard 0.94"
    /// beside it gives the user something to decide on.
    private static func withAdvisory(_ reasons: [String], for command: String) -> [String] {
        guard let advisory = CommandGate.advisoryReason(for: command) else { return reasons }
        return reasons + [advisory]
    }

    // MARK: - MCP Risk Analysis

    /// Analyzes an MCP tool name and its arguments for high-risk verbs/nouns.
    public static func assessMcpTool(name: String, arguments: [String: String]) -> ToolRiskAssessment {
        let rawToolName = name.components(separatedBy: "__").last ?? name
        let normalized = rawToolName.replacingOccurrences(of: "([a-z])([A-Z])", with: "$1_$2", options: .regularExpression).lowercased()
        let parts = Set(normalized.components(separatedBy: CharacterSet(charactersIn: "_-.")))

        var reasons: [String] = []

        // 1. Check for destructive verbs
        for verb in mcpDestructiveVerbs {
            if parts.contains(verb) || normalized.contains(verb) {
                reasons.append("MCP tool contains destructive action verb '\(verb)'")
                break
            }
        }

        // 2. Check for execution verbs
        for verb in mcpExecutionVerbs {
            if parts.contains(verb) || normalized.contains(verb) {
                reasons.append("MCP tool executes arbitrary commands or scripts ('\(verb)')")
                break
            }
        }

        // 3. Check for privilege / payment verbs
        for verb in mcpPrivilegeVerbs {
            if parts.contains(verb) || normalized.contains(verb) {
                reasons.append("MCP tool modifies permissions, privileges, or billing ('\(verb)')")
                break
            }
        }

        // 4. Check for sensitive nouns
        for noun in mcpSensitiveNouns {
            if parts.contains(noun) || normalized.contains(noun) {
                reasons.append("MCP tool references sensitive entity '\(noun)'")
                break
            }
        }

        // 5. Inspect arguments for SQL mutation or sensitive paths
        for (k, v) in arguments {
            let lowerVal = v.lowercased()
            if lowerVal.contains("delete from") || lowerVal.contains("drop table") || lowerVal.contains("truncate table") {
                reasons.append("MCP argument '\(k)' contains destructive SQL query")
            }
            if isSensitivePath(v) {
                reasons.append("MCP argument '\(k)' references sensitive path '\(v)'")
            }
        }

        if !reasons.isEmpty {
            return ToolRiskAssessment(level: .high, category: .mcp, reasons: reasons)
        }

        return ToolRiskAssessment(level: .low, category: .mcp, reasons: ["MCP tool call '\(name)'"])
    }

    // MARK: - Helpers

    /// Checks whether a given path references sensitive system files or credentials.
    public static func isSensitivePath(_ path: String) -> Bool {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        let fileName = (trimmed as NSString).lastPathComponent.lowercased()
        if sensitiveFileNames.contains(fileName) || fileName.hasPrefix(".env") {
            return true
        }

        for prefix in sensitivePathPrefixes {
            if trimmed == prefix || trimmed.hasPrefix(prefix + "/") {
                return true
            }
        }

        if trimmed.contains("/.ssh/") || trimmed.contains("/.gnupg/") || trimmed.contains("/.aws/") {
            return true
        }

        return false
    }
}
