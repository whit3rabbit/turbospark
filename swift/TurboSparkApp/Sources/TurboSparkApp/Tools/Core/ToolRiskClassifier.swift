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
        #":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"#
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
             "search_code", "grep", "search", "skill", "todowrite", "todo_write", "tasklist", "task_list",
             "askuserquestion", "ask_user_question", "question", "taskcreate", "task_create":
            // Check if read_file is accessing a sensitive credential path
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
                if let url = URL(string: urlString) {
                    let host = url.host?.lowercased() ?? ""
                    if host == "169.254.169.254" || host == "metadata.google.internal" || host == "localhost" || host == "127.0.0.1" || host.hasPrefix("192.168.") || host.hasPrefix("10.") {
                        return ToolRiskAssessment(
                            level: .high,
                            category: category,
                            reasons: ["Accessing private network or cloud metadata endpoint: '\(urlString)'"]
                        )
                    }
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

        // Safe collapsible or read-only commands
        if TerminalCommandClassifier.isCollapsible(trimmed) {
            return ToolRiskAssessment(level: .safe, category: .terminal, reasons: [])
        }

        // Benign development commands (cargo, npm, pytest, git status/diff/add/commit)
        return ToolRiskAssessment(level: .low, category: .terminal, reasons: ["Standard workspace terminal command"])
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
