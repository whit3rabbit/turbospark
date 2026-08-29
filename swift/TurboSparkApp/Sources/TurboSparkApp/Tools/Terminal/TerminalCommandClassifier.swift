import Foundation

/// Analyzes shell commands to identify collapsible search/read operations vs mutating executions.
public enum TerminalCommandClassifier {
    /// Read-only search commands whose outputs are suitable for collapsible presentation.
    public static let searchCommands: Set<String> = [
        "find", "grep", "rg", "ag", "ack", "locate", "which", "whereis"
    ]

    /// Read-only inspection commands.
    public static let readCommands: Set<String> = [
        "cat", "head", "tail", "less", "more", "wc", "stat", "file", "strings",
        "jq", "awk", "cut", "sort", "uniq", "tr", "ls", "tree"
    ]

    /// Git inspection subcommands that are read-only.
    public static let gitReadSubcommands: Set<String> = [
        "status", "diff", "log", "show", "branch", "tag", "remote"
    ]

    /// Determines if a shell command string is purely a search or read operation.
    public static func isCollapsible(_ command: String) -> Bool {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let firstWord = trimmed.components(separatedBy: .whitespaces).first?.lowercased() else {
            return false
        }
        let baseCmd = (firstWord as NSString).lastPathComponent

        if searchCommands.contains(baseCmd) || readCommands.contains(baseCmd) {
            return true
        }

        if baseCmd == "git" {
            let parts = trimmed.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            if parts.count > 1 {
                let sub = parts[1].lowercased()
                return gitReadSubcommands.contains(sub)
            }
        }

        return false
    }

    /// Interprets a process termination status code.
    public static func interpretExitCode(_ code: Int32) -> String {
        switch code {
        case 0:
            return "success"
        case 124, 137, 143:
            return "terminated by timeout/signal"
        case 127:
            return "command not found"
        case 126:
            return "command invoked cannot execute"
        default:
            return "exit code \(code)"
        }
    }
}
