import Foundation

/// What a user must see before approving an MCP server: the command line AND
/// the environment, working directory and credentials-forwarding it carries.
/// `commandSummary` alone shows `npx -y pkg` while an `env` of
/// `NODE_OPTIONS=--require ./x.js` runs repo code before the package starts.
public struct McpApprovalSummary: Equatable, Sendable {
    /// Env names that make a child load or run code before `main`, or that
    /// redirect which binary is found. Matching is case-insensitive.
    static let loaderEnvNames: Set<String> = [
        "NODE_OPTIONS", "NODE_PATH", "PATH", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME",
        "RUBYOPT", "RUBYLIB", "PERL5OPT", "PERL5LIB", "BASH_ENV", "ENV", "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "GIT_SSH_COMMAND", "SHELL", "IFS",
    ]

    public static func isLoaderEnvName(_ name: String) -> Bool {
        let upper = name.uppercased()
        return loaderEnvNames.contains(upper) || upper.hasPrefix("DYLD_") || upper.hasPrefix("LD_")
    }

    /// Detail lines to render in a selectable block. The full argv is never
    /// truncated; loader-variable values are shown (they ARE the payload),
    /// other env values are masked because they are usually secrets.
    public let lines: [String]
    /// Loader-style env names present, sorted. Non-empty means warn.
    public let riskyEnvNames: [String]

    public init(transport: McpTransportSpec) {
        var lines: [String] = []
        var risky: [String] = []
        switch transport {
        case .stdio(let command, let args, let env, let cwd, let passthrough):
            lines.append("command: " + ([command] + args).joined(separator: " "))
            if let cwd, !cwd.isEmpty { lines.append("cwd: \(cwd)") }
            for name in env.keys.sorted() {
                if Self.isLoaderEnvName(name) {
                    risky.append(name)
                    lines.append("env (loader): \(name)=\(env[name] ?? "")")
                } else {
                    lines.append("env: \(name)=<hidden>")
                }
            }
            if !passthrough.isEmpty {
                lines.append("forwards host variables: " + passthrough.sorted().joined(separator: ", "))
                risky.append(contentsOf: passthrough.filter(Self.isLoaderEnvName))
            }
        case .sse(let url, let headers):
            lines.append("url: \(url.absoluteString)")
            if !headers.isEmpty {
                lines.append("sends headers: " + headers.keys.sorted().joined(separator: ", "))
            }
        }
        self.lines = lines
        self.riskyEnvNames = Array(Set(risky)).sorted()
    }

    /// All lines as one block.
    public var text: String { lines.joined(separator: "\n") }
}

extension McpServerConfig {
    public var approvalSummary: McpApprovalSummary { McpApprovalSummary(transport: transport) }
}
