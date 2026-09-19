import CTurboSpark
import Foundation

/// Status of the managed background server daemon (`turbospark start/stop`).
public struct TurboSparkDaemonStatus: Decodable, Sendable, Equatable {
    /// True when a server daemon process is actively running.
    public let running: Bool
    /// Process identifier of the server daemon, if running.
    public let pid: Int32?
    /// HTTP listen port of the server daemon, if running.
    public let port: UInt16?
    /// Complete API base endpoint URL (e.g. `http://127.0.0.1:8080/v1`).
    public let endpoint: String?
    /// Path to the server daemon log file (`~/.turbospark/logs/server.log`).
    public let logPath: String?

    public init(
        running: Bool,
        pid: Int32? = nil,
        port: UInt16? = nil,
        endpoint: String? = nil,
        logPath: String? = nil
    ) {
        self.running = running
        self.pid = pid
        self.port = port
        self.endpoint = endpoint
        self.logPath = logPath
    }
}

/// Inspection and lifecycle control for the managed background server daemon (`turbospark start/stop/status`).
public enum TurboSparkDaemon {
    /// Inspects whether a background server daemon is active and returns its status.
    public static func status() throws -> TurboSparkDaemonStatus {
        let json = try takeString { out in
            ts_daemon_status_json(out)
        }
        return try decode(TurboSparkDaemonStatus.self, from: json)
    }

    /// Stops the running background server daemon, if any.
    public static func stop() throws {
        try check(ts_daemon_stop())
    }

    /// Starts the background server daemon with optional arguments.
    public static func start(args: [String] = []) throws {
        let json = try JSONEncoder().encode(args)
        let jsonString = String(data: json, encoding: .utf8) ?? "[]"
        try check(jsonString.withCString { ts_daemon_start($0) })
    }

    /// Stops and restarts the background server daemon with optional arguments.
    public static func restart(args: [String] = []) throws {
        let json = try JSONEncoder().encode(args)
        let jsonString = String(data: json, encoding: .utf8) ?? "[]"
        try check(jsonString.withCString { ts_daemon_restart($0) })
    }
}

/// Coding agent connectors matching `turbospark start <agent>`.
public struct TurboSparkAgent: Sendable {
    public static let supportedAgents = [
        "claude", "codex", "opencode", "grok", "gemini", "hermes", "openclaw", "dsh",
    ]

    public static func isAgent(_ name: String) -> Bool {
        supportedAgents.contains(name.lowercased())
    }

    /// Returns the terminal export/launch command string for connecting an agent.
    ///
    /// Every interpolated value lands inside double quotes in a command the
    /// user pastes into a terminal, so each is escaped for that context: an
    /// unescaped `"`, `$` or backtick in a hand-typed key terminates the
    /// export or runs part of the key as a substitution. The engine-bound
    /// host never carries metacharacters today, but the helper does not
    /// have to be re-derived the day a bind address could.
    ///
    /// Pass the canonical id returned by an in-process server's `attach` as
    /// `canonicalModelID` to launch Claude Code with that backend's
    /// discovery alias. Omit it when the caller has no attached model to
    /// select yet; discovery is still enabled for `/model`.
    ///
    /// **CODEX DOES NOT READ `OPENAI_BASE_URL`.** Its custom-provider
    /// mechanism is a `[model_providers.<id>]` config table, which the
    /// `-c` overrides below spell on the command line: an in-memory
    /// override of `~/.codex/config.toml` for this one invocation, never a
    /// write to the file, so there is nothing to back up or restore -- the
    /// user's own config stays authoritative for every other codex run.
    /// The key rides `TURBOSPARK_API_KEY`, the env the table's `env_key`
    /// names, keeping it out of codex's argv as well.
    public static func launchCommand(
        for agent: String,
        host: String = "127.0.0.1",
        port: UInt16 = 8080,
        apiKey: String = "local",
        canonicalModelID: String? = nil
    ) -> String {
        let baseURL = "http://\(host):\(port)/v1"
        let base = shellDoubleQuoted(baseURL)
        switch agent.lowercased() {
        case "claude":
            let modelArgument = canonicalModelID.map {
                " --model \(shellDoubleQuoted("claude-turbospark-\($0)"))"
            } ?? ""
            let settings = shellSingleQuoted(claudeSettingsJSON(baseURL: baseURL, apiKey: apiKey))
            return "(settings_file=$(mktemp \"${TMPDIR:-/tmp}/turbospark-claude.XXXXXX\")"
                + " && chmod 600 \"$settings_file\""
                + " && trap 'rm -f \"$settings_file\"' EXIT HUP INT TERM"
                + " && printf '%s' \(settings) > \"$settings_file\""
                + " && claude --settings \"$settings_file\"\(modelArgument))"
        case "codex":
            // Each value is one `-c` argument, single-quoted because the
            // values themselves carry TOML's double quotes. `wire_api` is
            // "chat" because /v1/chat/completions is the wire this server
            // exercises end to end.
            let overrides = [
                "model_provider=\"turbospark\"",
                "model_providers.turbospark.name=\"TurboSpark\"",
                "model_providers.turbospark.base_url=\"\(baseURL)\"",
                "model_providers.turbospark.env_key=\"TURBOSPARK_API_KEY\"",
                "model_providers.turbospark.wire_api=\"chat\"",
            ]
            let modelArgument = canonicalModelID.map { " -m \(shellDoubleQuoted($0))" } ?? ""
            return "export TURBOSPARK_API_KEY=\(shellDoubleQuoted(apiKey)) && codex "
                + overrides.map { "-c \(shellSingleQuoted($0))" }.joined(separator: " ")
                + modelArgument
        case "opencode":
            return "export OPENAI_BASE_URL=\(base) && export OPENAI_API_KEY=\(shellDoubleQuoted(apiKey)) && opencode"
        case "hermes", "openclaw", "dsh":
            return "export OPENAI_BASE_URL=\(base) && export OPENAI_API_KEY=\(shellDoubleQuoted(apiKey))"
        default:
            return "export OPENAI_BASE_URL=\(base) && export OPENAI_API_KEY=\(shellDoubleQuoted(apiKey))"
        }
    }

    /// The `turbospark start <agent>` command a GUI launch button hands to
    /// Terminal.app. Unlike `launchCommand` (a self-contained paste for one
    /// already-running server), this routes through the CLI launcher so the
    /// daemon is started or model-switched before the agent execs, with the
    /// launcher's own per-agent wiring.
    ///
    /// `binaryPath` is spelled as the ABSOLUTE path found at button-enable
    /// time, so the command does not depend on the login shell's PATH; the
    /// agent binary itself is still resolved by the launcher inside that
    /// shell, whose PATH is the user's own.
    public static func cliLaunchCommand(
        for agent: String,
        binaryPath: String,
        modelArgument: String? = nil,
        port: UInt16? = nil
    ) -> String? {
        guard isAgent(agent) else { return nil }
        var tokens = [shellDoubleQuoted(binaryPath), "start", agent]
        if let modelArgument, !modelArgument.isEmpty {
            tokens += ["--model", shellDoubleQuoted(modelArgument)]
        }
        if let port {
            tokens += ["--port", String(port)]
        }
        return tokens.joined(separator: " ")
    }

    /// Wraps a shell command in the AppleScript that opens it in a new
    /// Terminal.app window. AppleScript string literals escape exactly two
    /// characters: the backslash and the closing double quote.
    public static func terminalDoScript(command: String) -> String {
        let escaped = command
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "tell application \"Terminal\" to do script \"\(escaped)\""
    }

    /// Escapes for a POSIX double-quoted string: backslash first, then the
    /// three characters a double-quoted shell context still treats
    /// specially -- the closing quote, parameter/command substitution, and
    /// command substitution's other spelling.
    private static func shellDoubleQuoted(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "$", with: "\\$")
            .replacingOccurrences(of: "`", with: "\\`")
        return "\"\(escaped)\""
    }

    /// This temporary-file overlay gives the ephemeral local server port an
    /// explicit one-session source without exposing the API key in Claude's
    /// process arguments.
    private static func claudeSettingsJSON(baseURL: String, apiKey: String) -> String {
        let settings: [String: [String: String]] = [
            "env": [
                "ANTHROPIC_BASE_URL": baseURL,
                "ANTHROPIC_API_KEY": apiKey,
                "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "true",
                "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
            ]
        ]
        let data = try! JSONSerialization.data(
            withJSONObject: settings,
            options: [.sortedKeys, .withoutEscapingSlashes])
        return String(decoding: data, as: UTF8.self)
    }

    private static func shellSingleQuoted(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
}
