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
    public static let supportedAgents = ["claude", "codex", "opencode", "hermes", "openclaw", "dsh"]

    public static func isAgent(_ name: String) -> Bool {
        supportedAgents.contains(name.lowercased())
    }

    /// Returns the terminal export/launch command string for connecting an agent.
    public static func launchCommand(
        for agent: String,
        host: String = "127.0.0.1",
        port: UInt16 = 8080,
        apiKey: String = "local"
    ) -> String {
        let base = "http://\(host):\(port)/v1"
        switch agent.lowercased() {
        case "claude":
            return "export ANTHROPIC_BASE_URL=\"\(base)\" && export ANTHROPIC_API_KEY=\"\(apiKey)\" && claude"
        case "codex":
            return "export OPENAI_BASE_URL=\"\(base)\" && export OPENAI_API_KEY=\"\(apiKey)\" && codex"
        case "opencode":
            return "export OPENAI_BASE_URL=\"\(base)\" && export OPENAI_API_KEY=\"\(apiKey)\" && opencode"
        case "hermes", "openclaw", "dsh":
            return "export OPENAI_BASE_URL=\"\(base)\" && export OPENAI_API_KEY=\"\(apiKey)\""
        default:
            return "export OPENAI_BASE_URL=\"\(base)\" && export OPENAI_API_KEY=\"\(apiKey)\""
        }
    }
}
