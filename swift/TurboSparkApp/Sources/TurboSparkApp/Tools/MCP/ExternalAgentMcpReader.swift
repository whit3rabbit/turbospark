import Foundation

/// One other-agent GLOBAL MCP config file and the servers found in it.
public struct ExternalAgentMcpSource: Identifiable, Sendable, Equatable {
    public var id: String { configPath }
    /// Which other tool owns the config file (drives the UI grouping).
    public let agent: SkillSourceAgent
    /// Display label for the tool, e.g. "Claude Code".
    public let label: String
    /// Absolute path of the config file that was read.
    public let configPath: String
    /// Servers parsed from the file, in file order.
    public let servers: [McpServerConfig]

    public init(
        agent: SkillSourceAgent, label: String, configPath: String, servers: [McpServerConfig]
    ) {
        self.agent = agent
        self.label = label
        self.configPath = configPath
        self.servers = servers
    }
}

/// Reads OTHER agent tools' global MCP configurations, for the Import
/// wizard only. Global servers are NOT discovered live anywhere: this
/// reader exists so importing them is an explicit user action, the twin of
/// the skills/agents roots the wizard scans.
///
/// Security posture inherited from `ProjectMcpDetector.parseConfigData`:
/// `autoApprove` is forced false (never trusted from disk), env expansion
/// is limited to the allowlisted variables with no project root, and the
/// wizard shows each server's command or URL before anything is added.
/// Imported servers are added DISABLED by the wizard.
public enum ExternalAgentMcpReader {
    /// JSON configs using the shared `mcpServers`-family root keys, parsed
    /// by the same code path as project configs.
    private static let jsonCandidates: [(agent: SkillSourceAgent, label: String, relPath: String)] = [
        (.claude, "Claude Code", ".claude.json"),
        (.cursor, "Cursor", ".cursor/mcp.json"),
        (.gemini, "Gemini CLI", ".gemini/settings.json")
    ]

    /// Reads every known global config that exists, returning only sources
    /// with at least one parseable server.
    public static func discoverSources(
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> [ExternalAgentMcpSource] {
        var sources: [ExternalAgentMcpSource] = []

        for candidate in jsonCandidates {
            let url = home.appendingPathComponent(candidate.relPath)
            guard let data = try? Data(contentsOf: url) else { continue }
            let servers = ProjectMcpDetector.parseConfigData(
                data, sourcePath: url.path, sourceLabel: candidate.label, rootURL: nil)
            if !servers.isEmpty {
                sources.append(ExternalAgentMcpSource(
                    agent: candidate.agent, label: candidate.label,
                    configPath: url.path, servers: servers))
            }
        }

        let codexURL = home.appendingPathComponent(".codex/config.toml")
        if let text = try? String(contentsOf: codexURL, encoding: .utf8) {
            let servers = parseCodexToml(text, sourcePath: codexURL.path)
            if !servers.isEmpty {
                sources.append(ExternalAgentMcpSource(
                    agent: .codex, label: "Codex",
                    configPath: codexURL.path, servers: servers))
            }
        }

        return sources
    }

    // MARK: - Codex TOML

    /// Parses `[mcp_servers.<name>]` tables out of Codex's config.toml.
    ///
    /// A targeted parser, deliberately NOT a TOML engine: it understands
    /// exactly the value shapes Codex's MCP tables use (string, string
    /// array, inline string table) and skips everything else. Anything it
    /// cannot interpret leaves that server out rather than guessing; the
    /// rest of the file is ignored whether or not it parses.
    static func parseCodexToml(_ text: String, sourcePath: String) -> [McpServerConfig] {
        var servers: [McpServerConfig] = []
        var order: [String] = []
        // String or array-of-string values per server; env is tracked
        // separately so `[mcp_servers.x.env]` subtables land in the right
        // place without a nested-dictionary state machine.
        var fields: [String: [String: TomlValue]] = [:]
        var envTables: [String: [String: String]] = [:]
        var currentServer: String?
        var inEnvSubtable = false

        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }

            if line.hasPrefix("[") {
                guard let closing = line.firstIndex(of: "]") else { continue }
                let header = String(line[line.index(after: line.startIndex)..<closing])
                // Split on TOP-LEVEL dots only: a quoted name may itself
                // contain dots ("github.tools"), which a naive split would
                // tear apart.
                let parts = splitTopLevel(header, separator: ".")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                guard parts.count >= 2, parts[0] == "mcp_servers" else {
                    currentServer = nil
                    inEnvSubtable = false
                    continue
                }
                // A quoted name keeps its dots; a bare name is used as-is.
                let name = stripTomlQuotes(parts[1])
                guard !name.isEmpty else {
                    currentServer = nil
                    inEnvSubtable = false
                    continue
                }
                if fields[name] == nil { order.append(name) }
                currentServer = name
                inEnvSubtable = parts.count >= 3 && parts[2] == "env"
                continue
            }

            guard let server = currentServer,
                  let eq = line.firstIndex(of: "=") else { continue }
            let key = stripTomlQuotes(
                String(line[..<eq]).trimmingCharacters(in: .whitespaces))
            let rawValue = String(line[line.index(after: eq)...])
                .trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty, let value = parseTomlValue(rawValue) else { continue }

            if inEnvSubtable {
                if case .string(let str) = value {
                    var env = envTables[server] ?? [:]
                    env[key] = ProjectMcpDetector.expandVariables(str, projectRoot: nil)
                    envTables[server] = env
                }
            } else {
                var dict = fields[server] ?? [:]
                dict[key] = value
                fields[server] = dict
            }
        }

        for name in order {
            guard let dict = fields[name] else { continue }
            let env = envTables[name] ?? [:]
            guard let config = codexServerConfig(
                name: name, fields: dict, inlineEnv: env, sourcePath: sourcePath)
            else { continue }
            servers.append(config)
        }
        return servers
    }

    /// Builds one server from its parsed fields, or nil when the table does
    /// not describe a server this app can run.
    private static func codexServerConfig(
        name: String,
        fields: [String: TomlValue],
        inlineEnv: [String: String],
        sourcePath: String
    ) -> McpServerConfig? {
        var env = inlineEnv
        if case .table(let table)? = fields["env"] {
            for (k, v) in table { env[k] = v }
        }

        // A remote server is a bare url; a local one is a command plus
        // optional args. Anything else is skipped, not guessed at.
        if case .string(let urlString)? = fields["url"], let url = URL(string: urlString) {
            var headers: [String: String] = [:]
            if case .table(let table)? = fields["headers"] {
                headers = table
            }
            return McpServerConfig(
                name: name,
                transport: .sse(url: url, headers: headers),
                isEnabled: true,
                autoApprove: false,
                sourcePath: sourcePath,
                serverDescription: "Imported from Codex config.toml")
        }

        guard case .string(let command)? = fields["command"], !command.isEmpty else {
            return nil
        }
        var args: [String] = []
        if case .array(let items)? = fields["args"] {
            args = items.map { ProjectMcpDetector.expandVariables($0, projectRoot: nil) }
        }
        let expandedCommand = ProjectMcpDetector.expandVariables(command, projectRoot: nil)
        return McpServerConfig(
            name: name,
            transport: .stdio(
                command: expandedCommand,
                args: args,
                env: env.mapValues { ProjectMcpDetector.expandVariables($0, projectRoot: nil) }),
            isEnabled: true,
            autoApprove: false,
            sourcePath: sourcePath,
            serverDescription: "Imported from Codex config.toml")
    }

    // MARK: - TOML value primitives

    private enum TomlValue {
        case string(String)
        case array([String])
        case table([String: String])
    }

    /// Parses the value shapes Codex MCP tables use: basic string, string
    /// array, inline string table. Bare numbers/booleans/anything else are
    /// nil -- none of them appear in a server definition this app runs.
    private static func parseTomlValue(_ raw: String) -> TomlValue? {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("\"") || trimmed.hasPrefix("'") {
            // A trailing comment after the value is stripped; the quote
            // scanner below never crosses the closing quote, so a "#"
            // inside the string is safe.
            return .string(parseTomlString(trimmed) ?? "")
        }
        if trimmed.hasPrefix("["), trimmed.hasSuffix("]") {
            let inner = String(trimmed.dropFirst().dropLast())
            var items: [String] = []
            for piece in splitTopLevel(inner, separator: ",") {
                let item = piece.trimmingCharacters(in: .whitespaces)
                guard !item.isEmpty else { continue }
                guard let str = parseTomlString(item) else { return nil }
                items.append(str)
            }
            return .array(items)
        }
        if trimmed.hasPrefix("{"), trimmed.hasSuffix("}") {
            let inner = String(trimmed.dropFirst().dropLast())
            var table: [String: String] = [:]
            for piece in splitTopLevel(inner, separator: ",") {
                let entry = piece.trimmingCharacters(in: .whitespaces)
                guard let eq = entry.firstIndex(of: "=") else { continue }
                let key = stripTomlQuotes(
                    String(entry[..<eq]).trimmingCharacters(in: .whitespaces))
                let value = parseTomlString(
                    String(entry[entry.index(after: eq)...]).trimmingCharacters(in: .whitespaces))
                guard !key.isEmpty, let value else { return nil }
                table[key] = value
            }
            return .table(table)
        }
        return nil
    }

    /// Reads one TOML basic/literal string, returning nil when the token is
    /// not a well-formed string. Escaped quotes are honored; other escapes
    /// pass through verbatim (Codex MCP values do not rely on them).
    private static func parseTomlString(_ raw: String) -> String? {
        guard let quote = raw.first, quote == "\"" || quote == "'" else { return nil }
        var result = ""
        var iterator = raw.indices.makeIterator()
        _ = iterator.next() // opening quote
        var closed = false
        while let index = iterator.next() {
            let char = raw[index]
            if char == "\\" && quote == "\"" {
                guard let nextIndex = iterator.next() else { return nil }
                result.append(raw[nextIndex])
                continue
            }
            if char == quote {
                closed = true
                break
            }
            result.append(char)
        }
        guard closed else { return nil }
        return result
    }

    private static func stripTomlQuotes(_ raw: String) -> String {
        parseTomlString(raw) ?? raw
    }

    /// Splits on `separator` at bracket depth zero and outside quotes, so
    /// `"a,b" , "c]"` splits on the real commas only.
    private static func splitTopLevel(_ text: String, separator: Character) -> [String] {
        var pieces: [String] = []
        var current = ""
        var depth = 0
        var quote: Character? = nil
        var previous: Character? = nil
        for char in text {
            if let open = quote {
                current.append(char)
                if char == open && previous != "\\" {
                    quote = nil
                }
            } else if char == "\"" || char == "'" {
                quote = char
                current.append(char)
            } else if char == "[" || char == "{" {
                depth += 1
                current.append(char)
            } else if char == "]" || char == "}" {
                depth -= 1
                current.append(char)
            } else if char == separator && depth == 0 {
                pieces.append(current)
                current = ""
            } else {
                current.append(char)
            }
            previous = char
        }
        pieces.append(current)
        return pieces
    }
}
