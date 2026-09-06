import Foundation

/// Detected MCP configuration file from a project directory.
public struct DetectedProjectMcpFile: Identifiable, Sendable, Equatable {
    public var id: String { fileURL.path }
    /// Name of the config file format (e.g. "Claude Code (.mcp.json)", "OpenCode (opencode.json)").
    public var formatLabel: String
    /// Absolute URL to the configuration file on disk.
    public var fileURL: URL
    /// Relative path within the project.
    public var relativePath: String
    /// Parsed server configurations found in this file.
    public var servers: [McpServerConfig]

    public init(
        formatLabel: String,
        fileURL: URL,
        relativePath: String,
        servers: [McpServerConfig]
    ) {
        self.formatLabel = formatLabel
        self.fileURL = fileURL
        self.relativePath = relativePath
        self.servers = servers
    }
}

/// Project-scoped discovery and import engine for MCP servers (compatible with agent-config, Claude Code, and OpenCode).
public enum ProjectMcpDetector {
    private static let candidateLocations: [(relPath: String, label: String)] = [
        (".mcp.json", "Claude Code / Standard (.mcp.json)"),
        ("opencode.json", "OpenCode (opencode.json)"),
        (".opencode/opencode.json", "OpenCode (.opencode/opencode.json)"),
        (".cursor/mcp.json", "Cursor (.cursor/mcp.json)"),
        (".vscode/mcp.json", "VS Code (.vscode/mcp.json)"),
        (".claude/mcp.json", "Claude (.claude/mcp.json)"),
        (".agents/mcp_config.json", "Antigravity / AGY (.agents/mcp_config.json)"),
        (".gemini/mcp.json", "Gemini (.gemini/mcp.json)")
    ]

    /// Scans a project root directory for all known MCP configuration files and extracts server definitions.
    public static func detectInProject(rootURL: URL) -> [DetectedProjectMcpFile] {
        var results: [DetectedProjectMcpFile] = []
        let fileManager = FileManager.default

        for candidate in candidateLocations {
            let fileURL = rootURL.appendingPathComponent(candidate.relPath)
            var isDir: ObjCBool = false
            if fileManager.fileExists(atPath: fileURL.path, isDirectory: &isDir), !isDir.boolValue {
                let servers = parseConfigFile(at: fileURL, rootURL: rootURL)
                if !servers.isEmpty {
                    results.append(
                        DetectedProjectMcpFile(
                            formatLabel: candidate.label,
                            fileURL: fileURL,
                            relativePath: candidate.relPath,
                            servers: servers
                        )
                    )
                }
            }
        }

        return results
    }

    /// Parses a JSON configuration file into a list of McpServerConfig items.
    public static func parseConfigFile(at url: URL, rootURL: URL?) -> [McpServerConfig] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return parseConfigData(data, sourcePath: url.path, sourceLabel: url.lastPathComponent, rootURL: rootURL)
    }

    /// The data-level form of `parseConfigFile`, for configs that came from
    /// somewhere other than a file -- a plugin manifest's inline
    /// `mcpServers` record, wrapped in `{"mcpServers": ...}` by its caller.
    public static func parseConfigData(
        _ data: Data, sourcePath: String, sourceLabel: String, rootURL: URL?
    ) -> [McpServerConfig] {
        // Sanitize trailing commas / comments if possible before parsing JSON
        let sanitizedData = sanitizeJsonComments(data)

        guard let jsonObject = try? JSONSerialization.jsonObject(with: sanitizedData) as? [String: Any] else {
            return []
        }

        // Support root keys: "mcpServers" (Claude/Cursor), "mcp" (OpenCode), "servers" (VSCode)
        let serversDict = (jsonObject["mcpServers"] as? [String: Any])
            ?? (jsonObject["mcp"] as? [String: Any])
            ?? (jsonObject["servers"] as? [String: Any])
            ?? [:]

        var configs: [McpServerConfig] = []

        for (serverName, rawValue) in serversDict {
            guard let serverDict = rawValue as? [String: Any] else { continue }

            // Handle enabled/disabled variations (OpenCode: "enabled", Claude: "disabled")
            let isEnabled: Bool
            if let enabledVal = serverDict["enabled"] as? Bool {
                isEnabled = enabledVal
            } else if let disabledVal = serverDict["disabled"] as? Bool {
                isEnabled = !disabledVal
            } else {
                isEnabled = true
            }

            // NEVER trust `autoApprove`/`auto_approve`/`alwaysAllow` from a
            // repo-controlled config file: `AppToolPermissionEngine` reads a
            // server's `autoApprove` flag to skip the confirmation prompt
            // entirely for non-high-risk calls, so honoring whatever a
            // cloned `.mcp.json` declares would let a malicious repository
            // grant itself silent tool execution the moment its config is
            // imported. Auto-approval requires an explicit, in-app opt-in
            // after import, never a value read straight off disk.
            let autoApprove = false

            let env = (serverDict["env"] as? [String: String]) ?? [:]
            let expandedEnv = env.mapValues { expandVariables($0, projectRoot: rootURL) }

            // 1. Stdio transport with command array: ["npx", "-y", "..."] (OpenCode standard)
            if let commandArray = serverDict["command"] as? [String], let firstCmd = commandArray.first {
                let cmdArgs = Array(commandArray.dropFirst())
                let additionalArgs = (serverDict["args"] as? [String]) ?? []
                let allArgs = (cmdArgs + additionalArgs).map { expandVariables($0, projectRoot: rootURL) }

                let spec = McpServerConfig(
                    name: serverName,
                    transport: .stdio(command: firstCmd, args: allArgs, env: expandedEnv),
                    isEnabled: isEnabled,
                    autoApprove: autoApprove,
                    sourcePath: sourcePath,
                    serverDescription: "Imported from \(sourceLabel)"
                )
                configs.append(spec)
            }
            // 2. Stdio transport with command string: "npx" + args array (Claude / Cursor standard)
            else if let command = serverDict["command"] as? String {
                let args = (serverDict["args"] as? [String]) ?? []
                let expandedArgs = args.map { expandVariables($0, projectRoot: rootURL) }

                let spec = McpServerConfig(
                    name: serverName,
                    transport: .stdio(command: command, args: expandedArgs, env: expandedEnv),
                    isEnabled: isEnabled,
                    autoApprove: autoApprove,
                    sourcePath: sourcePath,
                    serverDescription: "Imported from \(sourceLabel)"
                )
                configs.append(spec)
            }
            // 3. SSE transport: url + headers (OpenCode type: "remote" or Cursor/Claude url)
            else if let urlString = (serverDict["url"] as? String) ?? (serverDict["endpoint"] as? String),
                    let sseURL = URL(string: urlString) {
                let headers = (serverDict["headers"] as? [String: String]) ?? [:]

                let spec = McpServerConfig(
                    name: serverName,
                    transport: .sse(url: sseURL, headers: headers),
                    isEnabled: isEnabled,
                    autoApprove: autoApprove,
                    sourcePath: sourcePath,
                    serverDescription: "Imported from \(sourceLabel)"
                )
                configs.append(spec)
            }
        }

        return configs
    }

    /// Allowlist of safe system environment variable keys that may be expanded.
    public static let allowedEnvironmentVariableKeys: Set<String> = [
        "PATH", "HOME", "LANG", "TMPDIR"
    ]

    /// Expands safe variable placeholders (${workspaceFolder}, ${projectRoot}, ${cwd}, and allowlisted env vars).
    ///
    /// Arbitrary `${env:...}` and bare `$KEY` expansion over the parent process environment is
    /// intentionally NOT performed on repo-controlled configs: doing so would allow an untrusted
    /// repository to exfiltrate host credentials (e.g. `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`) by
    /// embedding them in server args or env definitions that get persisted or spawned.
    public static func expandVariables(_ input: String, projectRoot: URL?) -> String {
        var str = input
        if let rootPath = projectRoot?.path {
            str = str.replacingOccurrences(of: "${workspaceFolder}", with: rootPath)
            str = str.replacingOccurrences(of: "${projectRoot}", with: rootPath)
            str = str.replacingOccurrences(of: "${cwd}", with: rootPath)
        }
        let env = ProcessInfo.processInfo.environment
        for key in allowedEnvironmentVariableKeys {
            if let val = env[key] {
                str = str.replacingOccurrences(of: "${env:\(key)}", with: val)
            }
        }
        return str
    }

    /// Strips line comments (`// ...`) to make JSON parser resilient to JSONC formats.
    private static func sanitizeJsonComments(_ data: Data) -> Data {
        guard let text = String(data: data, encoding: .utf8) else { return data }
        var outputLines: [String] = []
        let lines = text.components(separatedBy: .newlines)
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("//") {
                continue
            }
            // Simple inline comment stripping if not inside quotes
            if let commentIndex = line.range(of: "//")?.lowerBound {
                let prefix = String(line[..<commentIndex])
                let quoteCount = prefix.filter { $0 == "\"" }.count
                if quoteCount % 2 == 0 {
                    outputLines.append(prefix)
                    continue
                }
            }
            outputLines.append(line)
        }
        let sanitized = outputLines.joined(separator: "\n")
        return sanitized.data(using: .utf8) ?? data
    }
}
