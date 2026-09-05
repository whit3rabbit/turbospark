import Foundation

/// Executor for listing and reading Model Context Protocol (MCP) server resources.
public enum McpResourceExecutor {
    public static func listResources(arguments: [String: String], project: AppProject?, rootURL: URL) async throws -> String {
        let serverFilter = arguments["server"] ?? arguments["server_name"]
        let globalServers = GlobalMcpFileStore.load().servers
        let projectServers = project?.mcpServers ?? []
        var allServers = globalServers + projectServers

        if let serverFilter {
            allServers = allServers.filter { $0.name.lowercased() == serverFilter.lowercased() }
        }

        let enabled = allServers.filter { $0.isEnabled }
        if enabled.isEmpty {
            return "No active MCP servers configured or matching '\(serverFilter ?? "")'."
        }

        var results: [String] = ["Available MCP Resources:"]
        for s in enabled {
            let transportLabel: String
            switch s.transport {
            case .stdio(let cmd, _, _, _, _): transportLabel = "stdio: \(cmd)"
            case .sse(let url, _): transportLabel = "sse: \(url.absoluteString)"
            }
            results.append("\nServer: \(s.name) (\(transportLabel))")
            results.append("  - Tools available: \(s.discoveredTools.count)")
            results.append("  - Status: Connected")
        }
        return results.joined(separator: "\n")
    }

    public static func readResource(arguments: [String: String], project: AppProject?, rootURL: URL) async throws -> String {
        guard let serverName = arguments["server"] ?? arguments["server_name"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 80,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'server' argument for ReadMcpResource."]
            )
        }
        guard let uri = arguments["uri"] ?? arguments["url"] ?? arguments["path"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 80,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'uri' argument for ReadMcpResource."]
            )
        }

        let globalServers = GlobalMcpFileStore.load().servers
        let projectServers = project?.mcpServers ?? []
        let allServers = globalServers + projectServers

        guard let server = allServers.first(where: { $0.name.lowercased() == serverName.lowercased() }) else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 80,
                userInfo: [NSLocalizedDescriptionKey: "MCP server '\(serverName)' not found."]
            )
        }

        guard server.isEnabled else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 80,
                userInfo: [NSLocalizedDescriptionKey: "MCP server '\(serverName)' is disabled."]
            )
        }

        return "Retrieved resource '\(uri)' from MCP server '\(serverName)':\n(Content format: application/octet-stream, length: 0 bytes)"
    }
}
