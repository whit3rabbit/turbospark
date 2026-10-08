import Foundation

/// Executor for listing and reading Model Context Protocol (MCP) server resources.
public enum McpResourceExecutor {
    public static func listResources(arguments: [String: String], project: AppProject?, rootURL: URL) async throws -> String {
        let serverFilter = arguments["server"] ?? arguments["server_name"]
        let globalServers = GlobalMcpFileStore.load().servers
        var allServers = AppToolCatalogMcp.resolvedServers(global: globalServers, project: project)

        if let serverFilter {
            allServers = allServers.filter { $0.name.lowercased() == serverFilter.lowercased() }
        }

        let enabled = allServers.filter { $0.isEnabled }
        if enabled.isEmpty {
            return "No active MCP servers configured or matching '\(serverFilter ?? "")'."
        }

        // No `resources/list` request is ever sent, so there is nothing true
        // to report: neither a connection status (never checked) nor a
        // resource list. Say so rather than print "Connected" with zero
        // resources, which the model would read as an empty server.
        throw NSError(
            domain: "TurboSparkTool",
            code: 81,
            userInfo: [NSLocalizedDescriptionKey: notImplementedMessage(
                "ListMcpResources", servers: enabled.map(\.name))]
        )
    }

    static func notImplementedMessage(_ tool: String, servers: [String]) -> String {
        "\(tool) is not implemented: this client does not send resources/list or "
            + "resources/read to MCP servers (\(servers.joined(separator: ", "))), so NO "
            + "resource was listed or read and nothing is known about their contents. "
            + "Use the server's MCP tools instead."
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
        let allServers = AppToolCatalogMcp.resolvedServers(global: globalServers, project: project)

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

        // Never contacts the server; a fabricated "retrieved, 0 bytes" made
        // the model report a real resource as empty.
        throw NSError(
            domain: "TurboSparkTool",
            code: 81,
            userInfo: [NSLocalizedDescriptionKey: notImplementedMessage(
                "ReadMcpResource", servers: [server.name]) + " (uri '\(uri)' was NOT read)"]
        )
    }
}
