import Foundation

/// Persistent container for global MCP server configurations.
public struct GlobalMcpArchive: Codable, Sendable {
    public var servers: [McpServerConfig]

    public init(servers: [McpServerConfig] = []) {
        self.servers = servers
    }

    private enum CodingKeys: String, CodingKey { case servers }

    /// Element-level tolerance: one row this build cannot decode (a transport
    /// from a newer build) costs that row, not the whole list that the next
    /// save would then overwrite.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        servers = try c.decodeLossyArray(McpServerConfig.self, forKey: .servers)
    }

    public static func empty() -> GlobalMcpArchive {
        GlobalMcpArchive(servers: [])
    }
}

/// Filesystem storage utilities for saving and loading global application MCP servers.
public enum GlobalMcpFileStore {
    private static var storageDirectory: URL {
        AppStorageRoot.directory
    }

    private static var archiveFileURL: URL {
        storageDirectory.appendingPathComponent("global_mcp_servers.json")
    }

    /// Loads the saved global MCP servers archive from disk or returns an empty default.
    public static func load() -> GlobalMcpArchive {
        AppJSONStore.load(
            GlobalMcpArchive.self, from: archiveFileURL, label: "global MCP server archive")
            ?? GlobalMcpArchive.empty()
    }

    /// Persists the global MCP servers archive to disk atomically.
    public static func save(_ archive: GlobalMcpArchive) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        AppJSONStore.save(
            archive, to: archiveFileURL, label: "Global MCP servers", encoder: encoder)
    }
}
