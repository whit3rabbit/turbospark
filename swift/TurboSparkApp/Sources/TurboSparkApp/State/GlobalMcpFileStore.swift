import Foundation

/// Persistent container for global MCP server configurations.
public struct GlobalMcpArchive: Codable, Sendable {
    public var servers: [McpServerConfig]

    public init(servers: [McpServerConfig] = []) {
        self.servers = servers
    }

    public static func empty() -> GlobalMcpArchive {
        GlobalMcpArchive(servers: [])
    }
}

/// Filesystem storage utilities for saving and loading global application MCP servers.
public enum GlobalMcpFileStore {
    private static var storageDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = appSupport.appendingPathComponent("TurboSpark", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static var archiveFileURL: URL {
        storageDirectory.appendingPathComponent("global_mcp_servers.json")
    }

    /// Loads the saved global MCP servers archive from disk or returns an empty default.
    public static func load() -> GlobalMcpArchive {
        guard let data = try? Data(contentsOf: archiveFileURL),
              let archive = try? JSONDecoder().decode(GlobalMcpArchive.self, from: data) else {
            return GlobalMcpArchive.empty()
        }
        return archive
    }

    /// Persists the global MCP servers archive to disk atomically.
    public static func save(_ archive: GlobalMcpArchive) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(archive) {
            try? data.write(to: archiveFileURL, options: .atomic)
        }
    }
}
