import Foundation

/// Transport protocol used to communicate with an MCP server.
public enum McpTransportSpec: Codable, Sendable, Equatable {
    /// Local subprocess communicating over standard input and standard output via line-delimited JSON-RPC 2.0.
    case stdio(command: String, args: [String] = [], env: [String: String] = [:])
    /// Remote server communicating over HTTP / Server-Sent Events (SSE).
    case sse(url: URL, headers: [String: String] = [:])

    enum CodingKeys: String, CodingKey {
        case type, command, args, env, url, headers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        if type == "sse" {
            let url = try container.decode(URL.self, forKey: .url)
            let headers = try container.decodeIfPresent([String: String].self, forKey: .headers) ?? [:]
            self = .sse(url: url, headers: headers)
        } else {
            let command = try container.decode(String.self, forKey: .command)
            let args = try container.decodeIfPresent([String].self, forKey: .args) ?? []
            let env = try container.decodeIfPresent([String: String].self, forKey: .env) ?? [:]
            self = .stdio(command: command, args: args, env: env)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .stdio(let command, let args, let env):
            try container.encode("stdio", forKey: .type)
            try container.encode(command, forKey: .command)
            try container.encode(args, forKey: .args)
            try container.encode(env, forKey: .env)
        case .sse(let url, let headers):
            try container.encode("sse", forKey: .type)
            try container.encode(url, forKey: .url)
            try container.encode(headers, forKey: .headers)
        }
    }
}

/// Description of a single tool discovered from an MCP server.
public struct McpDiscoveredTool: Identifiable, Codable, Sendable, Equatable {
    public var id: String { "\(serverName)::\(name)" }
    /// Canonical tool name as published by the server.
    public var name: String
    /// Human-readable tool description.
    public var description: String
    /// Raw JSON Schema string describing parameters.
    public var inputSchemaJSON: String
    /// Owning server identifier.
    public var serverName: String

    public init(name: String, description: String, inputSchemaJSON: String = "{}", serverName: String) {
        self.name = name
        self.description = description
        self.inputSchemaJSON = inputSchemaJSON
        self.serverName = serverName
    }
}

/// Persistent configuration model for an MCP server instance.
public struct McpServerConfig: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID
    /// Unique server display name.
    public var name: String
    /// Transport specification (stdio subprocess or remote SSE).
    public var transport: McpTransportSpec
    /// Whether this server is active and included in agent turns.
    public var isEnabled: Bool
    /// Whether tool calls from this server run automatically without interactive prompts.
    public var autoApprove: Bool
    /// Optional origin file path (e.g. if imported from `.mcp.json` or `opencode.json`).
    public var sourcePath: String?
    /// Optional user notes or documentation.
    public var serverDescription: String?
    /// Cached list of discovered tools.
    public var discoveredTools: [McpDiscoveredTool]
    /// Timestamp when configured.
    public var createdAt: Date
    /// Timestamp when last updated or verified.
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        name: String,
        transport: McpTransportSpec,
        isEnabled: Bool = true,
        autoApprove: Bool = false,
        sourcePath: String? = nil,
        serverDescription: String? = nil,
        discoveredTools: [McpDiscoveredTool] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.transport = transport
        self.isEnabled = isEnabled
        self.autoApprove = autoApprove
        self.sourcePath = sourcePath
        self.serverDescription = serverDescription
        self.discoveredTools = discoveredTools
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    enum CodingKeys: String, CodingKey {
        case id, name, transport, isEnabled, autoApprove, sourcePath, serverDescription, discoveredTools, createdAt, updatedAt
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        // `name` is tolerant and `transport` is NOT, deliberately (state#45).
        // A server with no readable transport cannot be contacted, dialled or
        // repaired, so the honest outcome is that this ROW fails to decode --
        // which `decodeLossyArray` at every call site turns into one dropped
        // server rather than a lost archive.
        self.name = try container.decodeIfPresent(String.self, forKey: .name) ?? "Unnamed Server"
        self.transport = try container.decode(McpTransportSpec.self, forKey: .transport)
        self.isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
        self.autoApprove = try container.decodeIfPresent(Bool.self, forKey: .autoApprove) ?? false
        self.sourcePath = try container.decodeIfPresent(String.self, forKey: .sourcePath)
        self.serverDescription = try container.decodeIfPresent(String.self, forKey: .serverDescription)
        self.discoveredTools = try container.decodeLossyArray(
            McpDiscoveredTool.self, forKey: .discoveredTools)
        self.createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        self.updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
    }

    /// Single line summary of command or endpoint for UI display.
    public var commandSummary: String {
        switch transport {
        case .stdio(let cmd, let args, _):
            if args.isEmpty {
                return cmd
            }
            return "\(cmd) \(args.joined(separator: " "))"
        case .sse(let url, _):
            return url.absoluteString
        }
    }
}
