import Foundation

/// Transport protocol used to communicate with an MCP server.
public enum McpTransportSpec: Codable, Sendable, Equatable {
    /// Local subprocess communicating over standard input and standard output via line-delimited JSON-RPC 2.0.
    ///
    /// `cwd` is the directory the child is spawned in. Nil means the caller's
    /// own choice stands, which for a tool call is the project root
    /// (`AppToolRegistry.executeMcpCall`) and for a connection test is nothing.
    ///
    /// `envPassthrough` NAMES host variables to forward, and is an allowlist
    /// rather than a switch on purpose. The child environment is built from
    /// scratch (see `McpClientEngine.childEnvironment`) precisely so a
    /// third-party server binary does not receive every credential this app
    /// was launched with, and a wildcard here would undo that.
    case stdio(
        command: String,
        args: [String] = [],
        env: [String: String] = [:],
        cwd: String? = nil,
        envPassthrough: [String] = [])
    /// Remote server communicating over HTTP / Server-Sent Events (SSE).
    case sse(url: URL, headers: [String: String] = [:])

    enum CodingKeys: String, CodingKey {
        case type, command, args, env, cwd, envPassthrough, url, headers
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
            // Both added after the first servers were written to disk, so both
            // are `decodeIfPresent` with a default (root Gotcha 13). An archive
            // predating them decodes with no working directory and an empty
            // passthrough, which is exactly the old behaviour.
            let cwd = try container.decodeIfPresent(String.self, forKey: .cwd)
            let envPassthrough =
                try container.decodeIfPresent([String].self, forKey: .envPassthrough) ?? []
            self = .stdio(
                command: command, args: args, env: env, cwd: cwd, envPassthrough: envPassthrough)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .stdio(let command, let args, let env, let cwd, let envPassthrough):
            try container.encode("stdio", forKey: .type)
            try container.encode(command, forKey: .command)
            try container.encode(args, forKey: .args)
            try container.encode(env, forKey: .env)
            try container.encodeIfPresent(cwd, forKey: .cwd)
            try container.encode(envPassthrough, forKey: .envPassthrough)
        case .sse(let url, let headers):
            try container.encode("sse", forKey: .type)
            try container.encode(url, forKey: .url)
            try container.encode(headers, forKey: .headers)
        }
    }
}

/// The MCP tool annotations a server may declare beside a tool in its
/// `tools/list` response.
///
/// Hints, not guarantees: the permission engine folds them into risk
/// assessment as one input beside the name heuristics, and the name
/// heuristics win on conflict. A server declaring `readOnly` on a tool
/// named `delete_everything` is describing itself, and the classifier
/// does not take its word for it.
public struct McpToolAnnotations: Codable, Sendable, Equatable {
    public var title: String?
    public var readOnly: Bool?
    public var destructiveHint: Bool?
    public var idempotentHint: Bool?
    public var openWorldHint: Bool?

    public init(
        title: String? = nil,
        readOnly: Bool? = nil,
        destructiveHint: Bool? = nil,
        idempotentHint: Bool? = nil,
        openWorldHint: Bool? = nil
    ) {
        self.title = title
        self.readOnly = readOnly
        self.destructiveHint = destructiveHint
        self.idempotentHint = idempotentHint
        self.openWorldHint = openWorldHint
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
    /// Server-declared behaviour hints, when the server sends them.
    ///
    /// Optional and defaulted so archives written before the field
    /// existed decode unchanged (synthesized Codable uses
    /// `decodeIfPresent` for optionals).
    public var annotations: McpToolAnnotations?

    public init(
        name: String,
        description: String,
        inputSchemaJSON: String = "{}",
        serverName: String,
        annotations: McpToolAnnotations? = nil
    ) {
        self.name = name
        self.description = description
        self.inputSchemaJSON = inputSchemaJSON
        self.serverName = serverName
        self.annotations = annotations
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

    /// Whether `name` is already taken among `existing`.
    ///
    /// **NAME IS THE IDENTITY KEY, NOT `id`.** A model addresses a server by
    /// name (`mcp__<server>__<tool>`), `AppToolRegistry.executeMcpCall` resolves
    /// it with `first(where:)`, and `AppToolPermissionEngine.evaluate` resolves
    /// its auto-approve arm the same way. Two servers sharing a name means the
    /// second can never be dialled, while the approval card shows only the name
    /// and so cannot tell the user which one they are about to run.
    ///
    /// The comparison is case-insensitive because both resolvers lowercase, and
    /// `excludingID` is what lets an EDIT of a server keep its own name.
    public static func nameIsTaken(_ name: String, among existing: [String]) -> Bool {
        let candidate = normalizedName(name)
        guard !candidate.isEmpty else { return false }
        return existing.contains { normalizedName($0) == candidate }
    }

    /// The same question against configs, minus one row by id, which is what
    /// lets an EDIT of a server keep its own name.
    public static func nameIsTaken(
        _ name: String,
        among existing: [McpServerConfig],
        excludingID: UUID?
    ) -> Bool {
        let names = existing
            .filter { excludingID == nil || $0.id != excludingID }
            .map(\.name)
        return nameIsTaken(name, among: names)
    }

    /// How both resolvers see a name: trimmed and lowercased.
    public static func normalizedName(_ name: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Single line summary of command or endpoint for UI display.
    public var commandSummary: String {
        switch transport {
        case .stdio(let cmd, let args, _, _, _):
            if args.isEmpty {
                return cmd
            }
            return "\(cmd) \(args.joined(separator: " "))"
        case .sse(let url, _):
            return url.absoluteString
        }
    }
}
