import Foundation

// MARK: - MCP Tool Definitions & Models

/// Input payload for listing data resources across connected MCP servers.
public struct ListMcpResourcesInput: Codable, Sendable, Equatable {
    /// Optional server name filter.
    public var server: String?

    public init(server: String? = nil) {
        self.server = server
    }
}

/// Metadata description of a single resource provided by an MCP server.
public struct McpResourceSummary: Codable, Sendable, Equatable {
    /// Unique resource URI.
    public var uri: String
    /// Display name of the resource.
    public var name: String
    /// MIME type of the content.
    public var mimeType: String?
    /// Description of the resource.
    public var description: String?
    /// Server identifier providing the resource.
    public var server: String

    public init(
        uri: String,
        name: String,
        mimeType: String? = nil,
        description: String? = nil,
        server: String
    ) {
        self.uri = uri
        self.name = name
        self.mimeType = mimeType
        self.description = description
        self.server = server
    }
}

/// Output payload containing discovered MCP resources.
public struct ListMcpResourcesOutput: Codable, Sendable, Equatable {
    /// Array of resource summaries.
    public var resources: [McpResourceSummary]

    public init(resources: [McpResourceSummary] = []) {
        self.resources = resources
    }
}

/// Input payload for reading an MCP resource by URI.
public struct ReadMcpResourceInput: Codable, Sendable, Equatable {
    /// Server identifier hosting the resource.
    public var server: String
    /// Target resource URI.
    public var uri: String

    public init(server: String, uri: String) {
        self.server = server
        self.uri = uri
    }
}

/// Content payload extracted from an MCP resource.
public struct McpResourceContent: Codable, Sendable, Equatable {
    /// Resource URI.
    public var uri: String
    /// MIME type.
    public var mimeType: String?
    /// Plain text content.
    public var text: String?
    /// Local file path if binary blob was saved to disk.
    public var blobSavedTo: String?

    public init(
        uri: String,
        mimeType: String? = nil,
        text: String? = nil,
        blobSavedTo: String? = nil
    ) {
        self.uri = uri
        self.mimeType = mimeType
        self.text = text
        self.blobSavedTo = blobSavedTo
    }
}

/// Output payload containing retrieved resource contents.
public struct ReadMcpResourceOutput: Codable, Sendable, Equatable {
    /// Extracted content items.
    public var contents: [McpResourceContent]
    /// Error message string if retrieval failed.
    public var error: String?

    public init(contents: [McpResourceContent] = [], error: String? = nil) {
        self.contents = contents
        self.error = error
    }
}

/// Input payload for listing child entries under an MCP directory URI.
public struct ReadMcpResourceDirInput: Codable, Sendable, Equatable {
    /// MCP server name.
    public var server: String
    /// Directory resource URI.
    public var uri: String

    public init(server: String, uri: String) {
        self.server = server
        self.uri = uri
    }
}

/// Output payload containing child resource entries in a directory.
public struct ReadMcpResourceDirOutput: Codable, Sendable, Equatable {
    /// Discovered child resources.
    public var resources: [McpResourceSummary]
    /// Error message if directory listing failed.
    public var error: String?

    public init(resources: [McpResourceSummary] = [], error: String? = nil) {
        self.resources = resources
        self.error = error
    }
}

/// Input payload for triggering tool schema refresh across connected MCP servers.
public struct RefreshMcpToolsInput: Codable, Sendable, Equatable {
    /// Optional server name to refresh specifically.
    public var server: String?

    public init(server: String? = nil) {
        self.server = server
    }
}

/// Refresh status report for an individual MCP server.
public struct McpServerRefreshStatus: Codable, Sendable, Equatable {
    /// Server identifier.
    public var server: String
    /// Status description.
    public var status: String
    /// Number of active tools.
    public var toolCount: Int?
    /// Tool names added.
    public var added: [String]?
    /// Tool names removed.
    public var removed: [String]?
    /// Error message if refresh failed.
    public var error: String?

    public init(
        server: String,
        status: String,
        toolCount: Int? = nil,
        added: [String]? = nil,
        removed: [String]? = nil,
        error: String? = nil
    ) {
        self.server = server
        self.status = status
        self.toolCount = toolCount
        self.added = added
        self.removed = removed
        self.error = error
    }
}

/// Output payload from refreshing MCP server tool schemas.
public struct RefreshMcpToolsOutput: Codable, Sendable, Equatable {
    /// Per-server refresh status items.
    public var servers: [McpServerRefreshStatus]

    public init(servers: [McpServerRefreshStatus] = []) {
        self.servers = servers
    }
}

/// Input payload for executing a generic dynamic MCP tool call.
public struct McpGenericInput: Codable, Sendable, Equatable {
    /// Server identifier.
    public var server: String
    /// Tool name.
    public var toolName: String
    /// Tool string arguments dictionary.
    public var arguments: [String: String]

    public init(server: String, toolName: String, arguments: [String: String] = [:]) {
        self.server = server
        self.toolName = toolName
        self.arguments = arguments
    }
}

/// Output returned from dynamic MCP tool execution.
public struct McpGenericOutput: Codable, Sendable, Equatable {
    /// Output text payload.
    public var text: String
    /// Whether the tool execution resulted in an error.
    public var isError: Bool

    public init(text: String, isError: Bool = false) {
        self.text = text
        self.isError = isError
    }
}

// MARK: - OpenAI Tool Definitions for MCP

/// OpenAI tool definition schemas for MCP resource queries and tool refresh.
public enum McpToolDefinitions {
    public static let listMcpResources = OpenAITool.function(
        name: "ListMcpResources",
        description: "List available data resources provided by connected Model Context Protocol (MCP) servers.",
        parameters: .object(
            properties: [
                "server": .string(description: "Optional server name filter.")
            ],
            required: []
        )
    )

    public static let readMcpResource = OpenAITool.function(
        name: "ReadMcpResource",
        description: "Read the text or binary content of a specific MCP resource URI.",
        parameters: .object(
            properties: [
                "server": .string(description: "The MCP server name."),
                "uri": .string(description: "The resource URI to read.")
            ],
            required: ["server", "uri"]
        )
    )

    public static let readMcpResourceDir = OpenAITool.function(
        name: "ReadMcpResourceDir",
        description: "List the direct children and subdirectories of an MCP directory resource.",
        parameters: .object(
            properties: [
                "server": .string(description: "The MCP server name."),
                "uri": .string(description: "The directory URI to inspect.")
            ],
            required: ["server", "uri"]
        )
    )

    public static let refreshMcpTools = OpenAITool.function(
        name: "RefreshMcpTools",
        description: "Re-query connected MCP servers to update active tool definitions.",
        parameters: .object(
            properties: [
                "server": .string(description: "Optional server name to refresh specifically.")
            ],
            required: []
        )
    )

    public static let all: [OpenAITool] = [
        listMcpResources, readMcpResource, readMcpResourceDir, refreshMcpTools
    ]
}
