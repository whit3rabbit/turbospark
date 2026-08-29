import Foundation

// MARK: - MCP Tool Definitions & Models

public struct ListMcpResourcesInput: Codable, Sendable, Equatable {
    public var server: String?

    public init(server: String? = nil) {
        self.server = server
    }
}

public struct McpResourceSummary: Codable, Sendable, Equatable {
    public var uri: String
    public var name: String
    public var mimeType: String?
    public var description: String?
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

public struct ListMcpResourcesOutput: Codable, Sendable, Equatable {
    public var resources: [McpResourceSummary]

    public init(resources: [McpResourceSummary] = []) {
        self.resources = resources
    }
}

public struct ReadMcpResourceInput: Codable, Sendable, Equatable {
    public var server: String
    public var uri: String

    public init(server: String, uri: String) {
        self.server = server
        self.uri = uri
    }
}

public struct McpResourceContent: Codable, Sendable, Equatable {
    public var uri: String
    public var mimeType: String?
    public var text: String?
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

public struct ReadMcpResourceOutput: Codable, Sendable, Equatable {
    public var contents: [McpResourceContent]
    public var error: String?

    public init(contents: [McpResourceContent] = [], error: String? = nil) {
        self.contents = contents
        self.error = error
    }
}

public struct ReadMcpResourceDirInput: Codable, Sendable, Equatable {
    public var server: String
    public var uri: String

    public init(server: String, uri: String) {
        self.server = server
        self.uri = uri
    }
}

public struct ReadMcpResourceDirOutput: Codable, Sendable, Equatable {
    public var resources: [McpResourceSummary]
    public var error: String?

    public init(resources: [McpResourceSummary] = [], error: String? = nil) {
        self.resources = resources
        self.error = error
    }
}

public struct RefreshMcpToolsInput: Codable, Sendable, Equatable {
    public var server: String?

    public init(server: String? = nil) {
        self.server = server
    }
}

public struct McpServerRefreshStatus: Codable, Sendable, Equatable {
    public var server: String
    public var status: String
    public var toolCount: Int?
    public var added: [String]?
    public var removed: [String]?
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

public struct RefreshMcpToolsOutput: Codable, Sendable, Equatable {
    public var servers: [McpServerRefreshStatus]

    public init(servers: [McpServerRefreshStatus] = []) {
        self.servers = servers
    }
}

public struct McpGenericInput: Codable, Sendable, Equatable {
    public var server: String
    public var toolName: String
    public var arguments: [String: String]

    public init(server: String, toolName: String, arguments: [String: String] = [:]) {
        self.server = server
        self.toolName = toolName
        self.arguments = arguments
    }
}

public struct McpGenericOutput: Codable, Sendable, Equatable {
    public var text: String
    public var isError: Bool

    public init(text: String, isError: Bool = false) {
        self.text = text
        self.isError = isError
    }
}

// MARK: - OpenAI Tool Definitions for MCP

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
