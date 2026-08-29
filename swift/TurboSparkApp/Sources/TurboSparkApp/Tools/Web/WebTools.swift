import Foundation

// MARK: - WebSearch Tool

public struct WebSearchInput: Codable, Sendable, Equatable {
    public var query: String
    public var allowedDomains: [String]?
    public var blockedDomains: [String]?

    enum CodingKeys: String, CodingKey {
        case query
        case allowedDomains = "allowed_domains"
        case blockedDomains = "blocked_domains"
    }

    public init(query: String, allowedDomains: [String]? = nil, blockedDomains: [String]? = nil) {
        self.query = query
        self.allowedDomains = allowedDomains
        self.blockedDomains = blockedDomains
    }
}

public struct WebSearchResultItem: Codable, Sendable, Equatable {
    public var title: String
    public var url: String
    public var snippet: String?

    public init(title: String, url: String, snippet: String? = nil) {
        self.title = title
        self.url = url
        self.snippet = snippet
    }
}

public struct WebSearchOutput: Codable, Sendable, Equatable {
    public var query: String
    public var results: [WebSearchResultItem]
    public var durationSeconds: Double
    public var searchCount: Int?

    public init(
        query: String,
        results: [WebSearchResultItem] = [],
        durationSeconds: Double = 0.0,
        searchCount: Int? = nil
    ) {
        self.query = query
        self.results = results
        self.durationSeconds = durationSeconds
        self.searchCount = searchCount
    }
}

// MARK: - WebFetch Tool

public struct WebFetchInput: Codable, Sendable, Equatable {
    public var url: String
    public var prompt: String

    public init(url: String, prompt: String) {
        self.url = url
        self.prompt = prompt
    }
}

public struct WebFetchOutput: Codable, Sendable, Equatable {
    public var bytes: Int
    public var code: Int
    public var codeText: String
    public var result: String
    public var durationMs: Double
    public var url: String

    public init(
        bytes: Int,
        code: Int,
        codeText: String,
        result: String,
        durationMs: Double,
        url: String
    ) {
        self.bytes = bytes
        self.code = code
        self.codeText = codeText
        self.result = result
        self.durationMs = durationMs
        self.url = url
    }
}

// MARK: - OpenAI Tool Definitions for Web Operations

public enum WebToolDefinitions {
    public static let webSearch = OpenAITool.function(
        name: "WebSearch",
        description: "Perform a web search query for documentation, articles, and APIs.",
        parameters: .object(
            properties: [
                "query": .string(description: "The search query string."),
                "allowed_domains": .array(
                    items: .string(),
                    description: "Optional list of domains to restrict search results to."
                ),
                "blocked_domains": .array(
                    items: .string(),
                    description: "Optional list of domains to exclude from search results."
                )
            ],
            required: ["query"]
        )
    )

    public static let webFetch = OpenAITool.function(
        name: "WebFetch",
        description: "Fetch web page content from a URL and extract relevant information.",
        parameters: .object(
            properties: [
                "url": .string(description: "The URL of the webpage to fetch."),
                "prompt": .string(description: "Instruction for what information to extract from the page.")
            ],
            required: ["url", "prompt"]
        )
    )

    public static let all: [OpenAITool] = [
        webSearch, webFetch
    ]
}
