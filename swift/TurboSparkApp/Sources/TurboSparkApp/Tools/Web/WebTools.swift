import Foundation

// MARK: - WebSearch Tool

/// Input parameters for web search queries.
public struct WebSearchInput: Codable, Sendable, Equatable {
    /// Search query string.
    public var query: String
    /// Optional whitelist of domains to restrict results to.
    public var allowedDomains: [String]?
    /// Optional blacklist of domains to filter out from results.
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

/// Single result item returned from a web search query.
public struct WebSearchResultItem: Codable, Sendable, Equatable {
    /// Web page title.
    public var title: String
    /// Target URL link.
    public var url: String
    /// Contextual snippet or summary of the result.
    public var snippet: String?

    public init(title: String, url: String, snippet: String? = nil) {
        self.title = title
        self.url = url
        self.snippet = snippet
    }
}

/// Output payload from a web search invocation.
public struct WebSearchOutput: Codable, Sendable, Equatable {
    /// Executed search query.
    public var query: String
    /// Discovered search result items.
    public var results: [WebSearchResultItem]
    /// Duration of search request in seconds.
    public var durationSeconds: Double
    /// Total count of matching search hits if available.
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

/// Input payload for fetching and scraping a web page.
public struct WebFetchInput: Codable, Sendable, Equatable {
    /// URL to retrieve.
    public var url: String
    /// Extraction instruction guiding what content or answer to extract.
    public var prompt: String

    public init(url: String, prompt: String) {
        self.url = url
        self.prompt = prompt
    }
}

/// Output payload containing fetched web page content.
public struct WebFetchOutput: Codable, Sendable, Equatable {
    /// Number of bytes fetched.
    public var bytes: Int
    /// HTTP status code.
    public var code: Int
    /// HTTP status description.
    public var codeText: String
    /// Extracted page markdown or text content.
    public var result: String
    /// Request round-trip time in milliseconds.
    public var durationMs: Double
    /// Final fetched URL after redirects.
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

/// OpenAI tool definition schemas for web search and content fetching.
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
