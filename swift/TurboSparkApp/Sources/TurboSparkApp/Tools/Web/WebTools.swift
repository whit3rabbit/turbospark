import Foundation

// MARK: - WebSearch Tool

/// Input parameters for web search queries.
public struct WebSearchInput: Codable, Sendable, Equatable {
    /// Search query string.
    public var query: String
    /// Optional limit on the number of results to return.
    public var numResults: Int?
    /// Optional whitelist of domains to restrict results to.
    public var allowedDomains: [String]?
    /// Optional blacklist of domains to filter out from results.
    public var blockedDomains: [String]?

    enum CodingKeys: String, CodingKey {
        case query
        case numResults = "num_results"
        case allowedDomains = "allowed_domains"
        case blockedDomains = "blocked_domains"
    }

    public init(
        query: String,
        numResults: Int? = nil,
        allowedDomains: [String]? = nil,
        blockedDomains: [String]? = nil
    ) {
        self.query = query
        self.numResults = numResults
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

    /// Formats the search results as clean Markdown for LLM synthesis.
    public func formatMarkdown() -> String {
        if results.isEmpty {
            return "No web search results found for query: \"\(query)\"."
        }
        var lines: [String] = ["### Web Search Results for: \"\(query)\""]
        for (index, item) in results.enumerated() {
            lines.append("")
            lines.append("\(index + 1). [\(item.title)](\(item.url))")
            if let snippet = item.snippet, !snippet.isEmpty {
                lines.append("   \(snippet)")
            }
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - WebFetch Tool

/// Input payload for fetching and scraping a web page.
public struct WebFetchInput: Codable, Sendable, Equatable {
    /// URL to retrieve.
    public var url: String
    /// The format to return the content in: "markdown" (default), "text", or "html".
    public var format: String?
    /// Optional timeout in seconds (maximum: 120, default: 30).
    public var timeout: Int?
    /// Optional extraction instruction guiding what content or answer to extract.
    public var prompt: String?

    enum CodingKeys: String, CodingKey {
        case url
        case format
        case timeout
        case prompt
    }

    public init(url: String, format: String? = "markdown", timeout: Int? = nil, prompt: String? = nil) {
        self.url = url
        self.format = format
        self.timeout = timeout
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
                "num_results": .integer(description: "Optional number of search results to return (default: 5, maximum: 20)."),
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
        description: "Fetch content from an HTTP or HTTPS URL and return it as text, markdown, or HTML. Markdown is the default.",
        parameters: .object(
            properties: [
                "url": .string(description: "The HTTP or HTTPS URL to fetch content from."),
                "format": .string(description: "The format to return the content in: 'markdown', 'text', or 'html'. Defaults to markdown."),
                "timeout": .integer(description: "Optional timeout in seconds (maximum: 120, default: 30).")
            ],
            required: ["url"]
        )
    )

    public static let all: [OpenAITool] = [
        webSearch, webFetch
    ]
}
