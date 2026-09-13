import Foundation

// MARK: - HttpRequest Tool

/// Input payload for executing arbitrary HTTP/REST requests.
public struct HttpRequestInput: Codable, Sendable, Equatable {
    /// Destination URL.
    public var url: String
    /// HTTP method: GET, POST, PUT, DELETE, PATCH, HEAD. Defaults to GET.
    public var method: String?
    /// Request headers as key-value pairs.
    public var headers: [String: String]?
    /// Request body string (JSON, text, or form data).
    public var body: String?
    /// Authentication type: 'bearer', 'basic', 'api_key', 'custom'.
    public var authType: String?
    /// Authentication token or credentials.
    public var authToken: String?
    /// Output format: 'raw', 'json', 'markdown'.
    public var format: String?
    /// Timeout in seconds (default 30, max 120).
    public var timeout: Int?

    enum CodingKeys: String, CodingKey {
        case url
        case method
        case headers
        case body
        case authType = "auth_type"
        case authToken = "auth_token"
        case format
        case timeout
    }

    public init(
        url: String,
        method: String? = "GET",
        headers: [String: String]? = nil,
        body: String? = nil,
        authType: String? = nil,
        authToken: String? = nil,
        format: String? = nil,
        timeout: Int? = nil
    ) {
        self.url = url
        self.method = method
        self.headers = headers
        self.body = body
        self.authType = authType
        self.authToken = authToken
        self.format = format
        self.timeout = timeout
    }
}

/// Output payload from executing an HTTP request.
public struct HttpRequestOutput: Codable, Sendable, Equatable {
    public var statusCode: Int
    public var statusText: String
    public var durationMs: Double
    public var bytes: Int
    public var headers: [String: String]
    public var body: String

    public init(
        statusCode: Int,
        statusText: String,
        durationMs: Double,
        bytes: Int,
        headers: [String: String],
        body: String
    ) {
        self.statusCode = statusCode
        self.statusText = statusText
        self.durationMs = durationMs
        self.bytes = bytes
        self.headers = headers
        self.body = body
    }

    public func formatResponse() -> String {
        var lines: [String] = [
            "HTTP \(statusCode) \(statusText) (\(String(format: "%.1f", durationMs))ms, \(bytes) bytes)"
        ]
        if !headers.isEmpty {
            lines.append("Headers:")
            for (key, val) in headers.sorted(by: { $0.key < $1.key }).prefix(15) {
                lines.append("  \(key): \(val)")
            }
            if headers.count > 15 {
                lines.append("  ... and \(headers.count - 15) more header(s)")
            }
            lines.append("")
        }
        lines.append("Body:")
        lines.append(body)
        return lines.joined(separator: "\n")
    }
}

// MARK: - OpenAI Tool Definitions for HttpRequest

public enum HttpRequestToolDefinitions {
    public static let httpRequest = OpenAITool.function(
        name: "HttpRequest",
        description: "Send an HTTP/REST request to an external or local API with customizable method, headers, auth (Bearer, Basic, API key), and body payload.",
        parameters: .object(
            properties: [
                "url": .string(description: "Destination URL (http:// or https://)."),
                "method": .string(description: "HTTP method: GET, POST, PUT, DELETE, PATCH, HEAD. Defaults to GET."),
                "headers": .object(properties: [:], description: "Optional dictionary of HTTP headers."),
                "body": .string(description: "Request body content for POST, PUT, or PATCH."),
                "auth_type": .string(description: "Authentication type: 'bearer', 'basic', 'api_key', 'custom'."),
                "auth_token": .string(description: "Authentication token, key, or credentials."),
                "format": .string(description: "Response format: 'raw', 'json' (pretty-printed), or 'markdown' (converts HTML to Markdown)."),
                "timeout": .integer(description: "Request timeout in seconds (default: 30, maximum: 120).")
            ],
            required: ["url"]
        )
    )

    public static let all: [OpenAITool] = [httpRequest]
}
