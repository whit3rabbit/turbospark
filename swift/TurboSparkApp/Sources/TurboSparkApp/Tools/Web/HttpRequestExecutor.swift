import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Engine for executing arbitrary HTTP REST requests with method, header, auth, and payload support.
public enum HttpRequestExecutor {
    public static let defaultTimeoutSeconds: Int = 30
    public static let maxTimeoutSeconds: Int = 120
    public static let maxResponseBytes: Int = 5 * 1024 * 1024 // 5 MB

    /// Executes an HTTP request and returns formatted response text.
    public static func execute(
        url urlString: String,
        method: String = "GET",
        headers: [String: String]? = nil,
        body: String? = nil,
        authType: String? = nil,
        authToken: String? = nil,
        format: String? = nil,
        timeout: Int? = nil,
        customSession: URLSession? = nil
    ) async throws -> String {
        let trimmedUrl = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmedUrl), let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw NSError(
                domain: "TurboSparkHttpRequest",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "URL must use http:// or https://: '\(urlString)'"]
            )
        }

        try HttpRequestDestinationValidator.validate(url)

        let resolvedMethod = method.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let validMethods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        guard validMethods.contains(resolvedMethod) else {
            throw NSError(
                domain: "TurboSparkHttpRequest",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported HTTP method: '\(method)'. Supported methods: \(validMethods.joined(separator: ", "))"]
            )
        }

        let resolvedTimeout = min(maxTimeoutSeconds, max(1, timeout ?? defaultTimeoutSeconds))
        var req = URLRequest(url: url)
        req.httpMethod = resolvedMethod
        req.timeoutInterval = TimeInterval(resolvedTimeout)
        req.setValue(WebFetchExecutor.browserUserAgent, forHTTPHeaderField: "User-Agent")

        // Set custom headers
        if let headers = headers {
            for (k, v) in headers {
                req.setValue(v, forHTTPHeaderField: k)
            }
        }

        // Set authentication
        if let token = authToken, !token.isEmpty {
            let auth = (authType ?? "bearer").lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            switch auth {
            case "bearer", "token":
                req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            case "basic":
                let encoded = Data(token.utf8).base64EncodedString()
                req.setValue("Basic \(encoded)", forHTTPHeaderField: "Authorization")
            case "api_key", "apikey":
                req.setValue(token, forHTTPHeaderField: "X-API-Key")
            case "custom":
                req.setValue(token, forHTTPHeaderField: "Authorization")
            default:
                req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
        }

        // Set body for non-GET/HEAD methods
        if let body = body, !body.isEmpty, resolvedMethod != "GET" && resolvedMethod != "HEAD" {
            let bodyData = Data(body.utf8)
            req.httpBody = bodyData
            if req.value(forHTTPHeaderField: "Content-Type") == nil && req.value(forHTTPHeaderField: "content-type") == nil {
                let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
                if (trimmed.hasPrefix("{") && trimmed.hasSuffix("}")) || (trimmed.hasPrefix("[") && trimmed.hasSuffix("]")) {
                    req.setValue("application/json; charset=utf-8", forHTTPHeaderField: "Content-Type")
                } else {
                    req.setValue("text/plain; charset=utf-8", forHTTPHeaderField: "Content-Type")
                }
            }
        }

        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.timeoutIntervalForRequest = TimeInterval(resolvedTimeout)
        sessionConfig.timeoutIntervalForResource = TimeInterval(resolvedTimeout)
        let startTime = Date()
        let (data, response) = try await performRequest(
            req,
            configuration: sessionConfig,
            customSession: customSession
        )
        let elapsedMs = Date().timeIntervalSince(startTime) * 1000.0

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(
                domain: "TurboSparkHttpRequest",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Non-HTTP response received."]
            )
        }

        var responseHeaders: [String: String] = [:]
        for (k, v) in httpResponse.allHeaderFields {
            if let keyStr = k as? String, let valStr = v as? String {
                responseHeaders[keyStr] = valStr
            }
        }

        let contentType = httpResponse.value(forHTTPHeaderField: "Content-Type") ?? httpResponse.value(forHTTPHeaderField: "content-type") ?? ""
        let rawBody = String(decoding: data, as: UTF8.self)

        let formattedBody: String
        let selectedFormat = (format ?? "auto").lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if selectedFormat == "markdown" || (selectedFormat == "auto" && contentType.lowercased().contains("text/html")) {
            formattedBody = WebFetchExecutor.convert(content: rawBody, contentType: contentType, format: "markdown")
        } else if selectedFormat == "json" || (selectedFormat == "auto" && contentType.lowercased().contains("application/json")) {
            if let obj = try? JSONSerialization.jsonObject(with: data, options: []),
               let prettyData = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]),
               let prettyStr = String(data: prettyData, encoding: .utf8) {
                formattedBody = prettyStr
            } else {
                formattedBody = rawBody
            }
        } else {
            formattedBody = rawBody
        }

        let statusText = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
        let output = HttpRequestOutput(
            statusCode: httpResponse.statusCode,
            statusText: statusText,
            durationMs: elapsedMs,
            bytes: data.count,
            headers: responseHeaders,
            body: AppToolRegistry.compactOutput(formattedBody, maxLines: 150)
        )
        return output.formatResponse()
    }

    private static func performRequest(
        _ request: URLRequest,
        configuration: URLSessionConfiguration,
        customSession: URLSession?
    ) async throws -> (Data, URLResponse) {
        // Injected sessions are test transports. Production redirects are
        // stopped and replayed only after validating the new destination.
        if let customSession {
            let result = try await customSession.data(for: request)
            if let finalURL = result.1.url {
                try HttpRequestDestinationValidator.validate(finalURL)
            }
            return result
        }

        let session = URLSession(configuration: configuration)
        var nextRequest: URLRequest? = request
        for _ in 0..<10 {
            guard let currentRequest = nextRequest, let currentURL = currentRequest.url else {
                throw requestError(6, "Redirect produced a malformed URL.")
            }
            try HttpRequestDestinationValidator.validate(currentURL)

            let redirect = HttpRequestRedirectDelegate()
            let result = try await session.data(for: currentRequest, delegate: redirect)
            if let redirectedRequest = redirect.takeRedirect() {
                nextRequest = redirectedRequest
                continue
            }
            if let finalURL = result.1.url {
                try HttpRequestDestinationValidator.validate(finalURL)
            }
            return result
        }
        throw requestError(7, "Too many HTTP redirects (maximum 10).")
    }

    private static func requestError(_ code: Int, _ message: String) -> NSError {
        NSError(
            domain: "TurboSparkHttpRequest",
            code: code,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }
}

private final class HttpRequestRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var redirectedRequest: URLRequest?

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        lock.lock()
        redirectedRequest = request
        lock.unlock()
        completionHandler(nil)
    }

    func takeRedirect() -> URLRequest? {
        lock.lock()
        defer { lock.unlock() }
        return redirectedRequest
    }
}
