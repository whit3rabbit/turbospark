import Foundation

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

        guard let host = url.host, !host.isEmpty else {
            throw NSError(
                domain: "TurboSparkHttpRequest",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Malformed URL with missing host: '\(urlString)'"]
            )
        }

        if AppToolSandbox.isPrivateOrMetadataHost(host) {
            throw NSError(
                domain: "TurboSparkHttpRequest",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Access to private network or metadata host '\(host)' is denied."]
            )
        }

        try AppToolSandbox.validateDomain(host)

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
        let session = customSession ?? URLSession(configuration: sessionConfig)

        let startTime = Date()
        let (data, response) = try await session.data(for: req)
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
}
