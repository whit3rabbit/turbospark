import Foundation

/// Engine for executing web searches and formatting results for model consumption.
///
/// Combines the zero-config MCP-backed search backends from OpenCode (`websearch.ts`)
/// with the multi-provider sanitization, domain gating, and SSRF defenses of omlx (`websearch.py`).
public enum WebSearchExecutor {
    public static let exaUrl = "https://mcp.exa.ai/mcp"
    public static let parallelUrl = "https://search.parallel.ai/mcp"
    public static let braveUrl = "https://api.search.brave.com/res/v1/web/search"

    public static let defaultNumResults: Int = 5
    public static let maxNumResults: Int = 20
    public static let maxQueryChars: Int = 300
    public static let maxUrlChars: Int = 2048
    public static let maxTitleChars: Int = 200
    public static let maxSnippetChars: Int = 800
    public static let maxResponseBytes: Int = 8 * 1024 * 1024
    public static let defaultTimeoutSeconds: Int = 25

    public static let userAgent: String = "turbospark-web/1.0"

    static func validateResponseSize(_ byteCount: Int) throws {
        guard byteCount <= maxResponseBytes else {
            throw NSError(
                domain: "TurboSparkWebSearch",
                code: 50,
                userInfo: [NSLocalizedDescriptionKey: "Web search response exceeded the 8 MiB limit."]
            )
        }
    }

    private static func fetchLimited(
        request: URLRequest,
        session: URLSession
    ) async throws -> (Data, URLResponse) {
        let (bytes, response) = try await session.bytes(for: request)
        if response.expectedContentLength >= 0 {
            try validateResponseSize(Int(response.expectedContentLength))
        }

        var data = Data()
        data.reserveCapacity(min(max(0, Int(response.expectedContentLength)), maxResponseBytes))
        for try await byte in bytes {
            try validateResponseSize(data.count + 1)
            data.append(byte)
        }
        return (data, response)
    }

    // MARK: - Primary Search Entrypoint

    /// Executes a web search query with optional domain filtering and provider configuration.
    public static func search(
        query rawQuery: String,
        numResults: Int? = nil,
        allowedDomains: [String]? = nil,
        blockedDomains: [String]? = nil,
        provider: String? = nil,
        exaApiKey: String? = nil,
        parallelApiKey: String? = nil,
        braveApiKey: String? = nil,
        searxngUrl: String? = nil,
        tavilyApiKey: String? = nil,
        customSession: URLSession? = nil
    ) async throws -> WebSearchOutput {
        let startTime = Date()
        let trimmedQuery = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedQuery.isEmpty else {
            throw NSError(
                domain: "TurboSparkWebSearch",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Search query cannot be empty."]
            )
        }

        let boundedQuery = String(trimmedQuery.prefix(maxQueryChars))
        let requestedCount = min(maxNumResults, max(1, numResults ?? defaultNumResults))
        let selectedProvider = (provider ?? "auto").lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        var rawItems: [WebSearchResultItem] = []

        switch selectedProvider {
        case "brave":
            rawItems = try await searchBrave(
                query: boundedQuery,
                count: requestedCount,
                apiKey: braveApiKey,
                customSession: customSession
            )

        case "searxng":
            rawItems = try await searchSearXNG(
                query: boundedQuery,
                baseUrl: searxngUrl,
                customSession: customSession
            )

        case "parallel":
            rawItems = try await searchParallel(
                query: boundedQuery,
                apiKey: parallelApiKey,
                customSession: customSession
            )

        case "exa":
            rawItems = try await searchExa(
                query: boundedQuery,
                count: requestedCount,
                apiKey: exaApiKey,
                customSession: customSession
            )

        case "tavily":
            rawItems = try await searchTavily(
                query: boundedQuery,
                count: requestedCount,
                apiKey: tavilyApiKey,
                customSession: customSession
            )

        default:
            // "auto": Try Exa first; if it fails, fallback to Parallel.
            // If user supplied a Brave key and no other provider is set, Brave is also eligible.
            if let bKey = braveApiKey, !bKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                do {
                    rawItems = try await searchBrave(
                        query: boundedQuery,
                        count: requestedCount,
                        apiKey: bKey,
                        customSession: customSession
                    )
                } catch {
                    rawItems = try await fallbackSearch(
                        query: boundedQuery,
                        count: requestedCount,
                        exaApiKey: exaApiKey,
                        parallelApiKey: parallelApiKey,
                        customSession: customSession
                    )
                }
            } else {
                rawItems = try await fallbackSearch(
                    query: boundedQuery,
                    count: requestedCount,
                    exaApiKey: exaApiKey,
                    parallelApiKey: parallelApiKey,
                    customSession: customSession
                )
            }
        }

        // Apply domain filtering, sanitization, and SSRF guards
        var sanitizedItems: [WebSearchResultItem] = []
        for item in rawItems {
            guard let cleaned = sanitize(
                item: item,
                allowedDomains: allowedDomains,
                blockedDomains: blockedDomains
            ) else {
                continue
            }
            sanitizedItems.append(cleaned)
            if sanitizedItems.count >= requestedCount {
                break
            }
        }

        let duration = Date().timeIntervalSince(startTime)
        return WebSearchOutput(
            query: boundedQuery,
            results: sanitizedItems,
            durationSeconds: duration,
            searchCount: sanitizedItems.count
        )
    }

    private static func fallbackSearch(
        query: String,
        count: Int,
        exaApiKey: String?,
        parallelApiKey: String?,
        customSession: URLSession?
    ) async throws -> [WebSearchResultItem] {
        do {
            return try await searchExa(
                query: query,
                count: count,
                apiKey: exaApiKey,
                customSession: customSession
            )
        } catch {
            return try await searchParallel(
                query: query,
                apiKey: parallelApiKey,
                customSession: customSession
            )
        }
    }

    // MARK: - Exa MCP Provider

    public static func searchExa(
        query: String,
        count: Int,
        apiKey: String? = nil,
        customSession: URLSession? = nil
    ) async throws -> [WebSearchResultItem] {
        var endpoint = exaUrl
        if let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !key.isEmpty {
            if var components = URLComponents(string: exaUrl) {
                components.queryItems = [URLQueryItem(name: "exaApiKey", value: key)]
                if let u = components.url?.absoluteString {
                    endpoint = u
                }
            }
        }

        guard let url = URL(string: endpoint) else {
            throw NSError(domain: "TurboSparkWebSearch", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid Exa MCP URL."])
        }

        let rpcBody: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": [
                "name": "web_search_exa",
                "arguments": [
                    "query": query,
                    "type": "auto",
                    "numResults": count,
                    "livecrawl": "fallback"
                ] as [String: Any]
            ] as [String: Any]
        ]

        let postData = try JSONSerialization.data(withJSONObject: rpcBody)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = postData
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = TimeInterval(defaultTimeoutSeconds)

        let session = customSession ?? URLSession.shared
        let (data, response) = try await fetchLimited(request: request, session: session)

        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "TurboSparkWebSearch", code: 3, userInfo: [NSLocalizedDescriptionKey: "Non-HTTP response from Exa."])
        }
        guard (200...299).contains(http.statusCode) else {
            throw NSError(domain: "TurboSparkWebSearch", code: 4, userInfo: [NSLocalizedDescriptionKey: "Exa search failed with HTTP \(http.statusCode)."])
        }

        let rawResponse = String(decoding: data, as: UTF8.self)
        return parseExaPayload(rawResponse)
    }

    public static func parseExaPayload(_ raw: String) -> [WebSearchResultItem] {
        // Exa MCP returns either direct JSON {"result":{"content":[{"type":"text","text":"..."}]}}
        // or SSE lines: data: {"result":{"content":[{"type":"text","text":"..."}]}}
        var textPayload = ""
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmed.hasPrefix("{"),
           let data = trimmed.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let result = json["result"] as? [String: Any],
           let content = result["content"] as? [[String: Any]] {
            for item in content {
                if let t = item["text"] as? String {
                    textPayload += t + "\n"
                }
            }
        } else {
            for line in trimmed.components(separatedBy: "\n") {
                let stripped = line.trimmingCharacters(in: .whitespaces)
                if stripped.hasPrefix("data:") {
                    let jsonPart = stripped.dropFirst(5).trimmingCharacters(in: .whitespaces)
                    if let data = jsonPart.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let result = json["result"] as? [String: Any],
                       let content = result["content"] as? [[String: Any]] {
                        for item in content {
                            if let t = item["text"] as? String {
                                textPayload += t + "\n"
                            }
                        }
                    }
                }
            }
        }

        if textPayload.isEmpty {
            textPayload = trimmed
        }

        return parseExaFormattedText(textPayload)
    }

    /// Parses Exa's returned textual format (e.g. `Title: ... \nURL: ... \nHighlights: ...`).
    public static func parseExaFormattedText(_ text: String) -> [WebSearchResultItem] {
        var items: [WebSearchResultItem] = []
        let blocks = text.components(separatedBy: "\n\n")

        var currentTitle = ""
        var currentUrl = ""
        var currentSnippet = ""

        func flushCurrent() {
            if !currentUrl.isEmpty {
                let title = currentTitle.isEmpty ? currentUrl : currentTitle
                items.append(WebSearchResultItem(
                    title: title,
                    url: currentUrl,
                    snippet: currentSnippet.isEmpty ? nil : currentSnippet.trimmingCharacters(in: .whitespacesAndNewlines)
                ))
            }
            currentTitle = ""
            currentUrl = ""
            currentSnippet = ""
        }

        for line in text.components(separatedBy: "\n") {
            let l = line.trimmingCharacters(in: .whitespaces)
            if l.hasPrefix("Title: ") {
                if !currentUrl.isEmpty { flushCurrent() }
                currentTitle = String(l.dropFirst(7))
            } else if l.hasPrefix("URL: ") {
                currentUrl = String(l.dropFirst(5))
            } else if l.hasPrefix("Highlights:") || l.hasPrefix("Snippet:") {
                continue
            } else if !l.isEmpty && !currentUrl.isEmpty {
                if currentSnippet.isEmpty {
                    currentSnippet = l
                } else if currentSnippet.count < maxSnippetChars {
                    currentSnippet += " " + l
                }
            }
        }
        flushCurrent()

        // If line-by-line parsing did not find entries, try parsing raw blocks
        if items.isEmpty {
            for block in blocks {
                let lines = block.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                var bTitle = ""
                var bUrl = ""
                var bSnippet = ""
                for line in lines {
                    if line.hasPrefix("http://") || line.hasPrefix("https://") {
                        bUrl = line
                    } else if bTitle.isEmpty {
                        bTitle = line
                    } else {
                        bSnippet += (bSnippet.isEmpty ? "" : " ") + line
                    }
                }
                if !bUrl.isEmpty {
                    items.append(WebSearchResultItem(
                        title: bTitle.isEmpty ? bUrl : bTitle,
                        url: bUrl,
                        snippet: bSnippet.isEmpty ? nil : bSnippet
                    ))
                }
            }
        }

        return items
    }

    // MARK: - Parallel MCP Provider

    public static func searchParallel(
        query: String,
        apiKey: String? = nil,
        customSession: URLSession? = nil
    ) async throws -> [WebSearchResultItem] {
        guard let url = URL(string: parallelUrl) else {
            throw NSError(domain: "TurboSparkWebSearch", code: 5, userInfo: [NSLocalizedDescriptionKey: "Invalid Parallel MCP URL."])
        }

        let sessionId = "turbospark_" + UUID().uuidString.prefix(8).lowercased()
        let rpcBody: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": [
                "name": "web_search",
                "arguments": [
                    "objective": query,
                    "search_queries": [query],
                    "session_id": sessionId
                ] as [String: Any]
            ] as [String: Any]
        ]

        let postData = try JSONSerialization.data(withJSONObject: rpcBody)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = postData
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        if let key = apiKey, !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        request.timeoutInterval = TimeInterval(defaultTimeoutSeconds)

        let session = customSession ?? URLSession.shared
        let (data, response) = try await fetchLimited(request: request, session: session)

        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "TurboSparkWebSearch", code: 6, userInfo: [NSLocalizedDescriptionKey: "Parallel search failed with HTTP \(code)."])
        }

        let rawResponse = String(decoding: data, as: UTF8.self)
        return parseParallelPayload(rawResponse)
    }

    public static func parseParallelPayload(_ raw: String) -> [WebSearchResultItem] {
        // Parallel MCP returns JSON with result.content[0].text containing JSON:
        // {"search_id": "...", "results": [{"url": "...", "title": "...", "excerpts": [...]}]}
        var items: [WebSearchResultItem] = []
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)

        var nestedJsonString = ""
        if let data = trimmed.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let result = json["result"] as? [String: Any],
           let content = result["content"] as? [[String: Any]] {
            for c in content {
                if let t = c["text"] as? String {
                    nestedJsonString = t
                    break
                }
            }
        }

        let targetJson = nestedJsonString.isEmpty ? trimmed : nestedJsonString

        if let data = targetJson.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let results = json["results"] as? [[String: Any]] {
            for r in results {
                let url = (r["url"] as? String) ?? ""
                let title = (r["title"] as? String) ?? url
                var snippet = ""
                if let excerpts = r["excerpts"] as? [String] {
                    snippet = excerpts.joined(separator: " ")
                } else if let s = r["snippet"] as? String {
                    snippet = s
                }
                if !url.isEmpty {
                    items.append(WebSearchResultItem(
                        title: title,
                        url: url,
                        snippet: snippet.isEmpty ? nil : snippet
                    ))
                }
            }
        }

        return items
    }

    // MARK: - Brave Search API Provider

    public static func searchBrave(
        query: String,
        count: Int,
        apiKey: String?,
        customSession: URLSession? = nil
    ) async throws -> [WebSearchResultItem] {
        guard let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !key.isEmpty else {
            throw NSError(
                domain: "TurboSparkWebSearch",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "No Brave Search API key configured. Provide an API key to use Brave Search."]
            )
        }

        guard var components = URLComponents(string: braveUrl) else {
            throw NSError(domain: "TurboSparkWebSearch", code: 8, userInfo: [NSLocalizedDescriptionKey: "Invalid Brave Search endpoint."])
        }

        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "count", value: String(count))
        ]

        guard let url = components.url else {
            throw NSError(domain: "TurboSparkWebSearch", code: 8, userInfo: [NSLocalizedDescriptionKey: "Malformed Brave search URL."])
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue(key, forHTTPHeaderField: "X-Subscription-Token")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = TimeInterval(defaultTimeoutSeconds)

        let session = customSession ?? URLSession.shared
        let (data, response) = try await fetchLimited(request: request, session: session)

        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "TurboSparkWebSearch", code: 9, userInfo: [NSLocalizedDescriptionKey: "Brave search failed with HTTP \(code)."])
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let web = json["web"] as? [String: Any],
              let results = web["results"] as? [[String: Any]] else {
            return []
        }

        var items: [WebSearchResultItem] = []
        for r in results {
            guard let url = r["url"] as? String, !url.isEmpty else { continue }
            let title = (r["title"] as? String) ?? url
            let description = r["description"] as? String
            items.append(WebSearchResultItem(title: title, url: url, snippet: description))
        }
        return items
    }

    // MARK: - SearXNG Provider

    public static func searchSearXNG(
        query: String,
        baseUrl: String?,
        customSession: URLSession? = nil
    ) async throws -> [WebSearchResultItem] {
        guard let base = baseUrl?.trimmingCharacters(in: .whitespacesAndNewlines), !base.isEmpty else {
            throw NSError(
                domain: "TurboSparkWebSearch",
                code: 10,
                userInfo: [NSLocalizedDescriptionKey: "No SearXNG instance URL configured."]
            )
        }

        let cleanBase = base.hasSuffix("/") ? String(base.dropLast()) : base
        guard var components = URLComponents(string: cleanBase + "/search") else {
            throw NSError(domain: "TurboSparkWebSearch", code: 11, userInfo: [NSLocalizedDescriptionKey: "Invalid SearXNG endpoint URL."])
        }

        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "format", value: "json")
        ]

        guard let url = components.url else {
            throw NSError(domain: "TurboSparkWebSearch", code: 11, userInfo: [NSLocalizedDescriptionKey: "Malformed SearXNG search URL."])
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = TimeInterval(defaultTimeoutSeconds)

        let session = customSession ?? URLSession.shared
        let (data, response) = try await fetchLimited(request: request, session: session)

        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "TurboSparkWebSearch", code: 12, userInfo: [NSLocalizedDescriptionKey: "SearXNG search failed with HTTP \(code)."])
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let results = json["results"] as? [[String: Any]] else {
            return []
        }

        var items: [WebSearchResultItem] = []
        for r in results {
            guard let url = r["url"] as? String, !url.isEmpty else { continue }
            let title = (r["title"] as? String) ?? url
            let content = r["content"] as? String
            items.append(WebSearchResultItem(title: title, url: url, snippet: content))
        }
        return items
    }

    // MARK: - Sanitization & Security

    /// Normalizes and validates a single search hit against security rules and domain lists.
    public static func sanitize(
        item: WebSearchResultItem,
        allowedDomains: [String]? = nil,
        blockedDomains: [String]? = nil
    ) -> WebSearchResultItem? {
        let trimmedUrl = item.url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedUrl.isEmpty, trimmedUrl.count <= maxUrlChars else { return nil }

        guard let url = URL(string: trimmedUrl),
              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              let host = url.host?.lowercased(), !host.isEmpty else {
            return nil
        }

        // Drop URLs containing embedded credentials (user:password@host)
        if url.user != nil || url.password != nil {
            return nil
        }

        // SSRF guard: deny private, loopback, or metadata addresses
        if AppToolSandbox.isPrivateOrMetadataHost(host) {
            return nil
        }

        // Check blocked domains
        if let blocked = blockedDomains, !blocked.isEmpty {
            let normalizedBlocked = blocked.map { $0.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) }
            for b in normalizedBlocked where !b.isEmpty {
                if host == b || host.hasSuffix("." + b) {
                    return nil
                }
            }
        }

        // Check allowed domains whitelist
        if let allowed = allowedDomains, !allowed.isEmpty {
            let normalizedAllowed = allowed.map { $0.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) }
            var isAllowed = false
            for a in normalizedAllowed where !a.isEmpty {
                if host == a || host.hasSuffix("." + a) {
                    isAllowed = true
                    break
                }
            }
            if !isAllowed {
                return nil
            }
        }

        let cleanTitle = String(item.title.trimmingCharacters(in: .whitespacesAndNewlines).prefix(maxTitleChars))
        let cleanSnippet = item.snippet.map { String($0.trimmingCharacters(in: .whitespacesAndNewlines).prefix(maxSnippetChars)) }

        return WebSearchResultItem(
            title: cleanTitle.isEmpty ? host : cleanTitle,
            url: trimmedUrl,
            snippet: cleanSnippet?.isEmpty == true ? nil : cleanSnippet
        )
    }

    // MARK: - Tavily Search, Extract, Crawl, and Map

    public static func searchTavily(
        query: String,
        count: Int,
        apiKey: String? = nil,
        customSession: URLSession? = nil
    ) async throws -> [WebSearchResultItem] {
        let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? ProcessInfo.processInfo.environment["TAVILY_API_KEY"]
        guard let resolvedKey = key, !resolvedKey.isEmpty else {
            throw NSError(
                domain: "TurboSparkWebSearch",
                code: 30,
                userInfo: [NSLocalizedDescriptionKey: "Tavily API key is missing. Set TAVILY_API_KEY environment variable or supply it in provider settings."]
            )
        }

        guard let url = URL(string: "https://api.tavily.com/search") else {
            throw NSError(domain: "TurboSparkWebSearch", code: 31, userInfo: [NSLocalizedDescriptionKey: "Invalid Tavily search URL."])
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(userAgent, forHTTPHeaderField: "User-Agent")

        let payload: [String: Any] = [
            "api_key": resolvedKey,
            "query": query,
            "max_results": count,
            "search_depth": "basic",
            "include_answer": true
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: payload, options: [])

        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.timeoutIntervalForRequest = TimeInterval(defaultTimeoutSeconds)
        let session = customSession ?? URLSession(configuration: sessionConfig)

        let (data, response) = try await fetchLimited(request: req, session: session)
        guard let httpResp = response as? HTTPURLResponse, (200...299).contains(httpResp.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "TurboSparkWebSearch", code: 32, userInfo: [NSLocalizedDescriptionKey: "Tavily search failed with HTTP \(status)."])
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }

        var items: [WebSearchResultItem] = []
        if let answer = json["answer"] as? String, !answer.isEmpty {
            items.append(WebSearchResultItem(
                title: "Tavily Direct Answer",
                url: "https://tavily.com",
                snippet: answer
            ))
        }

        if let results = json["results"] as? [[String: Any]] {
            for res in results {
                let title = (res["title"] as? String) ?? "Untitled"
                let urlStr = (res["url"] as? String) ?? ""
                let content = (res["content"] as? String)
                if !urlStr.isEmpty {
                    items.append(WebSearchResultItem(title: title, url: urlStr, snippet: content))
                }
            }
        }
        return items
    }

    public static func extractTavily(
        urls: [String],
        apiKey: String? = nil,
        customSession: URLSession? = nil
    ) async throws -> String {
        let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? ProcessInfo.processInfo.environment["TAVILY_API_KEY"]
        guard let resolvedKey = key, !resolvedKey.isEmpty else {
            throw NSError(
                domain: "TurboSparkWebSearch",
                code: 33,
                userInfo: [NSLocalizedDescriptionKey: "Tavily API key is missing. Set TAVILY_API_KEY environment variable."]
            )
        }

        guard let endpoint = URL(string: "https://api.tavily.com/extract") else {
            throw NSError(domain: "TurboSparkWebSearch", code: 34, userInfo: [NSLocalizedDescriptionKey: "Invalid Tavily extract URL."])
        }

        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(userAgent, forHTTPHeaderField: "User-Agent")

        let payload: [String: Any] = [
            "api_key": resolvedKey,
            "urls": urls
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: payload, options: [])

        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.timeoutIntervalForRequest = TimeInterval(defaultTimeoutSeconds)
        let session = customSession ?? URLSession(configuration: sessionConfig)

        let (data, response) = try await fetchLimited(request: req, session: session)
        guard let httpResp = response as? HTTPURLResponse, (200...299).contains(httpResp.statusCode) else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "TurboSparkWebSearch", code: 35, userInfo: [NSLocalizedDescriptionKey: "Tavily extract failed with HTTP \(status)."])
        }

        return String(decoding: data, as: UTF8.self)
    }
}
