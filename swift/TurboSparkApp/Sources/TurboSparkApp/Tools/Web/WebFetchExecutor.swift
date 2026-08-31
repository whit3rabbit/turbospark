import Foundation

/// Engine for fetching web content and transforming HTML into markdown or text.
///
/// Ported from and compatible with OpenCode's `webfetch` tool (`packages/core/src/tool/webfetch.ts`).
public enum WebFetchExecutor {
    public static let maxResponseBytes: Int = 5 * 1024 * 1024 // 5 MB
    public static let defaultTimeoutSeconds: Int = 30
    public static let maxTimeoutSeconds: Int = 120

    public static let browserUserAgent: String =
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"

    public static let fallbackUserAgent: String = "turbospark"

    public static func acceptHeader(for format: String) -> String {
        switch format.lowercased() {
        case "markdown":
            return "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1"
        case "text":
            return "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1"
        case "html":
            return "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, text/markdown;q=0.7, */*;q=0.1"
        default:
            return "*/*"
        }
    }

    public static func isImageAttachment(mime: String) -> Bool {
        mime.hasPrefix("image/") && mime != "image/svg+xml" && mime != "image/vnd.fastbidsheet"
    }

    public static func isTextualMime(mime: String) -> Bool {
        if mime.isEmpty { return true }
        if mime.hasPrefix("text/") { return true }
        if mime == "application/json" || mime.hasSuffix("+json") { return true }
        if mime == "application/xml" || mime.hasSuffix("+xml") || mime == "application/xhtml+xml" { return true }
        if mime == "application/javascript" || mime == "application/x-javascript" { return true }
        if mime == "application/csv" || mime == "application/x-yaml" || mime == "application/yaml" { return true }
        return false
    }

    public static func mimeFrom(contentType: String) -> String {
        contentType.components(separatedBy: ";").first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
    }

    /// Fetches content from an HTTP or HTTPS URL and returns it formatted as markdown, plain text, or HTML.
    public static func fetch(
        url urlString: String,
        format: String = "markdown",
        timeout: Int? = nil,
        customSession: URLSession? = nil
    ) async throws -> String {
        let trimmedUrl = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmedUrl), let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "URL must use http:// or https://: '\(urlString)'"]
            )
        }

        guard let host = url.host, !host.isEmpty else {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Malformed URL with missing host: '\(urlString)'"]
            )
        }

        if AppToolSandbox.isPrivateOrMetadataHost(host) {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Access to private network or metadata host '\(host)' is denied."]
            )
        }

        try AppToolSandbox.validateDomain(host)

        let targetFormat = ["text", "html"].contains(format.lowercased()) ? format.lowercased() : "markdown"
        let resolvedTimeout = min(Self.maxTimeoutSeconds, max(1, timeout ?? Self.defaultTimeoutSeconds))

        func createRequest(userAgent: String) -> URLRequest {
            var req = URLRequest(url: url)
            req.httpMethod = "GET"
            req.timeoutInterval = TimeInterval(resolvedTimeout)
            req.setValue(userAgent, forHTTPHeaderField: "User-Agent")
            req.setValue(acceptHeader(for: targetFormat), forHTTPHeaderField: "Accept")
            req.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")
            return req
        }

        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.timeoutIntervalForRequest = TimeInterval(resolvedTimeout)
        sessionConfig.timeoutIntervalForResource = TimeInterval(resolvedTimeout)
        let session = customSession ?? URLSession(configuration: sessionConfig)

        var (data, response) = try await session.data(for: createRequest(userAgent: browserUserAgent))
        var httpResponse = response as? HTTPURLResponse

        // Cloudflare challenge fallback retry
        if let resp = httpResponse, resp.statusCode == 403 {
            let cfHeader = resp.value(forHTTPHeaderField: "cf-mitigated")?.lowercased()
            if cfHeader == "challenge" || resp.statusCode == 403 {
                let retryReq = createRequest(userAgent: fallbackUserAgent)
                if let (retryData, retryResp) = try? await session.data(for: retryReq),
                   let retryHttp = retryResp as? HTTPURLResponse,
                   (200...299).contains(retryHttp.statusCode) {
                    data = retryData
                    response = retryResp
                    httpResponse = retryHttp
                }
            }
        }

        guard let finalHttp = httpResponse else {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Unable to fetch \(urlString): Non-HTTP response received."]
            )
        }

        guard (200...299).contains(finalHttp.statusCode) else {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "HTTP request failed with status code \(finalHttp.statusCode)."]
            )
        }

        if data.count > maxResponseBytes {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Response too large (exceeds \(maxResponseBytes) byte limit)."]
            )
        }

        let contentType = finalHttp.value(forHTTPHeaderField: "Content-Type") ?? finalHttp.value(forHTTPHeaderField: "content-type") ?? ""
        let mime = mimeFrom(contentType: contentType)

        if isImageAttachment(mime: mime) {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported fetched image content type: \(mime)"]
            )
        }

        if !isTextualMime(mime: mime) {
            throw NSError(
                domain: "TurboSparkWebFetch",
                code: 8,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported fetched file content type: \(mime)"]
            )
        }

        let rawContent: String
        if let utf8Str = String(data: data, encoding: .utf8) {
            rawContent = utf8Str
        } else if let latin1Str = String(data: data, encoding: .isoLatin1) {
            rawContent = latin1Str
        } else {
            rawContent = String(decoding: data, as: UTF8.self)
        }

        return convert(content: rawContent, contentType: contentType, format: targetFormat)
    }

    public static func convert(content: String, contentType: String, format: String) -> String {
        let isHtml = contentType.lowercased().contains("text/html") || contentType.lowercased().contains("application/xhtml+xml")
        if !isHtml {
            return content
        }
        switch format.lowercased() {
        case "markdown":
            return convertHTMLToMarkdown(content)
        case "text":
            return extractTextFromHTML(content)
        case "html":
            return content
        default:
            return convertHTMLToMarkdown(content)
        }
    }

    // MARK: - HTML Entities

    public static func decodeHTMLEntities(_ text: String) -> String {
        var result = text
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&apos;", with: "'")
            .replacingOccurrences(of: "&mdash;", with: "---")
            .replacingOccurrences(of: "&ndash;", with: "--")
            .replacingOccurrences(of: "&hellip;", with: "...")
            .replacingOccurrences(of: "&copy;", with: "(c)")
            .replacingOccurrences(of: "&reg;", with: "(r)")
            .replacingOccurrences(of: "&trade;", with: "(tm)")

        // Decode decimal entities (&#123;)
        if let regex = try? NSRegularExpression(pattern: "&#(\\d+);", options: []) {
            let matches = regex.matches(in: result, options: [], range: NSRange(location: 0, length: (result as NSString).length))
            for match in matches.reversed() {
                if let range = Range(match.range(at: 1), in: result),
                   let code = UInt32(result[range]),
                   let scalar = UnicodeScalar(code) {
                    let fullRange = Range(match.range(at: 0), in: result)!
                    result.replaceSubrange(fullRange, with: String(Character(scalar)))
                }
            }
        }

        // Decode hex entities (&#x1F600;)
        if let regex = try? NSRegularExpression(pattern: "&#[xX]([0-9a-fA-F]+);", options: []) {
            let matches = regex.matches(in: result, options: [], range: NSRange(location: 0, length: (result as NSString).length))
            for match in matches.reversed() {
                if let range = Range(match.range(at: 1), in: result),
                   let code = UInt32(result[range], radix: 16),
                   let scalar = UnicodeScalar(code) {
                    let fullRange = Range(match.range(at: 0), in: result)!
                    result.replaceSubrange(fullRange, with: String(Character(scalar)))
                }
            }
        }

        return result
    }

    // MARK: - Plain Text Extraction

    public static func extractTextFromHTML(_ html: String) -> String {
        var text = html

        // Remove script, style, noscript, iframe, object, embed, svg tags and their content
        let removalTags = ["script", "style", "noscript", "iframe", "object", "embed", "svg", "head"]
        for tag in removalTags {
            if let regex = try? NSRegularExpression(pattern: "<\(tag)\\b[^>]*>[\\s\\S]*?<\\/\(tag)>", options: [.caseInsensitive]) {
                text = regex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: " ")
            }
        }

        // Replace break/paragraph tags with newlines
        text = text.replacingOccurrences(of: "(?i)<(br|hr|p|div|li|tr|h[1-6])[^>]*>", with: "\n", options: .regularExpression)

        // Strip remaining HTML tags
        if let tagRegex = try? NSRegularExpression(pattern: "<[^>]+>", options: []) {
            text = tagRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: " ")
        }

        text = decodeHTMLEntities(text)

        // Collapse excess whitespace while preserving line structure
        let lines = text.components(separatedBy: "\n").map { line in
            line.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }

        return lines.joined(separator: "\n")
    }

    // MARK: - HTML to Markdown Conversion

    public static func convertHTMLToMarkdown(_ html: String) -> String {
        var text = html

        // 1. Remove non-content tags & their inner contents
        let removalTags = ["script", "style", "noscript", "meta", "link", "svg", "head", "iframe", "object", "embed"]
        for tag in removalTags {
            if let regex = try? NSRegularExpression(pattern: "<\(tag)\\b[^>]*>[\\s\\S]*?<\\/\(tag)>", options: [.caseInsensitive]) {
                text = regex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "")
            }
            // Also remove self-closing / single instances of meta/link
            if let singleRegex = try? NSRegularExpression(pattern: "<\(tag)\\b[^>]*\\/?>", options: [.caseInsensitive]) {
                text = singleRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "")
            }
        }

        // 2. Preformatted code blocks: <pre><code( class="language-xyz")?>...</code></pre> or <pre>...</pre>
        if let preCodeRegex = try? NSRegularExpression(pattern: "<pre\\b[^>]*>\\s*<code(?:\\s+class=[\"'](?:language-)?([a-zA-Z0-9_-]+)[\"'])?[^>]*>([\\s\\S]*?)<\\/code>\\s*<\\/pre>", options: [.caseInsensitive]) {
            text = preCodeRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "\n```$1\n$2\n```\n")
        }
        if let preRegex = try? NSRegularExpression(pattern: "<pre\\b[^>]*>([\\s\\S]*?)<\\/pre>", options: [.caseInsensitive]) {
            text = preRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "\n```\n$1\n```\n")
        }

        // 3. Inline code: <code>...</code>
        if let codeRegex = try? NSRegularExpression(pattern: "<code\\b[^>]*>([\\s\\S]*?)<\\/code>", options: [.caseInsensitive]) {
            text = codeRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "`$1`")
        }

        // 4. Headings: <h1> to <h6>
        for level in 1...6 {
            let hashes = String(repeating: "#", count: level)
            if let hRegex = try? NSRegularExpression(pattern: "<h\(level)\\b[^>]*>([\\s\\S]*?)<\\/h\(level)>", options: [.caseInsensitive]) {
                text = hRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "\n\n\(hashes) $1\n\n")
            }
        }

        // 5. Bold: <strong>, <b>
        if let boldRegex = try? NSRegularExpression(pattern: "<(strong|b)\\b[^>]*>([\\s\\S]*?)<\\/\\1>", options: [.caseInsensitive]) {
            text = boldRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "**$2**")
        }

        // 6. Italic: <em>, <i>
        if let italicRegex = try? NSRegularExpression(pattern: "<(em|i)\\b[^>]*>([\\s\\S]*?)<\\/\\1>", options: [.caseInsensitive]) {
            text = italicRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "*$2*")
        }

        // 7. Links: <a href="url">text</a>
        if let linkRegex = try? NSRegularExpression(pattern: "<a\\b[^>]*\\bhref=[\"']([^\"']*)[\"'][^>]*>([\\s\\S]*?)<\\/a>", options: [.caseInsensitive]) {
            text = linkRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "[$2]($1)")
        }

        // 8. Images: <img ...>
        if let imgTagRegex = try? NSRegularExpression(pattern: "<img\\b([^>]*)>", options: [.caseInsensitive]) {
            let nsText = text as NSString
            let matches = imgTagRegex.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length))
            for match in matches.reversed() {
                let attrsString = nsText.substring(with: match.range(at: 1))
                var src = ""
                var alt = ""
                if let srcMatch = try? NSRegularExpression(pattern: "\\bsrc=[\"']([^\"']*)[\"']", options: [.caseInsensitive]).firstMatch(in: attrsString, options: [], range: NSRange(location: 0, length: (attrsString as NSString).length)),
                   let range = Range(srcMatch.range(at: 1), in: attrsString) {
                    src = String(attrsString[range])
                }
                if let altMatch = try? NSRegularExpression(pattern: "\\balt=[\"']([^\"']*)[\"']", options: [.caseInsensitive]).firstMatch(in: attrsString, options: [], range: NSRange(location: 0, length: (attrsString as NSString).length)),
                   let range = Range(altMatch.range(at: 1), in: attrsString) {
                    alt = String(attrsString[range])
                }
                let replacement = src.isEmpty ? "" : "![\(alt)](\(src))"
                if let fullRange = Range(match.range(at: 0), in: text) {
                    text.replaceSubrange(fullRange, with: replacement)
                }
            }
        }

        // 9. Blockquotes: <blockquote>...</blockquote>
        if let quoteRegex = try? NSRegularExpression(pattern: "<blockquote\\b[^>]*>([\\s\\S]*?)<\\/blockquote>", options: [.caseInsensitive]) {
            text = quoteRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "\n\n> $1\n\n")
        }

        // 10. Horizontal rules: <hr>
        text = text.replacingOccurrences(of: "(?i)<hr\\b[^>]*\\/?>", with: "\n\n---\n\n", options: .regularExpression)

        // 11. Lists: <li>...</li>
        if let liRegex = try? NSRegularExpression(pattern: "<li\\b[^>]*>([\\s\\S]*?)<\\/li>", options: [.caseInsensitive]) {
            text = liRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "\n- $1")
        }

        // 12. Paragraphs & line breaks
        text = text.replacingOccurrences(of: "(?i)<br\\b[^>]*\\/?>", with: "\n", options: .regularExpression)
        if let pRegex = try? NSRegularExpression(pattern: "<p\\b[^>]*>([\\s\\S]*?)<\\/p>", options: [.caseInsensitive]) {
            text = pRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "\n\n$1\n\n")
        }

        // 13. Divs & table tags
        text = text.replacingOccurrences(of: "(?i)<\\/(div|tr|table|ul|ol)>", with: "\n", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?i)<(td|th)\\b[^>]*>", with: " | ", options: .regularExpression)

        // 14. Strip any remaining HTML tags
        if let anyTagRegex = try? NSRegularExpression(pattern: "<[^>]+>", options: []) {
            text = anyTagRegex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: (text as NSString).length), withTemplate: "")
        }

        // 15. Decode HTML Entities
        text = decodeHTMLEntities(text)

        // 16. Clean up spacing and empty lines
        let rawLines = text.components(separatedBy: "\n")
        var cleanedLines: [String] = []
        var consecutiveEmpty = 0

        for rawLine in rawLines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                consecutiveEmpty += 1
                if consecutiveEmpty <= 1 {
                    cleanedLines.append("")
                }
            } else {
                consecutiveEmpty = 0
                cleanedLines.append(line)
            }
        }

        return cleanedLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
