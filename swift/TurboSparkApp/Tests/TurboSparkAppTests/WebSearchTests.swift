import XCTest
@testable import TurboSparkApp

final class WebSearchTests: XCTestCase {

    // MARK: - AppToolRegistry Implementation Registration

    func testAppToolRegistrySupportedToolNamesIncludesWebSearch() {
        XCTAssertTrue(AppToolRegistry.isImplemented("WebSearch"))
        XCTAssertTrue(AppToolRegistry.isImplemented("websearch"))
        XCTAssertTrue(AppToolRegistry.isImplemented("web_search"))
        XCTAssertTrue(AppToolRegistry.isImplemented("search_web"))
    }

    func testWebSearchDoesNotRequireProjectWorkspace() {
        XCTAssertFalse(
            AppToolRegistry.workspaceRootedToolNames.contains("websearch"),
            "WebSearch should be usable in rootless/projectless chats."
        )
        XCTAssertFalse(
            AppToolRegistry.workspaceRootedToolNames.contains("web_search")
        )
    }

    // MARK: - Argument Validation

    func testAppToolRegistryWebSearchExecutionMissingQuery() async {
        let call = AppToolCall(name: "WebSearch", arguments: [:], category: .web)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Missing 'query'"))
    }

    // MARK: - Result Sanitization & Security

    func testSanitizationRejectsPrivateAndMetadataHosts() {
        let privateHits = [
            WebSearchResultItem(title: "Local", url: "http://localhost:8080/search"),
            WebSearchResultItem(title: "Loopback", url: "http://127.0.0.1:3000/info"),
            WebSearchResultItem(title: "Metadata", url: "http://169.254.169.254/latest/meta-data"),
            WebSearchResultItem(title: "Internal", url: "http://app.local/admin"),
            WebSearchResultItem(title: "Decimal IP", url: "http://2130706433/"),
            WebSearchResultItem(title: "IPv6 Loopback", url: "http://[::1]:8080/")
        ]

        for hit in privateHits {
            let sanitized = WebSearchExecutor.sanitize(item: hit)
            XCTAssertNil(sanitized, "Private/metadata URL '\(hit.url)' should be rejected by SSRF checks.")
        }
    }

    func testSanitizationRejectsUrlsWithEmbeddedCredentials() {
        let credHit = WebSearchResultItem(title: "Creds", url: "https://user:password@example.com/docs")
        let sanitized = WebSearchExecutor.sanitize(item: credHit)
        XCTAssertNil(sanitized, "URLs with embedded credentials must be dropped.")
    }

    func testSanitizationRejectsNonHttpSchemes() {
        let invalidHits = [
            WebSearchResultItem(title: "File", url: "file:///etc/passwd"),
            WebSearchResultItem(title: "JS", url: "javascript:alert(1)"),
            WebSearchResultItem(title: "Data", url: "data:text/html,test")
        ]

        for hit in invalidHits {
            let sanitized = WebSearchExecutor.sanitize(item: hit)
            XCTAssertNil(sanitized, "Non-HTTP URL '\(hit.url)' must be dropped.")
        }
    }

    func testSanitizationDomainFiltering() {
        let hitA = WebSearchResultItem(title: "Apple Dev", url: "https://developer.apple.com/swift/")
        let hitB = WebSearchResultItem(title: "Swift Org", url: "https://swift.org/download/")
        let hitC = WebSearchResultItem(title: "Spam Site", url: "https://spam.example.org/swift")

        // Allowed domains test
        let allowedFilteredA = WebSearchExecutor.sanitize(item: hitA, allowedDomains: ["apple.com", "swift.org"])
        XCTAssertNotNil(allowedFilteredA)
        let allowedFilteredB = WebSearchExecutor.sanitize(item: hitB, allowedDomains: ["apple.com", "swift.org"])
        XCTAssertNotNil(allowedFilteredB)

        let allowedDenied = WebSearchExecutor.sanitize(item: hitC, allowedDomains: ["apple.com", "swift.org"])
        XCTAssertNil(allowedDenied, "Non-allowed domain should be rejected.")

        // Blocked domains test
        let blockedFiltered = WebSearchExecutor.sanitize(item: hitC, blockedDomains: ["example.org"])
        XCTAssertNil(blockedFiltered, "Blocked domain should be filtered out.")

        let blockedAllowed = WebSearchExecutor.sanitize(item: hitA, blockedDomains: ["example.org"])
        XCTAssertNotNil(blockedAllowed)
    }

    // MARK: - Parser Tests

    func testParseExaFormattedText() {
        let rawExa = """
        Title: Swift.org - Welcome to Swift
        URL: https://swift.org/
        Highlights:
        Swift is a robust and intuitive programming language created by Apple.
        It is designed to give developers more freedom than ever.

        Title: Swift - Apple Developer
        URL: https://developer.apple.com/swift/
        Highlights:
        Build apps for iOS, iPadOS, macOS, watchOS, and tvOS with Swift.
        """

        let results = WebSearchExecutor.parseExaFormattedText(rawExa)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].title, "Swift.org - Welcome to Swift")
        XCTAssertEqual(results[0].url, "https://swift.org/")
        XCTAssertTrue(results[0].snippet?.contains("programming language created by Apple") == true)
        XCTAssertEqual(results[1].title, "Swift - Apple Developer")
        XCTAssertEqual(results[1].url, "https://developer.apple.com/swift/")
    }

    func testParseExaSSEPayload() {
        let ssePayload = """
        event: message
        data: {"result":{"content":[{"type":"text","text":"Title: Swift Concurrency\\nURL: https://swift.org/concurrency\\nHighlights:\\nStructured concurrency in Swift."}]}}
        """

        let results = WebSearchExecutor.parseExaPayload(ssePayload)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].title, "Swift Concurrency")
        XCTAssertEqual(results[0].url, "https://swift.org/concurrency")
        XCTAssertTrue(results[0].snippet?.contains("Structured concurrency") == true)
    }

    func testParseParallelPayload() {
        let parallelJson = """
        {
          "jsonrpc": "2.0",
          "id": 1,
          "result": {
            "content": [
              {
                "type": "text",
                "text": "{\\"search_id\\": \\"s_123\\", \\"results\\": [{\\"url\\": \\"https://swift.org\\", \\"title\\": \\"Swift Org\\", \\"excerpts\\": [\\"Modern type-safe language.\\"]}]}"
              }
            ]
          }
        }
        """

        let results = WebSearchExecutor.parseParallelPayload(parallelJson)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].title, "Swift Org")
        XCTAssertEqual(results[0].url, "https://swift.org")
        XCTAssertEqual(results[0].snippet, "Modern type-safe language.")
    }

    // MARK: - Markdown Formatting

    func testWebSearchOutputMarkdownFormatting() {
        let output = WebSearchOutput(
            query: "swift async await",
            results: [
                WebSearchResultItem(
                    title: "Swift Concurrency Guide",
                    url: "https://docs.swift.org/concurrency",
                    snippet: "Comprehensive guide to async/await in Swift."
                ),
                WebSearchResultItem(
                    title: "Apple Developer Documentation",
                    url: "https://developer.apple.com/documentation/swift",
                    snippet: "Official API references."
                )
            ],
            durationSeconds: 0.42,
            searchCount: 2
        )

        let md = output.formatMarkdown()
        XCTAssertTrue(md.contains("### Web Search Results for: \"swift async await\""))
        XCTAssertTrue(md.contains("1. [Swift Concurrency Guide](https://docs.swift.org/concurrency)"))
        XCTAssertTrue(md.contains("Comprehensive guide to async/await in Swift."))
        XCTAssertTrue(md.contains("2. [Apple Developer Documentation](https://developer.apple.com/documentation/swift)"))
    }

    func testWebSearchOutputEmptyMarkdown() {
        let output = WebSearchOutput(query: "xyznonexistentquery123", results: [])
        let md = output.formatMarkdown()
        XCTAssertTrue(md.contains("No web search results found"))
    }

    func testTavilyMissingKeyFails() async {
        do {
            _ = try await WebSearchExecutor.searchTavily(query: "rust metal", count: 3, apiKey: "")
            XCTFail("Should fail when Tavily API key is empty")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Tavily API key is missing"))
        }
    }
}
