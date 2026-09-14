import XCTest
@testable import TurboSparkApp

final class WebFetchTests: XCTestCase {
    override func setUp() {
        super.setUp()
        AppToolRegistry.webToolsEnabledProvider = nil
    }

    // MARK: - HTML to Markdown Conversion

    func testConvertHeadings() {
        let html = "<h1>Title 1</h1><p>Some text</p><h2>Subtitle 2</h2><h3>Heading 3</h3>"
        let md = WebFetchExecutor.convertHTMLToMarkdown(html)
        XCTAssertTrue(md.contains("# Title 1"))
        XCTAssertTrue(md.contains("## Subtitle 2"))
        XCTAssertTrue(md.contains("### Heading 3"))
        XCTAssertTrue(md.contains("Some text"))
    }

    func testConvertLinksAndImages() {
        let html = """
        <p>Visit <a href="https://example.com/docs">Documentation</a> for info.</p>
        <img src="https://example.com/logo.png" alt="Logo" />
        """
        let md = WebFetchExecutor.convertHTMLToMarkdown(html)
        XCTAssertTrue(md.contains("[Documentation](https://example.com/docs)"))
        XCTAssertTrue(md.contains("![Logo](https://example.com/logo.png)"))
    }

    func testConvertTextFormatting() {
        let html = "<p><strong>Bold</strong>, <b>Also Bold</b>, <em>Italic</em>, <i>Also Italic</i>, and <code>inline code</code>.</p>"
        let md = WebFetchExecutor.convertHTMLToMarkdown(html)
        XCTAssertTrue(md.contains("**Bold**"))
        XCTAssertTrue(md.contains("**Also Bold**"))
        XCTAssertTrue(md.contains("*Italic*"))
        XCTAssertTrue(md.contains("*Also Italic*"))
        XCTAssertTrue(md.contains("`inline code`"))
    }

    func testConvertCodeBlocks() {
        let html = """
        <pre><code class="language-swift">
        func greet() {
            print("Hello, world!")
        }
        </code></pre>
        """
        let md = WebFetchExecutor.convertHTMLToMarkdown(html)
        XCTAssertTrue(md.contains("```swift"))
        XCTAssertTrue(md.contains("func greet()"))
        XCTAssertTrue(md.contains("```"))
    }

    func testConvertListsAndBlockquotes() {
        let html = """
        <blockquote>This is a quote</blockquote>
        <ul>
            <li>First item</li>
            <li>Second item</li>
        </ul>
        <hr>
        """
        let md = WebFetchExecutor.convertHTMLToMarkdown(html)
        XCTAssertTrue(md.contains("> This is a quote"))
        XCTAssertTrue(md.contains("- First item"))
        XCTAssertTrue(md.contains("- Second item"))
        XCTAssertTrue(md.contains("---"))
    }

    func testStripScriptStyleAndMetaTags() {
        let html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <style>body { color: red; }</style>
            <script>console.log('malicious script');</script>
            <meta name="description" content="Test">
        </head>
        <body>
            <noscript>Please enable javascript</noscript>
            <svg><path d="M0 0"/></svg>
            <h1>Visible Heading</h1>
            <p>Visible paragraph content.</p>
        </body>
        </html>
        """
        let md = WebFetchExecutor.convertHTMLToMarkdown(html)
        XCTAssertFalse(md.contains("color: red"))
        XCTAssertFalse(md.contains("console.log"))
        XCTAssertFalse(md.contains("Please enable javascript"))
        XCTAssertFalse(md.contains("<path"))
        XCTAssertTrue(md.contains("# Visible Heading"))
        XCTAssertTrue(md.contains("Visible paragraph content."))
    }

    func testDecodeHtmlEntities() {
        let html = "<p>&copy; 2026 &amp; Co. &lt;special&gt; &quot;quotes&#39; &mdash; text &nbsp; &#65; &#x42;</p>"
        let md = WebFetchExecutor.convertHTMLToMarkdown(html)
        XCTAssertTrue(md.contains("(c) 2026 & Co."))
        XCTAssertTrue(md.contains("<special>"))
        XCTAssertTrue(md.contains("\"quotes'"))
        XCTAssertTrue(md.contains("--- text"))
        XCTAssertTrue(md.contains("A B"))
    }

    func testExtractPlainTextFromHTML() {
        let html = """
        <div class="main">
            <h1>Welcome to TurboSpark</h1>
            <p>Fast and <strong>reliable</strong> local inference.</p>
            <script>alert(1);</script>
        </div>
        """
        let text = WebFetchExecutor.extractTextFromHTML(html)
        XCTAssertFalse(text.contains("<"))
        XCTAssertFalse(text.contains(">"))
        XCTAssertFalse(text.contains("alert"))
        XCTAssertTrue(text.contains("Welcome to TurboSpark"))
        XCTAssertTrue(text.contains("Fast and reliable local inference."))
    }

    // MARK: - MIME Classification

    func testMimeClassification() {
        XCTAssertTrue(WebFetchExecutor.isImageAttachment(mime: "image/png"))
        XCTAssertTrue(WebFetchExecutor.isImageAttachment(mime: "image/jpeg"))
        XCTAssertTrue(WebFetchExecutor.isImageAttachment(mime: "image/webp"))
        XCTAssertFalse(WebFetchExecutor.isImageAttachment(mime: "image/svg+xml"))
        XCTAssertFalse(WebFetchExecutor.isImageAttachment(mime: "text/html"))

        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "text/html"))
        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "text/plain"))
        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "text/markdown"))
        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "application/json"))
        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "application/problem+json"))
        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "application/xml"))
        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "application/javascript"))
        XCTAssertTrue(WebFetchExecutor.isTextualMime(mime: "application/xhtml+xml"))

        XCTAssertFalse(WebFetchExecutor.isTextualMime(mime: "application/octet-stream"))
        XCTAssertFalse(WebFetchExecutor.isTextualMime(mime: "application/zip"))
        XCTAssertFalse(WebFetchExecutor.isTextualMime(mime: "video/mp4"))
    }

    // MARK: - URL and Protocol Validation

    func testInvalidUrlProtocolsAreRejected() async {
        let invalidUrls = [
            "ftp://example.com/file.txt",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,hello",
            "not a url"
        ]

        for url in invalidUrls {
            do {
                _ = try await WebFetchExecutor.fetch(url: url)
                XCTFail("Should have thrown for invalid URL: \(url)")
            } catch {
                XCTAssertTrue(
                    error.localizedDescription.contains("http:// or https://") || error.localizedDescription.contains("Malformed"),
                    "Error should explain protocol violation for \(url): \(error.localizedDescription)"
                )
            }
        }
    }

    func testPrivateAndMetadataHostsAreRejected() async {
        let privateUrls = [
            "http://localhost:8080/api",
            "http://127.0.0.1:3000",
            "http://169.254.169.254/latest/meta-data/",
            "http://2130706433/",
            "http://0177.0.0.1/"
        ]

        for url in privateUrls {
            do {
                _ = try await WebFetchExecutor.fetch(url: url)
                XCTFail("Should have thrown for private/metadata host: \(url)")
            } catch {
                XCTAssertTrue(
                    error.localizedDescription.contains("private") || error.localizedDescription.contains("denied"),
                    "Error should explain private host rejection for \(url): \(error.localizedDescription)"
                )
            }
        }
    }

    // MARK: - AppToolRegistry Execution

    func testAppToolRegistryWebFetchExecutionMissingUrl() async {
        let call = AppToolCall(name: "WebFetch", arguments: [:], category: .web)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Missing 'url'"))
    }

    func testAppToolRegistryWebFetchPrivateHostRefused() async {
        let call = AppToolCall(name: "WebFetch", arguments: ["url": "http://127.0.0.1:8080"], category: .web)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("private") || result.output.contains("denied"))
    }

    func testAppToolRegistrySupportedToolNamesIncludesWebFetch() {
        XCTAssertTrue(AppToolRegistry.isImplemented("WebFetch"))
        XCTAssertTrue(AppToolRegistry.isImplemented("webfetch"))
        XCTAssertTrue(AppToolRegistry.isImplemented("web_fetch"))
        XCTAssertTrue(AppToolRegistry.isImplemented("fetch_url"))
        XCTAssertTrue(AppToolRegistry.isImplemented("read_url_content"))
    }
}
