import XCTest

@testable import TurboSparkApp

final class ArtifactWebViewSecurityTests: XCTestCase {
    func testNetworkAccessConvertsAFileDocumentToInlineHTML() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let page = folder.appendingPathComponent("page.html")
        let html = "<script>fetch('https://example.com')</script>"
        try html.write(to: page, atomically: true, encoding: .utf8)

        let document = ArtifactWebDocument.file(page: page, readAccessFolder: folder)

        XCTAssertEqual(document.isolatedForNetworkAccess(), .inline(html: html))
        XCTAssertNil(document.isolatedForNetworkAccess().readAccessFolder)
    }

    func testInlineDocumentRemainsInlineWhenNetworkAccessIsEnabled() {
        let document = ArtifactWebDocument.inline(html: "<p>preview</p>")

        XCTAssertEqual(document.isolatedForNetworkAccess(), document)
    }
}
