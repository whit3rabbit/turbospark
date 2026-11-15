import XCTest
@testable import TurboSparkApp

final class ImageFetchExecutorTests: NetworkPolicyPinnedTestCase {
    private var root: URL!

    private static let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0x0D, 0x49, 0x48, 0x44, 0x52])
    private static let jpeg = Data([0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, 0x4A, 0x46, 0x49, 0x46, 0, 1])
    private static let gif = Data("GIF89a".utf8) + Data(count: 10)
    private static let webp = Data("RIFF".utf8) + Data([0, 0, 0, 0]) + Data("WEBP".utf8) + Data(count: 4)

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImageFetchTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ImageSearchStubProtocol.seen = []
        ImageSearchStubProtocol.handler = nil
    }

    override func tearDownWithError() throws {
        ImageSearchStubProtocol.handler = nil
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    private func serve(_ body: Data, status: Int = 200, type: String = "image/png") {
        ImageSearchStubProtocol.handler = { _ in (status, ["Content-Type": type], body) }
    }

    private func fetch(
        _ path: String, url: String = "https://img.example.org/a.png", sourcePage: String? = nil
    ) async throws -> String {
        try await ImageFetchExecutor.fetch(
            url: url, path: path, sourcePage: sourcePage, rootURL: root,
            customSession: ImageSearchStubProtocol.session(), destinationValidator: { _ in })
    }

    private func exists(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent(path).path)
    }

    private func assertRefused(
        _ path: String, url: String = "https://img.example.org/a.png", contains: String,
        file: StaticString = #filePath, line: UInt = #line
    ) async {
        do {
            _ = try await fetch(path, url: url)
            XCTFail("expected a refusal containing '\(contains)'", file: file, line: line)
        } catch {
            XCTAssertTrue(
                contains.isEmpty || error.localizedDescription.contains(contains),
                error.localizedDescription, file: file, line: line)
        }
    }

    // MARK: - Saving

    func testSavesEachRasterKindAndReportsProvenance() async throws {
        for (path, data, kind) in [
            ("a/hawk.png", Self.png, "png"), ("b/hawk.jpg", Self.jpeg, "jpeg"),
            ("c/hawk.jpeg", Self.jpeg, "jpeg"), ("d/hawk.gif", Self.gif, "gif"), ("e/hawk.webp", Self.webp, "webp"),
        ] {
            serve(data)
            let report = try await fetch(path, sourcePage: "https://page.example.com/hawk")
            XCTAssertTrue(report.contains("Saved \(kind) image"), report)
            XCTAssertTrue(report.contains("Source page (cite this): https://page.example.com/hawk"), report)
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent(path)), data)
        }
    }

    func testReportWithoutSourcePageRemindsToCite() async throws {
        serve(Self.png)
        let report = try await fetch("x.png")
        XCTAssertTrue(report.contains("Cite the source page"), report)
    }

    // MARK: - Content checks (the header is not trusted)

    func testHTMLBodyIsRefusedEvenWithAnImageContentType() async {
        serve(Data("<html><script>alert(1)</script></html>".utf8), type: "image/png")
        await assertRefused("fake.png", contains: "not a PNG, JPEG, GIF or WebP")
        XCTAssertFalse(exists("fake.png"))
    }

    func testSVGIsRefused() async {
        serve(Data("<svg xmlns=\"http://www.w3.org/2000/svg\"><script/></svg>".utf8), type: "image/svg+xml")
        await assertRefused("logo.png", contains: "not a PNG, JPEG, GIF or WebP")
        XCTAssertFalse(exists("logo.png"))
    }

    func testExtensionMustMatchTheSniffedKind() async {
        serve(Self.png)
        await assertRefused("pic.jpg", contains: "use .png")
        XCTAssertFalse(exists("pic.jpg"))
    }

    func testNonImageExtensionIsRefusedBeforeAnyRequest() async {
        serve(Self.png)
        await assertRefused("script.sh", contains: "must end in")
        await assertRefused("noext", contains: "must end in")
        XCTAssertTrue(ImageSearchStubProtocol.seen.isEmpty, "a bad path must not cost a download")
    }

    func testHTTPErrorSavesNothing() async {
        serve(Data("nope".utf8), status: 404)
        await assertRefused("a.png", contains: "HTTP 404")
        XCTAssertFalse(exists("a.png"))
    }

    func testOversizeImageIsRefusedAndNotWritten() async {
        serve(Self.png + Data(count: ImageFetchExecutor.maxImageBytes + 1))
        await assertRefused("big.png", contains: "MiB limit")
        XCTAssertFalse(exists("big.png"))
    }

    // MARK: - Destination

    func testNeverOverwritesAnExistingFile() async throws {
        let original = Data("precious".utf8)
        try original.write(to: root.appendingPathComponent("keep.png"))
        serve(Self.png)
        await assertRefused("keep.png", contains: "already exists")
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("keep.png")), original)
    }

    func testPathCannotEscapeTheProjectRoot() async {
        serve(Self.png)
        for path in ["../escape.png", "/tmp/abs.png", "~/home.png", "a/../../escape.png"] {
            await assertRefused(path, contains: "")
        }
        XCTAssertTrue(ImageSearchStubProtocol.seen.isEmpty)
    }

    func testSymlinkedDirectoryCannotPivotOutOfTheRoot() async throws {
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImageFetchOutside-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("link"), withDestinationURL: outside)
        serve(Self.png)
        await assertRefused("link/pivot.png", contains: "escapes the project root")
        XCTAssertFalse(FileManager.default.fileExists(atPath: outside.appendingPathComponent("pivot.png").path))
    }

    func testSensitiveDestinationIsRefused() async {
        serve(Self.png)
        await assertRefused(".ssh/id.png", contains: "sensitive")
        XCTAssertTrue(ImageSearchStubProtocol.seen.isEmpty)
    }

    // MARK: - SSRF: the real destination validator, no injected hooks

    func testPrivateAndMetadataTargetsAreRefusedBeforeAnyRequest() async {
        serve(Self.png)
        for url in [
            "http://127.0.0.1/a.png", "http://localhost/a.png", "http://169.254.169.254/a.png",
            "http://[::1]/a.png", "http://10.0.0.5/a.png", "http://192.168.1.1/a.png",
        ] {
            do {
                _ = try await ImageFetchExecutor.fetch(
                    url: url, path: "ssrf.png", rootURL: root,
                    customSession: ImageSearchStubProtocol.session())
                XCTFail("\(url) must be refused")
            } catch {
                XCTAssertFalse(error.localizedDescription.isEmpty)
            }
            XCTAssertFalse(exists("ssrf.png"), url)
        }
        XCTAssertTrue(ImageSearchStubProtocol.seen.isEmpty, "the stub must never be reached: \(ImageSearchStubProtocol.seen)")
    }

    func testNonHttpAndCredentialedUrlsAreRefused() async {
        serve(Self.png)
        for url in ["file:///etc/passwd", "ftp://example.com/a.png", "https://u:p@example.com/a.png", "not a url"] {
            await assertRefused("n.png", url: url, contains: "")
        }
        XCTAssertTrue(ImageSearchStubProtocol.seen.isEmpty)
    }

    // MARK: - Registry integration

    func testRegistryNeedsAProjectRoot() async {
        let call = AppToolCall(
            name: "image_fetch", arguments: ["url": "https://img.example.org/a.png", "path": "a.png"], category: .web)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("needs a project workspace"), result.output)
    }
}
