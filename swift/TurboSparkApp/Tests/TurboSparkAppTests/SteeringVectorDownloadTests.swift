import Foundation
import XCTest

@testable import TurboSparkApp

final class SteeringVectorDownloadTests: XCTestCase {
    func testSourceAcceptsHuggingFaceURLsAndBuildsAStableResolveURL() {
        let source = SteeringVectorSource(
            repo: "https://huggingface.co/cfontes/Qwable-3.6-27B-refusal-control-vector/",
            file: "refusal.gguf",
            revision: "main")

        XCTAssertNil(source.validationError)
        XCTAssertEqual(
            source.identity,
            "cfontes/Qwable-3.6-27B-refusal-control-vector@main/refusal.gguf")
        XCTAssertEqual(
            source.downloadURL?.absoluteString,
            "https://huggingface.co/cfontes/Qwable-3.6-27B-refusal-control-vector/resolve/main/refusal.gguf")
    }

    func testSourceRejectsPathTraversalAndNonVectorFiles() {
        let traversal = SteeringVectorSource(
            repo: "owner/name",
            file: "../weights.gguf",
            revision: "main")
        XCTAssertNotNil(traversal.validationError)
        XCTAssertNil(traversal.downloadURL)

        let nonVector = SteeringVectorSource(
            repo: "owner/name",
            file: "README.md",
            revision: "main")
        XCTAssertEqual(nonVector.validationError, "The vector file must be a .gguf file.")
    }

    func testSourceRejectsARevisionThatCouldEscapeTheResolvePath() {
        let source = SteeringVectorSource(
            repo: "owner/name",
            file: "vector.gguf",
            revision: "refs/heads/main")
        XCTAssertEqual(
            source.validationError,
            "Revision must be a branch name or commit without path separators.")
    }

    func testManagedPathIsStableAndScopedToTheVectorStore() {
        let source = SteeringVectorSource(
            repo: "owner/name",
            file: "vector.gguf",
            revision: "abc123")

        let first = SteeringVectorDownloader.managedPath(for: source)
        let second = SteeringVectorDownloader.managedPath(for: source)

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.pathExtension, "gguf")
        XCTAssertTrue(first.path.hasPrefix(AppStorageRoot.subdirectory("steering-vectors").path))
    }

    func testBoundedDownloadRejectsAdvertisedOversizeBeforeReceivingBody() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        _ = FileManager.default.createFile(atPath: temporary.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: temporary) }

        var result: Result<Void, Error>?
        let delegate = SteeringVectorDownloader.BoundedDownloadDelegate(
            file: try FileHandle(forWritingTo: temporary),
            maxBytes: 4,
            completion: { result = $0 })
        let session = URLSession(configuration: .ephemeral)
        let task = session.dataTask(with: URL(string: "https://huggingface.co/vector.gguf")!)
        let response = HTTPURLResponse(
            url: task.originalRequest!.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Length": "5"])!
        var disposition: URLSession.ResponseDisposition?

        delegate.urlSession(session, dataTask: task, didReceive: response) {
            disposition = $0
        }

        XCTAssertEqual(disposition, .cancel)
        assertTooLarge(result)
        XCTAssertEqual(try Data(contentsOf: temporary), Data())
    }

    func testBoundedDownloadCancelsBeforeWritingAChunkPastTheLimit() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        _ = FileManager.default.createFile(atPath: temporary.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: temporary) }

        var result: Result<Void, Error>?
        let delegate = SteeringVectorDownloader.BoundedDownloadDelegate(
            file: try FileHandle(forWritingTo: temporary),
            maxBytes: 4,
            completion: { result = $0 })
        let session = URLSession(configuration: .ephemeral)
        let task = session.dataTask(with: URL(string: "https://huggingface.co/vector.gguf")!)

        delegate.urlSession(session, dataTask: task, didReceive: Data([1, 2, 3]))
        delegate.urlSession(session, dataTask: task, didReceive: Data([4, 5]))

        assertTooLarge(result)
        XCTAssertEqual(try Data(contentsOf: temporary), Data([1, 2, 3]))
    }

    func testTokenHostAllowlistAcceptsOnlyOfficialHttpsHosts() {
        for ok in ["https://huggingface.co/a", "https://cdn-lfs.huggingface.co/x",
                   "https://cas-bridge.xethub.hf.co/x", "https://hf.co/x"] {
            XCTAssertTrue(HfEndpointResolution.isTokenHost(URL(string: ok)), ok)
        }
        for bad in ["http://huggingface.co/a", "https://hf-mirror.com/a",
                    "https://evilhuggingface.co/a", "https://huggingface.co.evil.example/a",
                    "https://nothf.co/a", "http://127.0.0.1/a"] {
            XCTAssertFalse(HfEndpointResolution.isTokenHost(URL(string: bad)), bad)
        }
        XCTAssertFalse(HfEndpointResolution.isTokenHost(nil))
    }

    func testRedirectStripsAuthorizationOffHostButKeepsItOnHfHosts() {
        var off = URLRequest(url: URL(string: "https://evil.example/f")!)
        off.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        XCTAssertNil(SteeringVectorDownloader.BoundedDownloadDelegate.redirectedRequest(off)
            .value(forHTTPHeaderField: "Authorization"))
        var on = URLRequest(url: URL(string: "https://cdn-lfs.huggingface.co/f")!)
        on.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        XCTAssertEqual(
            SteeringVectorDownloader.BoundedDownloadDelegate.redirectedRequest(on)
                .value(forHTTPHeaderField: "Authorization"), "Bearer secret")
    }

    func testDownloadSendsTokenToHfAndStripsItOnCrossHostRedirect() async {
        RedirectStubProtocol.reset()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RedirectStubProtocol.self]
        let source = SteeringVectorSource(repo: "owner/name", file: "v.gguf", revision: "main")
        // The stub body is not a control vector, so the call throws after the
        // network phase; the recorded requests are what this test asserts.
        _ = try? await SteeringVectorDownloader.download(
            source: source, expectedHidden: nil, expectedLayers: nil,
            configuration: config, tokenProvider: { "hf_secret" })
        let seen = RedirectStubProtocol.seen
        XCTAssertEqual(seen.map { $0.url?.host }, ["huggingface.co", "evil.example"])
        XCTAssertEqual(seen.first?.value(forHTTPHeaderField: "Authorization"), "Bearer hf_secret")
        XCTAssertNil(seen.last?.value(forHTTPHeaderField: "Authorization"))
    }

    private func assertTooLarge(
        _ result: Result<Void, Error>?,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard case let .failure(error) = result,
              case SteeringVectorDownloadError.tooLarge = error
        else {
            XCTFail("Expected a too-large failure", file: file, line: line)
            return
        }
    }
}

final class RedirectStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var seen: [URLRequest] = []
    static func reset() { seen = [] }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}
    override func startLoading() {
        Self.seen.append(request)
        let url = request.url!
        if url.host == "huggingface.co" {
            let target = URL(string: "https://evil.example/file")!
            let resp = HTTPURLResponse(url: url, statusCode: 302, httpVersion: nil,
                                       headerFields: ["Location": target.absoluteString])!
            var next = URLRequest(url: target)
            next.allHTTPHeaderFields = request.allHTTPHeaderFields
            client?.urlProtocol(self, wasRedirectedTo: next, redirectResponse: resp)
        } else {
            let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: resp, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data([1, 2, 3]))
            client?.urlProtocolDidFinishLoading(self)
        }
    }
}
