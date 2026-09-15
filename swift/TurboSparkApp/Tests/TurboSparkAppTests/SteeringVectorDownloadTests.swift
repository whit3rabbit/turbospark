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
