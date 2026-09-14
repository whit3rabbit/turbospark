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
}
