import XCTest

@testable import TurboSparkApp

/// The read-only source viewer opened from a Read tool card (component
/// audit, source-viewer). It reads from disk only inside the project root,
/// so the containment and refusal cases matter as much as the range math.
final class SourceLineViewerTests: XCTestCase {

    private var root: URL!
    private var outside: URL!

    override func setUpWithError() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("SourceLineViewerTests-\(UUID().uuidString)")
        root = base.appendingPathComponent("project")
        outside = base.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("src"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }

    // MARK: - Requested range

    func testRangeReadsEverySpellingTheCardShows() {
        XCTAssertEqual(SourceLineViewer.requestedRange(["start_line": "3", "end_line": "9"]), 3...9)
        XCTAssertEqual(SourceLineViewer.requestedRange(["StartLine": "1", "EndLine": "2"]), 1...2)
        XCTAssertEqual(SourceLineViewer.requestedRange(["startLine": " 4 ", "endLine": "4"]), 4...4)
        XCTAssertEqual(SourceLineViewer.requestedRange(["start": "10", "end": "20"]), 10...20)
    }

    func testRangeIsNilWhenUnsure() {
        XCTAssertNil(SourceLineViewer.requestedRange([:]))
        XCTAssertNil(SourceLineViewer.requestedRange(["start_line": "3"]))
        XCTAssertNil(SourceLineViewer.requestedRange(["start_line": "a", "end_line": "9"]))
        XCTAssertNil(SourceLineViewer.requestedRange(["start_line": "0", "end_line": "9"]))
        XCTAssertNil(SourceLineViewer.requestedRange(["start_line": "9", "end_line": "3"]))
    }

    // MARK: - Clamp and window

    func testClampTrimsToTheFileAndRefusesAStartPastTheEnd() {
        XCTAssertEqual(SourceLineViewer.clamp(5...50, lineCount: 20), 5...20)
        XCTAssertEqual(SourceLineViewer.clamp(1...3, lineCount: 20), 1...3)
        XCTAssertNil(SourceLineViewer.clamp(21...30, lineCount: 20))
        XCTAssertNil(SourceLineViewer.clamp(1...3, lineCount: 0))
    }

    func testWindowAddsContextInsideTheFile() {
        let c = SourceLineViewer.contextLines
        XCTAssertEqual(SourceLineViewer.window(for: 1...2, lineCount: 10), 1...10)
        XCTAssertEqual(
            SourceLineViewer.window(for: 500...510, lineCount: 10_000),
            (500 - c)...(510 + c))
    }

    func testWindowIsCappedAndStartsBeforeTheRange() {
        let cap = SourceLineViewer.maxRenderedLines
        let window = SourceLineViewer.window(for: 1000...5000, lineCount: 10_000)
        XCTAssertEqual(window.count, cap)
        XCTAssertEqual(window.lowerBound, 1000 - SourceLineViewer.contextLines)
    }

    // MARK: - Containment

    func testRelativePathInsideTheProjectResolves() throws {
        let file = root.appendingPathComponent("src/a.swift")
        try "x".write(to: file, atomically: true, encoding: .utf8)
        guard case .file(let url) = SourceLineViewer.resolve(path: "src/a.swift", projectRoot: root) else {
            return XCTFail("expected a file")
        }
        XCTAssertEqual(url, PathContainment.canonical(file))
    }

    func testNoProjectMeansNoRead() {
        XCTAssertEqual(SourceLineViewer.resolve(path: "src/a.swift", projectRoot: nil), .noProject)
    }

    func testTraversalAbsoluteAndTildePathsAreRefused() throws {
        try "secret".write(to: outside.appendingPathComponent("b.txt"), atomically: true, encoding: .utf8)
        XCTAssertEqual(SourceLineViewer.resolve(path: "../outside/b.txt", projectRoot: root), .outsideProject)
        XCTAssertEqual(
            SourceLineViewer.resolve(path: outside.appendingPathComponent("b.txt").path, projectRoot: root),
            .outsideProject)
        XCTAssertEqual(SourceLineViewer.resolve(path: "~/b.txt", projectRoot: root), .outsideProject)
    }

    func testSymlinkOutOfTheProjectIsRefused() throws {
        try "secret".write(to: outside.appendingPathComponent("b.txt"), atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("link"), withDestinationURL: outside)
        XCTAssertEqual(SourceLineViewer.resolve(path: "link/b.txt", projectRoot: root), .outsideProject)
    }

    func testAbsolutePathInsideTheProjectResolves() throws {
        let file = root.appendingPathComponent("src/a.swift")
        try "x".write(to: file, atomically: true, encoding: .utf8)
        XCTAssertEqual(
            SourceLineViewer.resolve(path: file.path, projectRoot: root),
            .file(PathContainment.canonical(file)))
    }

    func testSensitivePathIsRefusedEvenInsideTheProject() throws {
        try "K=V".write(to: root.appendingPathComponent(".env"), atomically: true, encoding: .utf8)
        XCTAssertEqual(SourceLineViewer.resolve(path: ".env", projectRoot: root), .sensitive)
    }

    // MARK: - Load

    func testLoadSplitsLinesLikeTheReadTool() throws {
        let file = root.appendingPathComponent("a.txt")
        try "one\r\ntwo\nthree".write(to: file, atomically: true, encoding: .utf8)
        guard case .success(let loaded) = SourceLineViewer.load(file) else { return XCTFail("expected text") }
        XCTAssertEqual(loaded.lines.count, 3)
        XCTAssertEqual(SourceLineViewer.displayLine(loaded.lines[0]), "one")
    }

    func testLoadRefusesOversizedMissingDirectoryAndBinary() throws {
        let big = root.appendingPathComponent("big.txt")
        try String(repeating: "a", count: 33).write(to: big, atomically: true, encoding: .utf8)
        XCTAssertEqual(failure(SourceLineViewer.load(big, maxBytes: 32)), .tooLarge)
        XCTAssertEqual(failure(SourceLineViewer.load(root.appendingPathComponent("none"))), .notFound)
        XCTAssertEqual(failure(SourceLineViewer.load(root.appendingPathComponent("src"))), .notAFile)
        let bin = root.appendingPathComponent("b.bin")
        try Data([0xff, 0xfe, 0x00, 0xc3]).write(to: bin)
        XCTAssertEqual(failure(SourceLineViewer.load(bin)), .notText)
    }

    private func failure(_ result: Result<SourceLineViewer.Loaded, SourceLineViewer.LoadFailure>)
        -> SourceLineViewer.LoadFailure?
    {
        if case .failure(let f) = result { return f }
        return nil
    }

    // MARK: - Changed, display, copy

    func testChangedOnlyWhenModifiedAfterTheRead() {
        let read = Date(timeIntervalSince1970: 1000)
        XCTAssertFalse(SourceLineViewer.mayHaveChanged(modifiedAt: read, readAt: read))
        XCTAssertFalse(SourceLineViewer.mayHaveChanged(modifiedAt: nil, readAt: read))
        XCTAssertTrue(SourceLineViewer.mayHaveChanged(modifiedAt: read.addingTimeInterval(5), readAt: read))
    }

    func testDisplayLineTruncatesVeryLongLines() {
        let long = String(repeating: "x", count: SourceLineViewer.maxDisplayedLineLength + 10)
        let shown = SourceLineViewer.displayLine(long)
        XCTAssertEqual(shown.count, SourceLineViewer.maxDisplayedLineLength + 4)
        XCTAssertTrue(shown.hasSuffix(" ..."))
    }

    func testCopyTextIsTheHighlightedRangeOnly() {
        let lines = ["a", "b\r", "c", "d"]
        XCTAssertEqual(SourceLineViewer.copyText(lines: lines, range: 2...3), "b\nc")
    }
}
