import XCTest
@testable import TurboSparkApp

/// "View file" on a turn's file-tree row opens the same read-only
/// `SourceLineViewerSheet` the Read card opens, with no range to highlight.
/// The gates (containment, sensitive paths, the 2 MB cap) are the sheet's
/// own and are covered by `SourceLineViewerTests`; this covers what the row
/// adds: which rows offer it and which lines a whole-file view draws.
final class FileTreeSourceViewerTests: XCTestCase {
    private func entry(_ change: TurnFileTree.Change) -> TurnFileTree.Entry {
        TurnFileTree.Entry(
            path: "src/a.swift", change: change, additions: nil, deletions: nil,
            origin: TurnSourceOrigin(messageID: UUID(), callID: UUID()))
    }

    func testEveryRowButADeleteOffersTheViewer() {
        for change in [TurnFileTree.Change.added, .modified, .written, .read] {
            XCTAssertTrue(entry(change).canViewFile, "\(change)")
        }
        XCTAssertFalse(entry(.deleted).canViewFile, "a deleted file has nothing to show")
    }

    func testWholeFileWindowStartsAtLineOneAndIsCapped() {
        XCTAssertEqual(SourceLineViewer.wholeFileWindow(lineCount: 12), 1...12)
        XCTAssertEqual(
            SourceLineViewer.wholeFileWindow(lineCount: SourceLineViewer.maxRenderedLines + 50),
            1...SourceLineViewer.maxRenderedLines)
        // An empty file splits into one empty line; the window still exists.
        XCTAssertEqual(SourceLineViewer.wholeFileWindow(lineCount: 0), 1...1)
    }

    /// The row resolves its path through the sheet's own gate: a path that
    /// climbs out of the project or names a key file is refused, not read.
    func testRowPathsGoThroughTheViewerGate() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertEqual(SourceLineViewer.resolve(path: "../escape.txt", projectRoot: root), .outsideProject)
        XCTAssertEqual(SourceLineViewer.resolve(path: ".env", projectRoot: root), .sensitive)
        XCTAssertEqual(SourceLineViewer.resolve(path: "a.swift", projectRoot: nil), .noProject)
    }
}
