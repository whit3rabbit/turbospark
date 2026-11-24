import XCTest
@testable import TurboSparkApp

/// The artifact chip's hover peek: title, kind, revision and the first few
/// lines of a text file. Read-only, bounded, and silent about content it
/// cannot read with certainty.
final class ArtifactChipPeekTests: XCTestCase {
    private typealias State = ArtifactChipState

    func testPeekLinesSkipLeadingBlanksAndCapCount() {
        let text = "\n\n# Title\r\nline two\nline three\nline four\nline five\n"
        XCTAssertEqual(State.peekLines(text), ["# Title", "line two", "line three", "line four"])
    }

    func testPeekLinesCutLongLines() throws {
        let long = String(repeating: "a", count: State.peekLineLength + 40)
        let line = try XCTUnwrap(State.peekLines(long).first)
        XCTAssertEqual(line, String(repeating: "a", count: State.peekLineLength) + "...")
    }

    func testPeekLinesOfBlankTextAreEmpty() {
        XCTAssertEqual(State.peekLines(" \n\t\n"), [])
        XCTAssertEqual(State.peekLines(""), [])
    }

    /// A head cut mid-character still decodes; a binary file does not.
    func testDecodeHeadToleratesCutMultibyteButRefusesBinary() {
        var data = Data("caf\u{00E9}".utf8)
        data.removeLast()
        XCTAssertEqual(State.decodeHead(data, truncated: true), "caf")
        XCTAssertNil(State.decodeHead(data, truncated: false))
        XCTAssertNil(State.decodeHead(Data([0x41, 0x00, 0x42]), truncated: false))
    }

    func testPeekTextNamesTitleKindAndVersion() {
        let text = State.peekText(
            title: "plan.md", formatLabel: "Markdown", revision: 3, fileName: "plan.md",
            lines: ["# Plan"])
        let rows = text.components(separatedBy: "\n")
        XCTAssertEqual(rows.first, "plan.md")
        XCTAssertTrue(rows[1].hasPrefix("Markdown"))
        XCTAssertTrue(rows[1].contains("3"), rows[1])
        XCTAssertTrue(rows.contains("# Plan"))
    }

    func testPeekTextWithoutLinesHasNoContentBlock() {
        let text = State.peekText(
            title: "a.md", formatLabel: "Markdown", revision: 1, fileName: "a.md", lines: [])
        XCTAssertFalse(text.contains("\n\n\n"))
        XCTAssertEqual(text.components(separatedBy: "\n").count, 4, text)
    }

    /// Only text kinds on a plain path are read: an image or PDF has no
    /// lines, and an encrypted-profile asset would be decrypted by a hover.
    func testPeekURLOnlyForPlainTextKinds() {
        let chat = UUID()
        let md = AppArtifact(chatID: chat, path: "/tmp/x/plan.md", title: "plan.md", origin: .fileWrite)
        XCTAssertEqual(State.peekURL(for: md)?.path, "/tmp/x/plan.md")
        let png = AppArtifact(chatID: chat, path: "/tmp/x/a.png", title: "a.png", origin: .fileWrite)
        XCTAssertNil(State.peekURL(for: png))
        let ghost = AppArtifact(chatID: chat, path: nil, title: "Plan", origin: .plan)
        XCTAssertNil(State.peekURL(for: ghost))
    }

    func testReadHeadReadsBoundedPrefixOfRegularFile() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("big.md")
        try String(repeating: "x", count: State.peekByteLimit * 3).write(to: file, atomically: true, encoding: .utf8)
        XCTAssertEqual(State.readHead(file)?.count, State.peekByteLimit)
        XCTAssertNil(State.readHead(dir), "a directory is not read")
        XCTAssertNil(State.readHead(dir.appendingPathComponent("missing.md")))
    }
}
