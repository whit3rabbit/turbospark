import XCTest

@testable import TurboSparkApp

/// How an attachment is PREVIEWED and THUMBNAILED, as values.
///
/// Sibling of `ImageAttachmentTests`, which covers the path from picker to
/// wire. Everything here is a display decision, and every case is shaped so it
/// needs no view: a rendering rule locked inside a `body` is a rule no test in
/// this suite can reach (`swift/CLAUDE.md` Gotcha 26), which is how
/// `FileRowView` came to spell its own subtitle and drift from the model's.
final class AttachmentPreviewTests: XCTestCase {

    private func attachment(
        named name: String,
        label: String = "Image",
        text: String = "",
        path: String? = nil,
        bytes: Int? = nil
    ) -> AppPromptAttachment {
        AppPromptAttachment(
            fileName: name,
            formatLabel: label,
            extractedText: text,
            wasTruncatedDuringExtraction: false,
            sourcePath: path,
            sourceByteSize: bytes)
    }

    /// A temporary directory holding one real file, so `sourceExists` answers
    /// truthfully. `previewKind` stats the filesystem and cannot be faked.
    private func withRealFile(named name: String, _ body: (URL) -> Void) {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-preview-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: Data([0x25, 0x50]))
        body(url)
    }

    // MARK: - previewKind

    /// The first coverage this property has ever had. It decides which
    /// renderer the pane reaches for, and a wrong answer is not an error: a
    /// PDF routed to the image arm renders nothing and reads as a broken file.
    func testPreviewKindResolvesPerExtension() {
        withRealFile(named: "page.pdf") { url in
            XCTAssertEqual(
                attachment(named: "page.pdf", label: "PDF", path: url.path).previewKind, .pdf)
        }
        withRealFile(named: "page.png") { url in
            XCTAssertEqual(attachment(named: "page.png", path: url.path).previewKind, .image)
        }
        withRealFile(named: "notes.txt") { url in
            XCTAssertEqual(
                attachment(named: "notes.txt", label: "Text", text: "hi", path: url.path)
                    .previewKind, .text)
        }
    }

    /// **A renderer pointed at a file that is gone draws an empty pane, not an
    /// error.** `.text` is the honest fallback because the extracted text is
    /// the one thing an attachment still HAS once its source has moved. The
    /// QuickLook arm depends on this: it fires only inside `.text`.
    func testPreviewKindFallsBackToTextWhenTheSourceIsGone() {
        XCTAssertEqual(
            attachment(named: "page.pdf", label: "PDF", path: "/nope/page.pdf").previewKind,
            .text)
        XCTAssertEqual(
            attachment(named: "page.png", path: "/nope/page.png").previewKind, .text)
        // No path at all is the archive-back-compat shape, and reads the same.
        XCTAssertEqual(attachment(named: "page.png").previewKind, .text)
    }

    // MARK: - Thumbnail eligibility

    /// A thumbnail is offered only where it beats the symbol it replaces.
    /// QuickLook renders a `.swift` file too, as a generic page icon that says
    /// less than the themed glyph and costs a generator round trip to say it.
    func testOnlyImagesAndPDFsAskForAThumbnail() {
        withRealFile(named: "page.png") { url in
            XCTAssertNotNil(attachment(named: "page.png", path: url.path).thumbnailSourceURL)
        }
        withRealFile(named: "page.pdf") { url in
            XCTAssertNotNil(
                attachment(named: "page.pdf", label: "PDF", path: url.path).thumbnailSourceURL)
        }
        withRealFile(named: "main.swift") { url in
            XCTAssertNil(
                attachment(named: "main.swift", label: "SWIFT", text: "x", path: url.path)
                    .thumbnailSourceURL)
        }
        // A missing source resolves `.text`, so it asks for no thumbnail and
        // the row keeps its symbol rather than flashing an empty box.
        XCTAssertNil(attachment(named: "page.png", path: "/nope/page.png").thumbnailSourceURL)
    }

    // MARK: - Cache identity

    /// **A rewritten file at the same path is a different picture.** Without
    /// the modification date in the key, an artifact the model overwrites
    /// serves its previous thumbnail for the life of the process, which reads
    /// as the write having failed.
    func testAThumbnailKeyChangesWhenTheFileIsRewritten() {
        let early = Date(timeIntervalSince1970: 1_000)
        let late = Date(timeIntervalSince1970: 2_000)
        let a = AttachmentThumbnailStore.cacheKey(path: "/a/b.png", modified: early, pixelSize: 28)
        let b = AttachmentThumbnailStore.cacheKey(path: "/a/b.png", modified: late, pixelSize: 28)
        XCTAssertNotEqual(a, b)

        // Same file, same instant: a cache hit is the whole point.
        XCTAssertEqual(
            a, AttachmentThumbnailStore.cacheKey(path: "/a/b.png", modified: early, pixelSize: 28))

        // An unreadable date must not collide with a real one.
        XCTAssertNotEqual(
            a, AttachmentThumbnailStore.cacheKey(path: "/a/b.png", modified: nil, pixelSize: 28))
    }

    /// The chip asks for 16 and the Files row for 28. One key for both would
    /// serve whichever rendered first, so a chip would show a 28pt image
    /// squeezed into 16 or a row a 16pt one blown up and soft.
    func testAThumbnailKeyChangesWithPixelSize() {
        let stamp = Date(timeIntervalSince1970: 1_000)
        XCTAssertNotEqual(
            AttachmentThumbnailStore.cacheKey(path: "/a/b.png", modified: stamp, pixelSize: 16),
            AttachmentThumbnailStore.cacheKey(path: "/a/b.png", modified: stamp, pixelSize: 28))
    }

    /// Two files must not share a key however alike their metadata.
    func testAThumbnailKeyIsPerPath() {
        let stamp = Date(timeIntervalSince1970: 1_000)
        XCTAssertNotEqual(
            AttachmentThumbnailStore.cacheKey(path: "/a/b.png", modified: stamp, pixelSize: 28),
            AttachmentThumbnailStore.cacheKey(path: "/a/c.png", modified: stamp, pixelSize: 28))
    }

    // MARK: - Files list subtitle

    /// The Files row's sibling of
    /// `ImageAttachmentTests.testAnImageChipDoesNotAdvertiseAZeroCharacterCount`.
    /// That case pins the CHIP's subtitle, and the row used to assemble its
    /// own, appending a character count unconditionally: an image row read
    /// "Image - 51 KB - 0 chars - Untitled" on the one surface no test could
    /// see. Composing from `detailText` is what makes the two agree.
    func testAnImageRowDoesNotAdvertiseAZeroCharacterCount() {
        let reference = AppAttachmentReference(
            attachment: attachment(named: "page.png", bytes: 51_200),
            chatID: UUID(),
            chatTitle: "Some chat")
        XCTAssertFalse(reference.listDetailText.contains("chars"), reference.listDetailText)
        XCTAssertTrue(reference.listDetailText.contains("Image"), reference.listDetailText)
        // The row still says which conversation it came from; that is the one
        // thing the list adds over the chip.
        XCTAssertTrue(reference.listDetailText.contains("Some chat"), reference.listDetailText)
    }

    /// A real document still reports its count, so the fix above narrowed
    /// nothing: the count was only ever wrong on a file that extracts no text.
    func testADocumentRowStillReportsItsCharacterCount() {
        let reference = AppAttachmentReference(
            attachment: attachment(named: "notes.txt", label: "Text", text: "some words"),
            chatID: UUID(),
            chatTitle: "Some chat")
        XCTAssertTrue(reference.listDetailText.contains("10 chars"), reference.listDetailText)
    }
}
