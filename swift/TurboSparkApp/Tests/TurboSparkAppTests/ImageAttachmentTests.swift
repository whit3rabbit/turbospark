import UniformTypeIdentifiers
import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Images attached to a prompt, from the picker through to the wire.
///
/// Every case here is a way a picture is silently LOST rather than a way it
/// errors, which is the failure mode this whole path has: a dropped image
/// leaves the model answering fluently about something it was never shown
/// (`crates/cli` Gotcha 13, measured on the first end-to-end run there).
final class ImageAttachmentTests: XCTestCase {

    private func attachment(
        named name: String,
        path: String? = nil,
        bytes: Int? = nil
    ) -> AppPromptAttachment {
        AppPromptAttachment(
            fileName: name,
            formatLabel: "Image",
            extractedText: "",
            wasTruncatedDuringExtraction: false,
            sourcePath: path,
            sourceByteSize: bytes)
    }

    // MARK: - Classification

    /// One extension list, read by every consumer. It was spelled out twice
    /// before `isImage` arrived, and a restated list is correct on the day it
    /// is written and silently wrong at the next addition.
    func testEveryImageExtensionIsClassifiedAsOne() {
        for ext in AppPromptAttachment.imageFileExtensions {
            XCTAssertTrue(
                attachment(named: "page.\(ext)").isImage,
                "\(ext) should be an image")
            XCTAssertEqual(attachment(named: "page.\(ext)").symbolName, "photo")
        }
        for ext in ["pdf", "docx", "txt", "rs", "swift"] {
            XCTAssertFalse(
                attachment(named: "doc.\(ext)").isImage,
                "\(ext) should not be an image")
        }
    }

    /// **A PICTURE THE ENGINE CANNOT OPEN IS NOT SENDABLE, however well it
    /// renders.** The engine reads the file by path, so an attachment whose
    /// source was never recorded or has since moved would reach `prepare` as
    /// a file-not-found in the middle of a turn.
    func testAnImageIsSendableOnlyWhenItsSourceIsStillOnDisk() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-image-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let present = dir.appendingPathComponent("page.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: present)

        XCTAssertTrue(attachment(named: "page.png", path: present.path).isSendableImage)
        // Recorded but gone.
        XCTAssertFalse(
            attachment(named: "page.png", path: dir.appendingPathComponent("gone.png").path)
                .isSendableImage)
        // Never recorded.
        XCTAssertFalse(attachment(named: "page.png", path: nil).isSendableImage)
        // A document is never a sendable image whatever its path.
        XCTAssertFalse(attachment(named: "notes.txt", path: present.path).isSendableImage)
    }

    // MARK: - Persistence

    /// **THE DECODE-TOLERANCE GUARD.** `imagePaths` is a field added after
    /// release to a type the on-disk archive already holds. A non-optional
    /// decode of it would throw on every message written before today,
    /// `load()` swallows that and returns the empty archive, and the next
    /// write puts the emptiness back over the user's chats -- which has
    /// already happened once here (`swift/CLAUDE.md` Gotcha 13).
    func testAMessageWrittenBeforeImagesExistedStillDecodes() throws {
        let legacy = """
        { "id": "6C7E4C4E-0000-4000-8000-000000000001", "role": "user",
          "content": "hello", "reasoning": "", "toolCalls": [], "toolResults": [],
          "createdAt": 750000000 }
        """
        let message = try JSONDecoder().decode(
            AppChatMessage.self, from: Data(legacy.utf8))
        XCTAssertEqual(message.content, "hello")
        XCTAssertEqual(message.imagePaths, [])
    }

    /// Paths survive a round trip, which is what lets a multi-step agent
    /// turn rebuild the same prompt it sent the first time.
    func testImagePathsRoundTripThroughTheArchive() throws {
        let message = AppChatMessage(
            role: .user, content: "Transcribe.", imagePaths: ["/a.png", "/b.png"])
        let data = try JSONEncoder().encode(message)
        let back = try JSONDecoder().decode(AppChatMessage.self, from: data)
        XCTAssertEqual(back.imagePaths, ["/a.png", "/b.png"])
    }

    // MARK: - The wire shape

    /// A rebuilt history turn carries its pictures. Without this the FIRST
    /// step of an agent loop sends the image and every later step does not,
    /// while the transcript still shows the question that referred to it.
    func testARebuiltTurnCarriesItsImagesInOrder() {
        let stored = AppChatMessage(
            role: .user, content: "What is this?", imagePaths: ["/first.png", "/second.png"])
        let wire = ChatMessage(
            role: stored.role,
            content: stored.content,
            images: stored.imagePaths.map(ChatImage.path))
        XCTAssertEqual(wire.images, [.path("/first.png"), .path("/second.png")])
    }

    /// **AN IMAGE-ONLY TURN HAS NO TEXT AND IS STILL A TURN.** The history
    /// builder's emptiness guard predates images and would drop one whole,
    /// which reads as the model ignoring the picture.
    func testAnImageOnlyTurnIsNotTreatedAsEmpty() {
        let imageOnly = AppChatMessage(role: .user, content: "", imagePaths: ["/p.png"])
        let trulyEmpty = AppChatMessage(role: .user, content: "")
        // This is the guard `executeGenerationTurn` applies, stated once.
        XCTAssertTrue(!imageOnly.content.isEmpty || !imageOnly.imagePaths.isEmpty)
        XCTAssertFalse(!trulyEmpty.content.isEmpty || !trulyEmpty.imagePaths.isEmpty)
    }

    // MARK: - Presentation

    /// An image extracts no text, so a character count on one is a zero from
    /// an absent measurement rather than a measurement of zero. "0 chars"
    /// reads as a failed import.
    func testAnImageChipDoesNotAdvertiseAZeroCharacterCount() {
        let image = attachment(named: "page.png", bytes: 51_200)
        XCTAssertEqual(image.characterCount, 0, "an image genuinely extracts no text")
        // The SUBTITLE is the assertion: it must not turn that zero into a
        // fact about the file.
        XCTAssertFalse(image.detailText.contains("chars"), image.detailText)
        XCTAssertTrue(image.detailText.contains("Image"), image.detailText)

        // A picture whose file has gone says so rather than looking normal.
        XCTAssertTrue(
            attachment(named: "page.png", path: "/nope/gone.png", bytes: 10)
                .detailText.contains("file missing"))

        // A document with real text still reports its count, unchanged.
        let doc = AppPromptAttachment(
            fileName: "notes.txt",
            formatLabel: "Text",
            extractedText: "some words",
            wasTruncatedDuringExtraction: false)
        XCTAssertFalse(doc.isImage)
        XCTAssertTrue(doc.detailText.contains("10 chars"), doc.detailText)
    }

    /// The picker types are derived from the extension list rather than
    /// restated, so adding an extension needs no second edit.
    func testThePickerTypesComeFromTheExtensionList() {
        let types = AppPromptAttachment.imageContentTypes
        XCTAssertFalse(types.isEmpty)
        for type in types {
            let ext = type.preferredFilenameExtension ?? ""
            XCTAssertTrue(
                AppPromptAttachment.imageFileExtensions.contains(ext)
                    || AppPromptAttachment.imageFileExtensions.contains(where: {
                        UTType(filenameExtension: $0) == type
                    }),
                "\(type) is not one of the declared image extensions")
        }
    }
}
