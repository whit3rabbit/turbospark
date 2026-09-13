import Foundation
import XCTest

@testable import TurboSparkApp

/// Tests for send-time `@path` resolution: which tokens are mentions, what
/// they import, and the unguarded append path the resolver needs while a
/// submission is already `submitting`.
final class MentionResolverTests: XCTestCase {

    private func makeTempDir(_ label: String) -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-mention-\(label)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writeFile(_ text: String, to url: URL) {
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: - Token extraction

    func testEmailAddressesAreNotMentions() {
        XCTAssertEqual(
            MentionResolver.mentionTokens(in: "mail bob@example.com about it"),
            [])
    }

    func testMidWordAtIsNotAMention() {
        XCTAssertEqual(
            MentionResolver.mentionTokens(in: "handle user@name in the parser"),
            [])
    }

    func testTokensStripQuotesAndDeduplicate() {
        let draft = "see @a.txt and @\"b c.txt\" again @a.txt"
        XCTAssertEqual(
            MentionResolver.mentionTokens(in: draft),
            ["a.txt", "b c.txt"])
    }

    func testDoubleAtIsSkipped() {
        XCTAssertEqual(MentionResolver.mentionTokens(in: "literal @@ here"), [])
    }

    func testDraftContainsMentionsNeedsARealToken() {
        XCTAssertTrue(MentionResolver.draftContainsMentions("look at @src/a.rs"))
        XCTAssertFalse(MentionResolver.draftContainsMentions("no at sign here"))
        XCTAssertFalse(MentionResolver.draftContainsMentions("email a@b.com"))
    }

    // MARK: - Resolution

    /// The chat archive is redirected under XCTest but SHARED across the
    /// process (Gotcha 37's seam keys on the process, not the test), so every
    /// test targets an explicit fresh chat id and asserts on that row --
    /// `promptAttachments` reads whatever chat `loadChats()` restored.
    @MainActor
    private func attachments(model: AppModel, chatID: UUID) -> [AppPromptAttachment] {
        model.chats.first(where: { $0.id == chatID })?.draftAttachments ?? []
    }

    @MainActor
    func testFileMentionImportsAnAttachment() async throws {
        let root = makeTempDir("file")
        defer { try? FileManager.default.removeItem(at: root) }
        writeFile("hello mention", to: root.appendingPathComponent("a.txt"))

        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "look at @a.txt", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(failures, [])
        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 1)
        XCTAssertEqual(attachments(model: model, chatID: chatID).first?.fileName, "a.txt")
        XCTAssertEqual(attachments(model: model, chatID: chatID).first?.extractedText, "hello mention")
    }

    @MainActor
    func testUnresolvableTokenImportsNothingSilently() async throws {
        let root = makeTempDir("missing")
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "look at @does-not-exist.txt", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(failures, [], "a missing path is prose, not a failure")
        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 0)
    }

    @MainActor
    func testFolderMentionExpandsThroughTheBoundedWalk() async throws {
        let root = makeTempDir("folder")
        defer { try? FileManager.default.removeItem(at: root) }
        let sub = root.appendingPathComponent("sub")
        try? FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        writeFile("one", to: sub.appendingPathComponent("one.txt"))
        writeFile("two", to: sub.appendingPathComponent("two.md"))

        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "read @sub please", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(failures, [])
        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 2)
    }

    @MainActor
    func testQuotedTokenResolvesPathsWithSpaces() async throws {
        let root = makeTempDir("spaces")
        defer { try? FileManager.default.removeItem(at: root) }
        writeFile("spaced", to: root.appendingPathComponent("my note.txt"))

        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "read @\"my note.txt\"", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(failures, [])
        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 1)
        XCTAssertEqual(attachments(model: model, chatID: chatID).first?.fileName, "my note.txt")
    }

    @MainActor
    func testAbsoluteTokenOutsideTheProjectRootResolvesNothing() async throws {
        let root = makeTempDir("absolute")
        defer { try? FileManager.default.removeItem(at: root) }
        let outside = makeTempDir("outside")
        defer { try? FileManager.default.removeItem(at: outside) }
        writeFile("outside", to: outside.appendingPathComponent("far.txt"))

        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "read @\(outside.appendingPathComponent("far.txt").path)",
            projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(failures, [])
        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 0)
    }

    @MainActor
    func testProjectSymlinkOutsideTheRootResolvesNothing() async throws {
        let root = makeTempDir("symlink-root")
        defer { try? FileManager.default.removeItem(at: root) }
        let outside = makeTempDir("symlink-outside")
        defer { try? FileManager.default.removeItem(at: outside) }
        let secret = outside.appendingPathComponent("secret.txt")
        writeFile("private", to: secret)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("leak"), withDestinationURL: secret)

        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "read @leak", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(failures, [])
        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 0)
    }

    @MainActor
    func testRelativeTokenWithoutAProjectRootResolvesNothing() async throws {
        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "look at @a.txt", projectRoot: nil, chatID: chatID, into: model)

        XCTAssertEqual(failures, [])
        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 0)
    }

    // MARK: - The unguarded append path

    @MainActor
    func testSubmissionAppendWorksWhileInteractiveAppendIsRefused() {
        let model = AppModel()
        let chatID = UUID()
        model.submitting = true

        func makeAttachment(_ name: String) -> AppPromptAttachment {
            AppPromptAttachment(
                fileName: name, formatLabel: "Text", extractedText: "x",
                wasTruncatedDuringExtraction: false)
        }

        model.addPromptAttachment(makeAttachment("guarded.txt"), toChatID: chatID)
        XCTAssertEqual(
            attachments(model: model, chatID: chatID).count, 0,
            "the interactive path must keep its submitting guard")

        model.appendPromptAttachmentDuringSubmission(makeAttachment("resolved.txt"), toChatID: chatID)
        XCTAssertEqual(
            attachments(model: model, chatID: chatID).count, 1,
            "the resolver's own append must work mid-submission")

        model.submitting = false
    }
}
