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

    // MARK: - Typed context references

    /// Runs `git` in `dir`, failing the test if it does not succeed
    /// (mirrors ProjectLifecycleTests' fixture).
    private func git(_ args: [String], in dir: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.currentDirectoryURL = dir
        process.arguments = args
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, "git \(args.joined(separator: " ")) failed")
    }

    /// A real one-commit repository so the git references have something to
    /// read; an empty directory makes every `@diff` outcome vacuously empty.
    private func makeGitRepo(_ label: String) throws -> URL {
        let dir = makeTempDir(label)
        try git(["init", "-q"], in: dir)
        try git(["config", "user.email", "t@example.com"], in: dir)
        try git(["config", "user.name", "T"], in: dir)
        try "base\n".write(to: dir.appendingPathComponent("base.txt"), atomically: true, encoding: .utf8)
        try git(["add", "."], in: dir)
        try git(["commit", "-qm", "base commit"], in: dir)
        return dir
    }

    @MainActor
    func testDiffReferenceAttachesTheWorkingTreeDiff() async throws {
        let root = try makeGitRepo("diff")
        defer { try? FileManager.default.removeItem(at: root) }
        try "base\nUNSTAGED_MARKER\n".write(
            to: root.appendingPathComponent("base.txt"), atomically: true, encoding: .utf8)

        let model = AppModel()
        let chatID = UUID()
        let failures = await MentionResolver.resolveMentions(
            in: "summarize @diff", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(failures, [])
        let attached = attachments(model: model, chatID: chatID)
        XCTAssertEqual(attached.count, 1)
        XCTAssertEqual(attached.first?.fileName, "Working tree diff")
        XCTAssertEqual(attached.first?.formatLabel, "diff")
        XCTAssertTrue(attached.first?.extractedText.contains("UNSTAGED_MARKER") ?? false)
    }

    @MainActor
    func testStagedReferenceAttachesTheCachedDiff() async throws {
        let root = try makeGitRepo("staged")
        defer { try? FileManager.default.removeItem(at: root) }
        try "base\nSTAGED_MARKER\n".write(
            to: root.appendingPathComponent("base.txt"), atomically: true, encoding: .utf8)
        try git(["add", "."], in: root)

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "what is @staged", projectRoot: root, chatID: chatID, into: model)

        let attached = attachments(model: model, chatID: chatID)
        XCTAssertEqual(attached.count, 1)
        XCTAssertEqual(attached.first?.fileName, "Staged diff")
        XCTAssertTrue(attached.first?.extractedText.contains("STAGED_MARKER") ?? false)
    }

    @MainActor
    func testGitReferenceAttachesTheLastNCommits() async throws {
        let root = try makeGitRepo("gitref")
        defer { try? FileManager.default.removeItem(at: root) }
        try "second\n".write(to: root.appendingPathComponent("b.txt"), atomically: true, encoding: .utf8)
        try git(["add", "."], in: root)
        try git(["commit", "-qm", "second commit"], in: root)

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "recent work @git:2", projectRoot: root, chatID: chatID, into: model)

        let attached = attachments(model: model, chatID: chatID)
        XCTAssertEqual(attached.count, 1)
        XCTAssertEqual(attached.first?.fileName, "Last 2 commits")
        XCTAssertTrue(attached.first?.extractedText.contains("base commit") ?? false)
        XCTAssertTrue(attached.first?.extractedText.contains("second commit") ?? false)
    }

    @MainActor
    func testGitReferencesAreExplicitFailuresThatToast() async throws {
        // A directory that is NOT a repository: the failure is worth a
        // toast, unlike a missing bare path.
        let root = makeTempDir("notarepo")
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "summarize @diff", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 0)
        XCTAssertNotNil(model.activeToast, "an explicit reference that fails says so")
        XCTAssertTrue(model.activeToast?.message.contains("@diff") ?? false)
    }

    @MainActor
    func testDuplicateSchemeTokensAttachOnce() async throws {
        let root = try makeGitRepo("dup")
        defer { try? FileManager.default.removeItem(at: root) }
        try "base\nCHANGED\n".write(
            to: root.appendingPathComponent("base.txt"), atomically: true, encoding: .utf8)

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "@diff and again @diff", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 1)
    }

    @MainActor
    func testFileSchemeLineRangeSlicesTheExtractedText() async throws {
        let root = makeTempDir("range")
        defer { try? FileManager.default.removeItem(at: root) }
        writeFile("one\ntwo\nthree\nfour\nfive\n", to: root.appendingPathComponent("notes.txt"))

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "read @file:notes.txt:2-3", projectRoot: root, chatID: chatID, into: model)

        let attached = attachments(model: model, chatID: chatID)
        XCTAssertEqual(attached.count, 1)
        XCTAssertEqual(attached.first?.extractedText, "two\nthree")
    }

    @MainActor
    func testARangeShapedSuffixThatIsTheRealFileNameStillMentions() async throws {
        let root = makeTempDir("colonname")
        defer { try? FileManager.default.removeItem(at: root) }
        writeFile("colon file body", to: root.appendingPathComponent("notes:2024"))

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "read @notes:2024", projectRoot: root, chatID: chatID, into: model)

        let attached = attachments(model: model, chatID: chatID)
        XCTAssertEqual(attached.count, 1)
        XCTAssertEqual(attached.first?.fileName, "notes:2024")
        XCTAssertEqual(attached.first?.extractedText, "colon file body")
    }

    @MainActor
    func testSensitivePathsAreBlockedFromReferences() async throws {
        let root = makeTempDir("sensitive")
        defer { try? FileManager.default.removeItem(at: root) }
        // `secrets.json` is the DISCRIMINATING fixture: its extension is one
        // document extraction supports, so if the sensitive-path guard were
        // gone the import would succeed and this test would redden. `.env`
        // rides along as the real-world name, though extraction's own
        // format allowlist would refuse it anyway.
        writeFile("{\"k\":1}", to: root.appendingPathComponent("secrets.json"))
        writeFile("SECRET=1", to: root.appendingPathComponent(".env"))
        let sshDir = root.appendingPathComponent(".ssh")
        try? FileManager.default.createDirectory(at: sshDir, withIntermediateDirectories: true)
        writeFile("private", to: sshDir.appendingPathComponent("id_rsa"))

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "read @file:secrets.json and @.ssh/id_rsa", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(
            attachments(model: model, chatID: chatID).count, 0,
            "the read tool's sensitive-file list gates mentions too")
        XCTAssertNotNil(model.activeToast, "the explicit @file: refusal toasts")
    }

    @MainActor
    func testABareSensitivePathIsBlockedSilently() async throws {
        let root = makeTempDir("sensitive-bare")
        defer { try? FileManager.default.removeItem(at: root) }
        writeFile("SECRET=1", to: root.appendingPathComponent(".env"))

        let model = AppModel()
        let chatID = UUID()
        _ = await MentionResolver.resolveMentions(
            in: "read @.env", projectRoot: root, chatID: chatID, into: model)

        XCTAssertEqual(attachments(model: model, chatID: chatID).count, 0)
        XCTAssertNil(
            model.activeToast,
            "bare mentions keep the resolver's silence; the missing chip is the feedback")
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
