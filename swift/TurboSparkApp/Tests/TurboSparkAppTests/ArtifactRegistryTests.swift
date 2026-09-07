import XCTest

@testable import TurboSparkApp

/// The artifact policy and the dedupe, both as pure values.
///
/// Every case here runs without an `AppModel`, a window or a tool call, which
/// is deliberate and is the `ServerStatusRows` lesson (`swift/CLAUDE.md`
/// Gotcha 26): the decisions that decide whether a panel pops, and whether it
/// pops twice, used to be reachable only by running the app and looking.
final class ArtifactRegistryTests: XCTestCase {
    private let chat = UUID()

    private func artifact(
        _ path: String?,
        title: String = "",
        origin: AppArtifact.Origin = .fileWrite,
        toolCallID: UUID? = nil,
        bytes: Int? = nil,
        chatID: UUID? = nil
    ) -> AppArtifact {
        AppArtifact(
            chatID: chatID ?? chat,
            path: path,
            title: title,
            origin: origin,
            toolCallID: toolCallID,
            lastKnownByteSize: bytes)
    }

    private func produced(
        _ path: String,
        origin: AppArtifact.Origin,
        tool: String = "write_file"
    ) -> ArtifactRegistrar.ProducedFile {
        ArtifactRegistrar.ProducedFile(
            url: URL(fileURLWithPath: path),
            toolCallID: UUID(),
            toolName: tool,
            origin: origin)
    }

    // MARK: - Dedupe

    func testTheSamePathWrittenThreeTimesIsOneArtifactWhoseRevisionMoved() {
        var rows: [AppArtifact] = []
        let first = AppArtifact.upsert(artifact("/tmp/p/plan.md"), into: &rows)
        AppArtifact.upsert(artifact("/tmp/p/plan.md"), into: &rows)
        let third = AppArtifact.upsert(artifact("/tmp/p/plan.md"), into: &rows)

        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].revision, 3)
        // The id survives, which is what auto-open-once is keyed on.
        XCTAssertEqual(third.id, first.id)
        XCTAssertEqual(rows[0].id, first.id)
    }

    func testTwoDifferentPathsInOneTurnAreTwoArtifacts() {
        var rows: [AppArtifact] = []
        AppArtifact.upsert(artifact("/tmp/p/a.md"), into: &rows)
        AppArtifact.upsert(artifact("/tmp/p/b.md"), into: &rows)

        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(Set(rows.map(\.revision)), [1])
    }

    func testTheSamePathInTwoChatsIsTwoArtifacts() {
        var rows: [AppArtifact] = []
        AppArtifact.upsert(artifact("/tmp/p/plan.md"), into: &rows)
        AppArtifact.upsert(artifact("/tmp/p/plan.md", chatID: UUID()), into: &rows)

        XCTAssertEqual(rows.count, 2)
    }

    func testAnArtifactPathIsComparedStandardized() {
        var rows: [AppArtifact] = []
        AppArtifact.upsert(artifact("/tmp/p/docs/plan.md"), into: &rows)
        AppArtifact.upsert(artifact("/tmp/p/docs/../docs/plan.md"), into: &rows)

        XCTAssertEqual(rows.count, 1, "the two spellings name one file")
        XCTAssertEqual(rows[0].revision, 2)
    }

    func testARewriteMovesTheCardToTheLatestToolCall() {
        var rows: [AppArtifact] = []
        AppArtifact.upsert(artifact("/tmp/p/plan.md", toolCallID: UUID()), into: &rows)
        let latest = UUID()
        AppArtifact.upsert(artifact("/tmp/p/plan.md", toolCallID: latest), into: &rows)

        XCTAssertEqual(rows[0].toolCallID, latest)
    }

    func testARewriteDoesNotDemoteAPresentedFileToAPlainWrite() {
        var rows: [AppArtifact] = []
        AppArtifact.upsert(artifact("/tmp/p/report.md", origin: .sentToUser), into: &rows)
        AppArtifact.upsert(artifact("/tmp/p/report.md", origin: .fileWrite), into: &rows)

        XCTAssertEqual(rows[0].origin, .sentToUser)
    }

    func testTextBackedArtifactsNeverMergeBecauseTheyHaveNoPath() {
        var rows: [AppArtifact] = []
        AppArtifact.upsert(artifact(nil, title: "Plan A", origin: .plan), into: &rows)
        AppArtifact.upsert(artifact(nil, title: "Plan B", origin: .plan), into: &rows)

        XCTAssertEqual(rows.count, 2, "two ghost plans are two plans")
    }

    // MARK: - Policy

    func testRenderableDocumentsAndPresentedFilesBecomeArtifacts() {
        let files = [
            produced("/tmp/p/notes.md", origin: .fileWrite),
            produced("/tmp/p/page.html", origin: .fileWrite),
            produced("/tmp/p/src/lib.rs", origin: .fileWrite),
            produced("/tmp/p/report.docx", origin: .sentToUser, tool: "send_user_file"),
            produced("/tmp/p/plan.md", origin: .plan, tool: "exit_plan_mode"),
        ]

        let names = ArtifactRegistrar.artifactCandidates(from: files)
            .map { $0.url.lastPathComponent }

        XCTAssertEqual(names, ["notes.md", "page.html", "report.docx", "plan.md"])
        XCTAssertFalse(names.contains("lib.rs"), "a source edit must not pop a panel")
    }

    func testHTMLJoinsMarkdownAsAnAutoRegisteredWriteButCSSStaysSource() {
        let files = [
            produced("/tmp/p/index.html", origin: .fileWrite),
            produced("/tmp/p/style.css", origin: .fileWrite),
        ]

        let names = ArtifactRegistrar.artifactCandidates(from: files)
            .map { $0.url.lastPathComponent }

        // Both sets are read by the SAME policy, and each names a set the
        // panel renders from: html is a page, css is source text.
        XCTAssertEqual(names, ["index.html"])
    }

    func testMarkdownIsRecognisedRegardlessOfCase() {
        let files = [produced("/tmp/p/README.MD", origin: .fileWrite)]
        XCTAssertEqual(ArtifactRegistrar.artifactCandidates(from: files).count, 1)
    }

    func testHTMLIsRecognisedRegardlessOfCase() {
        let files = [produced("/tmp/p/INDEX.HTML", origin: .fileWrite)]
        XCTAssertEqual(ArtifactRegistrar.artifactCandidates(from: files).count, 1)
    }

    func testNothingIsReportedWhenNoFileSurvivesThePolicy() {
        var seen = 0
        ArtifactRegistrar.onArtifactsProduced = { _, _ in seen += 1 }
        defer { ArtifactRegistrar.onArtifactsProduced = nil }

        ArtifactRegistrar.report(
            chatID: chat, produced: [produced("/tmp/p/src/lib.rs", origin: .fileWrite)])

        XCTAssertEqual(seen, 0)
    }

    // MARK: - Presentation

    func testAMissingFileStatesItIsMissingAndQuotesNoSize() {
        let row = artifact("/tmp/definitely/not/here/plan.md", bytes: 4096)
        let detail = row.detailText

        XCTAssertTrue(detail.contains("file missing"), detail)
        XCTAssertFalse(
            detail.contains("KB") || detail.contains("4,096") || detail.contains("4096"),
            "a size recorded for a file that is gone is not a fact about disk: \(detail)")
    }

    func testATextBackedArtifactSaysItIsInMemoryRatherThanMissing() {
        let detail = artifact(nil, title: "Plan", origin: .plan).detailText
        XCTAssertTrue(detail.contains("in memory"), detail)
        XCTAssertFalse(detail.contains("missing"), detail)
    }

    func testTheRenderKindIsDerivedFromTheExtension() {
        XCTAssertEqual(AppArtifactRenderKind.forFileName("plan.md"), .markdown)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("page.html"), .html)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("page.htm"), .html)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("PAGE.HTML"), .html)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("notes.txt"), .text)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("shot.png"), .image)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("paper.pdf"), .pdf)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("report.docx"), .opaque)
    }

    func testHTMLIsARenderKindAndNoLongerASourceKind() {
        // One list with one reader: "html" living in BOTH sets would make
        // the extension render two ways depending on which check ran first.
        XCTAssertFalse(
            AppArtifactRenderKind.plainTextExtensions.contains("html"),
            "html cannot be both a rendered page and monospaced source")
        XCTAssertEqual(AppArtifactRenderKind.forFileName("page.html"), .html)
    }

    func testAnHTMLArtifactLabelsItselfHTML() {
        let row = artifact("/tmp/p/page.html")
        XCTAssertEqual(row.renderKind, .html)
        XCTAssertEqual(row.formatLabel, "HTML")
    }

    func testAGhostPlanRendersAsMarkdownDespiteHavingNoFileName() {
        XCTAssertEqual(artifact(nil, title: "Plan", origin: .plan).renderKind, .markdown)
    }

    func testAnUnrenderableArtifactStillOffersNothingItCannotDo() {
        // Gotcha 22: the external openers are gated on the file being there,
        // so there is no control that cannot fail.
        let row = artifact("/tmp/definitely/not/here/report.docx")
        XCTAssertFalse(row.canOpenExternally)
        XCTAssertFalse(row.hasReadableContent)
    }

    // MARK: - Decoding

    func testAnUnknownArtifactOriginDecodesToAFallbackRatherThanThrowing() throws {
        let json = """
        {"id":"\(UUID().uuidString)","chatID":"\(chat.uuidString)","path":"/tmp/p/plan.md",
         "title":"Plan","origin":"holographic","revision":2}
        """
        let row = try JSONDecoder().decode(AppArtifact.self, from: Data(json.utf8))

        XCTAssertEqual(row.origin, .fileWrite)
        XCTAssertEqual(row.revision, 2)
    }

    func testAnArtifactWrittenByAnOlderBuildDecodesWithoutItsNewerKeys() throws {
        let json = """
        {"chatID":"\(chat.uuidString)","path":"/tmp/p/plan.md","origin":"plan"}
        """
        let row = try JSONDecoder().decode(AppArtifact.self, from: Data(json.utf8))

        XCTAssertEqual(row.origin, .plan)
        XCTAssertEqual(row.revision, 1)
        XCTAssertNil(row.lastKnownByteSize)
    }
}
