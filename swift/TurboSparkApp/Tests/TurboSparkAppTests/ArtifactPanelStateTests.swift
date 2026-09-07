import XCTest

@testable import TurboSparkApp

/// The artifact panel's state invariants: the three right-column claimants
/// clear each other, a network grant is bound to the exact content, and
/// auto-open fires once per artifact.
///
/// The state lives on `AppModel` but every rule here is decidable without a
/// window, which is the `ServerStatusRows` lesson: the setters' invariant is
/// documented on `AppRightColumnClaimant` as the braces, and this file is
/// the test that keeps them braced.
@MainActor
final class ArtifactPanelStateTests: XCTestCase {
    private var model: AppModel!
    private let chatID = UUID()

    override func setUp() {
        super.setUp()
        model = AppModel()
        model.chats = [AppChat(id: chatID, title: "Panel tests")]
        model.selectedChatID = chatID
    }

    private func register(path: String, origin: AppArtifact.Origin = .fileWrite) -> AppArtifact? {
        let file = ArtifactRegistrar.ProducedFile(
            url: URL(fileURLWithPath: path),
            toolCallID: UUID(),
            toolName: "write_file",
            origin: origin)
        model.registerProducedArtifacts(chatID: chatID, files: [file])
        let standardized = AppArtifact.standardize(path)
        return model.chats[0].artifacts.first { $0.path == standardized }
    }

    // MARK: - Claimant invariants

    func testOpeningAnArtifactClosesEveryOtherPreview() {
        model.showPreview(attachmentID: UUID())
        model.htmlPreview = ArtifactHTMLPreview(title: "fence", html: "<p>hi</p>")
        XCTAssertNotNil(model.previewAttachmentID)
        XCTAssertNotNil(model.htmlPreview)

        let id = register(path: "/tmp/p/page.html")?.id ?? UUID()
        model.openArtifact(id: id)

        XCTAssertEqual(model.openArtifactID, id)
        XCTAssertNil(model.previewAttachmentID)
        XCTAssertNil(model.htmlPreview)
    }

    func testAFilePreviewClickClosesTheOtherPanels() {
        let id = register(path: "/tmp/p/page.html")?.id ?? UUID()
        model.openArtifact(id: id)
        model.htmlPreview = ArtifactHTMLPreview(title: "fence", html: "<p>hi</p>")
        XCTAssertEqual(model.openArtifactID, id)
        XCTAssertNotNil(model.htmlPreview)

        model.showPreview(attachmentID: UUID())

        XCTAssertNotNil(model.previewAttachmentID)
        XCTAssertNil(model.openArtifactID)
        XCTAssertNil(model.htmlPreview)
    }

    func testAnInlinePreviewClosesTheOtherPanels() {
        let id = register(path: "/tmp/p/page.html")?.id ?? UUID()
        model.openArtifact(id: id)
        model.showPreview(attachmentID: UUID())

        model.openHTMLPreview(title: "HTML preview", html: "<h1>hi</h1>")

        XCTAssertNotNil(model.htmlPreview)
        XCTAssertNil(model.openArtifactID)
        XCTAssertNil(model.previewAttachmentID)
    }

    func testDismissingOneClaimantLeavesTheOthersAlone() {
        model.openHTMLPreview(title: "HTML preview", html: "<h1>hi</h1>")
        model.dismissHTMLPreview()
        XCTAssertNil(model.openArtifactID)
        XCTAssertNil(model.previewAttachmentID)

        let id = register(path: "/tmp/p/page.html")?.id ?? UUID()
        model.openArtifact(id: id)
        model.dismissArtifact()
        XCTAssertNil(model.previewAttachmentID)
        XCTAssertNil(model.htmlPreview)
    }

    // MARK: - Inline preview identity

    func testTheSameFencePreviewedTwiceKeepsItsIdentity() {
        model.openHTMLPreview(title: "HTML preview", html: "<h1>hi</h1>")
        let first = model.htmlPreview
        model.openHTMLPreview(title: "HTML preview", html: "<h1>hi</h1>")

        XCTAssertEqual(model.htmlPreview?.id, first?.id)
    }

    func testAnEditedFenceMintsANewPreview() {
        model.openHTMLPreview(title: "HTML preview", html: "<h1>v1</h1>")
        let first = model.htmlPreview
        model.openHTMLPreview(title: "HTML preview", html: "<h1>v2</h1>")

        XCTAssertNotEqual(model.htmlPreview?.id, first?.id)
    }

    // MARK: - Network grants

    func testNetworkIsDeniedByDefaultForBothContentKinds() {
        let artifact = register(path: "/tmp/p/page.html")!
        model.openHTMLPreview(title: "HTML preview", html: "<h1>hi</h1>")

        XCTAssertFalse(model.isNetworkAllowed(for: artifact))
        XCTAssertFalse(model.isNetworkAllowed(for: model.htmlPreview!))
    }

    func testAGrantFollowsTheInlinePreviewItWasGivenTo() {
        model.openHTMLPreview(title: "HTML preview", html: "<h1>v1</h1>")
        model.allowNetworkForCurrentPreview()
        XCTAssertTrue(model.isNetworkAllowed(for: model.htmlPreview!))

        // The grant is keyed on the html hash, so an edited fence re-asks
        // rather than riding in on the older content's permission.
        model.openHTMLPreview(title: "HTML preview", html: "<h1>v2</h1>")
        XCTAssertFalse(model.isNetworkAllowed(for: model.htmlPreview!))
    }

    func testAGrantDoesNotSurviveARewrittenArtifact() {
        let first = register(path: "/tmp/p/page.html")!
        model.openArtifact(id: first.id)
        model.allowNetworkForCurrentPreview()
        XCTAssertTrue(model.isNetworkAllowed(for: first))

        // The rewrite bumps the revision, which moves the contentKey, which
        // is the grant key: the same path with new bytes is a new question.
        let second = register(path: "/tmp/p/page.html")!
        XCTAssertEqual(second.id, first.id, "upsert keeps the id stable")
        XCTAssertGreaterThan(second.revision, first.revision)
        XCTAssertFalse(model.isNetworkAllowed(for: second))
    }

    func testRevokingRemovesTheGrantForTheShownContent() {
        let artifact = register(path: "/tmp/p/page.html")!
        model.openArtifact(id: artifact.id)
        model.allowNetworkForCurrentPreview()
        XCTAssertTrue(model.isNetworkAllowed(for: artifact))

        model.revokeNetworkForCurrentPreview()
        XCTAssertFalse(model.isNetworkAllowed(for: artifact))
    }

    // MARK: - Auto-open

    func testAProducedHTMLArtifactAutoOpensExactlyOnce() {
        let first = register(path: "/tmp/p/page.html")

        XCTAssertEqual(model.openArtifactID, first?.id, "the first write pops the panel")

        model.dismissArtifact()
        let second = register(path: "/tmp/p/page.html")

        XCTAssertNil(model.openArtifactID, "a rewrite must not re-pop a panel the user closed")
        XCTAssertEqual(second?.id, first?.id)
    }

    func testAutoOpenIgnoresEverythingThatIsNotHTML() {
        let markdown = register(path: "/tmp/p/notes.md")
        XCTAssertNotNil(markdown)
        XCTAssertNil(model.openArtifactID)

        let source = register(path: "/tmp/p/lib.rs")
        XCTAssertNotNil(source)
        XCTAssertNil(model.openArtifactID)
    }

    func testAutoOpenNeverFiresOutsideTheProducingChat() {
        // A chat produced the file; the user then deleted it or switched.
        // The hook passes the TURN's chat id, and an unknown chat registers
        // and opens nothing rather than falling back to the selection.
        let file = ArtifactRegistrar.ProducedFile(
            url: URL(fileURLWithPath: "/tmp/p/page.html"),
            toolCallID: UUID(),
            toolName: "write_file",
            origin: .fileWrite)
        model.registerProducedArtifacts(chatID: UUID(), files: [file])

        XCTAssertNil(model.openArtifactID)
        XCTAssertTrue(model.chats[0].artifacts.isEmpty)
    }

    // MARK: - Lookup

    func testArtifactLookupFindsARowRegardlessOfSelection() {
        let artifact = register(path: "/tmp/p/page.html")!
        let otherChat = AppChat(id: UUID(), title: "somewhere else")
        model.chats.append(otherChat)
        model.selectedChatID = otherChat.id

        XCTAssertEqual(model.artifact(id: artifact.id)?.id, artifact.id)
    }

    func testArtifactLookupMissesReturnNothing() {
        XCTAssertNil(model.artifact(id: UUID()))
    }
}
