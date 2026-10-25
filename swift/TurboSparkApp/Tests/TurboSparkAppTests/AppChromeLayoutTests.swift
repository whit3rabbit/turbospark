import XCTest

@testable import TurboSparkApp

/// The right column's precedence and the window floor it implies.
///
/// **`minimumWindowWidth` had no test at all before this file.** It has one
/// caller, it is arithmetic over five constants, and it is the thing that
/// decides whether a newly-opened panel is drawn at its declared width or
/// squeezed -- which is only visible in a screenshot, i.e. the class of defect
/// `swift/CLAUDE.md` Gotcha 18 records for the composer.
final class AppChromeLayoutTests: XCTestCase {
    private let artifactID = UUID()
    private let attachmentID = UUID()
    private let htmlPreviewID = UUID()

    // MARK: - Precedence

    func testTheArtifactPanelClaimsTheRightColumnAboveThePreviewAndTheInspector() {
        let claimant = AppRightColumnClaimant.resolve(
            openArtifactID: artifactID,
            previewAttachmentID: attachmentID,
            isInspectorVisible: true)

        XCTAssertEqual(claimant, .artifact(artifactID))
    }

    func testThePreviewClaimsItAboveTheInspector() {
        let claimant = AppRightColumnClaimant.resolve(
            openArtifactID: nil,
            previewAttachmentID: attachmentID,
            isInspectorVisible: true)

        XCTAssertEqual(claimant, .filePreview(attachmentID))
    }

    func testTheInspectorClaimsItWhenNothingElseDoes() {
        XCTAssertEqual(
            AppRightColumnClaimant.resolve(
                openArtifactID: nil, previewAttachmentID: nil, isInspectorVisible: true),
            .inspector)
    }

    func testNobodyClaimsAnEmptyRightColumn() {
        let claimant = AppRightColumnClaimant.resolve(
            openArtifactID: nil, previewAttachmentID: nil, isInspectorVisible: false)

        XCTAssertEqual(claimant, .none)
        XCTAssertFalse(claimant.isVisible)
    }

    func testOnlyTheArtifactClaimantReportsItselfAsOne() {
        // `.toggleInspector` reads this to know what to close FIRST.
        XCTAssertTrue(AppRightColumnClaimant.artifact(artifactID).isArtifact)
        XCTAssertFalse(AppRightColumnClaimant.filePreview(attachmentID).isArtifact)
        XCTAssertFalse(AppRightColumnClaimant.inspector.isArtifact)
        XCTAssertFalse(AppRightColumnClaimant.none.isArtifact)
    }

    // MARK: - The inline html preview claimant

    func testAnInlinePreviewClaimsTheColumnAboveTheFilePreviewAndTheInspector() {
        let claimant = AppRightColumnClaimant.resolve(
            openArtifactID: nil,
            htmlPreviewID: htmlPreviewID,
            previewAttachmentID: attachmentID,
            isInspectorVisible: true)

        XCTAssertEqual(claimant, .htmlPreview(htmlPreviewID))
    }

    func testAnAutoOpenedArtifactStillOutranksAnInlinePreview() {
        // The ladder is the belt: the setters keep an artifact and a
        // preview from ever being set together, and the order that does
        // fire keeps the turn-driven panel from being displaced by a stale
        // click.
        let claimant = AppRightColumnClaimant.resolve(
            openArtifactID: artifactID,
            htmlPreviewID: htmlPreviewID,
            previewAttachmentID: nil,
            isInspectorVisible: false)

        XCTAssertEqual(claimant, .artifact(artifactID))
    }

    func testAnInlinePreviewIsAPreviewPaneTheInspectorShortcutMustClose() {
        // The shortcut closes what the user can see, whichever preview kind
        // it is; `.htmlPreview` joining the ladder must not drop out of it.
        XCTAssertTrue(AppRightColumnClaimant.htmlPreview(htmlPreviewID).isPreviewPane)
        XCTAssertTrue(AppRightColumnClaimant.artifact(artifactID).isPreviewPane)
        XCTAssertTrue(AppRightColumnClaimant.filePreview(attachmentID).isPreviewPane)
        XCTAssertFalse(AppRightColumnClaimant.inspector.isPreviewPane)
        XCTAssertFalse(AppRightColumnClaimant.none.isPreviewPane)
    }

    func testTheInlinePreviewTakesTheArtifactPanelWidth() {
        // Both web-backed claims are the same panel at the same width; a
        // different width would mean the two ladders disagree about what a
        // preview is.
        XCTAssertEqual(
            AppChromeLayout.rightColumnWidth(.htmlPreview(htmlPreviewID)),
            AppChromeLayout.artifactPanelWidth)
    }

    // MARK: - Widths

    func testTheMinimumWidthGrowsByExactlyTheArtifactPanelWhenItIsTheClaimant() {
        let bare = AppChromeLayout.minimumWindowWidth(
            isPanelVisible: true, rightColumn: .none)
        let withPanel = AppChromeLayout.minimumWindowWidth(
            isPanelVisible: true, rightColumn: .artifact(artifactID))

        // EQUALS, not "is larger": a wrong arm reusing `inspectorWidth` is
        // also larger, and the whole reason the panel is 420 rather than 320
        // is that this assertion can then tell the two apart.
        XCTAssertEqual(
            withPanel - bare,
            AppChromeLayout.artifactPanelWidth + AppChromeLayout.dividerWidth)
    }

    func testTheRailIsCountedInEveryStateAndThePanelOnlyWhenShown() {
        // ABSOLUTE, not a delta between two `minimumWindowWidth` calls: a
        // constant added to or dropped from both arms of a subtraction
        // cancels, so a relative test cannot see the rail's width going
        // missing. The rail is on screen in every section and state, so both
        // floors carry it; only the panel comes and goes.
        XCTAssertEqual(
            AppChromeLayout.minimumWindowWidth(isPanelVisible: true, rightColumn: .none),
            AppChromeLayout.navigationRailWidth
                + AppChromeLayout.sidebarPanelWidth
                + AppChromeLayout.dividerWidth
                + AppChromeLayout.primaryMinimumWidth)

        XCTAssertEqual(
            AppChromeLayout.minimumWindowWidth(isPanelVisible: false, rightColumn: .none),
            AppChromeLayout.navigationRailWidth
                + AppChromeLayout.dividerWidth
                + AppChromeLayout.primaryMinimumWidth)
    }

    func testLeftChromeIsTheRailPlusThePanelWhenShown() {
        XCTAssertEqual(
            AppChromeLayout.leftChromeWidth(isPanelVisible: false),
            AppChromeLayout.navigationRailWidth)
        XCTAssertEqual(
            AppChromeLayout.leftChromeWidth(isPanelVisible: true),
            AppChromeLayout.navigationRailWidth + AppChromeLayout.sidebarPanelWidth)
    }

    func testExactlyTheSectionsWithPanelContentHaveASidebarPanel() {
        // The window floor and the sidebar both follow this, so a section
        // listed here wrongly reserves (or drops) 260pt of width. Files,
        // Discover and Humanize have nothing to list beside the rail.
        let withPanel = AppModel.AppNavigationSection.allCases.filter(\.hasSidebarPanel)
        XCTAssertEqual(Set(withPanel), [.chat, .audio, .images, .modelManager, .server])
    }

    func testEveryPanelButChatsHasARailMenuForWhenItIsHidden() {
        // Hiding the panel must not strand a section's modes (Gotcha 17,
        // "navigation never hides"): each mode-listing panel has a rail menu.
        // Chat's panel is its conversation list, which the pane does not need.
        let withMenu = AppModel.AppNavigationSection.allCases
            .filter(NavigationRailView.hasDestinationMenu)
        XCTAssertEqual(Set(withMenu), [.audio, .images, .modelManager, .server])
    }

    func testTheArtifactPanelIsWiderThanTheInspector() {
        XCTAssertGreaterThan(
            AppChromeLayout.artifactPanelWidth,
            AppChromeLayout.inspectorWidth,
            "the panel renders prose; a fenced code block wraps at the inspector's width")
    }

    func testEachClaimantTakesItsOwnWidth() {
        XCTAssertEqual(AppChromeLayout.rightColumnWidth(.none), 0)
        XCTAssertEqual(
            AppChromeLayout.rightColumnWidth(.artifact(artifactID)),
            AppChromeLayout.artifactPanelWidth)
        XCTAssertEqual(
            AppChromeLayout.rightColumnWidth(.filePreview(attachmentID)),
            AppChromeLayout.inspectorWidth)
        XCTAssertEqual(
            AppChromeLayout.rightColumnWidth(.inspector, isExpandedWorktree: true),
            AppChromeLayout.expandedInspectorWidth)
    }
}
