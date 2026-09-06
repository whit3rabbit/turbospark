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

    // MARK: - Widths

    func testTheMinimumWidthGrowsByExactlyTheArtifactPanelWhenItIsTheClaimant() {
        let bare = AppChromeLayout.minimumWindowWidth(
            isChatSidebarVisible: true, rightColumn: .none)
        let withPanel = AppChromeLayout.minimumWindowWidth(
            isChatSidebarVisible: true, rightColumn: .artifact(artifactID))

        // EQUALS, not "is larger": a wrong arm reusing `inspectorWidth` is
        // also larger, and the whole reason the panel is 420 rather than 320
        // is that this assertion can then tell the two apart.
        XCTAssertEqual(
            withPanel - bare,
            AppChromeLayout.artifactPanelWidth + AppChromeLayout.dividerWidth)
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

    func testTheNewMinimumAgreesWithTheOldOneOnTheCasesTheyShare() {
        // The Bool-shaped overload is still live at its one call site until
        // RootView moves; the two must not disagree about an inspector.
        XCTAssertEqual(
            AppChromeLayout.minimumWindowWidth(
                isChatSidebarVisible: true, rightColumn: .inspector),
            AppChromeLayout.minimumWindowWidth(
                isChatSidebarVisible: true, isInspectorVisible: true))
        XCTAssertEqual(
            AppChromeLayout.minimumWindowWidth(
                isChatSidebarVisible: false, rightColumn: .none),
            AppChromeLayout.minimumWindowWidth(
                isChatSidebarVisible: false, isInspectorVisible: false))
    }
}
