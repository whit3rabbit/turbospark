import Foundation

/// Who owns the right column right now.
///
/// **The right column is ONE SLOT** (`swift/CLAUDE.md` Gotcha 17), and this
/// type is what a third claimant JOINS rather than becoming a fourth child of
/// `RootView`'s working-area `HStack`. Two claimants were arbitrated by an
/// `if/else if` chain reading two of `AppModel`'s properties directly; a
/// third makes that chain a decision worth naming, and naming it is what lets
/// a test pin the order at all.
public enum AppRightColumnClaimant: Equatable, Sendable {
    case none
    case artifact(UUID)
    /// An inline fence preview (`AppModel.htmlPreview`): bytes held in
    /// memory, opened by an explicit click on a "Preview" button.
    case htmlPreview(UUID)
    case filePreview(UUID)
    case inspector

    /// Precedence: artifact, then the two explicit-click previews, then
    /// inspector.
    ///
    /// **This ladder is the belt and not the braces.** The braces are an
    /// invariant in the SETTERS -- `openArtifact(id:)` clears the preview and
    /// `showPreview(attachmentID:)` clears the artifact -- so at most one of
    /// the first three is ever set and the order between them almost never
    /// fires. That is deliberate, and it is Gotcha 26's rule ("there is no
    /// second flag that could disagree") applied here: the ladder exists so
    /// the UI still resolves to something sane if a future writer forgets
    /// the invariant, and so the whole question is testable without a
    /// window.
    ///
    /// Artifact above the previews for the case where it does fire: an
    /// artifact panel opens as a RESULT of the turn the user is watching,
    /// while a preview is opened by an explicit click, and a click has
    /// already cleared the artifact by the time this is read.
    public static func resolve(
        openArtifactID: UUID?,
        htmlPreviewID: UUID? = nil,
        previewAttachmentID: UUID?,
        isInspectorVisible: Bool
    ) -> AppRightColumnClaimant {
        if let openArtifactID { return .artifact(openArtifactID) }
        if let htmlPreviewID { return .htmlPreview(htmlPreviewID) }
        if let previewAttachmentID { return .filePreview(previewAttachmentID) }
        if isInspectorVisible { return .inspector }
        return .none
    }

    /// Whether anything at all occupies the column.
    public var isVisible: Bool { self != .none }

    /// Whether the artifact panel owns it, which is what `.toggleInspector`
    /// has to close FIRST (a shortcut that toggles a pane hidden behind
    /// another one reads as a broken shortcut).
    public var isArtifact: Bool {
        if case .artifact = self { return true }
        return false
    }

    /// Whether ANY web/file preview the inspector shortcut must close before
    /// toggling. Both previews are explicit clicks; the shortcut closes what
    /// the user can see, whichever kind it is.
    public var isPreviewPane: Bool {
        switch self {
        case .artifact, .htmlPreview, .filePreview: return true
        case .none, .inspector: return false
        }
    }
}

extension AppChromeLayout {
    /// Wider than the inspector's 320 because this panel renders PROSE.
    ///
    /// A fenced code block wraps at 320 minus padding, which is the one thing
    /// a markdown reader must not do. It is also what makes the width test
    /// able to fail at all: at 320 == 320 the "the artifact panel widens the
    /// window" assertion could not distinguish a correct arm from one that
    /// reused `inspectorWidth`, which is Gotcha 26's only-loopback-cases
    /// lesson applied to a layout constant.
    public static let artifactPanelWidth: CGFloat = 420

    /// The width the right column takes for a given claimant.
    public static func rightColumnWidth(
        _ claimant: AppRightColumnClaimant,
        isExpandedWorktree: Bool = false
    ) -> CGFloat {
        switch claimant {
        case .none: return 0
        case .artifact, .htmlPreview: return artifactPanelWidth
        case .filePreview: return inspectorWidth
        case .inspector: return inspectorWidth(isExpanded: isExpandedWorktree)
        }
    }

    /// Minimum window width with the right column resolved rather than
    /// reduced to a Bool.
    ///
    /// The older `minimumWindowWidth(isChatSidebarVisible:isInspectorVisible:)`
    /// folded the preview into the inspector's term, which was correct while
    /// the two were the same width and silently wrong for a claimant that is
    /// not.
    /// The label says EXPANDED rather than visible, and the arithmetic adds
    /// ONE left column rather than two.
    ///
    /// Before the merge this summed a permanent 52pt rail and a conditional
    /// 260pt sidebar, so an expanded window needed 313pt of left chrome. The
    /// two are one column in two presentations now, so the sidebar's width
    /// REPLACES the rail's instead of stacking on it, and the expanded
    /// minimum drops by exactly `navigationRailWidth + dividerWidth`.
    public static func minimumWindowWidth(
        isSidebarExpanded: Bool,
        rightColumn: AppRightColumnClaimant,
        isExpandedWorktree: Bool = false
    ) -> CGFloat {
        let columnWidth = rightColumnWidth(rightColumn, isExpandedWorktree: isExpandedWorktree)
        return sidebarColumnWidth(isExpanded: isSidebarExpanded)
            + dividerWidth
            + primaryMinimumWidth
            + (rightColumn.isVisible ? columnWidth + dividerWidth : 0)
    }
}
