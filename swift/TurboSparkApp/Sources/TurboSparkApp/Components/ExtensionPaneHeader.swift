import SwiftUI

/// Layout constants shared by the extension panes (MCP Servers, Skills,
/// Agents, Plugins, Hooks, Scheduled Tasks).
enum ExtensionPaneLayout {
    /// Around content that sits under an `ExtensionPaneHeader`. The header's
    /// own padding and hairline already separate it from the shell title, so
    /// the top inset is smaller than `SettingsLayout.contentInsets`.
    static let contentInsets = EdgeInsets(
        top: 18, leading: SettingsLayout.horizontalInset,
        bottom: SettingsLayout.contentInsets.bottom, trailing: SettingsLayout.horizontalInset)
    /// The list column of a list/detail pane. Adjustable rather than fixed:
    /// a fixed 300pt column left long agent names truncated on a wide window
    /// and starved the detail column on a narrow one.
    static let listMinWidth: CGFloat = 220
    static let listIdealWidth: CGFloat = 300
    static let listMaxWidth: CGFloat = 440
    static let detailMinWidth: CGFloat = 320
}

/// The header every extension pane opens with, so the five read as one
/// family: an optional purpose line (with its documentation link), then one
/// adaptive row of `[scope][search] ... [secondary][primary]`.
///
/// The 2026-10-11 audit found each pane laying this row out by hand: a fixed
/// 320pt picker here, a 200pt search field there, and three or four actions
/// that truncated at the Settings window's minimum width. The row is an
/// `AppAdaptiveHeader`, so it wraps to two rows and then to `compactActions`
/// (primary visible, the rest behind one menu) instead of clipping. The
/// Settings shell already shows the pane title, so there is none here.
@MainActor
struct ExtensionPaneHeader<Intro: View, Filters: View, Actions: View, CompactActions: View>: View {
    @ViewBuilder let intro: () -> Intro
    @ViewBuilder let filters: () -> Filters
    @ViewBuilder let actions: () -> Actions
    @ViewBuilder let compactActions: () -> CompactActions

    init(
        @ViewBuilder intro: @escaping () -> Intro,
        @ViewBuilder filters: @escaping () -> Filters,
        @ViewBuilder actions: @escaping () -> Actions,
        @ViewBuilder compactActions: @escaping () -> CompactActions
    ) {
        self.intro = intro
        self.filters = filters
        self.actions = actions
        self.compactActions = compactActions
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            intro()
            AppAdaptiveHeader {
                HStack(spacing: AppMetrics.actionSpacing) { filters() }
            } actions: {
                actions()
            } compactActions: {
                compactActions()
            }
        }
        .padding(.horizontal, SettingsLayout.horizontalInset)
        .padding(.vertical, SettingsLayout.headerBarVerticalPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appPage)
        .overlay(alignment: .bottom) {
            Rectangle().fill(.appBorder).frame(height: 1)
        }
    }
}

extension ExtensionPaneHeader where CompactActions == Actions {
    /// A header whose actions are already compact (one menu and the primary).
    init(
        @ViewBuilder intro: @escaping () -> Intro,
        @ViewBuilder filters: @escaping () -> Filters,
        @ViewBuilder actions: @escaping () -> Actions
    ) {
        self.init(intro: intro, filters: filters, actions: actions, compactActions: actions)
    }
}

extension ExtensionPaneHeader where Intro == EmptyView, Actions == EmptyView, CompactActions == EmptyView {
    /// A header that only carries filters (the project-scope MCP view, whose
    /// actions live in the embedded project manager).
    init(@ViewBuilder filters: @escaping () -> Filters) {
        self.init(intro: { EmptyView() }, filters: filters, actions: { EmptyView() }, compactActions: { EmptyView() })
    }
}

/// The purpose line under the shell title, with an optional documentation
/// link. A real `Link` rather than a link-styled button: it is announced as a
/// link, opens in the browser, and shows the URL on hover like any link.
struct ExtensionPaneIntro: View {
    let text: Text
    var documentation: URL?
    var documentationHint: Text?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            text
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if let documentation {
                Link(destination: documentation) {
                    Text("Documentation", bundle: .module)
                }
                .appLinkTint()
                .themedFont(.small)
                .fixedSize()
                .help(documentationHint ?? Text("Documentation", bundle: .module))
                .accessibilityHint(documentationHint ?? Text("Documentation", bundle: .module))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The header's labeled overflow for secondary sources (other agents'
/// folders, a marketplace, a local folder): one control instead of a row of
/// bordered buttons, which is what crowded the Skills and MCP headers.
struct ExtensionImportMenu<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        Menu {
            content()
        } label: {
            Label { Text("Import", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .appMenuAction()
    }
}

/// The compact layout's overflow: every secondary action behind one labeled
/// menu so the primary action stays visible at any width or text size.
struct ExtensionMoreMenu<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        Menu {
            content()
        } label: {
            Label { Text("More", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
        }
        .appMenuAction()
    }
}

/// The rescan action every extension pane has. Icon-only, so the label is
/// both the VoiceOver name and the tooltip (it used to be a tooltip alone).
struct ExtensionRefreshButton: View {
    let label: Text
    let action: () -> Void

    var body: some View {
        AppIconButton("arrow.clockwise", label: label, action: action)
            .appAction(.secondary)
    }
}

/// A list/detail split whose list column the user can resize, within bounds
/// that keep both columns usable.
struct ExtensionSplitView<ListContent: View, DetailContent: View>: View {
    @ViewBuilder let list: () -> ListContent
    @ViewBuilder let detail: () -> DetailContent

    var body: some View {
        HSplitView {
            list()
                .frame(
                    minWidth: ExtensionPaneLayout.listMinWidth,
                    idealWidth: ExtensionPaneLayout.listIdealWidth,
                    maxWidth: ExtensionPaneLayout.listMaxWidth,
                    maxHeight: .infinity)
                .background(.appSurface)
            detail()
                .frame(minWidth: ExtensionPaneLayout.detailMinWidth, maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

/// What a list shows when a search filters every row out. Distinct from the
/// pane's "nothing installed" state, which offers ways to add something.
struct ExtensionNoMatchesState: View {
    let query: String

    var body: some View {
        AppEmptyState(
            systemImage: "magnifyingglass",
            title: Text("No Results", bundle: .module),
            message: Text("Nothing matches \"\(query)\".", bundle: .module))
    }
}
