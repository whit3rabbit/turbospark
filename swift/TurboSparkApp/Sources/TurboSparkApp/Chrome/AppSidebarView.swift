import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The window's left chrome: one unified labeled sidebar when expanded, or
/// the compact icon rail when collapsed.
///
/// Showing one column instead of a permanent rail beside a panel reclaims
/// 52pt of horizontal space and pairs every navigation icon with a readable
/// label. Collapsing the sidebar drops cleanly to `NavigationRailView` so
/// top-level navigation is never stranded.
///
/// `isPanelVisible` comes from `RootView`, which feeds the same value to the
/// window minimum. Do not recompute it here from the raw preference.
@MainActor
struct AppSidebarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let isPanelVisible: Bool
    /// The user's panel preference, which the collapsed rail's menus read.
    let isPanelPreferred: Bool

    var body: some View {
        Group {
            if isPanelVisible {
                SidebarPanelSurface(model: model)
            } else {
                NavigationRailView(
                    model: model, isPanelVisible: false, isPanelPreferred: isPanelPreferred)
            }
        }
        .frame(width: AppChromeLayout.leftChromeWidth(isPanelVisible: isPanelVisible))
        .frame(maxHeight: .infinity)
        .font(theme.ui(.small))
        .foregroundStyle(.appText)
        .clipped()
    }
}

/// The unified expanded sidebar: labeled workspace navigation, the active
/// section's content (chats, projects, or sub-modes), and the profile/settings
/// footer. Both the docked sidebar and the OC-06 hover peek render this view.
@MainActor
struct SidebarPanelSurface: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    /// The docked panel runs up behind the top bar's height; the peek starts
    /// below the top bar so the toggle stays uncovered.
    var clearsTopBar = true

    var body: some View {
        VStack(spacing: 0) {
            if model.activeSection == .chat {
                ChatSidebarView(model: model)
            } else {
                SidebarNavigationList(model: model)
                if model.activeSection.hasSidebarPanel {
                    SidebarDivider()
                    sectionPanel
                } else {
                    Spacer(minLength: 0)
                }
            }

            SidebarFooterView(model: model)
        }
        .font(theme.ui(.small))
        .foregroundStyle(.appText)
        .frame(width: AppChromeLayout.sidebarPanelWidth)
        // Starts below the top bar's height so the first row
        // clears the traffic lights, which a `.hiddenTitleBar`
        // window draws over the sidebar's first 44pt.
        .padding(.top, clearsTopBar ? AppChromeLayout.topBarHeight + 4 : 4)
        .frame(maxHeight: .infinity)
        .background(TurboSparkTheme.sidebarBackgroundColor(
            isDark: theme.isDark, reduceTransparency: theme.reduceTransparency))
    }

    @ViewBuilder
    private var sectionPanel: some View {
        switch model.activeSection {
        case .audio:
            AudioSidebarPanel(model: model, controller: model.audioWorkspace)
        case .images:
            ImagesSidebarPanel(model: model)
        case .modelManager, .modelHub:
            InstalledSidebarPanel(model: model)
        case .server:
            APISidebarPanel(model: model)
        default:
            EmptyView()
        }
    }
}
