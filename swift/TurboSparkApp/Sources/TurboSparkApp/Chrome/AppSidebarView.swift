import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The window's left chrome: the icon rail, always, and beside it the
/// selected section's panel when it has one.
///
/// ## Why this is two columns again
///
/// From 2026-09-09 this was ONE column in two presentations: expanded, the
/// sections were labeled rows stacked above the chat list; collapsed, it was
/// the rail. That fixed an earlier rail-plus-chat-sidebar pair that read as
/// two vertical bars, but the labeled rows cost roughly 300pt of chat history.
/// The pair is back on three conditions, and they are what keeps it from
/// regressing into the old look:
///
/// - **The panel never repeats the section list.** The rail answers "where
///   am I"; the panel shows only what is in that section (the chat list, the
///   Audio pages). The old pair duplicated navigation.
/// - **One surface.** Rail and panel share `sidebarBackgroundColor`, split by
///   a 0.5pt hairline the rail draws, so they read as one surface in two parts.
/// - **Navigation never hides.** The sidebar toggle hides the PANEL; the rail
///   stays in every section and state. Sections without a panel
///   (`AppNavigationSection.hasSidebarPanel`) show the rail alone.
///
/// `isPanelVisible` comes from `RootView`, which feeds the same value to the
/// window minimum. Do not recompute it here from the raw preference.
@MainActor
struct AppSidebarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let isPanelVisible: Bool
    /// The user's panel preference, which the rail's Audio item reads.
    let isPanelPreferred: Bool

    var body: some View {
        HStack(spacing: 0) {
            NavigationRailView(
                model: model, isPanelVisible: isPanelVisible, isPanelPreferred: isPanelPreferred)
            if isPanelVisible {
                panel
                    .frame(width: AppChromeLayout.sidebarPanelWidth)
                    // Starts below the top bar's height so the first row
                    // clears the traffic lights, which a `.hiddenTitleBar`
                    // window draws over the rail and the panel's first 14pt.
                    .padding(.top, AppChromeLayout.topBarHeight + 6)
                    .frame(maxHeight: .infinity)
                    // The rail paints its own background; painting this one
                    // under both would stack two translucent layers there.
                    .background(TurboSparkTheme.sidebarBackgroundColor(
                        isDark: theme.isDark, reduceTransparency: theme.reduceTransparency))
            }
        }
        .frame(width: AppChromeLayout.leftChromeWidth(isPanelVisible: isPanelVisible))
        .frame(maxHeight: .infinity)
        .font(theme.ui(.small))
        .foregroundStyle(.appText)
        .clipped()
    }

    @ViewBuilder
    private var panel: some View {
        switch model.activeSection {
        case .chat:
            ChatSidebarView(model: model)
        case .audio:
            AudioSidebarPanel(model: model, controller: model.audioWorkspace)
        default:
            // Unreachable while `hasSidebarPanel` lists only the two above. A
            // section added there needs its own case here.
            EmptyView()
        }
    }
}
