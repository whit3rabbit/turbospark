import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The window's single left column, in one of two presentations.
///
/// **This replaced TWO separate bands**, an always-visible 52pt icon rail and
/// a 260pt chat sidebar that appeared beside it, which read as two vertical
/// bars stacked against each other. They are now one surface: expanded, the
/// sections are labeled rows above the chat list; collapsed, they fall back to
/// the icon rail. `NavigationRailView` is unchanged and IS the collapsed
/// presentation.
///
/// ## Why collapsing does not hide it
///
/// `NavigationRailView`'s doc records the constraint the old split existed to
/// satisfy: sections lived in the rail so the chat sidebar could be hidden
/// without stranding navigation. Folding navigation INTO a hideable sidebar
/// would strand it twice over -- once on Cmd+B, and once in the four sections
/// that never showed a chat list at all.
///
/// So the toggle no longer hides anything. It switches between the two
/// presentations of the same column, and the collapsed one is exactly the rail
/// that used to be permanent. Navigation is reachable in every state, the
/// transcript still gets its width back, and the constraint is met by the
/// COLUMN rather than by having two of them.
///
/// ## What moved out of `ChatSidebarView`
///
/// The profile/settings footer. It is sidebar chrome rather than chat chrome:
/// it has to be present in Files and Server too, where there is no chat list
/// to hang it off. `ChatSidebarView` is now the chat list alone and this view
/// owns the footer, which is also why the `AppLanguage` binding lives here.
@MainActor
struct AppSidebarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let isExpanded: Bool

    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    var body: some View {
        Group {
            if isExpanded {
                expanded
            } else {
                NavigationRailView(model: model)
            }
        }
        .frame(width: AppChromeLayout.sidebarColumnWidth(isExpanded: isExpanded))
        .frame(maxHeight: .infinity)
        .background(
            isExpanded
                ? TurboSparkTheme.sidebarBackgroundColor(
                    isDark: theme.isDark, reduceTransparency: theme.reduceTransparency)
                : TurboSparkTheme.railBackgroundColor(
                    isDark: theme.isDark, reduceTransparency: theme.reduceTransparency))
        .clipped()
    }

    private var expanded: some View {
        VStack(spacing: 0) {
            SidebarSectionNavView(model: model)

            Divider()
                .padding(.top, 8)

            // The chat list is only meaningful beside a conversation. In every
            // other section the nav rows and the footer are the whole sidebar,
            // and the Spacer is what keeps the footer on the floor.
            if model.activeSection == .chat {
                ChatSidebarView(model: model)
            } else {
                Spacer(minLength: 0)
            }

            ChatSidebarFooterView(
                model: model,
                // "local chats stored" is a machine-wide figure, so it counts
                // every persisted chat regardless of the selected project or
                // the archive filter the list above applies.
                chatCount: model.chats.filter { !$0.isGhost }.count,
                languageRawValue: $languageRawValue)
        }
        // Starts below the top bar's height so the first row clears the
        // traffic lights, which a `.hiddenTitleBar` window draws over this
        // column. Same reason and same figure as the rail's own inset.
        .padding(.top, AppChromeLayout.topBarHeight + 6)
    }
}
