import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Ghost Mode: start a temporary chat, or end the one being viewed.
///
/// It sat beside the profile menu in the old sidebar footer. That footer is
/// gone (the profile menu lives in the rail), and a temporary chat is a chat
/// action, so it now sits beside "New chat" in the chat panel.
@MainActor
struct GhostChatButton: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var confirmingEndGhost = false

    var body: some View {
        Button {
            if model.isInGhostChat {
                if model.ghostChatHasContent {
                    confirmingEndGhost = true
                } else {
                    model.endGhostChat()
                }
            } else {
                model.enterGhostChat()
            }
        } label: {
            GhostGlyph(size: 14 * theme.sidebarScale, color: model.isInGhostChat ? theme.accent : theme.secondaryText)
                .frame(width: theme.sidebarRowHeight, height: theme.sidebarRowHeight)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // `enterGhostChat()`/`endGhostChat()` both no-op under exactly this
        // guard, matching the File menu's "New Temporary Chat".
        .disabled(model.generating || model.submitting || model.pendingToolCall != nil)
        .help((model.isInGhostChat ? Text("Discard Temporary Chat", bundle: .module) : Text("New Temporary Chat", bundle: .module)))
        .accessibilityLabel((model.isInGhostChat ? Text("Discard Temporary Chat", bundle: .module) : Text("New Temporary Chat", bundle: .module)))
        .accessibilityAddTraits(model.isInGhostChat ? [.isButton, .isSelected] : .isButton)
        .confirmationDialog(
            Text("Discard Temporary Chat", bundle: .module), isPresented: $confirmingEndGhost,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                model.endGhostChat()
            } label: { Text("Discard Temporary Chat", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This conversation exists only in memory and cannot be recovered once discarded.", bundle: .module)
        }
    }
}
