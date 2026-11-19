import SwiftUI

/// The session tab strip (OC-04): one tab per open chat, a draft glyph on a
/// chat that has no messages yet, a plus button, and a labeled menu that lists
/// every tab and the Recently Closed stack.
///
/// A row of the working-panes band under the top bar, not a fifth band
/// (DESIGN.md section 3), drawn only with two or more tabs open so a window
/// that never opens a second tab is unchanged. Tab titles are never truncated
/// to nothing: the row scrolls, and the "All Tabs" menu names every tab for
/// keyboard and VoiceOver use. Hover is never the only affordance: the close
/// button is dimmed, not removed, and the context menu and Cmd+Shift+W close
/// too.
struct SessionTabStripView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ScaledMetric private var stripHeight: CGFloat = 34

    var body: some View {
        HStack(spacing: 4) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(model.sessionTabs.tabs, id: \.self) { chatID in
                        if let chat = model.chats.first(where: { $0.id == chatID }) {
                            SessionTabItemView(
                                model: model, chat: chat,
                                isSelected: model.sessionTabs.selectedChatID == chatID)
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
            newTabButton
            allTabsMenu
        }
        .padding(.horizontal, 10)
        .frame(minHeight: stripHeight)
        .background(.appSurface)
        .overlay(alignment: .bottom) {
            Rectangle().fill(.appBorder).frame(height: AppChromeLayout.dividerWidth)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Session tabs", bundle: .module))
    }

    private var newTabButton: some View {
        Button {
            model.openNewSessionTab()
        } label: {
            Label { Text("New Tab", bundle: .module) } icon: { Image(systemName: "plus") }
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.plain)
        // The icon-only label drops the text VoiceOver needs; the name is
        // set here so it cannot drift from the tooltip.
        .accessibilityLabel(Text("New Tab", bundle: .module))
        .help(Text("New Tab", bundle: .module))
    }

    private var allTabsMenu: some View {
        Menu {
            ForEach(model.sessionTabs.tabs, id: \.self) { chatID in
                Button {
                    model.activateSessionTab(chatID: chatID)
                } label: {
                    Text(verbatim: model.sessionTabTitle(for: chatID))
                }
            }
            Divider()
            Menu {
                ForEach(model.sessionTabs.closed.reversed(), id: \.chatID) { entry in
                    Button {
                        model.reopenClosedSessionTab(chatID: entry.chatID)
                    } label: {
                        Text(verbatim: model.sessionTabTitle(for: entry.chatID))
                    }
                }
            } label: {
                Text("Recently Closed", bundle: .module)
            }
            .disabled(model.sessionTabs.closed.isEmpty)
        } label: {
            Label { Text("All Tabs", bundle: .module) } icon: { Image(systemName: "chevron.down") }
                .labelStyle(.iconOnly)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        // Named for VoiceOver, like the new-tab button beside it.
        .accessibilityLabel(Text("All Tabs", bundle: .module))
        .help(Text("All Tabs", bundle: .module))
    }
}

/// One tab: select button, status glyph, close button.
private struct SessionTabItemView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let chat: AppChat
    let isSelected: Bool
    @State private var isHovered = false

    private var badge: WorkspaceTabBadge { model.sessionTabBadge(for: chat.id) }
    private var isDraft: Bool { chat.messages.isEmpty }
    private var title: String { model.sessionTabTitle(for: chat.id) }

    var body: some View {
        HStack(spacing: 4) {
            Button {
                model.activateSessionTab(chatID: chat.id)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isDraft ? "square.and.pencil" : "bubble.left")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                        .accessibilityHidden(true)
                    Text(verbatim: title)
                        .font(theme.ui(.small, weight: isSelected ? .semibold : nil))
                        .foregroundStyle(.appText)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    badgeGlyph
                }
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .help(Text("Switch to \(title)", bundle: .module))
            .accessibilityLabel(Text(verbatim: title))
            .accessibilityValue(accessibilityValue)
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            Button {
                model.closeSessionTab(chatID: chat.id)
            } label: {
                Label { Text("Close Tab", bundle: .module) } icon: { Image(systemName: "xmark") }
                    .labelStyle(.iconOnly)
                    .font(theme.ui(.micro, weight: .medium))
                    .foregroundStyle(.appSecondary)
                    // Always laid out so a hover does not shift the title.
                    .opacity(isSelected || isHovered ? 1 : 0.35)
            }
            .buttonStyle(.plain)
            .help(Text("Close Tab", bundle: .module))
            .accessibilityLabel(Text("Close \(title) tab", bundle: .module))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .fill(isSelected ? AnyShapeStyle(.appElevated) : AnyShapeStyle(Color.clear))
        )
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(isSelected ? AnyShapeStyle(.appBorder) : AnyShapeStyle(Color.clear), lineWidth: 1)
        )
        .onHover { isHovered = $0 }
        .tsAnimation(TSMotion.hover, value: isHovered)
        .contextMenu {
            Button {
                model.closeSessionTab(chatID: chat.id)
            } label: {
                Text("Close Tab", bundle: .module)
            }
        }
    }

    @ViewBuilder
    private var badgeGlyph: some View {
        switch badge {
        case .pending:
            Image(systemName: "hand.raised.fill")
                .font(theme.ui(.tiny))
                .foregroundStyle(AppAttention.color)
                .help(Text("Needs Approval", bundle: .module))
                .accessibilityHidden(true)
        case .running:
            Image(systemName: "flame.fill")
                .font(theme.ui(.tiny))
                .foregroundStyle(theme.accent)
                .tsPulse(active: true)
                .help(Text("Generating", bundle: .module))
                .accessibilityHidden(true)
        case .unread:
            Circle()
                .fill(theme.accent)
                .frame(width: 8, height: 8)
                .help(Text("Unread", bundle: .module))
                .accessibilityHidden(true)
        case .none:
            EmptyView()
        }
    }

    private var accessibilityValue: Text {
        switch badge {
        case .pending: return Text("Needs Approval", bundle: .module)
        case .running: return Text("Generating", bundle: .module)
        case .unread: return Text("Unread", bundle: .module)
        case .none: return isDraft ? Text("Draft", bundle: .module) : Text(verbatim: "")
        }
    }
}
