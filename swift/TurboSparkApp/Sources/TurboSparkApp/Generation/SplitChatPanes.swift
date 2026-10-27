import SwiftUI

/// The split-view pane column (qwen-code `SplitView` parity): the secondary
/// chats beside the main conversation, separated by a resizable divider.
/// Panes are live follow surfaces -- a running turn in a pane's chat streams
/// here as it lands -- and read-only; the header's promote button makes a
/// pane's chat the main conversation.
struct SplitChatPanesView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        let panes = model.splitPaneChats
        if !panes.isEmpty {
            HSplitView {
                ForEach(panes) { chat in
                    SplitChatPaneView(model: model, chatID: chat.id)
                        .frame(minWidth: 280, idealWidth: 380, maxWidth: 620)
                        .frame(maxHeight: .infinity)
                }
            }
            .frame(maxHeight: .infinity)
        }
    }
}

/// One split pane: title bar, then the chat's transcript rendered read-only.
/// The row renderer is a compact sibling of the main transcript's
/// `MessageRowView`, not a reuse of it -- that view is wired to the model's
/// editing, branching and speaking state for the SELECTED chat, and a pane
/// bound to those seams would act on whichever chat is main, not its own.
private struct SplitChatPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let chatID: UUID

    private var chat: AppChat? {
        model.chats.first { $0.id == chatID }
    }

    private var isChatRunning: Bool {
        model.isChatRunning(chatID: chatID)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if let chat, !chat.messages.isEmpty {
                transcript(chat)
            } else {
                emptyState
            }
        }
        .background(.appPage)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(chat.map { Text("Split pane: \($0.title)", bundle: .module) } ?? Text("Split pane", bundle: .module))
    }

    private var header: some View {
        AppAdaptiveHeader {
            HStack(spacing: 6) {
                Circle()
                    .fill(isChatRunning ? AnyShapeStyle(theme.accent) : AnyShapeStyle(theme.border))
                    .frame(width: 6, height: 6)
                    .accessibilityHidden(true)
                (chat.map { Text($0.title) } ?? Text("Chat", bundle: .module))
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(Text(verbatim: chat?.title ?? ""))
            }
        } actions: {
            HStack(spacing: 8) {
                Button {
                    model.promoteSplitPane(chatID: chatID)
                } label: {
                    Image(systemName: "arrow.up.left.to.line")
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .disabled(model.isRunning)
                .help(model.isRunning
                    ? Text("Cannot switch chats while generation is running", bundle: .module)
                    : Text("Make this the main conversation", bundle: .module))
                .accessibilityLabel(Text("Promote this chat to the main conversation", bundle: .module))

                Button {
                    model.closeSplitPane(chatID: chatID)
                } label: {
                    Image(systemName: "xmark")
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .help(Text("Close split pane", bundle: .module))
                .accessibilityLabel(Text("Close split pane", bundle: .module))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private func transcript(_ chat: AppChat) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(chat.messages) { message in
                        SplitPaneMessageRow(message: message)
                    }
                    Color.clear.frame(height: 1).id("pane-bottom-\(chatID.uuidString)")
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 14)
            }
            .onAppear {
                proxy.scrollTo("pane-bottom-\(chatID.uuidString)", anchor: .bottom)
            }
            .onChange(of: chat.messages.count) { _, _ in
                proxy.scrollTo("pane-bottom-\(chatID.uuidString)", anchor: .bottom)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(theme.ui(.title3))
                .foregroundStyle(.appSecondary)
            Text("No messages yet", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// A read-only transcript row for a split pane: user bubble, assistant
/// markdown, or a one-line tool-call summary. Tool RESULTS collapse to a
/// status line here on purpose -- a pane is for following along, and the
/// full cards are one promote-click away.
private struct SplitPaneMessageRow: View {
    @Environment(\.appTheme) private var theme
    let message: AppChatMessage

    var body: some View {
        switch message.role {
        case .user:
            HStack(alignment: .top, spacing: 0) {
                Spacer(minLength: 32)
                // Head-only preview (the marker inside states the hidden
                // count): a pane is read-only follow-along, and the full
                // text is one promote-click away in the main transcript.
                Text(
                    MessageContentPreview.make(message.content, includesTail: false)?.visible
                        ?? message.content
                )
                .themedFont(.small)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.appSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                        .stroke(theme.border, lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                if !message.content.isEmpty {
                    ChatMessageMarkdownView(
                        MessageContentPreview.make(message.content, includesTail: false)?
                            .visible ?? message.content
                    )
                }
                ForEach(message.toolCalls) { call in
                    HStack(spacing: 5) {
                        Image(systemName: "wrench.and.screwdriver")
                            .themedFont(.tiny)
                            .foregroundStyle(.appAccent)
                            .accessibilityHidden(true)
                        Text(call.name)
                            .themedCode(.tiny)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                    }
                }
            }
        default:
            EmptyView()
        }
    }
}
