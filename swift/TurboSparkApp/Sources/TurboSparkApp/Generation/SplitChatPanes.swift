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
        .background(TurboSparkTheme.pageBackgroundColor)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Split pane: \(chat?.title ?? "chat")")
    }

    private var header: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(isChatRunning ? TurboSparkTheme.accentColor : Color.primary.opacity(0.2))
                .frame(width: 6, height: 6)
                .accessibilityHidden(true)
            Text(chat?.title ?? "Chat")
                .themedFont(.small, weight: .semibold)
                .lineLimit(1)
                .truncationMode(.tail)
                .help(chat?.title ?? "")
            Spacer(minLength: 8)
            Button {
                model.promoteSplitPane(chatID: chatID)
            } label: {
                Image(systemName: "arrow.up.left.to.line")
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .disabled(model.isRunning)
            .help("Make this the main conversation")
            .accessibilityLabel("Promote this chat to the main conversation")
            Button {
                model.closeSplitPane(chatID: chatID)
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .help("Close split pane")
            .accessibilityLabel("Close split pane")
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
                .font(theme.ui(points: 18))
                .foregroundStyle(.tertiary)
            Text("No messages yet", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// A read-only transcript row for a split pane: user bubble, assistant
/// markdown, or a one-line tool-call summary. Tool RESULTS collapse to a
/// status line here on purpose -- a pane is for following along, and the
/// full cards are one promote-click away.
private struct SplitPaneMessageRow: View {
    let message: AppChatMessage

    var body: some View {
        switch message.role {
        case .user:
            HStack(alignment: .top, spacing: 0) {
                Spacer(minLength: 32)
                Text(message.content)
                    .themedFont(.small)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(nsColor: .controlBackgroundColor).opacity(0.85))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 6) {
                if !message.content.isEmpty {
                    ChatMessageMarkdownView(message.content)
                }
                ForEach(message.toolCalls) { call in
                    HStack(spacing: 5) {
                        Image(systemName: "wrench.and.screwdriver")
                            .themedFont(.tiny)
                            .foregroundStyle(TurboSparkTheme.accentColor)
                            .accessibilityHidden(true)
                        Text(call.name)
                            .themedCode(.tiny)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        default:
            EmptyView()
        }
    }
}
