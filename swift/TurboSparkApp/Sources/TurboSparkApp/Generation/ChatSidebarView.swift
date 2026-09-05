import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct ChatSidebarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    @State private var chatBeingRenamed: AppChat?
    @State private var chatForSystemPrompt: AppChat?
    @State private var renameText = ""
    @State private var chatPendingDeletion: AppChat?
    @State private var showingProjectSettingsSheet = false
    @State private var projectBeingEdited: AppProject?
    @State private var projectForMcpSettings: AppProject?

    var body: some View {
        VStack(spacing: 0) {
            // No app title and no section list here: the rail owns navigation
            // and the top bar owns the model. This pane is conversations only.
            newChatButton
                .padding(.horizontal, 10)
                .padding(.top, 10)
                .padding(.bottom, 8)
            ChatSidebarProjectsSectionView(
                model: model,
                showingProjectSettingsSheet: $showingProjectSettingsSheet,
                projectBeingEdited: $projectBeingEdited,
                projectForMcpSettings: $projectForMcpSettings
            )
            Divider()
            chatList
            Divider()
            ChatSidebarFooterView(
                chatCount: historyChats.count,
                languageRawValue: $languageRawValue
            )
        }
        .sheet(isPresented: $showingProjectSettingsSheet) {
            ProjectSettingsSheet(
                model: model,
                editingProject: projectBeingEdited,
                onDismiss: {
                    showingProjectSettingsSheet = false
                    projectBeingEdited = nil
                }
            )
        }
        .sheet(item: $projectForMcpSettings) { proj in
            ProjectMcpSettingsSheet(
                model: model,
                projectID: proj.id,
                onDismiss: {
                    projectForMcpSettings = nil
                }
            )
        }
        .sheet(item: $chatForSystemPrompt) { chat in
            ChatSystemPromptSheet(
                model: model,
                chatID: chat.id,
                onDismiss: { chatForSystemPrompt = nil }
            )
        }
        .alert(
            "Rename chat",
            isPresented: renameAlertPresented,
            presenting: chatBeingRenamed
        ) { chat in
            TextField("Chat name", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Rename") {
                model.renameChat(id: chat.id, title: renameText)
            }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        } message: { _ in
            Text("Choose a name that identifies this chat.")
        }
        .alert(
            "Delete chat?",
            isPresented: deletionAlertPresented,
            presenting: chatPendingDeletion
        ) { chat in
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                model.deleteChat(id: chat.id)
            }
        } message: { chat in
            Text("\"\(chat.title)\" and its conversation history will be removed.")
        }
    }

    private var newChatButton: some View {
        Button {
            model.createChat()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "square.and.pencil")
                    .font(theme.ui(points: 12))
                    .accessibilityHidden(true)
                Text("New chat")
                    .font(theme.ui(points: 12, weight: .medium))
                Spacer()
                Text("\u{2318}N")
                    .font(theme.ui(points: 10))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, minHeight: 30)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background(
            TurboSparkTheme.surfaceColor,
            in: .rect(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        }
        .disabled(model.isRunning)
        .help("Create a new chat (\u{2318}N)")
        .accessibilityHint("Starts a fresh conversation. Disabled while the model is generating.")
    }

    private var chatList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 3) {
                Text(model.selectedProject != nil ? "\(model.selectedProject!.name) Chats" : "Chats")
                    .font(theme.ui(points: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.top, 10)
                    .padding(.bottom, 3)
                    .accessibilityAddTraits(.isHeader)

                if historyChats.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "bubble.left.and.bubble.right")
                            .font(theme.ui(points: 20))
                            .foregroundStyle(.tertiary)
                            .padding(.top, 18)
                            .padding(.bottom, 2)
                            .accessibilityHidden(true)
                        Text("No chats yet")
                            .font(theme.ui(points: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                        Text("Start a conversation to see history here")
                            .font(theme.ui(points: 10))
                            .foregroundStyle(.tertiary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .padding(.horizontal, 12)
                    // Read the empty state as a single block rather than
                    // three sequential announcements.
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("No chats yet. Start a conversation to see history here.")
                } else {
                    ForEach(historyChats) { chat in
                        ChatSidebarChatRowView(
                            model: model,
                            chat: chat,
                            chatBeingRenamed: $chatBeingRenamed,
                            renameText: $renameText,
                            chatPendingDeletion: $chatPendingDeletion,
                            chatForSystemPrompt: $chatForSystemPrompt
                        )
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var historyChats: [AppChat] {
        sortedChats.filter { !$0.messages.isEmpty }
    }

    private var sortedChats: [AppChat] {
        model.filteredChats.sorted { lhs, rhs in
            lhs.updatedAt > rhs.updatedAt
        }
    }

    private var renameAlertPresented: Binding<Bool> {
        Binding(
            get: { chatBeingRenamed != nil },
            set: { if !$0 { chatBeingRenamed = nil } })
    }

    private var deletionAlertPresented: Binding<Bool> {
        Binding(
            get: { chatPendingDeletion != nil },
            set: { if !$0 { chatPendingDeletion = nil } })
    }
}
