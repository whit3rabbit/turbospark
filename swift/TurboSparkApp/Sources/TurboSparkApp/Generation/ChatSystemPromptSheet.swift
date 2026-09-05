import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Editor for ONE chat's own system prompt, overriding the app-wide default.
@MainActor
public struct ChatSystemPromptSheet: View {
    @ObservedObject var model: AppModel
    let chatID: UUID
    let onDismiss: () -> Void

    @State private var text: String = ""

    public init(model: AppModel, chatID: UUID, onDismiss: @escaping () -> Void) {
        self.model = model
        self.chatID = chatID
        self.onDismiss = onDismiss
    }

    private var chat: AppChat? {
        model.chats.first { $0.id == chatID }
    }

    /// What this chat is actually sending right now, and where it came from.
    ///
    /// Stated rather than left to inference: with two places a prompt can come
    /// from and an empty editor meaning "inherit", a user cannot otherwise
    /// tell a chat that has no prompt from one inheriting a long default.
    private var effectiveSourceDescription: String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return "This chat sends the prompt below."
        }
        let fallback = model.defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if fallback.isEmpty {
            return "This chat sends no system prompt. No app-wide default is set."
        }
        return "Empty, so this chat inherits the app-wide default "
            + "(\(fallback.count) characters, set in Settings > Engine)."
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                TextEditor(text: $text)
                    .font(.system(.body, design: .monospaced))
                    .frame(minWidth: 460, minHeight: 220)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.25)))
                Text(effectiveSourceDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(16)
            Divider()
            footer
        }
        .onAppear {
            text = chat?.systemPrompt ?? ""
        }
    }

    private var header: some View {
        HStack {
            Label("Chat System Prompt", systemImage: "text.bubble")
                .font(.headline)
            Spacer()
        }
        .padding(16)
    }

    private var footer: some View {
        HStack {
            // Clearing is the way BACK to the default, so it needs its own
            // control: a user who has typed a prompt cannot otherwise tell
            // that emptying the editor and saving is what restores it.
            Button("Use Default") {
                text = ""
            }
            .disabled(text.isEmpty)
            Spacer()
            Button("Cancel") { onDismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Save") {
                model.setChatSystemPrompt(id: chatID, prompt: text)
                onDismiss()
            }
            .keyboardShortcut(.defaultAction)
        }
        .padding(16)
    }
}
