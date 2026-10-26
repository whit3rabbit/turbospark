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
    private var effectiveSourceDescription: Text {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return Text("This chat sends the prompt below.", bundle: .module)
        }
        let fallback = model.defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if fallback.isEmpty {
            return Text("This chat sends no system prompt. No app-wide default is set.", bundle: .module)
        }
        return Text("Empty, so this chat inherits the app-wide default (\(fallback.count) characters, set in Settings > Engine).", bundle: .module)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                TextEditor(text: $text)
                    .themedCode(.base)
                    // The editor is the only thing that sizes this sheet's width.
                    .frame(minWidth: 460)
                    .appEditorSurface(minHeight: 220)
                effectiveSourceDescription
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
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
            Label { Text("Chat System Prompt", bundle: .module) } icon: { Image(systemName: "text.bubble") }
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Spacer()
        }
        .padding(16)
    }

    private var footer: some View {
        HStack {
            // Clearing is the way BACK to the default, so it needs its own
            // control: a user who has typed a prompt cannot otherwise tell
            // that emptying the editor and saving is what restores it.
            Button {
                text = ""
            } label: { Text("Use Default", bundle: .module) }
            .disabled(text.isEmpty)
            .help(
                text.isEmpty
                    ? Text("Use Default is only available when a custom prompt is set", bundle: .module)
                    : Text("Use Default", bundle: .module)
            )
            Spacer()
            Button { onDismiss() } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
            Button {
                model.setChatSystemPrompt(id: chatID, prompt: text)
                onDismiss()
            } label: { Text("Save", bundle: .module) }
            .keyboardShortcut(.defaultAction)
        }
        .padding(16)
    }
}
