import SwiftUI

/// General settings for the context engine (`swift/docs/SWIFT_CONTEXT_ENGINE.md`):
/// the global switch, conversation recall with its storage disclosure, and
/// clearing everything it stored. Each project opts its files in from the
/// project sheet (`ProjectContextEngineSection`).
struct ContextEngineSettingsSection: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var confirmingClear = false

    var body: some View {
        Section {
            Toggle(isOn: Binding(
                get: { model.contextEngineEnabled },
                set: { model.setContextEngineEnabled($0) }
            )) {
                Text("Add relevant context to each message", bundle: .module)
            }
            .settingsControl("Add relevant context to each message", pane: .general, timing: .nextTurn)
            Text(
                "Retrieves passages from earlier in the conversation, and from a project's files when the project turns it on, and adds them to the message within the model's context window.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            Toggle(isOn: Binding(
                get: { model.conversationRecallEnabled },
                set: { model.setConversationRecallEnabled($0) }
            )) {
                Text("Recall earlier turns of each chat", bundle: .module)
            }
            .settingsControl("Recall earlier turns of each chat", pane: .general, timing: .nextTurn)
            .disabled(!model.contextEngineEnabled)
            Text(
                "Records finished turns so older ones can come back after they leave the context window. They are stored unencrypted on this Mac, outside the encrypted profile. Ghost chats are never recorded. Turning this off deletes what was recorded.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            Button(role: .destructive) {
                confirmingClear = true
            } label: {
                Text("Clear Context Engine Data", bundle: .module)
            }
            .appAction(.destructive, size: .small)
            .settingsControl("Clear Context Engine Data", pane: .general, timing: .immediate)
        } header: {
            Text("Context Engine", bundle: .module)
        }
        .confirmationDialog(
            Text("Clear Context Engine Data?", bundle: .module), isPresented: $confirmingClear
        ) {
            Button(role: .destructive) {
                model.clearContextEngineData()
            } label: { Text("Clear Data", bundle: .module) }
        } message: {
            Text(
                "This deletes every project's context index and all recorded conversation turns. Projects are indexed again when next used.",
                bundle: .module)
        }
    }
}
