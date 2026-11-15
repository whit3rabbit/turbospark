import SwiftUI

/// The composer's Remote section (ZP-22): one menu beside the tool chips that
/// lists enabled remote provider models and points THIS chat at one. Local is
/// the default for every chat and stays the model picker in the top bar; this
/// pill does not exist until a provider with a key and an enabled model is
/// configured.
///
/// Remote use is explicit. Choosing a model raises a confirmation naming the
/// provider host, and while a chat is on a remote model a persistent label
/// says its content leaves this Mac.
struct RemoteModelPill: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var store: RemoteProviderStore

    private var selection: (provider: String, model: String, host: String)? {
        model.remoteSelectionLabel(for: model.selectedChatID)
    }

    var body: some View {
        let entries = store.pickerEntries()
        if !entries.isEmpty || selection != nil {
            HStack(spacing: 6) {
                menu(entries: entries)
                if let selection {
                    Label {
                        Text("Content leaves this Mac", bundle: .module)
                    } icon: {
                        Image(systemName: "arrow.up.right.circle")
                    }
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(.appWarning)
                    .help(Text("Prompts and tool results in this chat are sent to \(selection.host).", bundle: .module))
                    .accessibilityElement(children: .combine)
                }
            }
            .confirmationDialog(
                confirmationTitle,
                isPresented: Binding(
                    get: { model.remoteConfirmation != nil },
                    set: { if !$0 { model.cancelRemoteConfirmation() } }),
                titleVisibility: .visible
            ) {
                Button { model.confirmRemoteSelection() } label: { Text("Send to remote", bundle: .module) }
                Button(role: .cancel) { model.cancelRemoteConfirmation() } label: { Text("Keep local", bundle: .module) }
            } message: {
                if let request = model.remoteConfirmation {
                    Text("Prompts and tool results in this chat will be sent to \(request.providerName) (\(request.host)) and leave this machine. The provider's own terms apply. Local models stay the default for other chats.", bundle: .module)
                }
            }
        }
    }

    private var confirmationTitle: Text {
        if let request = model.remoteConfirmation {
            return Text("Send this chat to \(request.providerName)?", bundle: .module)
        }
        return Text(verbatim: "")
    }

    private func menu(entries: [(provider: RemoteProvider, model: RemoteProviderModel)]) -> some View {
        Menu {
            if selection != nil {
                Button {
                    model.useLocalModel(chatID: model.selectedChatID)
                } label: {
                    Label { Text("Use local model", bundle: .module) } icon: { Image(systemName: "desktopcomputer") }
                }
                Divider()
            }
            Section(header: Text("Remote (content leaves this Mac)", bundle: .module)) {
                ForEach(entries, id: \.model.id) { entry in
                    Button {
                        model.requestRemoteModel(
                            providerID: entry.provider.id, modelID: entry.model.id,
                            chatID: model.selectedChatID)
                    } label: {
                        let current = model.remoteSelections[model.selectedChatID]
                        let label = "\(entry.provider.name): \(entry.model.displayName)"
                        if current?.providerID == entry.provider.id, current?.modelID == entry.model.id {
                            Label(label, systemImage: "checkmark")
                        } else {
                            Text(verbatim: label)
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 3) {
                Image(systemName: "cloud")
                    .font(theme.ui(.small))
                if let selection {
                    Text(verbatim: "\(selection.provider): \(selection.model)")
                        .font(theme.ui(.small, weight: .medium))
                        .lineLimit(1)
                }
                Image(systemName: "chevron.up.chevron.down")
                    .font(theme.ui(.micro, weight: .bold))
            }
            .foregroundStyle(.appSecondary)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(model.isRunningInSelectedChat)
        .help(Text("Remote (content leaves this Mac)", bundle: .module))
        .accessibilityLabel(Text("Remote (content leaves this Mac)", bundle: .module))
    }
}
