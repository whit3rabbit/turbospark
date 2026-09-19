import SwiftUI

@MainActor
struct ChatShareButton: View {
    @ObservedObject var model: AppModel
    @State private var options = AppChatExportOptions()

    var body: some View {
        Menu {
            Toggle(isOn: $options.includeToolDetails) {
                Text("Include tool details", bundle: .module)
            }
            Divider()
            Button { share(.markdown) } label: { Text("Markdown (.md)", bundle: .module) }
            Button { share(.docx) } label: { Text("Word document (.docx)", bundle: .module) }
            Button { share(.html) } label: { Text("HTML page (.html)", bundle: .module) }
        } label: {
            Label { Text("Share", bundle: .module) } icon: { Image(systemName: "square.and.arrow.up") }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .themedFont(.small)
        .disabled(model.selectedChat.isGhost || !model.hasOutputTranscript)
        .help(model.selectedChat.isGhost
              ? Text("Temporary chats cannot be exported.", bundle: .module)
              : Text("Save a local copy of this conversation", bundle: .module))
        .accessibilityLabel(Text("Share conversation", bundle: .module))
        .onChange(of: model.selectedChatID) { _, _ in options = .init() }
    }

    private func share(_ format: AppChatExportFormat) {
        model.shareSelectedChat(format: format, options: options)
    }
}
