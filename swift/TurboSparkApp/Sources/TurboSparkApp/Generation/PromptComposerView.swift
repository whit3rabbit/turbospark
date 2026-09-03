import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @ObservedObject var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var isImportingDocuments = false
    @State private var isExtractingDocuments = false
    @State private var documentImportError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.promptAttachments.isEmpty {
                PromptAttachmentsView(model: model)
            }
            if let documentImportError {
                Text(documentImportError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            PromptComposerEditor(model: model, promptFocused: $promptFocused)
            PromptComposerFooter(
                model: model,
                showingPromptTips: $showingPromptTips,
                isExtractingDocuments: isExtractingDocuments,
                onAttach: {
                    documentImportError = nil
                    isImportingDocuments = true
                },
                promptFocused: $promptFocused
            )
        }
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(TurboSparkTheme.surfaceColor)
                .overlay {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
                }
        }
        .fileImporter(
            isPresented: $isImportingDocuments,
            allowedContentTypes: model.attachmentContentTypes,
            allowsMultipleSelection: true,
            onCompletion: handleDocumentSelection)
        .onReceive(NotificationCenter.default.publisher(for: .focusPrompt)) { _ in
            // Cmd+L from the menu moves focus to the prompt editor so a
            // keyboard-only user can start typing without reaching for the
            // mouse.
            promptFocused = true
        }
    }

    private func handleDocumentSelection(_ result: Result<[URL], any Error>) {
        switch result {
        case .success(let urls):
            importDocuments(urls)
        case .failure(let error):
            documentImportError = error.localizedDescription
        }
    }

    private func importDocuments(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        let targetChatID = model.selectedChatID
        isExtractingDocuments = true
        documentImportError = nil

        Task {
            let outcome = await AttachmentImporter.importDocuments(
                urls,
                into: model,
                chatID: targetChatID)
            documentImportError = outcome.errorText
            isExtractingDocuments = false
        }
    }

}
