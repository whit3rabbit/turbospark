import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @ObservedObject var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var isImportingDocuments = false
    @State private var isExtractingDocuments = false
    @State private var documentImportError: String?

    @ScaledMetric private var iconButtonSize: CGFloat = 28
    @ScaledMetric private var emptyEditorMinHeight: CGFloat = 44
    @ScaledMetric private var filledEditorMinHeight: CGFloat = 76
    @ScaledMetric private var emptyEditorMaxHeight: CGFloat = 60
    @ScaledMetric private var filledEditorMaxHeight: CGFloat = 180


    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.promptAttachments.isEmpty {
                attachments
            }
            if let documentImportError {
                Text(documentImportError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            editor
            footer
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .fileImporter(
            isPresented: $isImportingDocuments,
            allowedContentTypes: DocumentTextExtractor.supportedContentTypes,
            allowsMultipleSelection: true,
            onCompletion: handleDocumentSelection)
        .onReceive(NotificationCenter.default.publisher(for: .focusPrompt)) { _ in
            // Cmd+L from the menu moves focus to the prompt editor so a
            // keyboard-only user can start typing without reaching for the
            // mouse.
            promptFocused = true
        }
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Prompt")
            .accessibilityHint("Type your message here. Press Return to send, Shift-Return for a new line.")
            .font(.body)
            .scrollContentBackground(.hidden)
            .focused($promptFocused)
            .frame(minHeight: editorMinHeight, maxHeight: editorMaxHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    Text("Ask a question, request code, or explore ideas...")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            .onKeyPress(phases: .down) { press in
                if press.key == .escape {
                    if model.isRunning && model.canCancel {
                        model.cancel()
                        return .handled
                    }
                    promptFocused = false
                    return .handled
                }
                if press.key == .return {
                    if press.modifiers.contains(.shift) {
                        return .ignored
                    }
                    if model.canRun {
                        model.run()
                        return .handled
                    }
                    return .handled
                }
                return .ignored
            }

    }

    private var editorMinHeight: CGFloat {
        model.promptText.isEmpty ? emptyEditorMinHeight : filledEditorMinHeight
    }

    private var editorMaxHeight: CGFloat {
        model.promptText.isEmpty ? emptyEditorMaxHeight : filledEditorMaxHeight
    }


    private var footer: some View {
        HStack(spacing: 10) {
            attachDocumentAction
            promptTips
            if let project = model.selectedProject {
                HStack(spacing: 4) {
                    Image(systemName: project.agentType.systemImage)
                        .font(.caption2)
                        .accessibilityHidden(true)
                    Text("\(project.name) (\(project.agentType.label))")
                        .font(.caption2.weight(.medium))
                        .lineLimit(1)
                }
                .foregroundStyle(TurboSparkTheme.accentColor)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(TurboSparkTheme.accentColor.opacity(0.1), in: Capsule())
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Active project: \(project.name), \(project.agentType.label)")
            }
            if model.estimatedPromptTokens > 0 {
                Text("\(model.estimatedPromptTokens) tokens")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                    .accessibilityLabel("Estimated prompt length: \(model.estimatedPromptTokens) tokens")
            }
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var attachments: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(model.promptAttachments) { attachment in
                    HStack(spacing: 7) {
                        Image(systemName: "doc.text")
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(attachment.fileName)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                            Text(attachmentDetail(attachment))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            model.removePromptAttachment(id: attachment.id)
                        } label: {
                            Label("Remove \(attachment.fileName)", systemImage: "xmark.circle.fill")
                                .labelStyle(.iconOnly)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .disabled(model.isRunning)
                        .accessibilityLabel("Remove attachment \(attachment.fileName)")
                        .accessibilityHint("Detaches this document from the prompt")
                    }
                    .padding(.leading, 10)
                    .padding(.trailing, 7)
                    .padding(.vertical, 7)
                    .background(.quaternary.opacity(0.35), in: .capsule)
                    .overlay {
                        Capsule().stroke(.separator.opacity(0.4), lineWidth: 0.5)
                    }
                }
            }
        }
        .scrollIndicators(.hidden)
    }

    private var attachDocumentAction: some View {
        Button {
            documentImportError = nil
            isImportingDocuments = true
        } label: {
            Group {
                if isExtractingDocuments {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Attach documents", systemImage: "paperclip")
                        .labelStyle(.iconOnly)
                }
            }
            .frame(width: iconButtonSize, height: iconButtonSize)
            .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .disabled(model.isRunning || isExtractingDocuments)
        .help("Attach PDF, Word, Excel, code, or text files")
        .accessibilityLabel(isExtractingDocuments
                            ? "Extracting document text"
                            : "Attach documents")
        .accessibilityHint("Opens a file picker to attach documents to this prompt")
    }

    private var promptTips: some View {
        Button {
            showingPromptTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: iconButtonSize, height: iconButtonSize)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")

        .accessibilityLabel("Prompt tips")
        .accessibilityHint("Shows a popover with prompt writing guidance")
        .popover(isPresented: $showingPromptTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            promptGuide
        }
    }

    private var promptGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting tips")
                .font(.headline)

            tipSection("Clear task & constraints",
                       "State what you want created, explained, or transformed. Specify length, style, or output structure.")
            tipSection("Provide types & interfaces",
                       "For code tasks, provide signatures, expected inputs/outputs, or small working scaffolds.")
            tipSection("Attach relevant documents",
                       "Attach PDFs, spreadsheets, or code files for local reasoning and question answering.")
        }
        .font(.callout)
        .frame(width: 360, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).fontWeight(.semibold)
            Text(detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func attachmentDetail(_ attachment: AppPromptAttachment) -> String {
        let count = attachment.characterCount.formatted(.number.notation(.compactName))
        let suffix = attachment.wasTruncatedDuringExtraction ? " • truncated" : ""
        return "\(attachment.formatLabel) • \(count) chars\(suffix)"
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
            let outcomes = await Task.detached(priority: .userInitiated) {
                urls.map { url -> (URL, Result<ExtractedPromptDocument, Error>) in
                    do {
                        return (url, .success(try DocumentTextExtractor.extract(from: url)))
                    } catch {
                        return (url, .failure(error))
                    }
                }
            }.value

            var failures: [String] = []
            for (url, outcome) in outcomes {
                switch outcome {
                case .success(let document):
                    model.addPromptAttachment(
                        AppPromptAttachment(
                            fileName: document.fileName,
                            formatLabel: document.formatLabel,
                            extractedText: document.text,
                            wasTruncatedDuringExtraction: document.wasTruncated),
                        toChatID: targetChatID)
                case .failure(let error):
                    failures.append("\(url.lastPathComponent): \(error.localizedDescription)")
                }
            }
            documentImportError = failures.isEmpty ? nil : failures.joined(separator: "\n")
            isExtractingDocuments = false
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label("Clear prompt", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: iconButtonSize, height: iconButtonSize)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear prompt")
            .accessibilityLabel("Clear prompt")
            .accessibilityHint("Empties the prompt editor")
        } else if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear chat history", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: iconButtonSize, height: iconButtonSize)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear chat history")
            .accessibilityLabel("Clear chat history")
            .accessibilityHint("Removes the current chat transcript")
        }
    }

}
