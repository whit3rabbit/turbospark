import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @ObservedObject var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var isImportingDocuments = false
    @State private var isExtractingDocuments = false
    @State private var documentImportError: String?
    @State private var measuredTextHeight: CGFloat = 0

    @ScaledMetric private var iconButtonSize: CGFloat = 28
    @ScaledMetric private var editorMinHeight: CGFloat = 34
    @ScaledMetric private var editorMaxHeight: CGFloat = 200


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
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(TurboSparkTheme.surfaceColor)
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
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

    /// The editor grows with its text between a one-line floor and a scroll
    /// ceiling.
    ///
    /// `TextEditor` is greedy vertically and has no intrinsic height, so a
    /// plain `maxHeight` frame makes an empty composer as tall as a full one.
    /// So the height is MEASURED: a hidden `Text` with the same font and
    /// insets, sized with `fixedSize(vertical:)` at the editor's own width,
    /// reports its ideal height through a preference and the editor gets that
    /// height clamped into range. Two shapes that look like they would work
    /// and do not: a ZStack sizes to its largest child, which is the greedy
    /// editor, and `.frame(minHeight:maxHeight:)` is FLEXIBLE -- it reports
    /// the proposal clamped, not the child's ideal, so it always lands on the
    /// maximum.
    private var editor: some View {
        Color.clear
            .frame(height: min(max(measuredTextHeight, editorMinHeight), editorMaxHeight))
            .background(alignment: .topLeading) {
                Text(model.promptText.isEmpty ? " " : model.promptText)
                    .font(.body)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .background {
                        GeometryReader { geometry in
                            Color.clear.preference(
                                key: PromptTextHeightKey.self,
                                value: geometry.size.height)
                        }
                    }
                    .hidden()
                    .accessibilityHidden(true)
            }
            .onPreferenceChange(PromptTextHeightKey.self) { height in
                measuredTextHeight = height
            }
            .overlay(alignment: .topLeading) {
                TextEditor(text: $model.promptText)
                .accessibilityLabel("Prompt")
                .accessibilityHint("Type your message here. Press Return to send, Shift-Return for a new line.")
                .font(.body)
                .scrollContentBackground(.hidden)
                .focused($promptFocused)
                .overlay(alignment: .topLeading) {
                    if model.promptText.isEmpty {
                        Text("Ask a question, request code, or explore ideas...")
                            .font(.body)
                            .foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .padding(.vertical, 8)
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
    }


    private var footer: some View {
        HStack(spacing: 10) {
            attachDocumentAction
            promptTips
            forgeGuardrailsControl
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
                .help("Active project: \(project.name) (\(project.agentType.label))")
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Active project: \(project.name), \(project.agentType.label)")
            }
            if model.estimatedPromptTokens > 0 {
                Text("\(model.estimatedPromptTokens) tokens")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                    .help("Estimated prompt length: \(model.estimatedPromptTokens) tokens")
                    .accessibilityLabel("Estimated prompt length: \(model.estimatedPromptTokens) tokens")
            }
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var forgeGuardrailsControl: some View {
        let isEnabled = model.effectiveForgeGuardrailsEnabled
        let isGlobalFixed = model.guardrailsMode == .alwaysOn || model.guardrailsMode == .alwaysOff
        return HStack(spacing: 4) {
            Button {
                if !isGlobalFixed {
                    model.setForgeGuardrailsEnabled(!isEnabled)
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: isEnabled ? "shield.checkmark.fill" : "shield.slash")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(isEnabled ? TurboSparkTheme.accentColor : Color.secondary)

                    Text("Guardrails: \(isEnabled ? "On" : "Off")")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(isEnabled ? Color.primary : Color.secondary)
                }
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(
                    isEnabled ? TurboSparkTheme.accentColor.opacity(0.12) : Color.primary.opacity(0.04),
                    in: Capsule()
                )
                .overlay(
                    Capsule()
                        .stroke(isEnabled ? TurboSparkTheme.accentColor.opacity(0.3) : TurboSparkTheme.hairlineColor, lineWidth: 0.5)
                )
            }
            .buttonStyle(.plain)
            .disabled(isGlobalFixed)
            .help(isGlobalFixed
                  ? "Forge Guardrails is fixed to \(model.guardrailsMode.label) in Settings"
                  : (isEnabled ? "Forge Guardrails active: click to disable" : "Forge Guardrails disabled: click to enable"))
            .accessibilityLabel("Forge Guardrails: \(isEnabled ? "Enabled" : "Disabled")")
            .accessibilityHint(isGlobalFixed ? "Managed by global settings" : "Toggles tool-call guardrails for this context")

            Button {
                model.openSettings(tab: .engine)
            } label: {
                Image(systemName: "info.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Tool-call guardrails are active by default for models supporting tools. You can change this behavior in Settings. Click to open Settings.")
            .accessibilityLabel("Forge Guardrails information")
            .accessibilityHint("Opens Engine Settings to configure Guardrails")
        }
    }

    private var attachments: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 6) {
                ForEach(model.promptAttachments) { attachment in
                    attachmentChip(attachment)
                }
            }
            .padding(.bottom, 2)
        }
        .scrollIndicators(.hidden)
    }

    /// The chip body OPENS THE PREVIEW rather than doing nothing: a document
    /// the model is about to read should be readable by the user too, and the
    /// character count alone never said whether extraction worked.
    private func attachmentChip(_ attachment: AppPromptAttachment) -> some View {
        let isPreviewing = model.previewAttachmentID == attachment.id
        return HStack(spacing: 6) {
            Button {
                model.showPreview(attachmentID: attachment.id)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: attachment.symbolName)
                        .font(.system(size: 11))
                        .foregroundStyle(isPreviewing ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(attachment.fileName)
                            .font(.system(size: 11, weight: .medium))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(attachmentDetail(attachment))
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Preview \(attachment.fileName)")
            .accessibilityLabel("Preview \(attachment.fileName)")
            .accessibilityValue(attachmentDetail(attachment))
            .accessibilityHint("Opens this document in the preview pane")

            Button {
                model.removePromptAttachment(id: attachment.id)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
                    .frame(width: 14, height: 14)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.tertiary)
            .disabled(model.isRunning)
            .help("Remove \(attachment.fileName)")
            .accessibilityLabel("Remove attachment \(attachment.fileName)")
            .accessibilityHint("Detaches this document from the prompt")
        }
        .frame(maxWidth: 220, alignment: .leading)
        .padding(.leading, 8)
        .padding(.trailing, 6)
        .padding(.vertical, 5)
        .background(
            isPreviewing ? TurboSparkTheme.accentColor.opacity(0.12) : Color.primary.opacity(0.05),
            in: .rect(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    isPreviewing ? TurboSparkTheme.accentColor.opacity(0.4) : TurboSparkTheme.hairlineColor,
                    lineWidth: 0.5)
        }
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
            let outcome = await AttachmentImporter.importDocuments(
                urls,
                into: model,
                chatID: targetChatID)
            documentImportError = outcome.errorText
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

/// Ideal height of the composer's text, measured off a hidden sizer.
private struct PromptTextHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
