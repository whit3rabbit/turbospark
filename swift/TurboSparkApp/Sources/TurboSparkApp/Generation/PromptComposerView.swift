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
    @Environment(\.appTheme) private var theme

    @ScaledMetric private var iconButtonSize: CGFloat = 28
    @ScaledMetric private var editorMinHeight: CGFloat = 34
    @ScaledMetric private var editorMaxHeight: CGFloat = 200

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

    /// The editor grows with its text between a one-line floor and a scroll
    /// ceiling.
    ///
    /// `TextEditor` is greedy vertically and has no intrinsic height, so a
    /// plain `maxHeight` frame makes an empty composer as tall as a full one.
    /// So the height is MEASURED: a hidden `Text` with the same font and
    /// insets, sized with `fixedSize(vertical:)` at the editor's own width,
    /// reports its ideal height through a preference and the editor gets that
    /// height clamped into range.
    private var editor: some View {
        Color.clear
            .frame(height: min(max(measuredTextHeight, editorMinHeight), editorMaxHeight))
            .background(alignment: .topLeading) {
                Text(model.promptText.isEmpty ? " " : model.promptText)
                    .font(theme.uiFont)
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
                    .font(theme.uiFont)
                    .scrollContentBackground(.hidden)
                    .focused($promptFocused)
                    .overlay(alignment: .topLeading) {
                        if model.promptText.isEmpty {
                            Text("Ask a question, request code, or explore ideas...")
                                .font(theme.uiFont)
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

    /// The footer is two rows, not one.
    ///
    /// Eight controls in a single `HStack` overflow the composer's 680pt and
    /// an `HStack` resolves that by COMPRESSING them, with no floor: the mode
    /// segment collapsed to one character per line and "Guardrails: On"
    /// wrapped mid-pill. The context pills wrap through `FlowLayout` instead,
    /// and the actions keep a fixed row so Generate stays pinned right.
    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            contextPills
            actionRow
        }
    }

    private var contextPills: some View {
        FlowLayout(spacing: 6, lineSpacing: 6) {
            PromptInteractionModeSegment(model: model)
            // Guarded here as well as inside the pill: a view that renders
            // nothing is still a subview, and FlowLayout would leave its
            // spacing behind as a phantom gap.
            if model.selectedProject != nil {
                PromptProjectContextPill(model: model)
            }
            PromptModelSelectorPill(model: model)
            // Needs a session: the levels are the checkpoint's own and are
            // read off its template at open.
            if model.reasoningPickerEnabled {
                PromptReasoningPillControl(model: model)
            }
            ForgeGuardrailsPillControl(model: model)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var actionRow: some View {
        HStack(spacing: 8) {
            PromptAttachDocumentButton(
                iconButtonSize: iconButtonSize,
                isRunning: model.isRunning,
                isExtracting: isExtractingDocuments,
                onAttach: {
                    documentImportError = nil
                    isImportingDocuments = true
                }
            )
            PromptTipsButton(
                iconButtonSize: iconButtonSize,
                showingTips: $showingPromptTips
            )
            if model.estimatedPromptTokens > 0 {
                Text("\(model.estimatedPromptTokens) tokens")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .fixedSize()
                    .help("Estimated prompt length: \(model.estimatedPromptTokens) tokens")
                    .accessibilityLabel("Estimated prompt length: \(model.estimatedPromptTokens) tokens")
            }
            Spacer(minLength: 8)
            clearAction
            GenerateControl(model: model)
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
