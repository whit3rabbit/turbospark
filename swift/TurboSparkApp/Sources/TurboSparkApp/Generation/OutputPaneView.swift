import AppKit
import SwiftUI
import TurboSpark

/// Main view displaying generated responses, conversation transcripts, or empty-state guidance.
struct OutputPaneView: View {
    @ObservedObject var model: AppModel
    @State private var responseCopyFeedbackID: UUID?
    @State private var lastRunningState = false

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .task(id: responseCopyFeedbackID) {
            guard responseCopyFeedbackID != nil else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                responseCopyFeedbackID = nil
            }
        }
        .contextMenu {
            Button("Copy response") {
                copyResponse()
            }
            .disabled(model.outputResponsePlainText.isEmpty)

            Button("Copy conversation") {
                copy(model.outputConversationPlainText)
            }
            .disabled(model.outputConversationPlainText.isEmpty)

            Divider()

            Button("Clear chat history") { model.clearOutput() }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
        .onChange(of: model.isRunning) { wasRunning, isRunning in
            // Announce when generation finishes so a screen-reader user knows
            // they can read the response. Streaming text itself is too
            // granular to announce per token, but the completion moment is.
            // This app is macOS-only (.macOS(.v14) in Package.swift), so the
            // static AccessibilityNotification.Announcement.post is the
            // cross-platform API to use here; the iOS-only
            // \.accessibilityAnnouncementQueue environment value is not.
            if wasRunning && !isRunning {
                let count = model.liveTokenCount
                let message = count > 0
                    ? "Generation finished. \(count) tokens."
                    : "Generation finished."
                _ = AccessibilityNotification.Announcement.post(.init(message))
            }
            lastRunningState = isRunning
        }
    }

    private var placeholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 38))
                .foregroundStyle(.quaternary)
                .accessibilityHidden(true)

            if model.session == nil {
                if model.installed.isEmpty {
                    Text("No models installed.")
                        .font(.headline)
                    Text("Click 'Install...' to browse the catalog and download a model.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Model ready to load.")
                        .font(.headline)
                    Text("Ask a question or enter a prompt to begin.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    if let selected = model.selected {
                        Button("Load \(selected.alias)") {
                            model.loadModel()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .accessibilityHint("Opens the selected model into memory")
                    }
                }
            } else {
                Text("Start a conversation")
                    .font(.headline)
                Text("Ask a question, write code, or explore ideas.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var transcript: some View {
        ChatTranscriptView(model: model)
            .id(model.selectedChatID)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func copyResponse() {
        copy(model.outputResponsePlainText)
        withAnimation(.easeIn(duration: 0.15)) {
            responseCopyFeedbackID = UUID()
        }
    }
}

/// Native SwiftUI transcript view rendering multi-turn conversations with Markdown formatting.
private struct ChatTranscriptView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 20) {
                    ForEach(model.selectedChat.messages) { message in
                        MessageRowView(model: model, message: message)
                    }

                    if model.isRunning || !model.outputText.isEmpty || !model.outputReasoningText.isEmpty {
                        ActiveStreamingRowView(
                            output: model.outputText,
                            reasoning: model.outputReasoningText,
                            isRunning: model.isRunning
                        )
                    }

                    Color.clear
                        .frame(height: 1)
                        .id("bottom")
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
            }
            .onChange(of: model.outputText) {
                if model.isRunning {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onChange(of: model.outputReasoningText) {
                if model.isRunning {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onAppear {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
        }
    }
}

/// View displaying a committed conversation message turn.
private struct MessageRowView: View {
    @ObservedObject var model: AppModel
    let message: AppChatMessage

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if message.role == .user {
                HStack {
                    Spacer(minLength: 40)
                    Text(message.content)
                        .font(.body)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(Color.primary.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .textSelection(.enabled)
                }
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    if !message.reasoning.isEmpty {
                        ReasoningDisclosureView(reasoning: message.reasoning)
                    }

                    if !message.toolCalls.isEmpty {
                        ForEach(message.toolCalls) { call in
                            let matchResult = message.toolResults.first(where: { $0.callID == call.id })
                            ToolCallCardView(model: model, call: call, result: matchResult)
                        }
                    }

                    if !message.content.isEmpty && message.toolCalls.isEmpty {
                        ChatMessageMarkdownView(message.content)
                    } else if !message.content.isEmpty && !message.content.starts(with: "<tool_call>") && !message.content.starts(with: "Invoking tool") {
                        ChatMessageMarkdownView(message.content)
                    }

                    HStack(spacing: 8) {
                        MessageCopyButton(text: message.content)
                        Spacer()
                    }
                    .padding(.top, 2)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

/// Small copy button displayed beneath message results.
private struct MessageCopyButton: View {
    let text: String
    @State private var isCopied = false

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            withAnimation(.easeInOut(duration: 0.15)) {
                isCopied = true
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 11, weight: .medium))
                    .accessibilityHidden(true)
                Text(isCopied ? "Copied" : "Copy")
                    .font(.caption2.weight(.medium))
            }
            .foregroundStyle(isCopied ? TurboSparkTheme.accentColor : Color.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Copy message text")
        .accessibilityLabel(isCopied ? "Copied" : "Copy message")
        .accessibilityHint("Copies this message to the clipboard")
        .task(id: isCopied) {
            guard isCopied else { return }
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                isCopied = false
            }
        }
    }
}

/// View rendering live streaming output and prefill animations.
private struct ActiveStreamingRowView: View {
    let output: String
    let reasoning: String
    let isRunning: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if isRunning && output.isEmpty && reasoning.isEmpty {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Thinking...")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 8)
                // Combine the spinner and "Thinking..." into one announcement.
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Thinking")
                .accessibilityHint("The model is reasoning before producing a response")
            } else {
                if !reasoning.isEmpty {
                    ReasoningDisclosureView(reasoning: reasoning, defaultExpanded: true)
                }
                if !output.isEmpty {
                    ChatMessageMarkdownView(output)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Collapsible disclosure view for model reasoning and chain of thought.
private struct ReasoningDisclosureView: View {
    let reasoning: String
    var defaultExpanded: Bool = false

    @State private var isExpanded: Bool = false

    var body: some View {
        DisclosureGroup(
            isExpanded: Binding(
                get: { defaultExpanded || isExpanded },
                set: { isExpanded = $0 }
            )
        ) {
            Text(reasoning)
                .font(.callout.monospaced())
                .foregroundStyle(.secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.03))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .textSelection(.enabled)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "brain")
                    .font(.caption)
                    .foregroundStyle(TurboSparkTheme.accentColor)
                    .accessibilityHidden(true)
                Text("Thought process")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
        .accessibilityLabel("Thought process")
        .accessibilityHint("Expands to reveal the model's chain-of-thought reasoning")
    }
}
