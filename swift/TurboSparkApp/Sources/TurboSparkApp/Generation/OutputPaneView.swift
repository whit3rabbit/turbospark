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
        ScrollView {
            WelcomeHeroView(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.vertical, 24)
        }
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
                            model: model,
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

/// View displaying a committed conversation message turn with Claude-style layout and hover actions.
private struct MessageRowView: View {
    @ObservedObject var model: AppModel
    let message: AppChatMessage
    @State private var isHovered = false
    @ObservedObject private var speechManager = AppSpeechSynthesizer.shared

    private var isCurrentlySpeakingThis: Bool {
        speechManager.isSpeaking && speechManager.speakingMessageID == message.id
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if message.role == .user {
                HStack(alignment: .top, spacing: 0) {
                    Spacer(minLength: 48)
                    VStack(alignment: .trailing, spacing: 6) {
                        CollapsibleMessageContentView(
                            text: message.content,
                            isUser: true
                        )
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.85))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        .shadow(color: Color.black.opacity(0.03), radius: 3, x: 0, y: 1)

                        if isHovered || isCurrentlySpeakingThis {
                            MessageActionBarView(
                                text: message.content,
                                messageID: message.id,
                                date: message.createdAt
                            )
                            .transition(.opacity.combined(with: .scale(scale: 0.98)))
                        }
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 6) {
                        Image(systemName: "sparkles")
                            .font(.caption.weight(.bold))
                            .foregroundStyle(TurboSparkTheme.accentColor)
                            .accessibilityHidden(true)
                        Text(model.selected?.alias ?? "TurboSpark")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                    .padding(.bottom, -2)

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
                        CollapsibleMessageContentView(text: message.content, isUser: false, maxHeight: 380)
                    } else if !message.content.isEmpty && !message.content.starts(with: "<tool_call>") && !message.content.starts(with: "Invoking tool") {
                        CollapsibleMessageContentView(text: message.content, isUser: false, maxHeight: 380)
                    }

                    if isHovered || isCurrentlySpeakingThis {
                        HStack(spacing: 6) {
                            MessageActionBarView(
                                text: message.content,
                                messageID: message.id,
                                date: message.createdAt
                            )
                            Spacer()
                        }
                        .padding(.top, 2)
                        .transition(.opacity.combined(with: .scale(scale: 0.98)))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
    }
}

/// View rendering live streaming output and prefill animations.
private struct ActiveStreamingRowView: View {
    @ObservedObject var model: AppModel
    let output: String
    let reasoning: String
    let isRunning: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(TurboSparkTheme.accentColor)
                    .accessibilityHidden(true)
                Text(model.selected?.alias ?? "TurboSpark")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .padding(.bottom, -2)

            if isRunning && output.isEmpty && reasoning.isEmpty {
                HStack(spacing: 8) {
                    TaskProgressFlameIcon(size: 16)
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

            if isRunning {
                StreamingStatusFooterView(model: model)
                    .padding(.top, 2)
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
        .help("Model thought process and reasoning trace")
        .accessibilityLabel("Thought process")
        .accessibilityHint("Expands to reveal the model's chain-of-thought reasoning")
    }
}
