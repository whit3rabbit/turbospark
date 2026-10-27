import AppKit
import SwiftUI
import TurboSpark

/// Main view displaying generated responses, conversation transcripts, or empty-state guidance.
struct OutputPaneView: View {
    @ObservedObject var model: AppModel
    @State private var responseCopyFeedbackID: UUID?
    @State private var lastRunningState = false
    @State private var showingMemoryCapture = false
    @State private var showingClearConfirmation = false

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: responseCopyFeedbackID) {
            guard responseCopyFeedbackID != nil else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled else { return }
            TSMotion.animate(TSMotion.select) {
                responseCopyFeedbackID = nil
            }
        }
        .contextMenu {
            Button {
                copyResponse()
            } label: { Text("Copy response", bundle: .module) }
            .disabled(model.outputResponsePlainText.isEmpty)

            Button {
                copy(model.outputConversationPlainText)
            } label: { Text("Copy conversation", bundle: .module) }
            .disabled(!model.hasOutputConversationText)

            if model.memoryEnabled {
                Button {
                    showingMemoryCapture = true
                } label: { Text("Add conversation to memory", bundle: .module) }
                .disabled(!model.hasOutputConversationText)
            }

            Divider()

            Button {
                model.showSessionStats = true
            } label: { Text("Session Stats...", bundle: .module) }
            Button {
                model.exportSelectedChat(format: .markdown)
            } label: { Text("Export Conversation...", bundle: .module) }
            Button {
                model.showHelpSheet = true
            } label: { Text("Help...", bundle: .module) }

            Divider()

            Button {
                model.showTasksSheet = true
            } label: { Text("Background Tasks...", bundle: .module) }
            Button {
                model.showContextSheet = true
            } label: { Text("Context Usage...", bundle: .module) }

            Divider()

            if model.collapsedTurnAnchors.isEmpty {
                Button {
                    model.collapseAllTurns()
                } label: { Text("Collapse finished turns", bundle: .module) }
                .disabled(!model.canCollapseTurns)
            } else {
                Button {
                    model.collapsedTurnAnchors.removeAll()
                } label: { Text("Expand collapsed turns", bundle: .module) }
            }

            Button(role: .destructive) {
                showingClearConfirmation = true
            } label: { Text("Clear chat history", bundle: .module) }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
        .confirmationDialog(
            Text("Clear Chat History", bundle: .module),
            isPresented: $showingClearConfirmation,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                model.clearOutput()
            } label: {
                Text("Clear Chat History", bundle: .module)
            }
            Button(role: .cancel) {} label: {
                Text("Cancel", bundle: .module)
            }
        } message: {
            Text("This will clear the current transcript. This cannot be undone.", bundle: .module)
        }
        .sheet(isPresented: $model.showSessionStats) {
            SessionStatsSheet(model: model)
        }
        .sheet(isPresented: $model.showHelpSheet) {
            HelpSheetView(model: model)
        }
        // The qwen-code local-command sheets (`/context`, `/tasks`,
        // `/tools`, `/status`, `/rewind`, `/recap`).
        .sheet(isPresented: $model.showContextSheet) {
            ContextUsageSheet(model: model)
        }
        .sheet(isPresented: $model.showTasksSheet) {
            TasksStatusSheet(model: model)
        }
        .sheet(isPresented: $model.showToolsSheet) {
            ToolsListSheet(model: model)
        }
        .sheet(isPresented: $model.showStatusSheet) {
            StatusSheet(model: model)
        }
        .sheet(isPresented: $model.showRewindSheet) {
            RewindSheet(model: model)
        }
        .sheet(isPresented: $model.showRecapSheet) {
            RecapSheet(model: model)
        }
        // The qwen-code git-command sheet (`/diff`, `/log`, `/prs`).
        .sheet(isPresented: $model.showGitSheet) {
            GitInfoSheet(model: model)
        }
        .sheet(isPresented: $showingMemoryCapture) {
            ProfileMemoryCaptureSheet(
                model: model,
                isPresented: $showingMemoryCapture,
                source: model.outputConversationPlainText)
        }
        // `/delete` asks first, exactly like the sidebar's Delete action.
        .alert(
            Text("Delete this chat?", bundle: .module),
            isPresented: $model.confirmDeleteChat
        ) {
            Button(role: .destructive) {
                model.deleteChat(id: model.selectedChatID)
            } label: { Text("Delete", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("The conversation '\(model.selectedChat.title)' will be removed. This cannot be undone.", bundle: .module)
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
                // A turn that ends by itself disarms a two-stage Esc that
                // never got its second press.
                model.disarmEscCancel()
                let count = model.liveTokenCount
                // Spoken, so in the UI language like any visible string.
                let message = count > 0
                    ? AppStrings.localized("Generation finished. \(count) tokens.")
                    : AppStrings.localized("Generation finished.")
                _ = AccessibilityNotification.Announcement.post(.init(message))
            } else if !wasRunning && isRunning {
                let message = model.reasoning != .off
                    ? AppStrings.localized("Generation started. Thinking.")
                    : AppStrings.localized("Generation started.")
                _ = AccessibilityNotification.Announcement.post(.init(message))
            }
            lastRunningState = isRunning
        }
    }

    private var placeholder: some View {
        GeometryReader { proxy in
            ScrollView(.vertical) {
                WelcomeHeroView(model: model)
                    .frame(maxWidth: .infinity)
                    .frame(minHeight: proxy.size.height)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
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
        TSMotion.animate(TSMotion.select) {
            responseCopyFeedbackID = UUID()
        }
    }
}
