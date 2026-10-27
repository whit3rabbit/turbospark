import SwiftUI
import TurboSpark

/// Dedicated Notes and Summary view for meeting takes.
///
/// Phase 2:
/// 1. Ground Truth Safety: Displays the executive summary and action items while
///    guaranteeing that the underlying transcript is never modified or overwritten.
/// 2. Model Selection & Recommendations: Lists installed chat models, highlighting
///    lightweight models like Qwen 3.5 1.5B / Qwen 1.5B for fast local inference.
/// 3. Interactive Audio Jumps: Clickable timestamps seek the audio transport directly.
/// 4. Export & Copy: One-click export to Markdown or clipboard.
@MainActor
struct AudioMeetingNotesView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    @State private var selectedModel: InstalledModel?
    @State private var notesText: String = ""
    @State private var isEditing: Bool = false
    @State private var copiedFeedback: Bool = false

    private var availableModels: [InstalledModel] {
        controller.availableChatModels?() ?? []
    }

    private var prioritizedModels: [InstalledModel] {
        AudioMeetingNotesAdvisor.prioritizedModels(availableModels)
    }

    private var activeModel: InstalledModel? {
        selectedModel ?? AudioMeetingNotesAdvisor.defaultModel(
            from: availableModels,
            currentSelected: controller.selectedChatModel?()
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            groundTruthBanner
            headerControls
            Divider()

            if let summary = item.summary {
                summaryContent(summary)
            } else {
                emptyState
            }
        }
        .onAppear {
            if let summary = item.summary {
                notesText = summary.text
            }
            if selectedModel == nil {
                selectedModel = activeModel
            }
        }
        .onChange(of: item.summary?.text) { _, newText in
            if !isEditing, let newText {
                notesText = newText
            }
        }
    }

    private var canRunSummarization: Bool {
        controller.canSummarizeMeeting && !controller.isBusy && !controller.hasRecordingActivity
    }

    // MARK: - Ground Truth Banner

    private var groundTruthBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "shield.checkmark.fill")
                .foregroundStyle(.appAccent)
            Text("Transcript preserved: ground truth is immutable.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Spacer()
            if let rev = item.preferredTranscript {
                (Text("\(rev.segments.count) segments", bundle: .module) + Text(verbatim: " (\(audioTime(item.duration)))"))
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .monospacedDigit()
            }
        }
        .appCardSurface(padding: 10)
    }

    // MARK: - Header Controls

    private var headerControls: some View {
        AppAdaptiveHeader(
            leading: { modelMenu },
            actions: {
                if item.summary != nil {
                    copyButton
                    exportButton
                }
                summarizeButton
            }
        )
    }

    private var modelMenu: some View {
        Menu {
            if prioritizedModels.isEmpty {
                Text("No chat models installed", bundle: .module)
            } else {
                ForEach(prioritizedModels) { model in
                    Button {
                        selectedModel = model
                        controller.selectChatModel?(model)
                    } label: {
                        HStack {
                            if activeModel?.id == model.id {
                                Image(systemName: "checkmark")
                            }
                            Text(verbatim: model.alias)
                            if let badge = AudioMeetingNotesAdvisor.recommendationBadge(for: model) {
                                Text(verbatim: " (") + Text(LocalizedStringKey(badge), bundle: .module) + Text(verbatim: ")")
                            }
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                if let model = activeModel {
                    Text(verbatim: model.alias)
                    if AudioMeetingNotesAdvisor.isRecommended(model: model) {
                        Image(systemName: "sparkles")
                            .foregroundStyle(.appAccent)
                    }
                } else {
                    Text("Select Model", bundle: .module)
                }
            }
        }
        .appMenuAction(size: .small)
        .help(Text("Select model for meeting summarization.", bundle: .module))
    }

    private var summarizeButton: some View {
        Button {
            controller.summarizeMeeting(with: activeModel)
        } label: {
            HStack(spacing: 6) {
                if controller.isBusy {
                    ProgressView()
                        .controlSize(.small)
                    Text("Summarizing...", bundle: .module)
                } else if item.summary == nil {
                    Image(systemName: "sparkles")
                    Text("Generate Notes", bundle: .module)
                } else {
                    Image(systemName: "arrow.clockwise")
                    Text("Re-generate", bundle: .module)
                }
            }
        }
        .appAction(.primary, size: .small)
        .help(
            controller.canSummarizeMeeting
                ? (item.summary == nil ? Text("Generate Notes", bundle: .module) : Text("Re-generate", bundle: .module))
                : Text("Load a local chat model to summarize this transcript.", bundle: .module)
        )
        .disabled(!canRunSummarization)
    }

    private var copyButton: some View {
        Button {
            copyToClipboard()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: copiedFeedback ? "checkmark" : "doc.on.doc")
                if copiedFeedback {
                    Text("Copied!", bundle: .module)
                } else {
                    Text("Copy", bundle: .module)
                }
            }
        }
        .appAction(.secondary, size: .small)
        .help(Text("Copy notes to clipboard.", bundle: .module))
    }

    private var exportButton: some View {
        Button {
            exportMarkdown()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.down.doc")
                Text("Export MD", bundle: .module)
            }
        }
        .appAction(.secondary, size: .small)
        .help(Text("Export notes as a Markdown file.", bundle: .module))
    }

    // MARK: - Summary Content

    @ViewBuilder
    private func summaryContent(_ summary: AudioMeetingSummary) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                if let alias = summary.modelAlias {
                    Text("Generated with \(alias)", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Button {
                    if isEditing {
                        controller.saveSummaryText(notesText, for: item.id)
                    }
                    isEditing.toggle()
                } label: {
                    Label {
                        if isEditing {
                            Text("Save Notes", bundle: .module)
                        } else {
                            Text("Edit Notes", bundle: .module)
                        }
                    } icon: {
                        Image(systemName: isEditing ? "checkmark" : "pencil")
                    }
                }
                .appAction(.secondary, size: .small)
            }

            if !summary.sourceTimes.isEmpty {
                timeJumps(summary.sourceTimes)
            }

            if isEditing {
                TextEditor(text: $notesText)
                    .themedFont(.base)
                    .appEditorSurface(minHeight: 280)
            } else {
                ScrollView {
                    Text(verbatim: notesText)
                        .themedFont(.base)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                }
                .appCardSurface(padding: 0)
                .frame(minHeight: 240)
            }
        }
    }

    private func timeJumps(_ times: [Double]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Jump to discussion:", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)

            FlowLayout(spacing: 6, lineSpacing: 6) {
                ForEach(Array(times.enumerated()), id: \.offset) { _, time in
                    Button {
                        controller.seek(to: time)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "play.circle.fill")
                            Text(verbatim: audioTime(time))
                                .monospacedDigit()
                        }
                    }
                    .appAction(.secondary, size: .small)
                    .help(
                        item.clips.isEmpty
                            ? Text("No clips available to play audio.", bundle: .module)
                            : Text(verbatim: audioTime(time))
                    )
                    .disabled(item.clips.isEmpty || controller.hasRecordingActivity)
                }
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        AppEmptyState(
            systemImage: "doc.text.badge.sparkles",
            title: Text("No Meeting Notes Yet", bundle: .module),
            message: Text("Generate structured notes with key decisions, action items, and topic overviews using a local model. Your spoken transcript remains untouched.", bundle: .module)
        ) {
            VStack(spacing: 12) {
                if let model = activeModel {
                    HStack(spacing: 6) {
                        Image(systemName: "sparkles")
                            .foregroundStyle(.appAccent)
                        Text("Model: \(model.alias)", bundle: .module)
                        if let badge = AudioMeetingNotesAdvisor.recommendationBadge(for: model) {
                            (Text(verbatim: "(") + Text(LocalizedStringKey(badge), bundle: .module) + Text(verbatim: ")"))
                                .foregroundStyle(.appSecondary)
                        }
                    }
                    .themedFont(.small)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.appSurface)
                    .clipShape(Capsule())
                }

                Button {
                    controller.summarizeMeeting(with: activeModel)
                } label: {
                    Label {
                        Text("Generate Notes & Summary", bundle: .module)
                    } icon: {
                        Image(systemName: "sparkles")
                    }
                }
                .appAction(.primary)
                .appPrimaryGlow(canRunSummarization)
                .help(
                    controller.canSummarizeMeeting
                        ? Text("Generate Notes & Summary", bundle: .module)
                        : Text("Load a local chat model to summarize this transcript.", bundle: .module)
                )
                .disabled(!canRunSummarization)
            }
        }
        .appCardSurface(padding: 24)
    }

    // MARK: - Actions

    private func copyToClipboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(notesText, forType: .string)
        copiedFeedback = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            copiedFeedback = false
        }
    }

    private func exportMarkdown() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "\(item.title) - Notes.md"
        if panel.runModal() == .OK, let url = panel.url {
            try? notesText.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
