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

    // MARK: - Ground Truth Banner

    private var groundTruthBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "shield.checkmark.fill")
                .foregroundStyle(.appAccent)
            Text(verbatim: "Transcript preserved: ground truth is immutable.")
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Spacer()
            if let rev = item.preferredTranscript {
                Text(verbatim: "\(rev.segments.count) segments (\(audioTime(item.duration)))")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .monospacedDigit()
            }
        }
        .appCardSurface(padding: 10)
    }

    // MARK: - Header Controls

    private var headerControls: some View {
        HStack(alignment: .center, spacing: 12) {
            modelMenu

            Spacer(minLength: 8)

            if item.summary != nil {
                copyButton
                exportButton
            }

            summarizeButton
        }
    }

    private var modelMenu: some View {
        Menu {
            if prioritizedModels.isEmpty {
                Text(verbatim: "No chat models installed")
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
                                Text(verbatim: " (\(badge))")
                            }
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                Text(verbatim: activeModel?.alias ?? "Select Model")
                if let model = activeModel, AudioMeetingNotesAdvisor.isRecommended(model: model) {
                    Image(systemName: "sparkles")
                        .foregroundStyle(.appAccent)
                }
            }
        }
        .appMenuAction(size: .small)
        .help(Text(verbatim: "Select model for meeting summarization."))
    }

    private var summarizeButton: some View {
        Button {
            controller.summarizeMeeting(with: activeModel)
        } label: {
            HStack(spacing: 6) {
                if controller.isBusy {
                    ProgressView()
                        .controlSize(.small)
                    Text(verbatim: "Summarizing...")
                } else if item.summary == nil {
                    Image(systemName: "sparkles")
                    Text(verbatim: "Generate Notes")
                } else {
                    Image(systemName: "arrow.clockwise")
                    Text(verbatim: "Re-generate")
                }
            }
        }
        .appAction(.primary, size: .small)
        .disabled(!controller.canSummarizeMeeting || controller.isBusy || controller.hasRecordingActivity)
    }

    private var copyButton: some View {
        Button {
            copyToClipboard()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: copiedFeedback ? "checkmark" : "doc.on.doc")
                Text(verbatim: copiedFeedback ? "Copied!" : "Copy")
            }
        }
        .appAction(.secondary, size: .small)
        .help(Text(verbatim: "Copy notes to clipboard."))
    }

    private var exportButton: some View {
        Button {
            exportMarkdown()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.down.doc")
                Text(verbatim: "Export MD")
            }
        }
        .appAction(.secondary, size: .small)
        .help(Text(verbatim: "Export notes as a Markdown file."))
    }

    // MARK: - Summary Content

    @ViewBuilder
    private func summaryContent(_ summary: AudioMeetingSummary) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                if let alias = summary.modelAlias {
                    Text(verbatim: "Generated with \(alias)")
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
                        Text(verbatim: isEditing ? "Save Notes" : "Edit Notes")
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
                    .frame(minHeight: 280)
                    .padding(8)
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(.appBorder, lineWidth: 1)
                    }
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
            Text(verbatim: "Jump to discussion:")
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
                    .disabled(item.clips.isEmpty || controller.hasRecordingActivity)
                }
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(alignment: .center, spacing: 16) {
            Image(systemName: "doc.text.badge.sparkles")
                .font(.system(size: 40))
                .foregroundStyle(.appSecondary)

            VStack(spacing: 6) {
                Text(verbatim: "No Meeting Notes Yet")
                    .themedFont(.title3, weight: .semibold)
                Text(verbatim: "Generate structured notes with key decisions, action items, and topic overviews using a local model. Your spoken transcript remains untouched.")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
            }

            if let model = activeModel {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .foregroundStyle(.appAccent)
                    Text(verbatim: "Model: \(model.alias)")
                    if let badge = AudioMeetingNotesAdvisor.recommendationBadge(for: model) {
                        Text(verbatim: "(\(badge))")
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
                    Text(verbatim: "Generate Notes & Summary")
                } icon: {
                    Image(systemName: "sparkles")
                }
            }
            .appAction(.primary)
            .appPrimaryGlow(true)
            .disabled(!controller.canSummarizeMeeting || controller.isBusy || controller.hasRecordingActivity)
        }
        .frame(maxWidth: .infinity)
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
