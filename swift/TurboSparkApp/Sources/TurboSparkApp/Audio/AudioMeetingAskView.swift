import SwiftUI

/// One question about meetings and its answer. Kept for the session only:
/// answers are model output and never join a take's saved record.
struct AudioAskEntry: Identifiable, Equatable {
    let id = UUID()
    let itemID: UUID
    let question: String
    let allMeetings: Bool
    var answer = ""
    var sources: [AudioAskPassage] = []
    var failure: String?
    var finished = false
}

@MainActor
final class AudioMeetingAskState: ObservableObject {
    @Published var entries: [AudioAskEntry] = []
    @Published var running: UUID?
    func entries(for itemID: UUID) -> [AudioAskEntry] { entries.filter { $0.itemID == itemID } }
}

extension AudioWorkspaceController {
    /// Answers a question from one meeting, or from every Meeting take.
    /// Retrieval runs here; the answer streams from the chat model through
    /// `promptHandler`. Nothing is saved.
    func askAboutMeetings(_ question: String, from item: AudioLibraryItem, allMeetings: Bool) {
        let question = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, !isBusy, !hasRecordingActivity, canSummarize, let promptHandler else { return }
        let scope = allMeetings
            ? SpeechToTextLibrary.items(for: .meeting, in: items, collections: preferences.presetCollections)
            : [item]
        let entry = AudioAskEntry(itemID: item.id, question: question, allMeetings: allMeetings)
        ask.entries.append(entry); ask.running = entry.id
        isBusy = true; error = nil
        let target = notesModel
        let token = epoch
        jobTask = Task { [weak self] in
            guard let self else { return }
            defer { if self.valid(token) { self.isBusy = false; self.ask.running = nil; self.finishAsk(entry.id) } }
            let passages = AudioMeetingRetriever.passages(for: scope)
            let meaning = await self.librarySearch.meaningHits(question, controller: self)
            let chosen = AudioMeetingRetriever.select(question: question, from: passages,
                                                      meaning: meaning.filter { hit in scope.contains { $0.id == hit.itemID } })
            self.updateAsk(entry.id) { $0.sources = chosen.filter { $0.kind == .transcript } }
            guard !chosen.isEmpty else {
                self.updateAsk(entry.id) { $0.failure = AppStrings.localized("There is no transcript or notes to answer from yet.") }
                return
            }
            let prompt = AudioAskPrompt.build(question: question, passages: chosen, meetingCount: Set(chosen.map(\.itemID)).count)
            do {
                let answer = try await promptHandler(prompt.system, prompt.user, target) { [weak self] text in
                    guard let self, self.valid(token) else { return }
                    self.updateAsk(entry.id) { $0.answer = text }
                }
                try Task.checkCancellation()
                self.updateAsk(entry.id) { $0.answer = answer }
            } catch {
                if self.valid(token), !Task.isCancelled { self.updateAsk(entry.id) { $0.failure = error.localizedDescription } }
            }
        }
    }

    private func updateAsk(_ id: UUID, _ change: (inout AudioAskEntry) -> Void) {
        guard let index = ask.entries.firstIndex(where: { $0.id == id }) else { return }
        change(&ask.entries[index])
    }

    private func finishAsk(_ id: UUID) { updateAsk(id) { $0.finished = true } }
}

/// The Ask tab of a meeting: questions answered from its transcript, the
/// user's notes and the summary, or from every meeting, with the moments
/// each answer drew on as buttons that play from there.
@MainActor
struct AudioMeetingAskView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @ObservedObject var ask: AudioMeetingAskState
    let item: AudioLibraryItem
    @State private var question = ""
    @State private var allMeetings = false
    @FocusState private var fieldFocused: Bool

    init(controller: AudioWorkspaceController, item: AudioLibraryItem) {
        self.controller = controller
        self.ask = controller.ask
        self.item = item
    }

    private var canAsk: Bool {
        controller.canSummarize && !controller.isBusy && !controller.hasRecordingActivity
            && !question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            let entries = ask.entries(for: item.id)
            if entries.isEmpty {
                Text("Ask about decisions, owners, numbers or anything said. Answers come from this meeting's transcript and notes, with the moments they draw on.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(entries) { entry in AudioAskEntryView(controller: controller, entry: entry, running: ask.running == entry.id) }
            composer
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                TextField(text: $question, prompt: Text("Ask about this meeting", bundle: .module)) {
                    Text("Question", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
                .focused($fieldFocused)
                .onSubmit(submit)
                Button(action: submit) {
                    Label { Text("Ask", bundle: .module) } icon: { Image(systemName: "arrow.up.circle.fill") }
                }
                .appAction(.primary, size: .small)
                .disabled(!canAsk)
                .help(askHelp)
            }
            Toggle(isOn: $allMeetings) { Text("Search all meetings", bundle: .module) }
                .toggleStyle(.checkbox)
                .themedFont(.small)
        }
    }

    private var askHelp: Text {
        if !controller.canSummarize { return Text("Install a chat model to ask about meetings.", bundle: .module) }
        if controller.isBusy || controller.hasRecordingActivity { return Text("Wait for the current audio job to finish.", bundle: .module) }
        if question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return Text("Type a question first.", bundle: .module) }
        return Text("Ask", bundle: .module)
    }

    private func submit() {
        guard canAsk else { return }
        controller.askAboutMeetings(question, from: item, allMeetings: allMeetings)
        question = ""
    }
}

@MainActor
private struct AudioAskEntryView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let entry: AudioAskEntry
    let running: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label { Text(verbatim: entry.question).textSelection(.enabled) } icon: { Image(systemName: "questionmark.bubble") }
                .themedFont(.base, weight: .semibold)
                .fixedSize(horizontal: false, vertical: true)
            if let failure = entry.failure {
                Label { Text(verbatim: failure) } icon: { Image(systemName: "exclamationmark.triangle") }
                    .themedFont(.small).foregroundStyle(.appDanger)
                    .fixedSize(horizontal: false, vertical: true)
            } else if entry.answer.isEmpty && running {
                HStack(spacing: 8) { ProgressView().controlSize(.small); Text("Reading the meeting...", bundle: .module) }
                    .themedFont(.small).foregroundStyle(.appSecondary)
            } else if !entry.answer.isEmpty {
                AudioMeetingNotesReader(text: entry.answer)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if !entry.sources.isEmpty, entry.finished || !entry.answer.isEmpty {
                sources
            }
        }
        .appCardSurface(padding: 14)
        .accessibilityElement(children: .contain)
    }

    private var sources: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Drawn from", bundle: .module).themedFont(.small, weight: .semibold).foregroundStyle(.appSecondary)
                .accessibilityAddTraits(.isHeader)
            FlowLayout(spacing: 6, lineSpacing: 6) {
                ForEach(entry.sources.prefix(12)) { passage in
                    Button { play(passage) } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "play.circle.fill").accessibilityHidden(true)
                            if entry.allMeetings { Text(verbatim: passage.title).lineLimit(1) }
                            Text(verbatim: audioTime(passage.start ?? 0)).monospacedDigit()
                        }
                    }
                    .appAction(.secondary, size: .small)
                    .help(Text(verbatim: String(passage.text.prefix(240))))
                    .accessibilityLabel(Text("Play \(passage.title) from \(audioTime(passage.start ?? 0))", bundle: .module))
                    .disabled(controller.hasRecordingActivity)
                }
            }
        }
    }

    private func play(_ passage: AudioAskPassage) {
        guard let target = controller.items.first(where: { $0.id == passage.itemID }), let start = passage.start else { return }
        if controller.selectedID != target.id { controller.select(target) }
        controller.seek(to: start, item: target)
    }
}
