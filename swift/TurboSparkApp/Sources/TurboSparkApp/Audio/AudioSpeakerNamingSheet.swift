import SwiftUI

/// Name the Speakers: one row per voice the final pass found, with its talk
/// time, its longest line, a few seconds of it and a name field. A name
/// applies to every line from that voice and to exports. The engine reports
/// no measure of how alike two voices sound, so merging is a plain choice of
/// target, never a suggestion. Nothing is saved until Apply Names.
@MainActor
struct AudioSpeakerNamingSheet: View {
    enum Purpose { case afterTake, edit }

    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    let purpose: Purpose
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme
    /// Merges as edited here. Names are typed into `drafts` and trimmed only
    /// on Apply, so a field never eats the space between two words.
    @State private var roster: AudioSpeakerRoster
    @State private var drafts: [String: String]
    @State private var suggestions: [String: AudioSpeakerMemory.Suggestion] = [:]
    @FocusState private var focusedKey: String?

    init(controller: AudioWorkspaceController, item: AudioLibraryItem, purpose: Purpose) {
        self.controller = controller
        self.item = item
        self.purpose = purpose
        let roster = AudioSpeakerRoster(item)
        _roster = State(initialValue: roster)
        _drafts = State(initialValue: roster.names)
    }

    private var scale: CGFloat { AppMetrics.textScale(theme) }
    private var segments: [AudioLibrarySegment] { item.preferredTranscript?.segments ?? [] }

    var body: some View {
        let summaries = AudioSpeakerSummary.summaries(segments, roster: roster)
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.horizontal, 24).padding(.top, 20).padding(.bottom, 14)
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    if summaries.isEmpty {
                        AppEmptyState(systemImage: "person.2",
                                      title: Text("No speakers labeled", bundle: .module),
                                      message: Text("Speaker labels appear here after a take's final transcript, when labels are on for its mode.", bundle: .module)) {}
                    }
                    ForEach(summaries) { summary in
                        row(summary, targets: summaries.filter { $0.mergedInto == nil && $0.id != summary.id }.map(\.id))
                    }
                }
                .padding(16)
            }
            Divider()
            footer
                .padding(.horizontal, 24).padding(.vertical, 14)
        }
        .frame(
            minWidth: (460 * scale).rounded(),
            idealWidth: (600 * scale).rounded(),
            maxWidth: (820 * scale).rounded(),
            minHeight: (360 * scale).rounded(),
            idealHeight: (560 * scale).rounded()
        )
        .background(.appPage)
        .themedFont(.base)
        .onAppear {
            focusedKey = summaries.first { $0.mergedInto == nil }?.id
            suggestions = AudioSpeakerMemory.suggestions(
                for: item,
                segments: segments,
                profiles: AudioSpeakerProfileStore.load(),
                store: controller.assets
            )
        }
        .onDisappear { if controller.speakerSampleKey != nil { controller.stopReferencePreview() } }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Name the Speakers", bundle: .module)
                .themedFont(.title2, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            HStack(spacing: AppMetrics.actionSpacing) {
                Text(verbatim: item.title).lineLimit(1).truncationMode(.middle).help(Text(verbatim: item.title))
                Text(verbatim: audioTime(item.duration)).monospacedDigit()
            }
            .themedFont(.small).foregroundStyle(.appSecondary)
            Text("A name applies to every line from that voice, and to exported transcripts and subtitles.", bundle: .module)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if let parts = item.speakerLabelParts, parts > 1 {
                Label { Text("Labeled in \(parts) parts", bundle: .module) } icon: { Image(systemName: "checkmark.circle.fill") }
                    .themedFont(.small, weight: .semibold)
                    .padding(.top, 2)
                Text("Parts are matched where they overlap. A voice that is quiet there can show up twice: merge it into the right speaker.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The name a voice shows here: the draft of the voice it shows as.
    private func name(_ key: String) -> String { ownName(roster.root(key)) }

    /// A voice's own draft, else its default label.
    private func ownName(_ key: String) -> String {
        let draft = (drafts[key] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return draft.isEmpty ? AudioSpeakerRoster.defaultName(key) : draft
    }

    private func row(_ summary: AudioSpeakerSummary, targets: [String]) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: summary.mergedInto == nil ? "person.circle" : "arrow.triangle.merge")
                .themedFont(.title2)
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    if let target = summary.mergedInto {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(verbatim: ownName(summary.id)).themedFont(.base, weight: .semibold)
                            Text("Merged into \(name(target))", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                        }
                    } else {
                        VStack(alignment: .leading, spacing: 6) {
                            nameField(summary.id)
                            if let suggestion = suggestions[summary.id], (drafts[summary.id] ?? "").isEmpty {
                                suggestionChip(for: summary.id, suggestion: suggestion)
                            }
                        }
                    }
                    Spacer(minLength: 8)
                    Text(verbatim: audioTime(summary.talkTime))
                        .monospacedDigit().themedFont(.small).foregroundStyle(.appSecondary)
                        .accessibilityLabel(Text("Talk time: \(audioTime(summary.talkTime))", bundle: .module))
                }
                if let quote = summary.quote {
                    Text(verbatim: quote)
                        .italic().themedFont(.small).foregroundStyle(.appSecondary)
                        .lineLimit(2).help(Text(verbatim: quote))
                }
                FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) {
                    sampleButton(summary)
                    if summary.mergedInto == nil { mergeMenu(summary.id, targets: targets) }
                    else { separateButton(summary.id) }
                }
            }
        }
        .appCardSurface(padding: 12)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder private func suggestionChip(for key: String, suggestion: AudioSpeakerMemory.Suggestion) -> some View {
        Button {
            drafts[key] = suggestion.name
        } label: {
            switch suggestion.reason {
            case .microphone:
                Label {
                    Text("You (Microphone)", bundle: .module)
                } icon: {
                    Image(systemName: "mic.fill")
                }
            case .voiceprintMatch(let profileName):
                Label {
                    Text(verbatim: "\(profileName) (\(Int(suggestion.confidence * 100))%)")
                } icon: {
                    Image(systemName: "sparkles")
                }
            }
        }
        .appAction(.secondary, size: .small)
        .help(Text("Use suggested speaker name.", bundle: .module))
    }

    private func nameField(_ key: String) -> some View {
        let placeholder = AudioSpeakerRoster.defaultName(key)
        return TextField(text: Binding(get: { drafts[key] ?? "" }, set: { drafts[key] = $0 }),
                         prompt: Text(verbatim: placeholder)) {
            Text("Name for \(placeholder)", bundle: .module)
        }
        .textFieldStyle(.roundedBorder)
        .focused($focusedKey, equals: key)
        .frame(minWidth: 160 * scale, maxWidth: 360 * scale)
        .accessibilityLabel(Text("Name for \(placeholder)", bundle: .module))
    }

    @ViewBuilder private func sampleButton(_ summary: AudioSpeakerSummary) -> some View {
        let playing = controller.speakerSampleKey == summary.id && controller.referencePlaying
        let label = name(summary.id)
        Button {
            if let sample = summary.sample { controller.playSpeakerSample(item, key: summary.id, range: sample) }
        } label: {
            Label {
                HStack(spacing: 6) {
                    if playing { Text("Stop Sample", bundle: .module) } else { Text("Play Sample", bundle: .module) }
                    if let sample = summary.sample {
                        Text(verbatim: audioTime(sample.upperBound - sample.lowerBound)).monospacedDigit().foregroundStyle(.appSecondary)
                    }
                }
            } icon: { Image(systemName: playing ? "stop.fill" : "play.fill").contentTransition(.symbolEffect(.replace)) }
        }
        .appAction(.secondary, size: .small)
        .tsAnimation(TSMotion.hover, value: playing)
        .disabled(summary.sample == nil || item.clips.isEmpty || controller.hasRecordingActivity)
        .accessibilityLabel(playing ? Text("Stop the sample of \(label)", bundle: .module) : Text("Play a sample of \(label)", bundle: .module))
        .help(summary.sample == nil ? Text("This voice has no timed line to play.", bundle: .module) : Text("Play a few seconds of this voice.", bundle: .module))
    }

    private func mergeMenu(_ key: String, targets: [String]) -> some View {
        Menu {
            ForEach(targets, id: \.self) { target in
                Button { TSMotion.animate(TSMotion.select) { merge(key, into: target) } } label: { Text(verbatim: name(target)) }
            }
        } label: { Text("Merge Into", bundle: .module) }
        .appMenuAction(size: .small)
        .disabled(targets.isEmpty)
        .help(targets.isEmpty ? Text("There is no other speaker to merge into.", bundle: .module)
              : Text("Show this voice as another speaker. You can separate them again.", bundle: .module))
    }

    private func separateButton(_ key: String) -> some View {
        Button { TSMotion.animate(TSMotion.select) { roster.separate(key) } } label: { Text("Separate", bundle: .module) }
            .appAction(.secondary, size: .small)
            .help(Text("Show this voice as its own speaker again.", bundle: .module))
    }

    private func merge(_ key: String, into target: String) {
        guard roster.merge(key, into: target) else { return }
        if focusedKey == key { focusedKey = nil }
    }

    private var footer: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            if purpose == .afterTake {
                Text("You can rename anyone later from the transcript.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 12)
            Button { dismiss() } label: {
                if purpose == .afterTake { Text("Not now", bundle: .module) } else { Text("Cancel", bundle: .module) }
            }
            .appAction(.secondary)
            .keyboardShortcut(.cancelAction)
            Button { apply() } label: { Text("Apply Names", bundle: .module) }
                .appAction(.primary)
                .appPrimaryGlow(true)
                .keyboardShortcut(.defaultAction)
        }
    }

    private func apply() {
        var final = roster
        for (key, draft) in drafts { final.rename(key, to: draft) }
        AudioSpeakerMemory.recordSpeakerNaming(item: item, renamedSpeakers: drafts)
        controller.applySpeakerRoster(final, to: item.id)
        dismiss()
    }
}
