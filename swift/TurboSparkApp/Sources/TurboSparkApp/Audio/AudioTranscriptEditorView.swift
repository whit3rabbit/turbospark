import SwiftUI

@MainActor
struct AudioTranscriptEditorView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    @State private var editingID: UUID?
    @State private var revisionID: UUID?
    @State private var segments: [AudioLibrarySegment] = []
    @State private var savedSegments: [AudioLibrarySegment] = []
    @State private var dirty = false
    @State private var namingSpeakers = false

    /// Names and merges live on the take, so every line, revision and
    /// export shows a rename at once.
    private var roster: AudioSpeakerRoster { AudioSpeakerRoster(item) }

    var body: some View {
        let roster = self.roster
        let speakers = roster.roots(in: segments)
        VStack(alignment: .leading, spacing: 12) {
            transcriptHeader
            if !speakers.isEmpty { speakersBar }
            if item.preferredTranscript?.source == .draft {
                Label { Text("Draft transcript", bundle: .module) } icon: { Image(systemName: "pencil") }
                    .foregroundStyle(.appSecondary)
            }
            LazyVStack(alignment: .leading, spacing: 16) {
                ForEach($segments) { $segment in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Button { controller.seek(to: segment.start) } label: {
                                Text(audioTime(segment.start)).monospacedDigit()
                            }.disabled(item.clips.isEmpty || controller.hasRecordingActivity)
                            speakerMenu(segment, roster: roster, speakers: speakers)
                            if segment.timing == "clip" || segment.timing == "window" {
                                Text("Approximate timing", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                            }
                        }
                        TextEditor(text: Binding(
                            get: { segment.text },
                            set: { segment.text = $0; stageEdits() }
                        )).themedFont(.base)
                            .frame(minHeight: 72)
                            .accessibilityLabel(Text("Transcript text", bundle: .module))
                            .padding(6).overlay { RoundedRectangle(cornerRadius: 6).stroke(.appBorder) }
                    }
                    .accessibilityElement(children: .contain)
                }
            }
            .accessibilityRotor(Text("Speakers", bundle: .module)) {
                ForEach(segments.filter { AudioSpeakerRoster.key($0) != nil }) { segment in
                    AccessibilityRotorEntry((roster.label(segment) ?? "") + ", " + audioTime(segment.start), id: segment.id)
                }
            }
            if let data = item.transcripts.first(where: { $0.id == revisionID })?.nativeResult,
               let result = String(data: data, encoding: .utf8) {
                DisclosureGroup {
                    Text(result).themedCode(.small).textSelection(.enabled).padding(.top, 8)
                } label: { Text("Advanced result details", bundle: .module) }
            }
        }
        .onAppear { load(item) }
        .onChange(of: item.id) { _, _ in flushEdits(); load(item) }
        .onChange(of: item.preferredTranscript?.id) { _, _ in
            if !dirty || item.preferredTranscript?.segments == segments { load(item) }
        }
        // Speaker labels land on the revision already shown, after the final
        // pass saved it, so its id does not change. Take them unless the user
        // has unsaved edits.
        .onChange(of: item.transcripts.first(where: { $0.id == revisionID })?.segments) { _, latest in
            guard !dirty, let latest, latest != savedSegments else { return }
            savedSegments = latest
            segments = latest
        }
        .onChange(of: segments) { _, _ in dirty = segments != savedSegments }
        .onDisappear { flushEdits() }
        .sheet(isPresented: $namingSpeakers) {
            AudioSpeakerNamingSheet(controller: controller, item: item, purpose: .edit)
        }
    }

    /// Opens Name the Speakers on the saved transcript, so it lists the
    /// voices as the lines show them.
    private var speakersBar: some View {
        FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) {
            Button { if flushEdits() { namingSpeakers = true } } label: {
                Label { Text("Name Speakers...", bundle: .module) } icon: { Image(systemName: "person.2") }
            }
            .appAction(.secondary, size: .small)
            .disabled(controller.hasRecordingActivity)
            if let parts = item.speakerLabelParts, parts > 1 {
                Label { Text("Labeled in \(parts) parts", bundle: .module) } icon: { Image(systemName: "checkmark.circle") }
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
    }

    /// Which voice a line is from. Picking one moves only this line; names
    /// are given once, in Name the Speakers.
    private func speakerMenu(_ segment: AudioLibrarySegment, roster: AudioSpeakerRoster, speakers: [String]) -> some View {
        let current = AudioSpeakerRoster.key(segment).map(roster.root)
        return Menu {
            ForEach(speakers, id: \.self) { key in
                Button { reassign(segment.id, to: key) } label: {
                    if key == current {
                        Label { Text(verbatim: roster.name(key)) } icon: { Image(systemName: "checkmark") }
                    } else {
                        Text(verbatim: roster.name(key))
                    }
                }
            }
            if !speakers.isEmpty { Divider() }
            Button { reassign(segment.id, to: AudioSpeakerRoster.nextID(in: item, also: segments)) } label: { Text("New Speaker", bundle: .module) }
            Button { reassign(segment.id, to: nil) } label: { Text("No Speaker", bundle: .module) }
                .disabled(current == nil)
        } label: {
            if let label = roster.label(segment) { Text(verbatim: label) } else { Text("No Speaker", bundle: .module) }
        }
        .appMenuAction(size: .small)
        .accessibilityLabel(Text("Speaker", bundle: .module))
        .accessibilityValue(roster.label(segment).map { Text(verbatim: $0) } ?? Text("No Speaker", bundle: .module))
    }

    private func reassign(_ id: UUID, to key: String?) {
        guard let index = segments.firstIndex(where: { $0.id == id }) else { return }
        segments[index] = AudioSpeakerRoster.assigning(segments[index], to: key)
        stageEdits()
    }

    private var transcriptHeader: some View {
        ViewThatFits(in: .horizontal) {
            HStack { revisionPicker; saveButton }
            VStack(alignment: .leading) { revisionPicker; saveButton }
        }
    }

    private var revisionPicker: some View {
        Picker(selection: Binding(get: { revisionID }, set: { id in
            guard flushEdits() else { return }
            if let revision = item.transcripts.first(where: { $0.id == id }) { loadRevision(revision) }
        })) {
            ForEach(item.transcripts) { revision in
                HStack {
                    Text(LocalizedStringKey(revision.source.audioTitle), bundle: .module)
                    Text(revision.createdAt, format: .dateTime.hour().minute().second())
                }.tag(Optional(revision.id))
            }
        } label: { Text("Transcript revision", bundle: .module) }
    }

    private var saveButton: some View {
        Button { flushEdits() } label: { Text("Save edits", bundle: .module) }
            .appAction(.secondary, size: .small)
            .disabled(!dirty)
    }

    private func load(_ value: AudioLibraryItem) {
        editingID = value.id
        if let revision = value.preferredTranscript { loadRevision(revision) }
        else { savedSegments = []; segments = []; revisionID = nil; dirty = false }
        if let pending = controller.transcriptDraft(for: value.id) {
            segments = pending
            dirty = segments != savedSegments
        }
    }

    private func loadRevision(_ revision: AudioTranscriptRevision) {
        revisionID = revision.id
        savedSegments = revision.segments
        segments = revision.segments
        dirty = false
    }

    private func stageEdits() {
        dirty = segments != savedSegments
        guard let editingID else { return }
        // Stage reversions too, so a pending autosave cannot restore text the user undid.
        controller.stageTranscriptEdits(segments, for: editingID)
    }

    @discardableResult private func flushEdits() -> Bool {
        guard dirty, let editingID else { return true }
        if controller.saveTranscript(segments, for: editingID) {
            savedSegments = segments
            dirty = false
            return true
        }
        return false
    }
}

private extension AudioTranscriptRevision.Source {
    var audioTitle: String {
        switch self {
        case .draft: return "Draft"
        case .final: return "Final"
        case .user: return "Edited"
        }
    }
}
