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

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            transcriptHeader
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
                            TextField(text: Binding(
                                get: { segment.speaker ?? "" },
                                set: { segment.speaker = $0.isEmpty ? nil : $0; stageEdits() }
                            )) { Text("Speaker", bundle: .module) }
                            .textFieldStyle(.roundedBorder)
                            .accessibilityLabel(Text("Speaker", bundle: .module))
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
                ForEach(segments.filter { $0.speaker != nil }) { segment in
                    AccessibilityRotorEntry((segment.speaker ?? "") + ", " + audioTime(segment.start), id: segment.id)
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
        .onChange(of: segments) { _, _ in dirty = segments != savedSegments }
        .onDisappear { flushEdits() }
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
