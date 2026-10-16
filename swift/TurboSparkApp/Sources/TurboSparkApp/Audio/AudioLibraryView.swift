import SwiftUI

@MainActor
struct AudioLibraryView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @State private var filter: AudioLibraryFilter = .all
    @State private var collection: String?
    @State private var tag: String?
    @State private var editingItem: AudioLibraryItem?
    @State private var editingCollection = false
    @State private var editedName = ""
    @State private var deletingItem: AudioLibraryItem?

    private var visibleItems: [AudioLibraryItem] {
        AudioLibraryOrganizer.filter(controller.filteredItems, by: filter, collection: collection, tag: tag)
    }

    var body: some View {
        GeometryReader { geometry in
            if geometry.size.width > 850 {
                HSplitView {
                    library.frame(minWidth: 260, idealWidth: 310, maxWidth: 400)
                    detail.frame(minWidth: 360)
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        library
                        if let item = controller.selectedItem {
                            AudioItemDetailView(controller: controller, item: item).padding(.horizontal, 20)
                        }
                    }.padding(.bottom, 20)
                }
            }
        }
        .sheet(item: $editingItem) { item in
            AudioNameSheet(title: editingCollection ? "Collection" : "Rename", value: $editedName,
                           allowEmpty: editingCollection) {
                if editingCollection { controller.organize(item, collection: editedName) }
                else { controller.rename(item, title: editedName) }
                editingItem = nil
            }
        }
        .modifier(AudioDeleteConfirmation(controller: controller, item: $deletingItem))
    }

    private var library: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Audio library", bundle: .module)
                .themedFont(.title2, weight: .semibold).accessibilityAddTraits(.isHeader)
            AppSearchField(text: $controller.search, prompt: Text("Search recordings and takes", bundle: .module), maxWidth: .infinity)
            AudioLibraryFilterBar(controller: controller, filter: $filter, collection: $collection, tag: $tag)
            HStack {
                Button { controller.importAudio() } label: {
                    Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                }
                .appAction(.secondary, size: .small)
                .disabled(controller.isBusy || controller.hasRecordingActivity)
                Button { controller.selectPage(.record) } label: {
                    Label { Text("Record", bundle: .module) } icon: { Image(systemName: "mic") }
                }
                .appAction(.secondary, size: .small)
            }
            if !controller.search.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                AudioSearchResultsView(controller: controller, search: controller.librarySearch)
            } else if visibleItems.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("No recordings or takes found", bundle: .module).themedFont(.base, weight: .semibold)
                    Text("Reopen recordings and takes, or start with an audio file.", bundle: .module)
                        .foregroundStyle(.appSecondary)
                }.padding(.vertical, 20)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(AudioLibraryOrganizer.grouped(visibleItems), id: \.group) { section in
                            Text(LocalizedStringKey(section.group.titleKey), bundle: .module)
                                .themedFont(.small, weight: .semibold).foregroundStyle(.appSecondary)
                                .accessibilityAddTraits(.isHeader)
                                .padding(.top, section.group == .today ? 0 : 6)
                            ForEach(section.items) { item in libraryRow(item) }
                        }
                    }
                }.frame(minHeight: 180)
            }
            Spacer(minLength: 0)
        }
        .padding(20)
        .onChange(of: controller.search) { _, query in controller.librarySearch.run(query, controller: controller) }
        .task { controller.librarySearch.scheduleIndexing(controller) }
    }

    @ViewBuilder private var detail: some View {
        if let item = controller.selectedItem {
            ScrollView {
                AudioItemDetailView(controller: controller, item: item)
                    .padding(24).frame(maxWidth: 900, alignment: .leading).frame(maxWidth: .infinity)
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "waveform").themedFont(.hero).foregroundStyle(.appSecondary)
                Text("Choose a recording or take", bundle: .module).themedFont(.title2, weight: .semibold)
                Text("Audio and edits are saved in your encrypted profile.", bundle: .module)
                    .foregroundStyle(.appSecondary).multilineTextAlignment(.center)
            }.padding(24).frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func libraryRow(_ item: AudioLibraryItem) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Button { controller.select(item) } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(item.title).themedFont(.base, weight: .medium).lineLimit(2)
                    HStack {
                        Text(item.createdAt, format: .dateTime.month(.abbreviated).day())
                        if item.hasPlayableAudio { Text(audioTime(item.duration)).monospacedDigit() }
                    }.themedFont(.small).foregroundStyle(.appSecondary)
                    if !item.collection.isEmpty { Text(item.collection).themedFont(.small).foregroundStyle(.appSecondary) }
                    if !item.tagList.isEmpty {
                        Label { Text(verbatim: item.tagList.joined(separator: ", ")).lineLimit(1) } icon: { Image(systemName: "tag") }
                            .themedFont(.small).foregroundStyle(.appSecondary)
                    }
                    if item.status != .completed {
                        Label { Text(LocalizedStringKey(item.status.audioTitle), bundle: .module) }
                            icon: { Image(systemName: item.status.audioSymbol) }
                            .themedFont(.small)
                    }
                }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(controller.selectedID == item.id ? .isSelected : [])
            .accessibilityAction(named: Text("Favorite", bundle: .module)) { controller.toggleFavorite(item) }
            .accessibilityAction(named: Text("Rename", bundle: .module)) { beginEditing(item, collection: false) }
            .accessibilityAction(named: Text("Delete", bundle: .module)) { deletingItem = item }
            VStack(spacing: 8) {
                Button { controller.toggleFavorite(item) } label: {
                    Image(systemName: item.favorite ? "star.fill" : "star")
                }.buttonStyle(.plain)
                    .accessibilityLabel(Text(item.favorite ? "Remove favorite" : "Favorite", bundle: .module))
                Menu {
                    Button { beginEditing(item, collection: false) } label: { Text("Rename", bundle: .module) }
                    Button { beginEditing(item, collection: true) } label: { Text("Collection", bundle: .module) }
                    Button { controller.select(item); controller.duplicateSelected() } label: {
                        Text("Duplicate with changes", bundle: .module)
                    }
                    Divider()
                    Button(role: .destructive) { deletingItem = item } label: { Text("Delete", bundle: .module) }
                } label: { Image(systemName: "ellipsis") }
                .menuStyle(.borderlessButton).fixedSize()
                .accessibilityLabel(Text("Audio actions", bundle: .module))
                .disabled(controller.hasRecordingActivity || controller.isBusy)
                AudioDeleteButton(controller: controller, item: item, compact: true)
            }
        }
        .padding(12)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
        .overlay { RoundedRectangle(cornerRadius: 10).stroke(controller.selectedID == item.id ? .appAccent : .appBorder, lineWidth: controller.selectedID == item.id ? 2 : 1) }
    }

    private func beginEditing(_ item: AudioLibraryItem, collection: Bool) {
        editingCollection = collection
        editedName = collection ? item.collection : item.title
        editingItem = item
    }
}

@MainActor
struct AudioItemDetailView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    private var comparisonItems: [AudioLibraryItem] {
        controller.items.filter { $0.id != item.id && !$0.clips.isEmpty }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(item.title).themedFont(.title2, weight: .semibold).textSelection(.enabled)
                        .accessibilityAddTraits(.isHeader)
                    Text(item.createdAt, format: .dateTime.month(.wide).day().year().hour().minute())
                        .themedFont(.small).foregroundStyle(.appSecondary)
                    AudioTagEditor(controller: controller, item: item).padding(.top, 4)
                }
                Spacer()
            }
            HStack(spacing: 12) {
                if item.hasPlayableAudio { AudioSaveAsButton(controller: controller, item: item) }
                if item.hasPlayableAudio && item.kind == .voiceover {
                    Button { controller.reuseVoice(item) } label: { Text("Use as reference voice", bundle: .module) }
                        .appAction(.secondary, size: .small)
                        .disabled(controller.isBusy || controller.hasRecordingActivity)
                }
                if item.preferredTranscript != nil { AudioExportMenu(controller: controller, item: item) }
                Spacer(minLength: 8)
                AudioDeleteButton(controller: controller, item: item)
            }
            if !item.clips.isEmpty { AudioTransportView(controller: controller) }
            if let failure = item.failure {
                Label { Text(failure).textSelection(.enabled) } icon: { Image(systemName: "exclamationmark.triangle") }
                    .foregroundStyle(.appSecondary)
            }
            itemActions
            if !item.markers.isEmpty { markers }
            if item.clips.count > 1 { clips }
            if !item.transcripts.isEmpty { AudioTranscriptEditorView(controller: controller, item: item) }
            if item.preferredTranscript != nil { meetingSummary }
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 8) {
                    if let modelID = item.recipe.modelID { Text(modelID).textSelection(.enabled) }
                    if !item.recipe.text.isEmpty { Text(item.recipe.text).textSelection(.enabled) }
                    if !item.recipe.caption.isEmpty { Text(item.recipe.caption).textSelection(.enabled) }
                    if !item.recipe.lyrics.isEmpty { Text(item.recipe.lyrics).textSelection(.enabled) }
                    HStack { Text("Seed", bundle: .module); Text(verbatim: String(item.recipe.seed)) }
                    if let parent = controller.items.first(where: { $0.id == item.sourceID }) {
                        Button { controller.select(parent) } label: {
                            Label { Text(parent.title) } icon: { Image(systemName: "arrow.turn.up.left") }
                        }
                    }
                }.themedFont(.small).padding(.top, 8)
            } label: { Text("Saved settings and source", bundle: .module) }
        }
    }

    private var itemActions: some View {
        ViewThatFits(in: .horizontal) {
            HStack { duplicateButton; runAgainButton; compareMenu }
            VStack(alignment: .leading, spacing: 8) { duplicateButton; runAgainButton; compareMenu }
        }
    }

    private var meetingSummary: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let summary = item.summary {
                Text("Meeting summary", bundle: .module)
                    .themedFont(.base, weight: .semibold).accessibilityAddTraits(.isHeader)
                Text(summary.text).textSelection(.enabled)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 90), alignment: .leading)], alignment: .leading) {
                    ForEach(Array(summary.sourceTimes.enumerated()), id: \.offset) { _, time in
                        Button { controller.seek(to: time) } label: {
                            Label { Text(audioTime(time)).monospacedDigit() } icon: { Image(systemName: "play.circle") }
                        }.disabled(item.clips.isEmpty || controller.hasRecordingActivity)
                    }
                }
            }
            Button { controller.summarizeMeeting() } label: {
                Text("Summarize meeting", bundle: .module)
            }
            .appAction(.secondary, size: .small)
            .disabled(!controller.canSummarizeMeeting || controller.isBusy || controller.hasRecordingActivity)
            if !controller.canSummarizeMeeting && !controller.isBusy {
                Text("Load a local chat model to summarize this transcript.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
    }

    private var duplicateButton: some View {
        Button { controller.duplicateSelected() } label: { Text("Duplicate with changes", bundle: .module) }
            .appAction(.secondary, size: .small)
            .disabled(controller.isBusy || controller.hasRecordingActivity)
    }

    private var runAgainButton: some View {
        Button { controller.runAgain(item) } label: { Text("Run again", bundle: .module) }
            .appAction(.secondary, size: .small)
            .disabled(controller.isBusy || controller.hasRecordingActivity || item.recipe.modelID == nil)
    }

    private var compareMenu: some View {
        Menu {
            ForEach(comparisonItems) { candidate in
                Button { controller.compare(with: candidate) } label: { Text(candidate.title) }
            }
        } label: { Text("Compare at this position", bundle: .module) }
        .appMenuAction(size: .small)
        .disabled(item.clips.isEmpty || comparisonItems.isEmpty || controller.hasRecordingActivity)
        .help(Text("Switch takes while keeping the playback position.", bundle: .module))
    }

    private var markers: some View {
        DisclosureGroup {
            ForEach(item.markers) { marker in
                Button { controller.seek(to: marker.time) } label: {
                    HStack {
                        Image(systemName: "bookmark")
                        Text(audioTime(marker.time)).monospacedDigit()
                        Text(marker.title)
                    }
                }.padding(.vertical, 4)
                    .disabled(controller.hasRecordingActivity)
            }
        } label: { Text("Markers", bundle: .module) }
        .accessibilityRotor(Text("Markers", bundle: .module)) {
            ForEach(item.markers) { marker in
                AccessibilityRotorEntry(marker.title + ", " + audioTime(marker.time), id: marker.id)
            }
        }
    }

    private var clips: some View {
        DisclosureGroup {
            ForEach(item.clips) { clip in
                HStack {
                    Text(clip.source)
                    Spacer()
                    Text(audioTime(clip.duration)).monospacedDigit().foregroundStyle(.appSecondary)
                    Menu {
                        Button { controller.exportClip(clip, preset: .nativeWAV) } label: { Text("WAV (original format)", bundle: .module) }
                        Button { controller.exportClip(clip, preset: .videoWAV) } label: { Text("WAV (48 kHz for video)", bundle: .module) }
                        Button { controller.exportClip(clip, preset: .m4a) } label: { Text("M4A", bundle: .module) }
                    } label: {
                        Label { Text("Export", bundle: .module) } icon: { Image(systemName: "square.and.arrow.up") }
                    }
                    .appMenuAction(size: .small)
                    .disabled(controller.hasRecordingActivity)
                }.padding(.vertical, 4)
            }
        } label: { Text("Tracks and sections", bundle: .module) }
    }
}

private extension AudioLibraryItem.Status {
    var audioTitle: String {
        switch self {
        case .draft: return "Draft"
        case .recording: return "Recording"
        case .processing: return "Processing"
        case .completed: return "Completed"
        case .interrupted: return "Interrupted"
        case .failed: return "Failed"
        }
    }
    var audioSymbol: String {
        switch self {
        case .draft: return "pencil"
        case .recording: return "record.circle"
        case .processing: return "hourglass"
        case .completed: return "checkmark.circle"
        case .interrupted: return "pause.circle"
        case .failed: return "exclamationmark.triangle"
        }
    }
}
