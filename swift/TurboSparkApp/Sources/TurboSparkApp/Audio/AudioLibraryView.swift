import SwiftUI

@MainActor
struct AudioLibraryView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @Environment(\.appTheme) private var theme
    @State private var filter: AudioLibraryFilter = .all
    @State private var collection: String?
    @State private var tag: String?

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    private var visibleItems: [AudioLibraryItem] {
        AudioLibraryOrganizer.filter(controller.filteredItems, by: filter, collection: collection, tag: tag)
    }

    var body: some View {
        GeometryReader { geometry in
            if geometry.size.width > (850 * scale).rounded() {
                HSplitView {
                    library.frame(
                        minWidth: (260 * scale).rounded(),
                        idealWidth: (310 * scale).rounded(),
                        maxWidth: (400 * scale).rounded()
                    )
                    detail.frame(minWidth: (360 * scale).rounded())
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
                AppEmptyState(
                    systemImage: "waveform",
                    title: Text("No recordings or takes found", bundle: .module),
                    message: Text("Reopen recordings and takes, or start with an audio file.", bundle: .module)
                ) {
                    Button { controller.importAudio() } label: {
                        Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                    }
                    .appAction(.secondary, size: .small)
                    .disabled(controller.isBusy || controller.hasRecordingActivity)
                }
                .padding(.vertical, 20)
            } else {
                ScrollView {
                    AudioLibraryList(controller: controller, items: visibleItems)
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
            AppEmptyState(
                systemImage: "waveform",
                title: Text("Choose a recording or take", bundle: .module),
                message: Text("Audio and edits are saved in your encrypted profile.", bundle: .module)
            )
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

@MainActor
struct AudioItemDetailView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    enum DetailViewTab: String, CaseIterable, Identifiable {
        case transcript = "Transcript"
        case notes = "Notes & Summary"
        var id: String { rawValue }
    }

    @State private var selectedTab: DetailViewTab = .transcript

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
            if !item.transcripts.isEmpty { detailContentSwitcher }
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

    private var detailContentSwitcher: some View {
        VStack(alignment: .leading, spacing: 14) {
            AudioDetailModeSwitcher(selection: $selectedTab, hasSummary: item.summary != nil)

            if selectedTab == .transcript {
                AudioTranscriptEditorView(controller: controller, item: item)
            } else {
                AudioMeetingNotesView(controller: controller, item: item)
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

/// Switches an item's detail between its transcript and its meeting notes.
/// An `AppPillPicker`, which never shrinks: the labeled segmented picker this
/// replaces was squeezed to 280 pt and wrapped "Detail View" one or two
/// letters per line.
@MainActor
struct AudioDetailModeSwitcher: View {
    @Binding var selection: AudioItemDetailView.DetailViewTab
    let hasSummary: Bool

    var body: some View {
        AppPillPicker(
            selection: $selection,
            items: [
                AppPillItem(value: .transcript, label: Text("Transcript", bundle: .module)),
                AppPillItem(
                    value: .notes,
                    label: Text("Notes & Summary", bundle: .module),
                    systemImage: hasSummary ? "checkmark.circle.fill" : nil),
            ],
            label: Text("Detail View", bundle: .module)
        )
    }
}
