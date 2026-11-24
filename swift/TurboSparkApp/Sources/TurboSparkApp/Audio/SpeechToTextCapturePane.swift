import SwiftUI

/// Notes and Meeting: the mode's own takes beside the selected one, or the
/// take being recorded while this mode's capture runs.
@MainActor
struct SpeechToTextCapturePane: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let mode: SpeechToTextMode
    @State private var query = ""

    private var items: [AudioLibraryItem] {
        SpeechToTextLibrary.items(for: mode, in: controller.items, collections: controller.preferences.presetCollections)
    }

    var body: some View {
        let items = self.items
        let live = controller.liveItem(for: mode)
        if items.isEmpty && live == nil && !controller.hasRecordingActivity {
            emptyState {
                recordBar(.inline).frame(maxWidth: 620 * AppMetrics.textScale(theme))
            }
        } else {
            VStack(spacing: 0) {
                if items.isEmpty && live == nil {
                    emptyState { EmptyView() }
                } else {
                    takes(items, live: live)
                }
                recordBar(.capsule)
                    .frame(maxWidth: 980 * AppMetrics.textScale(theme))
                    .padding(.horizontal, 24).padding(.vertical, 12)
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private func takes(_ items: [AudioLibraryItem], live: AudioLibraryItem?) -> some View {
        let selection = live?.id ?? items.first { $0.id == controller.selectedID }?.id
        let shown = SpeechToTextLibrary.matching(items, query: query)
        return SpeechToTextSplit(selectionID: selection, detailMaxWidth: mode == .meeting ? 1_240 : 900) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    listTitle.themedFont(.title3, weight: .semibold).accessibilityAddTraits(.isHeader)
                    Text(verbatim: "\(items.count)").monospacedDigit().themedFont(.small).foregroundStyle(.appSecondary)
                }
                AppSearchField(text: $query, prompt: searchPrompt, maxWidth: .infinity)
                if shown.isEmpty {
                    Text("No matches", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                } else {
                    AudioLibraryList(controller: controller, items: shown,
                                     selectedID: SpeechToTextLibrary.listSelection(controller.selectedID, in: controller.items))
                }
            }
        } detail: {
            if let live {
                SpeechToTextLiveTranscript(model: model, controller: controller, item: live, withNotes: mode == .meeting)
            } else if let item = controller.selectedItem, items.contains(where: { $0.id == item.id }) {
                AudioItemDetailView(controller: controller, item: item, meetingLayout: mode == .meeting)
            } else {
                AppEmptyState(systemImage: mode.symbol,
                              title: Text("Choose a recording or take", bundle: .module),
                              message: Text("Audio and edits are saved in your encrypted profile.", bundle: .module))
            }
        }
    }

    private var searchPrompt: Text {
        mode == .meeting ? Text("Search meetings", bundle: .module) : Text("Search voice notes", bundle: .module)
    }

    @ViewBuilder private var listTitle: some View {
        if mode == .meeting { Text("Meetings", bundle: .module) } else { Text("Voice Notes", bundle: .module) }
    }

    @ViewBuilder private func emptyState<Actions: View>(@ViewBuilder actions: @escaping () -> Actions) -> some View {
        if mode == .meeting {
            AppEmptyState(systemImage: mode.symbol,
                          title: Text("No meetings yet", bundle: .module),
                          message: Text("Record a call with your microphone and the app playing it. Speakers are labeled after you stop.", bundle: .module),
                          actions: actions)
        } else {
            AppEmptyState(systemImage: mode.symbol,
                          title: Text("No voice notes yet", bundle: .module),
                          message: Text("Record a voice note. The words appear as you speak, and the note keeps its audio.", bundle: .module),
                          actions: actions)
        }
    }

    private func recordBar(_ layout: AudioRecordBar<AudioCaptureSourcePickers>.Layout) -> some View {
        AudioRecordBar(controller: controller, layout: layout) {
            AudioCaptureSourcePickers(controller: controller, style: .menus, showsAccessWarning: layout == .inline)
        }
    }
}
