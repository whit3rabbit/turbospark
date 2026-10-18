import SwiftUI

/// Notes and Meeting: the mode's own takes beside the selected one, or the
/// take being recorded while this mode's capture runs.
@MainActor
struct SpeechToTextCapturePane: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let mode: SpeechToTextMode

    private var items: [AudioLibraryItem] {
        SpeechToTextLibrary.items(for: mode, in: controller.items, collections: controller.preferences.presetCollections)
    }

    var body: some View {
        let items = self.items
        let live = controller.liveItem(for: mode)
        if items.isEmpty && live == nil {
            emptyState
        } else {
            SpeechToTextSplit {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        listTitle.themedFont(.title3, weight: .semibold).accessibilityAddTraits(.isHeader)
                        Text(verbatim: "\(items.count)").monospacedDigit().themedFont(.small).foregroundStyle(.appSecondary)
                    }
                    AudioLibraryList(controller: controller, items: items,
                                     selectedID: SpeechToTextLibrary.listSelection(controller.selectedID, in: controller.items))
                }
            } detail: {
                if let live {
                    SpeechToTextLiveTranscript(model: model, controller: controller, item: live)
                } else if let item = controller.selectedItem, items.contains(where: { $0.id == item.id }) {
                    AudioItemDetailView(controller: controller, item: item)
                } else {
                    AppEmptyState(systemImage: mode.symbol,
                                  title: Text("Choose a recording or take", bundle: .module),
                                  message: Text("Audio and edits are saved in your encrypted profile.", bundle: .module))
                }
            }
        }
    }

    @ViewBuilder private var listTitle: some View {
        if mode == .meeting { Text("Meetings", bundle: .module) } else { Text("Voice Notes", bundle: .module) }
    }

    @ViewBuilder private var emptyState: some View {
        if mode == .meeting {
            AppEmptyState(systemImage: mode.symbol,
                          title: Text("No meetings yet", bundle: .module),
                          message: Text("Record a call with your microphone and the app playing it. Speakers are labeled after you stop.", bundle: .module))
        } else {
            AppEmptyState(systemImage: mode.symbol,
                          title: Text("No voice notes yet", bundle: .module),
                          message: Text("Record a voice note. The words appear as you speak, and the note keeps its audio.", bundle: .module))
        }
    }
}
