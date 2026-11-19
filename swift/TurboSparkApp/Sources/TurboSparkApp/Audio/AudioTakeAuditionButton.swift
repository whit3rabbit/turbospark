import SwiftUI

/// Inline playback also selects the take so its full player shows the same
/// audio. The playing ID can differ from selection during a comparison.
@MainActor
struct AudioTakeAuditionButton: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    private var isPlaying: Bool { controller.playbackItemID == item.id && controller.isPlaying }
    private var label: Text { isPlaying ? Text("Pause", bundle: .module) : Text("Play", bundle: .module) }

    var body: some View {
        AppIconButton(isPlaying ? "pause.fill" : "play.fill", label: label) {
            if controller.playbackItemID == item.id {
                controller.playPause(item: item)
            } else {
                controller.select(item)
                controller.play(item, at: 0)
            }
        }
        .appAction(.quiet, size: .small)
        .foregroundStyle(.appAccentText)
        .disabled(controller.hasRecordingActivity || !item.hasPlayableAudio)
        .help(controller.hasRecordingActivity ? Text("Recording", bundle: .module) : label)
    }
}
