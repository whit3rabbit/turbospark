import SwiftUI

/// The Record route's capture panel: what the take is for, its sources, the
/// shared record bar (AudioRecordBar.swift) and the live transcription model.
/// Split from AudioWorkspaceView to keep that file readable.
@MainActor
struct AudioRecordingPanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            AudioRecordingPresetPicker(controller: controller)
            AudioCaptureSourcePickers(controller: controller)
            AudioRecordBar(controller: controller)
            AudioLiveTextView(live: controller.live)
            DisclosureGroup {
                Toggle(isOn: $controller.liveTranscription) { Text("Show a live draft transcript", bundle: .module) }
                if controller.liveTranscription { AudioLiveTextModelRow(model: model, controller: controller).padding(.top, 4) }
                AudioModelPickerView(model: model, controller: controller).padding(.top, 8)
                Text("Live text is a draft. Saved audio remains available when transcription falls behind.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            } label: { Text("Transcription model", bundle: .module) }
        }
        .padding(20).background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 1) }
    }
}
