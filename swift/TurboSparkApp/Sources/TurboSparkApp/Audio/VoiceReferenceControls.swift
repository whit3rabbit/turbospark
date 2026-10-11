import SwiftUI

/// Choosing, previewing and trimming the reference voice. Shared by every
/// cloning engine; each engine view adds its own fields below it.
@MainActor
struct VoiceReferenceControls: View {
    @ObservedObject var controller: AudioWorkspaceController

    private func trim(_ path: WritableKeyPath<AudioVoiceReference, Double>) -> Binding<Double> {
        Binding(get: { controller.speechSettings.reference?[keyPath: path] ?? 0 }, set: {
            controller.stopReferencePreview()
            controller.speechSettings.reference?[keyPath: path] = $0
            // A transcript of a different region is invalid conditioning.
            controller.speechSettings.referenceText = ""
        })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Reference voice", bundle: .module).themedFont(.base, weight: .semibold)
            HStack {
                Button { controller.importAudio(asVoiceReference: true) } label: {
                    Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                }.disabled(controller.importTask != nil)
                Menu {
                    ForEach(controller.items.filter { $0.hasPlayableAudio && $0.status == .completed }) { item in
                        Button { controller.setVoiceReference(item) } label: { Text(item.title) }
                    }
                } label: { Text("Saved audio", bundle: .module) }
                Button { controller.selectPage(.record) } label: { Text("Record", bundle: .module) }
            }
            Text("Import MP3, WAV, M4A, FLAC, AIFF, or other audio formats supported by macOS.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            if let reference = controller.speechSettings.reference {
                HStack {
                    Image(systemName: "waveform").accessibilityHidden(true)
                    Text(reference.title).lineLimit(2)
                    Spacer()
                    Button { controller.previewVoiceReference() } label: {
                        if controller.referencePlaying { Text("Stop preview", bundle: .module) }
                        else { Text("Preview reference audio", bundle: .module) }
                    }.disabled(!reference.isValid)
                    Button { controller.clearVoiceReference() } label: { Text("Remove", bundle: .module) }
                }
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Start (seconds)", bundle: .module).themedFont(.small)
                        TextField(value: trim(\.start), format: .number) { Text("Start (seconds)", bundle: .module) }
                            .accessibilityLabel(Text("Start (seconds)", bundle: .module))
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Duration (seconds)", bundle: .module).themedFont(.small)
                        TextField(value: trim(\.duration), format: .number) { Text("Duration (seconds)", bundle: .module) }
                            .accessibilityLabel(Text("Duration (seconds)", bundle: .module))
                    }
                }.textFieldStyle(.roundedBorder)
                Text("Use a clear recording of one person speaking for 1 to 30 seconds.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
    }
}
