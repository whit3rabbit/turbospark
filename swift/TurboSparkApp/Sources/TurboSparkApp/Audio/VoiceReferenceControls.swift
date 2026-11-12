import SwiftUI

/// Choosing, previewing and trimming the reference voice. Shared by every
/// cloning engine; each engine view adds its own fields below it.
@MainActor
struct VoiceReferenceControls: View {
    @ObservedObject var controller: AudioWorkspaceController

    private func trim(_ path: WritableKeyPath<AudioVoiceReference, Double>) -> Binding<Double> {
        Binding(get: { controller.speechSettings.reference?[keyPath: path] ?? 0 },
                set: { controller.setVoiceReferenceTrim(path, to: $0) })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppSectionHeader(title: Text("Reference voice", bundle: .module))
            HStack {
                Button { controller.importAudio(asVoiceReference: true) } label: {
                    Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                }
                .appAction(.secondary, size: .small)
                .disabled(controller.importTask != nil)
                .help(controller.importTask != nil ? Text("Importing audio", bundle: .module) : Text("Import audio", bundle: .module))
                Menu {
                    ForEach(controller.items.filter { $0.hasPlayableAudio && $0.status == .completed }) { item in
                        Button { controller.setVoiceReference(item) } label: { Text(verbatim: item.title) }
                    }
                } label: {
                    Label { Text("Saved audio", bundle: .module) } icon: { Image(systemName: "books.vertical") }
                }
                .appMenuAction(size: .small)
                Button { controller.selectPage(.record) } label: {
                    Label { Text("Record", bundle: .module) } icon: { Image(systemName: "mic") }
                }
                .appAction(.secondary, size: .small)
            }
            Text("Import MP3, WAV, M4A, FLAC, AIFF, or other audio formats supported by macOS.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            if let reference = controller.speechSettings.reference {
                HStack {
                    Image(systemName: "waveform").accessibilityHidden(true)
                    Text(verbatim: reference.title).lineLimit(2)
                    Spacer()
                    // Playback looks the same here as in the transport and
                    // the API test panel: secondary, with a play/stop symbol.
                    Button { controller.previewVoiceReference() } label: {
                        Label {
                            if controller.referencePlaying { Text("Stop preview", bundle: .module) }
                            else { Text("Preview reference audio", bundle: .module) }
                        } icon: {
                            Image(systemName: controller.referencePlaying ? "stop.fill" : "play.fill")
                        }
                    }
                    .appAction(.secondary, size: .small)
                    .disabled(!reference.isValid)
                    .help(controller.referencePlaying ? Text("Stop preview", bundle: .module) : Text("Preview reference audio", bundle: .module))
                    Button { controller.clearVoiceReference() } label: { Text("Remove", bundle: .module) }
                        .appAction(.secondary, size: .small)
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
