import SwiftUI

/// Any Audio: bring audio in (import a file, or capture another app), list
/// it, and transcribe it. A capture streams live text like the other modes;
/// an imported file gets a transcript of its own.
@MainActor
struct SpeechToTextAnyAudioPane: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    private var items: [AudioLibraryItem] {
        SpeechToTextLibrary.items(for: .anyAudio, in: controller.items, collections: controller.preferences.presetCollections)
    }

    var body: some View {
        let items = self.items
        VStack(spacing: 0) {
            intake.padding(.horizontal, 24).padding(.vertical, 12)
            Divider()
            if items.isEmpty && controller.liveItem(for: .anyAudio) == nil {
                AppEmptyState(systemImage: SpeechToTextMode.anyAudio.symbol,
                              title: Text("No audio yet", bundle: .module),
                              message: Text("Import a file or capture audio from another app, then transcribe it.", bundle: .module)) {
                    importButton
                }
            } else {
                SpeechToTextSplit {
                    AudioLibraryList(controller: controller, items: items,
                                     selectedID: SpeechToTextLibrary.listSelection(controller.selectedID, in: controller.items))
                } detail: {
                    detail(items)
                }
            }
        }
    }

    // A menu, not a chip per app: ScreenCaptureKit lists every running app
    // (Dock, Spotlight, agents), which flooded the row as chips.
    private var intake: some View {
        VStack(alignment: .leading, spacing: 8) {
            FlowLayout(spacing: 8, lineSpacing: 8) {
                importButton
                Text("Capture from an app", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                AudioCaptureSourcePickers(controller: controller, style: .menus)
            }
            Text("TurboSpark playback is excluded. Only audio is saved.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var importButton: some View {
        Button { controller.importAudio() } label: {
            Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .appAction(.secondary, size: .small)
        .disabled(controller.isBusy || controller.hasRecordingActivity)
        .help(
            (controller.isBusy || controller.hasRecordingActivity)
                ? Text("Recording can start when the current audio job finishes.", bundle: .module)
                : Text("Import audio", bundle: .module)
        )
    }

    @ViewBuilder private func detail(_ items: [AudioLibraryItem]) -> some View {
        if let live = controller.liveItem(for: .anyAudio) {
            SpeechToTextLiveTranscript(model: model, controller: controller, item: live)
        } else if let selected = controller.selectedItem, selected.kind == .transcription {
            AudioItemDetailView(controller: controller, item: selected)
        } else if let source = controller.selectedItem, items.contains(where: { $0.id == source.id }) {
            if let transcript = SpeechToTextLibrary.transcriptItem(for: source, in: controller.items) {
                AudioItemDetailView(controller: controller, item: transcript)
            } else {
                VStack(alignment: .leading, spacing: 20) {
                    transcribeCard(source)
                    AudioItemDetailView(controller: controller, item: source)
                }
            }
        } else {
            AppEmptyState(systemImage: SpeechToTextMode.anyAudio.symbol,
                          title: Text("Choose a recording or take", bundle: .module),
                          message: Text("Audio and edits are saved in your encrypted profile.", bundle: .module))
        }
    }

    private var transcribeBlockedReason: Text? {
        controller.modelInstalled ? nil : Text("Download the audio model to transcribe.", bundle: .module)
    }

    private func transcribeCard(_ source: AudioLibraryItem) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "waveform").accessibilityHidden(true)
                Text(verbatim: source.title).lineLimit(2)
                Spacer(minLength: 8)
                Text(verbatim: audioTime(source.duration)).monospacedDigit()
            }
            .foregroundStyle(.appSecondary)
            AudioLanguagePicker(controller: controller)
            HStack(spacing: 12) {
                if let reason = transcribeBlockedReason {
                    reason.themedFont(.small).foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                Button { controller.run() } label: {
                    Label { Text("Transcribe", bundle: .module) } icon: { Image(systemName: "text.bubble") }
                }
                .appAction(.primary)
                .appPrimaryGlow(controller.canRun)
                .disabled(!controller.canRun)
                .help(transcribeBlockedReason ?? Text("Transcribe", bundle: .module))
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .appCardSurface()
    }
}
