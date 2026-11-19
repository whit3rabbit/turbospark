import SwiftUI

/// Keeps the reason beside the disabled action when setup or input is missing.
@MainActor
struct AudioRunControls<Actions: View>: View {
    @ObservedObject var controller: AudioWorkspaceController
    @ViewBuilder let actions: () -> Actions

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let reason = runBlockedReason {
                reason.themedFont(.small).foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            actionButtons
        }
    }

    private var actionButtons: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { actions(); Spacer(minLength: 16); runButton }
            VStack(alignment: .leading, spacing: 12) {
                actions()
                runButton
            }
        }
        .padding(.top, 4)
    }

    private var runBlockedReason: Text? {
        guard !controller.canRun else { return nil }
        if controller.hasRecordingActivity { return Text("Recording", bundle: .module) }
        if controller.isBusy { return Text("Wait for the current generation or model load to finish.", bundle: .module) }
        if controller.importTask != nil { return Text("Importing audio", bundle: .module) }
        if controller.isInstalling { return Text("Downloading audio model", bundle: .module) }
        if controller.selectedProfile == nil { return Text("Choose a model", bundle: .module) }
        if controller.page == .music && controller.modelGate != nil { return Text("Download the music model", bundle: .module) }
        if !controller.modelInstalled { return Text("Download this model or choose its checkpoint folder.", bundle: .module) }
        if !controller.selectedProfileIsRunnable { return Text("Runtime unavailable", bundle: .module) }
        if controller.task == .textToSpeech {
            if controller.recipe.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return Text("Write a script to generate speech.", bundle: .module)
            }
            if let message = controller.speechValidationMessage { return Text(verbatim: message) }
        } else if controller.task == .music {
            if let message = controller.musicValidationMessage { return Text(verbatim: message) }
            if controller.recipe.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return Text("Describe a soundtrack and keep each take for comparison.", bundle: .module)
            }
        } else if controller.selectedItem?.clips.isEmpty != false {
            return Text("Import or select an audio source first.", bundle: .module)
        }
        return nil
    }

    private var runButton: some View {
        Button { controller.run() } label: {
            Label { Text(LocalizedStringKey(controller.runTitle), bundle: .module) }
                icon: { Image(systemName: controller.page.symbol) }
        }
        .appAction(.primary).disabled(!controller.canRun)
        .help(runBlockedReason ?? Text(LocalizedStringKey(controller.runTitle), bundle: .module))
        .keyboardShortcut(.return, modifiers: .command)
    }
}
