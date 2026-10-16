import SwiftUI

/// The selected engine's own cards in the voice setup column: consent and
/// quality for MegaTTS3, transcript and sampling for Qwen3-TTS.
@MainActor
struct VoiceCloneEngineCards: View {
    @ObservedObject var controller: AudioWorkspaceController
    @ObservedObject var attention: VoiceCloneAttention
    let missing: [VoiceCloneStep]
    var transcriptFocused: FocusState<Bool>.Binding
    var consentFocused: FocusState<Bool>.Binding
    /// Called when consent is asked for before there is a reference to
    /// consent to, so the page can point at the reference card.
    let needsReference: () -> Void

    private var locked: Bool { controller.isBusy || controller.hasRecordingActivity }

    var body: some View {
        if controller.selectedProfileClonesNatively {
            consentCard.id(VoiceCloneStep.consent)
            qualityCard
        } else if controller.selectedProfile?.family == "qwen3_tts" {
            transcriptCard.id(VoiceCloneStep.transcript)
            qwenSettingsCard
        }
    }

    private var consentCard: some View {
        let given = controller.cloneConsentGiven
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: given ? "checkmark.shield.fill" : "shield")
                    .themedFont(.title3)
                    .foregroundStyle(given ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
                    .contentTransition(.symbolEffect(.replace))
                    .accessibilityHidden(true)
                AppSectionHeader(title: Text("Consent", bundle: .module))
            }
            VoiceCloneConsentControls(controller: controller, focus: consentFocused, onNeedsReference: needsReference)
            if attention.isOn(.consent, missing: missing), controller.speechSettings.reference?.isValid == true {
                AppAttentionNote(message: Text("Confirm you have the right to use this voice.", bundle: .module))
            } else if controller.speechSettings.reference?.isValid != true {
                Label { Text("Add a reference voice first. Consent applies to that exact clip.", bundle: .module).fixedSize(horizontal: false, vertical: true) }
                    icon: { Image(systemName: "info.circle") }
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
        .tsAnimation(TSMotion.select, value: given)
        .tsAnimation(TSMotion.select, value: attention.isOn(.consent, missing: missing))
        .disabled(locked)
        .appCardSurface()
        .overlay {
            if given {
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(.appAccent.opacity(0.35), lineWidth: 1)
                    .allowsHitTesting(false)
            }
        }
        .appAttention(attention.isOn(.consent, missing: missing), ping: attention.ping(.consent))
    }

    private var qualityCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppSectionHeader(title: Text("Quality", bundle: .module))
            VoiceCloneQualityControls(controller: controller, showsPickerLabel: false)
        }
        .disabled(locked)
        .appCardSurface()
    }

    private var transcriptCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppSectionHeader(title: Text("Transcript of reference speech", bundle: .module))
            Qwen3EmbeddingOnlyToggle(controller: controller)
            if !controller.speechSettings.embeddingOnly {
                AudioMeasuredTextEditor(
                    text: controller.speechBinding(\.referenceText),
                    label: Text("Transcript of reference speech", bundle: .module),
                    placeholder: Text("Transcript", bundle: .module),
                    minHeight: 72, maxHeight: 200, focus: transcriptFocused)
                    .padding(.horizontal, 6)
                    .background(.appPage.opacity(0.6), in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                    .appFocusGlow(transcriptFocused.wrappedValue)
                    .transition(.opacity)
                if attention.isOn(.transcript, missing: missing) {
                    AppAttentionNote(message: Text("Enter the exact transcript of the selected reference region.", bundle: .module))
                }
            }
        }
        .tsAnimation(TSMotion.pane, value: controller.speechSettings.embeddingOnly)
        .disabled(locked)
        .appCardSurface()
        .appAttention(attention.isOn(.transcript, missing: missing), ping: attention.ping(.transcript))
    }

    private var qwenSettingsCard: some View {
        Qwen3SpeechGenerationSettings(controller: controller)
            .disabled(locked)
            .appCardSurface()
    }
}
