import SwiftUI

/// The Voice Cloning script box, styled as the chat composer: radius 20, the
/// composer fill and the accent focus glow, with the run controls in its
/// footer so the reason Generate is disabled sits next to the button.
@MainActor
struct VoiceCloneScriptComposer: View {
    @ObservedObject var controller: AudioWorkspaceController
    @ObservedObject var attention: VoiceCloneAttention
    let missing: [VoiceCloneStep]
    var scriptFocused: FocusState<Bool>.Binding
    let showPromptLibrary: () -> Void
    let savePrompt: () -> Void
    /// Runs the job, or shows what is missing when only input is.
    let generate: () -> Void
    let showMissing: () -> Void

    @Environment(\.appTheme) private var theme
    @Environment(\.locale) private var locale

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header.padding(.horizontal, 16).padding(.top, 14).padding(.bottom, 4)
            AudioMeasuredTextEditor(
                text: $controller.recipe.text,
                label: Text("Script", bundle: .module),
                placeholder: Text("Type or paste what the cloned voice should say.", bundle: .module),
                minHeight: (150 * scale).rounded(), maxHeight: (360 * scale).rounded(),
                focus: scriptFocused)
                .padding(.horizontal, 11)
            Text("Separate sections with a blank line.", bundle: .module)
                .themedFont(.tiny).foregroundStyle(.appSecondary)
                .padding(.horizontal, 16).padding(.bottom, 10)
            if scriptNeedsAttention {
                AppAttentionNote(message: scriptMessage)
                    .padding(.horizontal, 16).padding(.bottom, 12)
            }
            if controller.selectedProfile != nil {
                Divider().opacity(0.6)
                settingsRow.padding(.horizontal, 16).padding(.vertical, 10)
            }
            Divider().opacity(0.6)
            actionRow.padding(.horizontal, 16).padding(.vertical, 12)
                .id(VoiceCloneStep.generate)
        }
        .appComposerSurface(focused: scriptFocused.wrappedValue)
        .appAttention(scriptNeedsAttention, ping: attention.ping(.script), radius: AppMetrics.composerRadius)
        .tsAnimation(TSMotion.select, value: scriptNeedsAttention)
    }

    private var scriptNeedsAttention: Bool { attention.isOn(.script, missing: missing) }

    private var scriptMessage: Text {
        controller.recipe.text.count > AudioWorkspaceController.speechTextCharacterLimit
            ? Text("Split speech text longer than 6,000 characters into sections.", bundle: .module)
            : Text("Write a script to generate speech.", bundle: .module)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Label { Text("Script", bundle: .module) } icon: { Image(systemName: "text.quote") }
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 8)
            counter
        }
    }

    private var counter: some View {
        let count = controller.recipe.text.count
        let limit = AudioWorkspaceController.speechTextCharacterLimit
        let over = count > limit
        return HStack(spacing: 4) {
            if over { Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange) }
            Text(verbatim: count.formatted(.number.locale(locale)) + " / " + limit.formatted(.number.locale(locale)))
                .monospacedDigit()
                .contentTransition(.numericText())
                .tsAnimation(TSMotion.hover, value: count)
        }
        .themedFont(.tiny)
        .foregroundStyle(over ? AnyShapeStyle(.orange) : AnyShapeStyle(.appSecondary))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Characters", bundle: .module))
        .accessibilityValue(Text("\(count) of \(limit)", bundle: .module))
    }

    private var settingsRow: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 18) {
                AudioLanguagePicker(controller: controller).fixedSize()
                AudioVoicePicker(controller: controller).fixedSize()
                AudioSpeedControl(controller: controller).frame(minWidth: 200, maxWidth: 300)
                Spacer(minLength: 0)
            }
            VStack(alignment: .leading, spacing: 10) {
                AudioLanguagePicker(controller: controller)
                AudioVoicePicker(controller: controller)
                AudioSpeedControl(controller: controller)
            }
        }
        .themedFont(.small)
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }

    private var actionRow: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { libraryActions; Spacer(minLength: 12); statusLabel; runButton }
            VStack(alignment: .leading, spacing: 12) {
                libraryActions
                statusLabel
                runButton
            }
        }
    }

    private var libraryActions: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            Button(action: showPromptLibrary) {
                Label { Text("Prompt library", bundle: .module) } icon: { Image(systemName: "text.book.closed") }
            }
            .appAction(.secondary, size: .small)
            Button(action: savePrompt) { Text("Save Prompt", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(!controller.canSavePrompt)
        }
        .fixedSize()
    }

    /// What stands between the user and a run, beside the button: "Ready" when
    /// nothing, a "steps left" button when only their input does (it shows
    /// them where), and otherwise the app's own reason.
    @ViewBuilder private var statusLabel: some View {
        if controller.canRun {
            Label { Text("Ready to generate", bundle: .module) } icon: { Image(systemName: "checkmark.circle.fill") }
                .themedFont(.small).foregroundStyle(.appSecondary)
                .transition(.opacity)
        } else if controller.voiceCloneRunNeedsInput {
            Button(action: showMissing) {
                Label { Text("Steps left: \(missing.count)", bundle: .module) } icon: { Image(systemName: "exclamationmark.triangle.fill") }
            }
            .appAction(.quiet, size: .small)
            .foregroundStyle(AppAttention.color)
            .help(Text("Show what is missing", bundle: .module))
            .fixedSize()
            .transition(.opacity)
        } else if let reason = blockerText {
            Label { reason.fixedSize(horizontal: false, vertical: true) } icon: { Image(systemName: "info.circle") }
                .themedFont(.small).foregroundStyle(.appSecondary)
                .lineLimit(3)
                .transition(.opacity)
        }
    }

    private var runButton: some View {
        Button(action: generate) {
            Label { Text("Generate speech", bundle: .module) } icon: { Image(systemName: "person.wave.2") }
        }
        .appAction(.primary, size: .large)
        .appPrimaryGlow(controller.canRun)
        .keyboardShortcut(.return, modifiers: .command)
        // Stays pressable while only the user's input is missing, so a press
        // can show what that is instead of leaving a dead button.
        .disabled(!controller.canRun && !controller.voiceCloneRunNeedsInput)
        .help(blockerText ?? Text("Generate (Cmd+Return)", bundle: .module))
        .fixedSize()
    }

    /// The reason the run is disabled, from `voiceCloneRunBlocker`.
    private var blockerText: Text? {
        guard let blocker = controller.voiceCloneRunBlocker else { return nil }
        switch blocker {
        case .recording: return Text("Stop recording before generating speech.", bundle: .module)
        case .importing: return Text("Importing audio", bundle: .module)
        case .installing: return Text("Downloading audio model", bundle: .module)
        case .noModel: return Text("Choose a voice cloning model first.", bundle: .module)
        case .notInstalled: return Text("Download this model or choose its checkpoint folder.", bundle: .module)
        case .runtimeUnavailable: return Text("Runtime unavailable", bundle: .module)
        case .emptyScript: return Text("Write a script to generate speech.", bundle: .module)
        case .invalid(let message): return Text(message)
        }
    }
}

/// A `TextEditor` that grows with its text between `minHeight` and
/// `maxHeight`, then scrolls. The height is measured from a hidden `Text`
/// with the same font and insets (the `PromptComposerEditor` pattern): a
/// `ZStack` sizer or a min/max frame both land on the maximum.
@MainActor
struct AudioMeasuredTextEditor: View {
    @Binding var text: String
    let label: Text
    let placeholder: Text
    let minHeight: CGFloat
    let maxHeight: CGFloat
    var focus: FocusState<Bool>.Binding

    @Environment(\.appTheme) private var theme
    @State private var measured: CGFloat = 0

    var body: some View {
        Color.clear
            .frame(height: min(max(measured, minHeight), maxHeight))
            .tsAnimation(TSMotion.hover, value: measured)
            .background(alignment: .topLeading) {
                Text(text.isEmpty ? " " : text)
                    .font(theme.uiFont)
                    .padding(.horizontal, 5).padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .background {
                        GeometryReader { geometry in
                            Color.clear.preference(key: AudioEditorHeightKey.self, value: geometry.size.height)
                        }
                    }
                    .hidden()
                    .accessibilityHidden(true)
            }
            .onPreferenceChange(AudioEditorHeightKey.self) { measured = $0 }
            .overlay(alignment: .topLeading) {
                TextEditor(text: $text)
                    .font(theme.uiFont)
                    .scrollContentBackground(.hidden)
                    .focused(focus)
                    .accessibilityLabel(label)
                    .overlay(alignment: .topLeading) {
                        if text.isEmpty {
                            placeholder
                                .font(theme.uiFont)
                                .foregroundStyle(.tertiary)
                                .padding(.leading, 5).padding(.vertical, 8)
                                .allowsHitTesting(false)
                                .accessibilityHidden(true)
                        }
                    }
            }
    }
}

private struct AudioEditorHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}
