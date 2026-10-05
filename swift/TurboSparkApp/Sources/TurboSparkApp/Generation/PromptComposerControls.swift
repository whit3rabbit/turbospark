import SwiftUI
import TurboSpark

/// Popover button providing prompt authoring tips.
struct PromptTipsButton: View {
    let iconButtonSize: CGFloat
    @Binding var showingTips: Bool

    var body: some View {
        Button {
            showingTips.toggle()
        } label: {
            Label { Text("Prompt tips", bundle: .module) } icon: { Image(systemName: "questionmark.circle") }
                .labelStyle(.iconOnly)
                .frame(width: iconButtonSize, height: iconButtonSize)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.appSecondary)
        .help(Text("Prompt tips", bundle: .module))
        .accessibilityLabel("Prompt tips")
        .accessibilityHint("Shows a popover with prompt writing guidance")
        .popover(isPresented: $showingTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            PromptTipsGuideView()
        }
    }
}

/// Content inside the prompt tips popover.
struct PromptTipsGuideView: View {
    @Environment(\.appTheme) private var theme
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting tips", bundle: .module)
                .font(theme.ui(.callout, weight: .semibold))

            tipSection("Clear task & constraints",
                       "State what you want created, explained, or transformed. Specify length, style, or output structure.")
            tipSection("Provide types & interfaces",
                       "For code tasks, provide signatures, expected inputs/outputs, or small working scaffolds.")
            tipSection("Attach relevant documents",
                       "Attach PDFs, spreadsheets, or code files for local reasoning and question answering.")
        }
        .font(theme.ui(.small))
        .frame(width: 360, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).fontWeight(.semibold)
            Text(detail).foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Button triggering document file attachment dialog.
struct PromptAttachDocumentButton: View {
    let iconButtonSize: CGFloat
    let isRunning: Bool
    let isExtracting: Bool
    let onAttach: () -> Void
    @State private var isHovered: Bool = false

    var body: some View {
        Button(action: onAttach) {
            Group {
                if isExtracting {
                    TaskProgressFlameIcon(size: 16)
                } else {
                    Image(systemName: "plus")
                        .themedFont(.callout, weight: .medium)
                }
            }
            .frame(width: iconButtonSize, height: iconButtonSize)
            .background(
                Color.primary.opacity(isHovered ? 0.08 : 0.04),
                in: Circle()
            )
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.appSecondary)
        .onHover { isHovered = $0 }
        .disabled(isRunning || isExtracting)
        .help("Attach PDF, Word, Excel, code, or text files")
        .accessibilityLabel(isExtracting
                            ? "Extracting document text"
                            : "Attach documents")
        .accessibilityHint("Opens a file picker to attach documents to this prompt")
    }
}

/// Web search toggle button in the chat composer bar.
struct SearchToggleButton: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var isHovered: Bool = false

    var body: some View {
        Button {
            model.webSearchEnabled.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "globe")
                    .themedFont(.tiny, weight: .medium)
                Text("Search", bundle: .module)
                    .font(theme.ui(.small, weight: .medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .foregroundStyle(model.webSearchEnabled ? Color.primary : Color.secondary)
            .background(
                model.webSearchEnabled
                    ? TurboSparkTheme.accentColor.opacity(0.14)
                    : Color.primary.opacity(isHovered ? 0.08 : 0.04),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .overlay {
                if model.webSearchEnabled {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(.appAccent.opacity(0.3), lineWidth: 0.5)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help(model.webSearchEnabled ? "Web search enabled: tools will search the web for real-time information" : "Enable web search")
        .accessibilityLabel("Web search: \(model.webSearchEnabled ? "Enabled" : "Disabled")")
    }
}

/// The composer's microphone (docs/AUDIO_UI.md, capability 1).
///
/// Click dictates; the context menu chooses a voice note, macOS Dictation,
/// or app-audio capture. With in-app audio off it is exactly the old
/// button: a forward to system dictation. While recording it turns into a
/// labelled red Stop, the one place the app uses red, always beside text.
struct PromptAudioInputButton: View {
    @ObservedObject var model: AppModel
    @FocusState.Binding var promptFocused: Bool
    var onCaptureAppAudio: () -> Void = {}
    @ObservedObject private var recorder = ComposerAudioRecorder.shared
    @ObservedObject private var capabilities = AudioCapabilities.shared
    @AppStorage(AudioPreferences.experimentalEnabledKey) private var audioEnabled = false
    @State private var isHovered: Bool = false

    private var isRecording: Bool {
        if case .recording = recorder.phase { return true }
        return false
    }

    private var isBusy: Bool {
        recorder.phase == .requestingPermission || recorder.phase == .transcribing
    }

    private var symbol: String {
        if isRecording { return "stop.circle.fill" }
        if case .denied = recorder.phase { return "mic.slash" }
        return "mic"
    }

    var body: some View {
        Button {
            if isRecording {
                recorder.finish(model: model)
            } else {
                recorder.start(mode: .dictate, model: model, promptFocused: $promptFocused)
            }
        } label: {
            ZStack {
                Image(systemName: symbol)
                    .themedFont(.callout, weight: .medium)
                    .foregroundStyle(isRecording ? Color.red : Color.secondary)
                    .opacity(isBusy ? 0.35 : 1)
                if isBusy {
                    ProgressView().controlSize(.small)
                }
            }
            .frame(width: 28, height: 28)
            .background(
                Color.primary.opacity(isHovered ? 0.08 : 0),
                in: Circle()
            )
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        // Enabled even when access was denied: the click is what surfaces
        // the denied notice and its System Settings button.
        .disabled(isBusy)
        .onHover { isHovered = $0 }
        .help(helpText)
        .accessibilityLabel(isRecording
            ? Text("Stop recording", bundle: .module)
            : Text("Voice input", bundle: .module))
        .accessibilityHint(Text("Records speech and inserts the transcript into the prompt", bundle: .module))
        .contextMenu {
            Button {
                recorder.start(mode: .dictate, model: model, promptFocused: $promptFocused)
            } label: {
                Label { Text("Dictate (insert text)", bundle: .module) } icon: { Image(systemName: "text.cursor") }
            }
            .disabled(!audioEnabled || isRecording)
            Button {
                recorder.start(mode: .voiceNote, model: model, promptFocused: $promptFocused)
            } label: {
                Label { Text("Record voice note (attach audio)", bundle: .module) } icon: { Image(systemName: "waveform") }
            }
            .disabled(!audioEnabled || isRecording)
            Button {
                onCaptureAppAudio()
            } label: {
                Label { Text("Record audio from an app...", bundle: .module) } icon: { Image(systemName: "macwindow.badge.plus") }
            }
            .disabled(!audioEnabled || !capabilities.snapshot.systemCapture.isUsable)
            Divider()
            Button {
                ComposerAudioRecorder.startSystemDictation(promptFocused: $promptFocused)
            } label: {
                Label { Text("Use macOS Dictation", bundle: .module) } icon: { Image(systemName: "keyboard") }
            }
            Button {
                model.openSettings(tab: .audio)
            } label: {
                Label { Text("Audio settings...", bundle: .module) } icon: { Image(systemName: "gearshape") }
            }
        }
        .onAppear { capabilities.refresh() }
    }

    private var helpText: Text {
        if isRecording { return Text("Stop and transcribe", bundle: .module) }
        if !audioEnabled { return Text("Start voice dictation", bundle: .module) }
        if let reason = capabilities.snapshot.micCapture.reason {
            return Text(verbatim: reason)
        }
        return Text("Dictate. Right-click for voice notes and app audio.", bundle: .module)
    }
}
