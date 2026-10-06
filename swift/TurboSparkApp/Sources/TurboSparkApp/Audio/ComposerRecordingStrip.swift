import SwiftUI

/// Replaces the prompt editor while the composer records, transcribes, or
/// reports a denied permission (docs/AUDIO_UI.md, capability 1 state table).
///
/// The footer stays put underneath, so the Stop button the user just
/// pressed is still where their pointer is.
// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct ComposerRecordingStrip: View {
    @ObservedObject var model: AppModel
    @ObservedObject var recorder: ComposerAudioRecorder
    @ObservedObject private var capture: AudioCaptureService
    @Environment(\.appTheme) private var theme

    init(model: AppModel, recorder: ComposerAudioRecorder) {
        self.model = model
        self.recorder = recorder
        self._capture = ObservedObject(wrappedValue: recorder.capture)
    }

    var body: some View {
        Group {
            switch recorder.phase {
            case .idle:
                EmptyView()
            case .requestingPermission:
                statusRow(Text("Waiting for microphone permission...", bundle: .module))
            case .recording(let mode):
                recordingRow(mode: mode)
            case .transcribing:
                transcribingRow
            case .denied(let kind):
                deniedRow(kind)
            }
        }
        .frame(minHeight: 44)
    }

    private func recordingRow(mode: ComposerAudioRecorder.Mode) -> some View {
        HStack(spacing: 10) {
            HStack(spacing: 5) {
                Circle().fill(Color.red).frame(width: 8, height: 8)
                Text("Recording", bundle: .module)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(Color.red)
            }
            .accessibilityElement(children: .combine)

            TSMotionContext { reduceMotion in
                if reduceMotion {
                    AudioLevelMeter(level: capture.levels.last ?? 0, tint: .red)
                } else {
                    AudioWaveformView(
                        samples: capture.levels, activeTint: .red, allActive: true)
                }
            }
            .frame(height: 28)

            Text(verbatim: WaveformMath.formatDuration(capture.elapsed))
                .themedCode(.small)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)
                .accessibilityLabel(Text("Elapsed time", bundle: .module))
                .accessibilityValue(WaveformMath.formatDuration(capture.elapsed))

            Button {
                recorder.cancel()
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.small, weight: .semibold)
                    .frame(width: 24, height: 24)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .keyboardShortcut(.cancelAction)
            .help(Text("Discard recording (Esc)", bundle: .module))
            .accessibilityLabel(Text("Discard recording", bundle: .module))

            Button {
                recorder.finish(model: model)
            } label: {
                Group {
                    if mode == .dictate {
                        Text("Insert text", bundle: .module)
                    } else {
                        Text("Attach", bundle: .module)
                    }
                }
                .themedFont(.tiny, weight: .semibold)
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(theme.accent.opacity(0.15), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
            .buttonStyle(.plain)
            .foregroundStyle(theme.accent)
            .keyboardShortcut(.defaultAction)
            .help(Text("Finish recording (Return)", bundle: .module))
        }
    }

    private var transcribingRow: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Transcribing...", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            if let provider = AudioCapabilities.shared.snapshot.transcription.provider {
                Text(verbatim: provider.label)
                    .themedFont(.micro)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
        }
        .accessibilityElement(children: .combine)
    }

    private func deniedRow(_ kind: AudioPermissionKind) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "mic.slash")
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            Group {
                switch kind {
                case .microphone:
                    Text("Microphone access is off for TurboSpark.", bundle: .module)
                case .speechRecognition:
                    Text("Speech recognition is off for TurboSpark.", bundle: .module)
                }
            }
            .themedFont(.small)
            Spacer()
            Button {
                AudioCapabilities.openPrivacySettings(for: kind)
            } label: {
                Text("Open System Settings", bundle: .module)
                    .themedFont(.tiny, weight: .medium)
            }
            .buttonStyle(.link)
            Button {
                recorder.dismissNotice()
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.micro, weight: .bold)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.tertiary)
            .accessibilityLabel(Text("Dismiss", bundle: .module))
        }
    }

    private func statusRow(_ text: Text) -> some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            text.themedFont(.small).foregroundStyle(.appSecondary)
            Spacer()
        }
    }
}
