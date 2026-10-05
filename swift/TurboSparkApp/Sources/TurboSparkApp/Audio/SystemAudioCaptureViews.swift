import AppKit
import SwiftUI

/// Pick an app (or all system audio) and record what it plays
/// (docs/AUDIO_UI.md, capability 3). The finished clip becomes an ordinary
/// audio attachment, so trim, transcript and export all apply.
struct SystemAudioCaptureSheet: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var service = SystemAudioCaptureService.shared
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme
    @State private var sources: [SystemAudioSource] = []
    @State private var selectedPID: pid_t = 0
    @State private var errorText: String?

    private var selected: SystemAudioSource? {
        sources.first { $0.pid == selectedPID }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Record App Audio", bundle: .module)
                .themedFont(.title3, weight: .semibold)
            Text("Only record audio you have the right to use. Protected streams may record as silence.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if service.isRecording {
                recordingPanel
            } else {
                sourceList
            }

            if let errorText {
                Text(verbatim: errorText)
                    .themedFont(.small)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Button {
                    sources = service.availableSources()
                } label: {
                    Text("Refresh", bundle: .module)
                }
                .disabled(service.isRecording)
                Spacer()
                Button {
                    if service.isRecording { service.cancel() }
                    dismiss()
                } label: {
                    if service.isRecording {
                        Text("Discard", bundle: .module)
                    } else {
                        Text("Cancel", bundle: .module)
                    }
                }
                .keyboardShortcut(.cancelAction)
                if service.isRecording {
                    Button {
                        service.finish(into: model)
                        dismiss()
                    } label: {
                        Text("Stop and Attach", bundle: .module)
                    }
                    .keyboardShortcut(.defaultAction)
                } else {
                    Button {
                        start()
                    } label: {
                        Text("Start Recording", bundle: .module)
                    }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!(selected?.hasAudioProcess ?? false))
                }
            }
        }
        .padding(20)
        .frame(width: 420)
        .onAppear {
            sources = service.availableSources()
            AudioCapabilities.shared.refresh()
        }
    }

    private var sourceList: some View {
        List(selection: Binding(
            get: { Optional(selectedPID) },
            set: { selectedPID = $0 ?? 0 })
        ) {
            ForEach(sources) { source in
                HStack(spacing: 8) {
                    icon(for: source)
                        .frame(width: 18, height: 18)
                        .accessibilityHidden(true)
                    Text(verbatim: source.name)
                    Spacer()
                    if !source.hasAudioProcess {
                        Text("Not playing audio", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                    }
                }
                .tag(Optional(source.pid))
                .opacity(source.hasAudioProcess ? 1 : 0.55)
            }
        }
        .frame(height: 220)
    }

    @ViewBuilder
    private func icon(for source: SystemAudioSource) -> some View {
        if source.isSystemWide {
            Image(systemName: "speaker.wave.2")
        } else if let app = NSRunningApplication(processIdentifier: source.pid), let image = app.icon {
            Image(nsImage: image).resizable()
        } else {
            Image(systemName: "app")
        }
    }

    private var recordingPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Circle().fill(Color.red).frame(width: 8, height: 8)
                Text("Recording", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(Color.red)
                Text(verbatim: service.source?.name ?? "")
                    .themedFont(.small)
                Spacer()
                Text(verbatim: WaveformMath.formatDuration(service.elapsed))
                    .themedCode(.small)
                    .monospacedDigit()
            }
            .accessibilityElement(children: .combine)
            TSMotionContext { reduceMotion in
                if reduceMotion {
                    AudioLevelMeter(level: service.levels.last ?? 0, tint: .red)
                } else {
                    AudioWaveformView(samples: service.levels, activeTint: .red, allActive: true)
                }
            }
            .frame(height: 40)
        }
        .padding(12)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func start() {
        guard let selected else { return }
        errorText = nil
        do {
            try service.start(source: selected)
        } catch {
            errorText = error.localizedDescription
        }
    }
}

/// The persistent consent indicator while app audio is being recorded: a
/// red dot, the source, the elapsed time, in the top bar of every section.
/// Clicking it stops and attaches, so the user is never one navigation away
/// from a recording they cannot find.
struct RecordingIndicatorPill: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var service = SystemAudioCaptureService.shared

    var body: some View {
        if service.isRecording {
            Button {
                service.finish(into: model)
            } label: {
                HStack(spacing: 5) {
                    Circle().fill(Color.red).frame(width: 7, height: 7)
                    Text(verbatim: service.source?.name ?? "")
                        .themedFont(.tiny, weight: .medium)
                        .lineLimit(1)
                    Text(verbatim: WaveformMath.formatDuration(service.elapsed))
                        .themedCode(.tiny)
                        .monospacedDigit()
                    Image(systemName: "stop.fill")
                        .themedFont(.micro)
                        .accessibilityHidden(true)
                }
                .foregroundStyle(Color.red)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.red.opacity(0.12), in: Capsule())
            }
            .buttonStyle(.plain)
            .help(Text("Recording app audio. Click to stop and attach.", bundle: .module))
            .accessibilityLabel(Text("Recording app audio", bundle: .module))
            .accessibilityHint(Text("Stops the recording and attaches it to the prompt", bundle: .module))
        }
    }
}
