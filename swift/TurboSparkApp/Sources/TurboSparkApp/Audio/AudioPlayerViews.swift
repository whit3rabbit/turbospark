import AppKit
import SwiftUI
import TurboSpark

/// `AudioWaveformView` over a file, with peaks from the engine
/// (`ts_audio_peaks_json` through `AudioPeaksCache`).
struct AudioFileWaveformView: View {
    let url: URL
    var buckets: Int = 160
    var progress: Double? = nil
    var selection: ClosedRange<Double>? = nil
    var onSeek: ((Double) -> Void)? = nil
    var onSelect: ((ClosedRange<Double>?) -> Void)? = nil
    var accessibilityValueText: String? = nil
    /// Reports the engine-measured duration once peaks load.
    var onLoaded: ((TimeInterval) -> Void)? = nil

    @State private var peaks: [Float] = []

    var body: some View {
        AudioWaveformView(
            samples: peaks, progress: progress, selection: selection,
            onSeek: onSeek, onSelect: onSelect,
            accessibilityValueText: accessibilityValueText)
            .task(id: "\(url.path)|\(buckets)") {
                let result = await AudioPeaksCache.shared.peaks(for: url, buckets: buckets)
                peaks = result?.peaks ?? []
                if let result { onLoaded?(result.durationSeconds) }
            }
    }
}

/// A sent voice note or audio file in a message bubble: play, scrub,
/// duration, export (docs/AUDIO_UI.md, capability 2). The transcript the
/// model read is already in the bubble text, so the row does not repeat it.
struct AudioAttachmentRow: View {
    let storedPath: String
    @ObservedObject private var player = AudioPlaybackController.shared
    @Environment(\.appTheme) private var theme
    @State private var duration: TimeInterval = 0
    @State private var showingExport = false

    private var url: URL? {
        let resolved = AppStorageRoot.resolveStoredPath(storedPath)
        guard !resolved.isEmpty, FileManager.default.fileExists(atPath: resolved) else { return nil }
        return URL(fileURLWithPath: resolved)
    }

    var body: some View {
        if let url {
            HStack(spacing: 8) {
                Button {
                    player.toggle(url: url, key: storedPath)
                } label: {
                    Image(systemName: player.isPlaying(key: storedPath) ? "pause.fill" : "play.fill")
                        .themedFont(.small, weight: .semibold)
                        .frame(width: 26, height: 26)
                        .background(theme.accent.opacity(0.14), in: Circle())
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(theme.accent)
                .accessibilityLabel(player.isPlaying(key: storedPath)
                    ? Text("Pause voice note", bundle: .module)
                    : Text("Play voice note", bundle: .module))

                AudioFileWaveformView(
                    url: url, buckets: 90,
                    progress: player.progress(for: storedPath) ?? 0,
                    onSeek: { player.seek(to: $0, url: url, key: storedPath) },
                    accessibilityValueText: positionText,
                    onLoaded: { duration = $0 })
                    .frame(width: 220, height: 26)

                Text(verbatim: positionText)
                    .themedCode(.tiny)
                    .monospacedDigit()
                    .foregroundStyle(.appSecondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.appSurface.opacity(0.85), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Color.primary.opacity(0.08), lineWidth: 1))
            .contextMenu {
                Button {
                    showingExport = true
                } label: {
                    Text("Export Audio...", bundle: .module)
                }
            }
            .accessibilityAction(named: Text("Play voice note", bundle: .module)) {
                player.toggle(url: url, key: storedPath)
            }
            .sheet(isPresented: $showingExport) {
                AudioExportSheet(
                    source: url,
                    suggestedName: (storedPath as NSString).lastPathComponent,
                    range: nil)
            }
        } else {
            Label {
                Text("Audio is no longer at its saved path", bundle: .module)
            } icon: {
                Image(systemName: "waveform.slash")
            }
            .themedFont(.small)
            .foregroundStyle(.secondary)
        }
    }

    private var positionText: String {
        let current = player.activeKey == storedPath ? player.currentTime : 0
        return "\(WaveformMath.formatDuration(current)) / \(WaveformMath.formatDuration(duration))"
    }
}

/// The preview pane for an audio attachment: waveform with playhead and
/// trim selection, transport, transcript (docs/AUDIO_UI.md, capability 2).
struct AudioPreviewView: View {
    @ObservedObject var model: AppModel
    let attachment: AppPromptAttachment
    @ObservedObject private var player = AudioPlaybackController.shared
    @ObservedObject private var transcriber = AudioAttachmentTranscriber.shared
    @Environment(\.appTheme) private var theme
    @State private var duration: TimeInterval = 0
    @State private var selection: ClosedRange<Double>?
    @State private var showingExport = false
    @State private var isTrimming = false

    private var key: String { attachment.id.uuidString }

    var body: some View {
        if let url = attachment.sourceURL {
            VStack(alignment: .leading, spacing: 14) {
                AudioFileWaveformView(
                    url: url, buckets: 240,
                    progress: player.progress(for: key) ?? 0,
                    selection: selection,
                    onSeek: { player.seek(to: $0, url: url, key: key) },
                    onSelect: { selection = $0 },
                    accessibilityValueText: timeText,
                    onLoaded: { duration = $0 })
                    .frame(height: 96)

                transport(url: url)
                selectionBar(url: url)
                Divider()
                transcriptSection
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .sheet(isPresented: $showingExport) {
                AudioExportSheet(
                    source: url, suggestedName: attachment.fileName,
                    range: selection.map { secondsRange($0) })
            }
        } else {
            Text("The source file is no longer at its original path.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .padding(20)
        }
    }

    /// No unmodified key shortcuts here: a bare Space or arrow registered as
    /// a key equivalent fires before the prompt editor sees the keystroke.
    private func transport(url: URL) -> some View {
        HStack(spacing: 10) {
            Button { player.skip(by: -5) } label: { Image(systemName: "gobackward.5") }
                .buttonStyle(.plain)
                .accessibilityLabel(Text("Back 5 seconds", bundle: .module))
            Button {
                player.toggle(url: url, key: key)
            } label: {
                Image(systemName: player.isPlaying(key: key) ? "pause.circle.fill" : "play.circle.fill")
                    .themedFont(.title2)
                    .foregroundStyle(theme.accent)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(player.isPlaying(key: key)
                ? Text("Pause", bundle: .module) : Text("Play", bundle: .module))
            Button { player.skip(by: 5) } label: { Image(systemName: "goforward.5") }
                .buttonStyle(.plain)
                .accessibilityLabel(Text("Forward 5 seconds", bundle: .module))

            Text(verbatim: timeText)
                .themedCode(.small)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)

            Spacer()

            Menu {
                ForEach(AudioPlaybackController.rates, id: \.self) { rate in
                    Button {
                        player.setRate(rate)
                    } label: {
                        Text(verbatim: Self.rateLabel(rate))
                    }
                }
            } label: {
                Text(verbatim: Self.rateLabel(player.rate))
                    .themedCode(.tiny)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help(Text("Playback speed", bundle: .module))

            Button {
                showingExport = true
            } label: {
                Text("Export...", bundle: .module).themedFont(.tiny, weight: .medium)
            }
        }
    }

    @ViewBuilder
    private func selectionBar(url: URL) -> some View {
        if let selection {
            let range = secondsRange(selection)
            HStack(spacing: 8) {
                Text(verbatim: "\(WaveformMath.formatDuration(range.lowerBound)) - \(WaveformMath.formatDuration(range.upperBound))")
                    .themedCode(.tiny)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Button {
                    self.selection = nil
                } label: {
                    Text("Clear selection", bundle: .module).themedFont(.tiny)
                }
                Button {
                    trim(url: url, range: range)
                } label: {
                    if isTrimming {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Use selection", bundle: .module).themedFont(.tiny, weight: .semibold)
                    }
                }
                .disabled(isTrimming || model.isRunning)
                .help(Text("Replace this attachment with the selected range and transcribe it again", bundle: .module))
            }
        } else {
            Text("Drag across the waveform to select a range to keep or export.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private var transcriptSection: some View {
        HStack(spacing: 6) {
            Text("Transcript", bundle: .module).themedFont(.small, weight: .semibold)
            if let provider = transcriber.providers[attachment.id] {
                Text(verbatim: provider.label)
                    .themedFont(.micro)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Button {
                transcriber.transcribe(attachment, chatID: nil, model: model)
            } label: {
                Group {
                    if attachment.extractedText.isEmpty {
                        Text("Transcribe", bundle: .module)
                    } else {
                        Text("Transcribe again", bundle: .module)
                    }
                }
                .themedFont(.tiny, weight: .medium)
            }
            .disabled(transcriber.isTranscribing(attachment.id))
        }
        if transcriber.isTranscribing(attachment.id) {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Transcribing...", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
            }
        } else if attachment.extractedText.isEmpty {
            Text(verbatim: transcriber.failures[attachment.id]
                ?? String(localized: "No transcript yet. The model receives the transcript, not the audio.", bundle: .module))
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            ScrollView {
                Text(verbatim: attachment.extractedText)
                    .themedFont(.callout)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var timeText: String {
        let current = player.activeKey == key ? player.currentTime : 0
        return "\(WaveformMath.formatDuration(current)) / \(WaveformMath.formatDuration(duration))"
    }

    private func secondsRange(_ fraction: ClosedRange<Double>) -> ClosedRange<TimeInterval> {
        (fraction.lowerBound * duration)...(fraction.upperBound * duration)
    }

    static func rateLabel(_ rate: Float) -> String {
        rate == rate.rounded() ? String(format: "%.0fx", rate) : String(format: "%.2gx", rate)
    }

    /// Trims through the engine, stores the result as a new managed asset,
    /// and points the attachment at it. The old transcript described the
    /// whole clip, so it is cleared and transcription starts again.
    private func trim(url: URL, range: ClosedRange<TimeInterval>) {
        isTrimming = true
        let id = attachment.id
        let name = attachment.fileName
        Task {
            do {
                let trimmed = try await Task.detached(priority: .userInitiated) {
                    try AudioEngineBridge.trimmedCopy(of: url, range: range)
                }.value
                defer { try? FileManager.default.removeItem(at: trimmed) }
                let stem = (name as NSString).deletingPathExtension
                let asset = try ManagedAssetStore.shared.store(
                    fileURL: trimmed, fileName: "\(stem)-trimmed.wav")
                player.stop()
                model.updatePromptAttachment(id: id, inChatID: nil) {
                    $0.fileName = asset.fileName
                    $0.sourcePath = asset.storedReference
                    $0.sourceByteSize = Int(asset.byteCount)
                    $0.extractedText = ""
                }
                selection = nil
                if let updated = model.promptAttachments.first(where: { $0.id == id }) {
                    transcriber.transcribe(updated, chatID: nil, model: model)
                }
            } catch {
                model.showToast(error.localizedDescription, style: .error)
            }
            isTrimming = false
        }
    }
}

/// Export to M4A (Apple AAC over an engine-rendered WAV) or WAV (engine).
struct AudioExportSheet: View {
    let source: URL
    let suggestedName: String
    let range: ClosedRange<TimeInterval>?

    @Environment(\.dismiss) private var dismiss
    @State private var options = AudioExportOptions()
    @State private var isExporting = false
    @State private var errorText: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Export Audio", bundle: .module).themedFont(.title3, weight: .semibold)
            if let range {
                Text(verbatim: "\(WaveformMath.formatDuration(range.lowerBound)) - \(WaveformMath.formatDuration(range.upperBound))")
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)
            }
            Form {
                Picker(selection: $options.format) {
                    ForEach(AudioExportFormat.allCases) { format in
                        Text(verbatim: format.displayName).tag(format)
                    }
                } label: { Text("Format", bundle: .module) }
                .pickerStyle(.segmented)

                if options.format == .m4a {
                    Picker(selection: $options.bitRate) {
                        ForEach(AudioExportOptions.aacBitRates, id: \.self) { rate in
                            Text(verbatim: "\(rate / 1_000) kbps").tag(rate)
                        }
                    } label: { Text("Quality", bundle: .module) }
                }

                Picker(selection: $options.sampleRate) {
                    Text("Original", bundle: .module).tag(Int?.none)
                    Text(verbatim: "44.1 kHz").tag(Int?.some(44_100))
                    Text(verbatim: "16 kHz").tag(Int?.some(16_000))
                } label: { Text("Sample rate", bundle: .module) }

                Toggle(isOn: $options.mono) { Text("Mono", bundle: .module) }
            }
            .formStyle(.grouped)

            if let errorText {
                Text(verbatim: errorText).themedFont(.small).foregroundStyle(.red)
            }

            HStack {
                Spacer()
                Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
                Button {
                    export()
                } label: {
                    if isExporting {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Export...", bundle: .module)
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(isExporting)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private func export() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [options.format.contentType]
        let stem = (suggestedName as NSString).deletingPathExtension
        panel.nameFieldStringValue = "\(stem).\(options.format.fileExtension)"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        var request = options
        request.range = range
        isExporting = true
        errorText = nil
        Task {
            do {
                try await Task.detached(priority: .userInitiated) {
                    try AudioEngineBridge.export(source: source, to: destination, options: request)
                }.value
                isExporting = false
                dismiss()
            } catch {
                isExporting = false
                errorText = error.localizedDescription
            }
        }
    }
}
