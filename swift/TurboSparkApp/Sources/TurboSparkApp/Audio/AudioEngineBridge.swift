import AVFoundation
import Foundation
import TurboSpark
import UniformTypeIdentifiers

/// The app side of the Rust audio engine (`crates/audio` via
/// `TurboSparkAudio`).
///
/// Rust owns decode, resampling, peaks, levels and trimmed WAV output; this
/// file only adapts those calls to app types and adds the one thing Apple
/// does better natively, AAC encoding. Nothing here decodes or resamples on
/// its own.

/// View-side waveform arithmetic. Fitting already-computed peaks to the
/// number of bars a view has room for is layout, not DSP, so it stays here.
enum WaveformMath {
    /// Max-reduces `samples` into exactly `count` bars; fewer samples than
    /// bars repeat, so the bar count never depends on clip length.
    static func resample(_ samples: [Float], to count: Int) -> [Float] {
        guard count > 0 else { return [] }
        guard !samples.isEmpty else { return Array(repeating: 0, count: count) }
        let ratio = Double(samples.count) / Double(count)
        return (0..<count).map { index in
            let start = Int(Double(index) * ratio)
            let end = max(start + 1, min(samples.count, Int(Double(index + 1) * ratio)))
            return samples[start..<end].max() ?? 0
        }
    }

    /// Playback-rate menu label: `1x`, `1.5x`, `0.75x`.
    static func rateLabel(_ rate: Float) -> String {
        rate == rate.rounded() ? String(format: "%.0fx", rate) : String(format: "%.2gx", rate)
    }

    /// `m:ss` under an hour, `h:mm:ss` otherwise. ASCII only.
    static func formatDuration(_ seconds: TimeInterval) -> String {
        guard seconds.isFinite, seconds > 0 else { return "0:00" }
        let total = Int(seconds.rounded(.down))
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%d:%02d", minutes, secs)
    }
}

/// What the importer should do with a file extension, per the ENGINE's
/// decoder list. One source of truth: a codec added in Rust appears in the
/// picker without a Swift edit.
enum AudioFileClass: Equatable, Sendable {
    case supported
    case refused
    case notAudio

    static func classify(extension ext: String) -> AudioFileClass {
        let lowered = ext.lowercased()
        guard let engine = AudioCapabilities.engine else { return .notAudio }
        if engine.decodeExtensions.contains(lowered) { return .supported }
        if engine.refusedExtensions.contains(lowered) { return .refused }
        return .notAudio
    }

    static var supportedExtensions: Set<String> {
        Set(AudioCapabilities.engine?.decodeExtensions ?? [])
    }

    /// Picker types derived from the engine's list.
    static var supportedContentTypes: [UTType] {
        supportedExtensions.sorted().compactMap { UTType(filenameExtension: $0) }
    }
}

/// Process-wide cache of engine peaks, keyed by path, modification date and
/// bucket count, so a trimmed file at the same path re-extracts.
actor AudioPeaksCache {
    static let shared = AudioPeaksCache()

    private var entries: [String: AudioPeaks] = [:]
    private var order: [String] = []
    private let capacity = 64

    func peaks(for url: URL, buckets: Int) async -> AudioPeaks? {
        let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate?.timeIntervalSince1970 ?? 0
        let key = "\(url.path)|\(modified)|\(buckets)"
        if let hit = entries[key] { return hit }
        // The engine call streams the whole file; never on an actor's
        // executor, never on the main actor.
        let result = await Task.detached(priority: .utility) {
            try? TurboSparkAudio.peaks(url, buckets: buckets)
        }.value
        guard let result else { return nil }
        entries[key] = result
        order.append(key)
        if order.count > capacity {
            entries.removeValue(forKey: order.removeFirst())
        }
        return result
    }
}

/// User-facing export formats.
///
/// WAV is written by the engine directly. M4A is the one native step: the
/// engine renders a float WAV (trim, rate, channels applied in Rust) and
/// `AVAudioFile` hands it to Apple's AAC encoder, which no pure-Rust crate
/// matches and which ships with every Mac.
enum AudioExportFormat: String, CaseIterable, Identifiable, Sendable {
    case m4a
    case wav

    var id: String { rawValue }
    var fileExtension: String { rawValue }
    var displayName: String {
        switch self {
        case .m4a: return "M4A (AAC)"
        case .wav: return "WAV"
        }
    }
    var contentType: UTType {
        switch self {
        case .m4a: return .mpeg4Audio
        case .wav: return .wav
        }
    }
}

/// One export request. Nil keeps the source's value.
struct AudioExportOptions: Equatable, Sendable {
    var format: AudioExportFormat = .m4a
    /// AAC only. 128 kbps is transparent for speech and small for music.
    var bitRate: Int = 128_000
    var sampleRate: Int? = nil
    var mono = false
    var range: ClosedRange<TimeInterval>? = nil

    static let aacBitRates = [64_000, 128_000, 256_000]
    static let sampleRateChoices: [Int?] = [nil, 44_100, 16_000]
}

enum AudioExportError: LocalizedError {
    case encoderUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .encoderUnavailable(let detail):
            return String(localized: "Audio export failed: \(detail)", bundle: .module)
        }
    }
}

/// Conversion entry points the views call. Every function blocks; call from
/// a detached task.
enum AudioEngineBridge {
    /// A unique path under the app's audio scratch directory.
    static func temporaryURL(extension ext: String) -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-audio", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("\(UUID().uuidString).\(ext)")
    }

    /// 16 kHz mono 16-bit WAV in a fresh temp file: the shape every speech
    /// provider, engine or Apple, is handed.
    static func speechCopy(of source: URL) throws -> URL {
        let destination = temporaryURL(extension: "wav")
        try TurboSparkAudio.convert(source, to: destination, options: .speech)
        return destination
    }

    /// Trims `source` to `range` (seconds) as a float WAV, keeping rate and
    /// channels. Used by the preview pane's "Use selection".
    static func trimmedCopy(of source: URL, range: ClosedRange<TimeInterval>) throws -> URL {
        let destination = temporaryURL(extension: "wav")
        try TurboSparkAudio.convert(
            source, to: destination,
            options: AudioConvertOptions(
                sampleFormat: .float32, startSeconds: range.lowerBound, endSeconds: range.upperBound))
        return destination
    }

    /// Writes `source` to `destination` per `options`.
    static func export(source: URL, to destination: URL, options: AudioExportOptions) throws {
        let engineOptions = AudioConvertOptions(
            sampleRate: options.sampleRate,
            channels: options.mono ? 1 : nil,
            sampleFormat: options.format == .wav ? .int16 : .float32,
            startSeconds: options.range?.lowerBound,
            endSeconds: options.range?.upperBound)
        try? FileManager.default.removeItem(at: destination)
        switch options.format {
        case .wav:
            try TurboSparkAudio.convert(source, to: destination, options: engineOptions)
        case .m4a:
            let rendered = temporaryURL(extension: "wav")
            defer { try? FileManager.default.removeItem(at: rendered) }
            try TurboSparkAudio.convert(source, to: rendered, options: engineOptions)
            try encodeAAC(wav: rendered, to: destination, bitRate: options.bitRate)
        }
    }

    /// AAC-LC caps the bit rate by sample rate and channel count (128 kbps
    /// is above what 16 kHz mono allows, and the encoder throws rather than
    /// clamping). Four bits per sample per channel stays inside every
    /// rate the export sheet offers.
    static func clampedAACBitRate(_ requested: Int, sampleRate: Double, channels: Int) -> Int {
        min(requested, Int(sampleRate * 4) * max(1, channels))
    }

    /// Apple's AAC encoder over an engine-rendered WAV. No resampling or
    /// remixing happens here: the WAV is already in its final shape, so the
    /// processing format is copied straight through.
    private static func encodeAAC(wav: URL, to destination: URL, bitRate: Int) throws {
        let input = try AVAudioFile(forReading: wav)
        let format = input.processingFormat
        // AAC encoders top out at 48 kHz.
        guard format.sampleRate <= 48_000 else {
            throw AudioExportError.encoderUnavailable("AAC supports at most 48 kHz; choose 44.1 kHz")
        }
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: Int(format.channelCount),
            AVEncoderBitRateKey: clampedAACBitRate(
                bitRate, sampleRate: format.sampleRate, channels: Int(format.channelCount)),
        ]
        let output = try AVAudioFile(
            forWriting: destination, settings: settings,
            commonFormat: format.commonFormat, interleaved: format.isInterleaved)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 32_768) else {
            throw AudioExportError.encoderUnavailable("buffer allocation")
        }
        while input.framePosition < input.length {
            try input.read(into: buffer)
            if buffer.frameLength == 0 { break }
            try output.write(from: buffer)
        }
    }
}
