import AVFoundation
import Foundation
import TurboSpark

/// Microphone recording to a float WAV through an `AVAudioEngine` input tap.
///
/// Capture is the OS half of audio, so it is Swift; everything after the
/// samples land (the meter level, normalization, peaks, transcription) is
/// the engine's. The file is WAV rather than CAF so the engine's decoder
/// reads it with no container special case.
///
/// The tap runs on a real-time audio thread. Everything it touches lives in
/// `CaptureSink`, which writes the buffer and asks the engine for one level
/// per buffer (`ts_audio_display_level` is real-time safe: no allocation,
/// lock or I/O); the tap closure never touches published state.
@MainActor
final class AudioCaptureService: ObservableObject {
    enum CaptureError: LocalizedError {
        case noInputDevice
        case permissionDenied

        var errorDescription: String? {
            switch self {
            case .noInputDevice:
                return String(localized: "No microphone is available.", bundle: .module)
            case .permissionDenied:
                return String(
                    localized: "Microphone access is off for TurboSpark. Turn it on in System Settings.",
                    bundle: .module)
            }
        }
    }

    /// Ring of recent display levels (0...1), oldest first.
    @Published private(set) var levels: [Float] = []
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var isRecording = false

    /// About 8 s of history at ~12 buffers per second.
    static let levelHistory = 96

    private var engine: AVAudioEngine?
    private var sink: CaptureSink?
    private var outputURL: URL?
    private var startedAt: Date?
    private var clock: Task<Void, Never>?
    /// Called on the main actor when the clip reaches its maximum length.
    var onLimitReached: (() -> Void)?

    /// Starts recording. The caller has already obtained permission.
    func start() throws {
        guard !isRecording else { return }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            throw CaptureError.permissionDenied
        }
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw CaptureError.noInputDevice
        }
        let url = AudioEngineBridge.temporaryURL(extension: "wav")
        let file = try AVAudioFile(
            forWriting: url, settings: Self.wavSettings(for: format),
            commonFormat: .pcmFormatFloat32, interleaved: false)
        let sink = CaptureSink(file: file) { [weak self] level in
            Task { @MainActor in self?.push(level: level) }
        }
        input.installTap(onBus: 0, bufferSize: 4_096, format: format) { buffer, _ in
            sink.consume(buffer)
        }
        engine.prepare()
        try engine.start()

        self.engine = engine
        self.sink = sink
        outputURL = url
        levels = Array(repeating: 0, count: Self.levelHistory)
        elapsed = 0
        startedAt = Date()
        isRecording = true
        startClock()
    }

    /// Float32 WAV at the device's own rate and channel count. No
    /// conversion here: the engine resamples later, once, for whoever needs
    /// a different shape.
    nonisolated static func wavSettings(for format: AVAudioFormat) -> [String: Any] {
        [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: Int(format.channelCount),
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
    }

    /// Stops and returns the finished file, or nil when nothing was recording.
    @discardableResult
    func stop() -> URL? {
        guard isRecording else { return nil }
        tearDown()
        let url = outputURL
        outputURL = nil
        return url
    }

    /// Stops and deletes the partial file.
    func cancel() {
        guard isRecording else { return }
        tearDown()
        if let outputURL { try? FileManager.default.removeItem(at: outputURL) }
        outputURL = nil
    }

    private func tearDown() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        sink?.close()
        sink = nil
        clock?.cancel()
        clock = nil
        isRecording = false
    }

    private func push(level: Float) {
        guard isRecording else { return }
        levels.append(level)
        if levels.count > Self.levelHistory {
            levels.removeFirst(levels.count - Self.levelHistory)
        }
    }

    private func startClock() {
        clock?.cancel()
        let limit = TimeInterval(AudioPreferences.maxClipSeconds)
        clock = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard let self, let startedAt = self.startedAt, self.isRecording else { return }
                self.elapsed = Date().timeIntervalSince(startedAt)
                if self.elapsed >= limit {
                    self.onLimitReached?()
                    return
                }
            }
        }
    }
}

/// The tap-side half: owns the file and reports levels. Used from the audio
/// thread until `close()`, which the main actor calls after the tap is gone.
/// Shared by microphone capture and the app-audio process tap.
final class CaptureSink: @unchecked Sendable {
    private let lock = NSLock()
    private var file: AVAudioFile?
    private let onLevel: @Sendable (Float) -> Void
    /// Frames since the last reported level. Levels go out about 12 times a
    /// second whatever the IO buffer size, so the meter's history spans the
    /// same time for the microphone (4096-frame taps) and a process tap
    /// (often 512 frames, ~90 callbacks a second).
    private var framesSinceLevel: AVAudioFrameCount = 0

    init(file: AVAudioFile, onLevel: @escaping @Sendable (Float) -> Void) {
        self.file = file
        self.onLevel = onLevel
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        try? file?.write(from: buffer)
        framesSinceLevel += buffer.frameLength
        let interval = AVAudioFrameCount(max(1, buffer.format.sampleRate / 12))
        let due = framesSinceLevel >= interval
        if due { framesSinceLevel = 0 }
        lock.unlock()
        guard due, let channel = buffer.floatChannelData?[0], buffer.frameLength > 0 else { return }
        // Interleaved buffers keep every channel in plane 0.
        let samples = Int(buffer.frameLength)
            * (buffer.format.isInterleaved ? Int(buffer.format.channelCount) : 1)
        onLevel(TurboSparkAudio.displayLevel(channel, count: samples))
    }

    /// Releasing the `AVAudioFile` finalizes the WAV header.
    func close() {
        lock.lock()
        file = nil
        lock.unlock()
    }
}
