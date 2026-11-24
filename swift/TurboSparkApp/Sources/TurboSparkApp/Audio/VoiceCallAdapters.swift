import AVFoundation
import Foundation
import TurboSpark

/// OC-08: the production halves of the call's seams. Tests use fakes; these
/// are the only call-path code that touches a microphone, a model or a
/// speech synthesizer.

// MARK: Speech

/// Reads replies with the system voice through a synthesizer the call owns.
///
/// The shared `AppSpeechSynthesizer` is keyed by message id and toggles one
/// whole message; a call queues sentence after sentence and has to know when
/// the queue drains, so it gets its own. It stays on-device: `AVSpeechSynthesizer`
/// with a system voice makes no network request.
@MainActor
final class VoiceCallSpeaker: NSObject, VoiceCallSpeaking, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    /// Utterances handed to the synthesizer and not yet finished, by identity.
    private var queued: [AVSpeechUtterance] = []
    var onIdle: (@MainActor () -> Void)?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    var isSpeaking: Bool { !queued.isEmpty }

    func speak(_ sentence: String) {
        let utterance = AVSpeechUtterance(string: sentence)
        if let voice = AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode()) {
            utterance.voice = voice
        }
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        queued.append(utterance)
        synthesizer.speak(utterance)
    }

    func stop() {
        // Drop the queue first: the cancel callbacks that follow find nothing
        // to match, so a deliberate stop never reads as the queue draining.
        queued.removeAll()
        if synthesizer.isSpeaking { synthesizer.stopSpeaking(at: .immediate) }
    }

    private func finished(_ utterance: AVSpeechUtterance) {
        guard let index = queued.firstIndex(where: { $0 === utterance }) else { return }
        queued.remove(at: index)
        if queued.isEmpty { onIdle?() }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        // The utterance is only compared by identity.
        nonisolated(unsafe) let ended = utterance
        Task { @MainActor in self.finished(ended) }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        nonisolated(unsafe) let ended = utterance
        Task { @MainActor in self.finished(ended) }
    }
}

// MARK: Microphone

/// The microphone for a call: `AudioCapture` in its non-persisting mode, its
/// live feed drained every 100 ms and converted to 16 kHz mono.
///
/// Capture only CHECKS microphone access and throws; the prompt belongs to
/// `RecordingPermissions` (the access sheet's Allow button).
@MainActor
final class AudioCaptureVoiceSource: VoiceCallAudioSource {
    private var capture: AudioCapture?
    private var pump: Pump?
    /// The device to open, resolved at each start so a saved microphone
    /// that was unplugged falls back to the system default (nil).
    private let microphoneID: @MainActor () -> UInt32?

    init(microphoneID: @escaping @MainActor () -> UInt32? = { nil }) {
        self.microphoneID = microphoneID
    }

    func start(onSamples: @escaping @Sendable ([Float]) -> Void,
               onFailure: @escaping @Sendable (String) -> Void) async throws {
        await stop()
        let feed = AudioLiveFeed(limitSeconds: 5)
        let capture = AudioCapture(
            onChunk: { _ in },
            onEvent: { event in
                switch event {
                case .level: break
                // The capture's own wording speaks of saved recordings.
                case .failure, .interruption:
                    onFailure(AppStrings.localized("The microphone stopped or changed, so the call ended."))
                }
            })
        capture.attachLiveFeed(feed)
        var configuration = AudioCapture.Configuration()
        configuration.capturesMicrophone = true
        configuration.microphoneID = microphoneID()
        configuration.systemSource = .none
        configuration.persistsChunks = false
        try await capture.start(configuration: configuration)
        self.capture = capture
        let pump = Pump(feed: feed, onSamples: onSamples)
        pump.start()
        self.pump = pump
    }

    func stop() async {
        pump?.stop()
        pump = nil
        let capture = self.capture
        self.capture = nil
        await capture?.stop()
    }

    /// Drains the feed on a utility queue and resamples. The converter keeps
    /// its filter state across blocks, so block edges do not click.
    private final class Pump: @unchecked Sendable {
        private let queue = DispatchQueue(label: "app.turbospark.voicecall.pump", qos: .userInitiated)
        private let feed: AudioLiveFeed
        private let onSamples: @Sendable ([Float]) -> Void
        private var converter = AudioLiveConverter()
        private var timer: DispatchSourceTimer?

        init(feed: AudioLiveFeed, onSamples: @escaping @Sendable ([Float]) -> Void) {
            self.feed = feed
            self.onSamples = onSamples
        }

        func start() {
            let timer = DispatchSource.makeTimerSource(queue: queue)
            timer.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
            timer.setEventHandler { [weak self] in self?.drain() }
            timer.resume()
            self.timer = timer
        }

        func stop() {
            queue.sync { timer?.cancel(); timer = nil }
        }

        private func drain() {
            var block: [Float] = []
            for packet in feed.drain().packets {
                block += converter.mono16k(packet.samples, sampleRate: packet.sampleRate, channels: packet.channels)
            }
            if !block.isEmpty { onSamples(block) }
        }
    }
}

// MARK: Recognizer

/// One utterance's recognizer: the existing `AudioLiveTranscriber` over the
/// call's 16 kHz blocks. A fresh one per turn, closed by `finish()` before
/// any generation starts.
@MainActor
final class VoiceCallLiveTranscriber: VoiceCallTranscribing {
    private let feed = AudioLiveFeed(limitSeconds: 60)
    private let transcriber: AudioLiveTranscriber
    private var offset = 0.0

    /// - Parameter open: opens the streaming model. A throw is reported as
    ///   `.stopped`, which ends the call with the reason shown.
    init(open: @escaping @Sendable () throws -> AudioLiveStreaming,
         onEvent: @escaping @MainActor (VoiceCallTranscriberEvent) -> Void) {
        transcriber = AudioLiveTranscriber(feed: feed, open: open) { event in
            switch event {
            case .ready: Task { @MainActor in onEvent(.ready) }
            case .stopped(let reason): Task { @MainActor in onEvent(.stopped(reason)) }
            // A call opens no voice cleaner, so cleaning never stops here.
            case .preparing, .update, .cleaningStopped: break
            }
        }
        transcriber.start()
    }

    func feed(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        feed.offer(samples, sampleRate: 16_000, channels: 1, offset: offset, source: "microphone")
        offset += Double(samples.count) / 16_000
    }

    func finish() async -> String {
        await transcriber.finishAndClose().text
    }

    func cancel() {
        transcriber.stopAndDrain()
    }
}

// MARK: Model status

/// Whether the call's recognizer can run, as a pure function of the facts so
/// the wording and the Download offer are testable.
enum VoiceCallSpeechModel {
    static func status(profileListed: Bool, canRun: Bool, unavailableReason: String?,
                       installed: Bool, english: Bool) -> VoiceCallSpeechModelStatus {
        guard profileListed else {
            return .unavailable(
                reason: AppStrings.localized("Streaming live text needs the live text model, which this catalog does not list."),
                canDownload: false)
        }
        guard canRun else {
            return .unavailable(
                reason: unavailableReason ?? AppStrings.localized("The live text model cannot run on this Mac."),
                canDownload: false)
        }
        // The streaming model writes fluent wrong English for other speech.
        guard english else {
            return .unavailable(reason: AppStrings.localized("Voice calls transcribe English only."), canDownload: false)
        }
        guard installed else {
            return .unavailable(
                reason: AppStrings.localized("Voice calls need the live text model. Download it to continue."),
                canDownload: true)
        }
        return .ready
    }
}
