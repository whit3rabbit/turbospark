import Foundation

/// An incremental speech edge from the call's microphone feed.
enum VoiceCallSpeechEvent: Equatable, Sendable {
    case speechStarted
    /// The silence window elapsed after speech.
    case speechEnded
}

/// OC-08: what the call needs from a voice activity detector. The only VAD
/// the app ships (`FluidAudioSpeechDetector`) is batch: it takes a finished
/// sample array and its models are an optional install. A call needs an
/// incremental onset and offset signal that works with no model, so v1 runs
/// this energy detector over the 16 kHz mono feed. A streaming wrapper over
/// Silero is a later swap behind this protocol.
///
/// Time is counted in samples, never read from a clock, so a test drives the
/// detector with synthetic audio and gets the same edges every run.
protocol VoiceCallSpeechDetecting {
    /// - Parameter onsetScale: multiplies the onset threshold. The call
    ///   raises it above 1 while the reply is being spoken, so the speaker's
    ///   own output does not read as the user (half-duplex barge-in).
    mutating func process(_ samples: [Float], onsetScale: Float) -> [VoiceCallSpeechEvent]
    mutating func reset()
}

/// Constants for the call. Not settings in v1: a setting would be a
/// `MacAppSettings` field with a default so old files decode.
enum VoiceCallTuning {
    /// Silence after speech that ends an utterance (OC-08 NFR default).
    static let silenceWindowSeconds: Double = 0.7
    /// Speech must sustain this long before it counts as an onset, so a
    /// cough or a key click does not open an utterance.
    static let onsetSeconds: Double = 0.12
    /// RMS (0...1) above which a 20 ms frame is speech with the room quiet.
    static let onsetLevel: Float = 0.02
    /// Barge-in threshold: the onset level times this while a reply is
    /// speaking, about +10 dB above the quiet-room onset. A sustained voice
    /// over headphones clears it, speaker bleed at normal volume does not.
    static let bargeInScale: Float = 3.0
    /// Audio the controller keeps while no recognizer is open and nobody is
    /// speaking, so the first words after an onset survive the recognizer
    /// opening.
    static let preRollSeconds: Double = 1.5
    /// The most it keeps once an utterance is open but its recognizer is
    /// not (a barge-in while the reply is still generating waits for the
    /// turn to finish before recognition may start).
    static let deferredUtteranceSeconds: Double = 30
    /// Recent utterances shown in the panel.
    static let captionLimit = 4
}

struct VoiceCallEnergyDetector: VoiceCallSpeechDetecting {
    struct Tuning: Equatable, Sendable {
        var sampleRate = 16_000
        /// 20 ms frames.
        var frameSamples = 320
        var onsetLevel = VoiceCallTuning.onsetLevel
        var onsetSeconds = VoiceCallTuning.onsetSeconds
        var silenceSeconds = VoiceCallTuning.silenceWindowSeconds
    }

    private let tuning: Tuning
    private var frame: [Float] = []
    private var inSpeech = false
    private var loudFrames = 0
    private var quietFrames = 0

    init(tuning: Tuning = Tuning()) {
        self.tuning = tuning
        frame.reserveCapacity(tuning.frameSamples)
    }

    private var framesPerSecond: Double { Double(tuning.sampleRate) / Double(tuning.frameSamples) }
    private var onsetFrames: Int { max(1, Int((tuning.onsetSeconds * framesPerSecond).rounded(.up))) }
    private var silenceFrames: Int { max(1, Int((tuning.silenceSeconds * framesPerSecond).rounded(.up))) }

    mutating func reset() {
        frame.removeAll(keepingCapacity: true)
        inSpeech = false
        loudFrames = 0
        quietFrames = 0
    }

    mutating func process(_ samples: [Float], onsetScale: Float) -> [VoiceCallSpeechEvent] {
        var events: [VoiceCallSpeechEvent] = []
        for sample in samples {
            frame.append(sample.isFinite ? sample : 0)
            guard frame.count == tuning.frameSamples else { continue }
            let level = Self.rms(frame)
            frame.removeAll(keepingCapacity: true)
            if inSpeech {
                // Hysteresis: once the user is talking, half the onset level
                // keeps the utterance open, so a soft word is not a gap.
                if level >= tuning.onsetLevel * 0.5 {
                    quietFrames = 0
                } else {
                    quietFrames += 1
                    if quietFrames >= silenceFrames {
                        inSpeech = false
                        quietFrames = 0
                        loudFrames = 0
                        events.append(.speechEnded)
                    }
                }
            } else if level >= tuning.onsetLevel * max(1, onsetScale) {
                loudFrames += 1
                if loudFrames >= onsetFrames {
                    inSpeech = true
                    loudFrames = 0
                    quietFrames = 0
                    events.append(.speechStarted)
                }
            } else {
                loudFrames = 0
            }
        }
        return events
    }

    private static func rms(_ frame: [Float]) -> Float {
        guard !frame.isEmpty else { return 0 }
        var sum: Float = 0
        for value in frame { sum += value * value }
        return (sum / Float(frame.count)).squareRoot()
    }
}
