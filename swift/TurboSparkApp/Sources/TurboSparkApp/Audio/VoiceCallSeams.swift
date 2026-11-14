import Foundation

/// OC-08: the seams between the call controller and everything it composes.
/// Each one is a protocol so a test drives the whole call with synthetic
/// audio, a fake recognizer, a fake speaker and a fake chat, and never opens
/// a microphone, a model or a speech synthesizer.

/// Where the call stands. Every state is drawn as a glyph plus text, never
/// color alone.
enum VoiceCallState: Equatable, Sendable {
    /// No call. The panel is closed and the microphone is released.
    case idle
    /// The recognizer is loading; the microphone is open.
    case preparing
    /// Microphone open, recognizer open, waiting for the user.
    case listening
    /// A chat turn is running for the user's message.
    case thinking
    /// The reply is being read aloud.
    case speaking
    /// Muted, or paused for a tool approval. The microphone is released.
    case paused
    /// The call ended on its own; the reason stays on screen until closed.
    case failed(String)

    var isActive: Bool { self != .idle }

    var symbol: String {
        switch self {
        case .idle: return "phone"
        case .preparing: return "hourglass"
        case .listening: return "waveform"
        case .thinking: return "ellipsis.bubble"
        case .speaking: return "speaker.wave.2.fill"
        case .paused: return "mic.slash.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    /// A string-catalog key, rendered through `Text(LocalizedStringKey(_), bundle:)`.
    var titleKey: String {
        switch self {
        case .idle: return "Voice call"
        case .preparing: return "Preparing"
        case .listening: return "Listening"
        case .thinking: return "Thinking"
        case .speaking: return "Speaking"
        case .paused: return "Muted"
        case .failed: return "Call failed"
        }
    }
}

/// One line in the panel's caption list.
struct VoiceCallCaption: Identifiable, Equatable, Sendable {
    enum Speaker: Equatable, Sendable { case you, reply, note }
    let id: UUID
    let speaker: Speaker
    var text: String
}

/// Why a Call press did not start a call. The state stays idle.
enum VoiceCallRefusal: Equatable, Sendable {
    /// macOS microphone access is not allowed. The access sheet handles this
    /// before Start; this is the controller's own re-check.
    case microphone
    case turnRunning
    case recordingActive
    case noSpeechModel(reason: String, canDownload: Bool)
}

/// What the call needs from the speech-to-text install.
enum VoiceCallSpeechModelStatus: Equatable, Sendable {
    case ready
    case unavailable(reason: String, canDownload: Bool)
}

/// What the call needs to know about the world, as closures.
struct VoiceCallEnvironment {
    /// Reads macOS status. Never prompts.
    var microphoneAllowed: @MainActor () -> Bool
    /// A Record-page capture is active (the call refuses beside it).
    var recordingActive: @MainActor () -> Bool
    var speechModel: @MainActor () async -> VoiceCallSpeechModelStatus
}

// MARK: Chat

enum VoiceCallChatEvent: Equatable, Sendable {
    /// A turn started in the bound chat, from any source (the composer's
    /// Send, a queued prompt).
    case turnStarted
    /// The live reply buffer, cumulative.
    case reply(String)
    /// The turn is over. `finalReply` is the chat's last assistant text.
    case turnFinished(finalReply: String?)
    /// A tool approval card or a question card opened.
    case approvalRaised
    /// The selected chat changed.
    case chatChanged
}

/// The call's whole view of the chat. It can read the turn state and write
/// the composer draft. There is deliberately no general `send`: the user's own
/// press of Send starts a turn (draft-first, so misheard or echoed speech never
/// reaches an agent that can run tools). The single exception is
/// `submitAutoVoiceUtterance`, reachable only with an explicit `allow` from the
/// submit decider while "Auto-send voice utterances" is on.
@MainActor
protocol VoiceCallChatPort: AnyObject {
    var selectedChatID: UUID { get }
    /// A turn holds the lane, a submission is awaiting its hook, or an
    /// approval is open.
    var isTurnBusy: Bool { get }
    var hasPendingApproval: Bool { get }
    var draft: String { get set }
    var onEvent: (@MainActor (VoiceCallChatEvent) -> Void)? { get set }
    /// OC-08 auto-send. Sends the composer's text as a turn only if every
    /// guard in `VoiceCallAutoSubmitGate` still holds. Returns whether a turn
    /// was started. The controller's one call site is after an `.allow`.
    func submitAutoVoiceUtterance(_ text: String, chatID: UUID, decision: VoiceCallSubmitDecision) -> Bool
}

// MARK: Audio

/// The microphone, as 16 kHz mono float blocks. A call persists no audio.
@MainActor
protocol VoiceCallAudioSource: AnyObject {
    func start(onSamples: @escaping @Sendable ([Float]) -> Void,
               onFailure: @escaping @Sendable (String) -> Void) async throws
    /// Releases the microphone. Safe to call when not started.
    func stop() async
}

enum VoiceCallTranscriberEvent: Equatable, Sendable {
    /// The recognizer finished loading and is consuming audio.
    case ready
    /// The recognizer died; the reason is user-facing.
    case stopped(String)
}

/// One utterance's recognizer. Created per turn and closed before any
/// generation starts, so recognition and the chat turn never overlap.
@MainActor
protocol VoiceCallTranscribing: AnyObject {
    /// Queues 16 kHz mono audio. Audio fed before the model is ready is
    /// kept and caught up.
    func feed(_ samples: [Float])
    /// Commits the tail, closes the recognizer and returns the final text.
    func finish() async -> String
    /// Closes without transcribing the rest.
    func cancel()
}

typealias VoiceCallTranscriberFactory =
    @MainActor (_ onEvent: @escaping @MainActor (VoiceCallTranscriberEvent) -> Void) -> VoiceCallTranscribing

// MARK: Speech

@MainActor
protocol VoiceCallSpeaking: AnyObject {
    /// Called when everything queued has been spoken. Not called by `stop()`.
    var onIdle: (@MainActor () -> Void)? { get set }
    var isSpeaking: Bool { get }
    func speak(_ sentence: String)
    /// Cuts playback and drops the queue.
    func stop()
}
