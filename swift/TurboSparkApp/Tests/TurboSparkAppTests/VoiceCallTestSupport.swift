import Foundation
import XCTest
@testable import TurboSparkApp

/// OC-08 test doubles. A call test never opens a microphone, a speech model
/// or a synthesizer: audio is a synthetic sample array, the recognizer, the
/// speaker and the chat are fakes, and time is counted in samples.

@MainActor
final class FakeCallChat: VoiceCallChatPort {
    var selectedChatID = UUID()
    var isTurnBusy = false
    var hasPendingApproval = false
    var draft = ""
    var onEvent: (@MainActor (VoiceCallChatEvent) -> Void)?
    func emit(_ event: VoiceCallChatEvent) { onEvent?(event) }
}

@MainActor
final class FakeCallAudio: VoiceCallAudioSource {
    var starts = 0
    var stops = 0
    var startError: Error?
    var onSamples: (@Sendable ([Float]) -> Void)?
    var onFailure: (@Sendable (String) -> Void)?
    func start(onSamples: @escaping @Sendable ([Float]) -> Void,
               onFailure: @escaping @Sendable (String) -> Void) async throws {
        starts += 1
        if let startError { throw startError }
        self.onSamples = onSamples
        self.onFailure = onFailure
    }
    func stop() async { stops += 1 }
}

@MainActor
final class FakeCallTranscriber: VoiceCallTranscribing {
    var fed: [Float] = []
    var text = "hello there"
    var finishCount = 0
    var cancelled = false
    var emit: (@MainActor (VoiceCallTranscriberEvent) -> Void)?
    func feed(_ samples: [Float]) { fed += samples }
    func finish() async -> String { finishCount += 1; return text }
    func cancel() { cancelled = true }
}

@MainActor
final class FakeCallSpeaker: VoiceCallSpeaking {
    var onIdle: (@MainActor () -> Void)?
    var isSpeaking = false
    var spoken: [String] = []
    var stops = 0
    func speak(_ sentence: String) { spoken.append(sentence); isSpeaking = true }
    func stop() { stops += 1; isSpeaking = false }
    /// The synthesizer drained its queue.
    func drain() { isSpeaking = false; onIdle?() }
}

/// A call with every seam faked. `autoReady` fires the recognizer's ready
/// event as soon as it is made, which is what a loaded model does.
@MainActor
final class CallHarness {
    let chat = FakeCallChat()
    let audio = FakeCallAudio()
    let speaker = FakeCallSpeaker()
    var transcribers: [FakeCallTranscriber] = []
    var autoReady = true
    var microphoneAllowed = true
    var recordingActive = false
    var model: VoiceCallSpeechModelStatus = .ready
    var nextText = "hello there"
    private(set) var controller: VoiceCallController!

    static let placeholders = VoiceCallSpeechPlanner.Placeholders(
        code: "Code block skipped.", path: "a file path", link: "a link")

    init() {
        let environment = VoiceCallEnvironment(
            microphoneAllowed: { [unowned self] in self.microphoneAllowed },
            recordingActive: { [unowned self] in self.recordingActive },
            speechModel: { [unowned self] in self.model })
        controller = VoiceCallController(
            chat: chat, audio: audio, speaker: speaker, environment: environment,
            makeTranscriber: { [unowned self] onEvent in
                let transcriber = FakeCallTranscriber()
                transcriber.text = self.nextText
                transcriber.emit = onEvent
                self.transcribers.append(transcriber)
                if self.autoReady { onEvent(.ready) }
                return transcriber
            },
            placeholders: Self.placeholders)
    }

    var live: FakeCallTranscriber? { transcribers.last.flatMap { $0.cancelled ? nil : $0 } }

    /// Lets main-actor tasks the controller started run to completion.
    func settle() async {
        for _ in 0..<30 { await Task.yield() }
    }

    func startCall() async {
        await controller.start()
        await settle()
    }

    /// `seconds` of loud audio in 100 ms blocks, then nothing.
    func talk(_ seconds: Double, amplitude: Float = 0.2) {
        feed(seconds, amplitude: amplitude)
    }

    func silence(_ seconds: Double) { feed(seconds, amplitude: 0) }

    private func feed(_ seconds: Double, amplitude: Float) {
        let blocks = Int((seconds * 10).rounded())
        for _ in 0..<blocks {
            let block = (0..<1_600).map { $0 % 2 == 0 ? amplitude : -amplitude }
            controller.ingest(block)
        }
    }

    /// Drives the call to speaking: a turn starts and the reply streams.
    func reachSpeaking() async {
        await startCall()
        chat.isTurnBusy = true
        chat.emit(.turnStarted)
        chat.emit(.reply("Here is the answer. And more"))
        XCTAssertEqual(controller.state, .speaking)
    }
}

/// Where the app sources live, for the source-scan tests.
enum VoiceCallSources {
    static func directory() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }
}
