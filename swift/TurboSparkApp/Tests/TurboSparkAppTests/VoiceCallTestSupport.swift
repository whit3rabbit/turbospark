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
    var autoSubmits: [(text: String, chatID: UUID, decision: VoiceCallSubmitDecision, addressed: Bool)] = []
    var autoSubmitAccepts = true
    func submitAutoVoiceUtterance(
        _ text: String, chatID: UUID, decision: VoiceCallSubmitDecision, addressed: Bool
    ) -> Bool {
        autoSubmits.append((text, chatID, decision, addressed))
        guard autoSubmitAccepts else { return false }
        // The real model raises the turn immediately after run().
        isTurnBusy = true
        onEvent?(.turnStarted)
        return true
    }
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
    /// The push-to-talk switch, on a fake clock the test advances.
    private(set) var talkSwitch: VoiceCallTalkSwitch!
    var clock: TimeInterval = 0
    var autoSendOn = false

    static let placeholders = VoiceCallSpeechPlanner.Placeholders(
        code: "Code block skipped.", path: "a file path", link: "a link")

    init(decider: VoiceCallSubmitDeciding? = nil, autoSend: Bool = false,
         sleep: @escaping VoiceCallDecisionRace.Sleep = { _ in try await Task.sleep(for: .seconds(3_600)) }) {
        autoSendOn = autoSend
        talkSwitch = VoiceCallTalkSwitch(now: { [unowned self] in self.clock })
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
            placeholders: Self.placeholders,
            submitDecider: decider,
            autoSendEnabled: { [unowned self] in self.autoSendOn },
            decisionTimeout: .milliseconds(10), decisionSleep: sleep, talk: talkSwitch)
    }

    /// Polls for a condition that depends on unstructured tasks.
    func eventually(_ condition: @MainActor () -> Bool) async -> Bool {
        for _ in 0..<300 {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return condition()
    }

    /// One spoken utterance, finished by the silence window. `addressed`
    /// says whether Talk (toggle form) is on for the whole utterance; it
    /// defaults to on exactly when auto-send is on, which is the only time a
    /// test of the older auto-send behavior needs it.
    func speakUtterance(_ text: String, addressed: Bool? = nil) async {
        let hold = addressed ?? autoSendOn
        if hold { talkSwitch.toggle() }
        nextText = text
        transcribers.last?.text = text
        talk(1)
        silence(1)
        await settle()
        if hold, talkSwitch.isTalking { talkSwitch.toggle() }
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
