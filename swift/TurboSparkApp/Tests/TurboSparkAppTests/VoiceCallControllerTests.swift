import XCTest
@testable import TurboSparkApp

/// OC-08. The state machine, with every seam faked: no microphone, no model,
/// no synthesizer. Time is counted in samples. The guards that matter (barge-in,
/// the declined-permission path, lane exclusion) each reddened under its own
/// mutation, recorded in the draft.
@MainActor
final class VoiceCallControllerTests: XCTestCase {
    // MARK: Turn-taking

    func testASpokenUtteranceLandsInTheDraftAndTheCaptionsAndNothingIsSent() async {
        let h = CallHarness()
        h.nextText = "what is the capital of France"
        await h.startCall()
        XCTAssertEqual(h.controller.state, .listening)
        h.talk(0.6)
        h.silence(0.8)
        await h.settle()
        XCTAssertEqual(h.chat.draft, "what is the capital of France")
        XCTAssertTrue(h.controller.draftReady)
        XCTAssertEqual(h.controller.captions.map(\.text), ["what is the capital of France"])
        XCTAssertEqual(h.controller.captions.first?.speaker, .you)
        // Draft-first: no turn started and the state did not move.
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertFalse(h.chat.isTurnBusy)
        // The recognizer for that utterance is closed, and a fresh one is open
        // for the next (lane: no turn is running).
        XCTAssertEqual(h.transcribers.count, 2)
        XCTAssertEqual(h.transcribers[0].finishCount, 1)
        XCTAssertNotNil(h.live)
    }

    func testASecondUtteranceAppendsToTheDraft() async {
        let h = CallHarness()
        await h.startCall()
        h.chat.draft = "typed first"
        h.talk(0.5); h.silence(0.8)
        await h.settle()
        XCTAssertEqual(h.chat.draft, "typed first hello there")
    }

    func testAnEmptyTranscriptChangesNothing() async {
        let h = CallHarness()
        h.nextText = "   "
        await h.startCall()
        h.talk(0.5); h.silence(0.8)
        await h.settle()
        XCTAssertEqual(h.chat.draft, "")
        XCTAssertFalse(h.controller.draftReady)
        XCTAssertTrue(h.controller.captions.isEmpty)
    }

    func testAnUtteranceThatEndsBeforeTheModelIsReadyWaitsForIt() async {
        let h = CallHarness()
        h.autoReady = false
        await h.startCall()
        XCTAssertEqual(h.controller.state, .preparing)
        h.talk(0.5); h.silence(0.8)
        await h.settle()
        XCTAssertEqual(h.transcribers[0].finishCount, 0, "a recognizer that is still loading must not be closed")
        h.transcribers[0].emit?(.ready)
        await h.settle()
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertEqual(h.transcribers[0].finishCount, 1)
        XCTAssertEqual(h.chat.draft, "hello there")
    }

    // MARK: Reply

    func testTheReplyIsSpokenSentenceBySentenceAsItStreams() async {
        let h = CallHarness()
        await h.startCall()
        h.chat.isTurnBusy = true
        h.chat.emit(.turnStarted)
        XCTAssertEqual(h.controller.state, .thinking)
        h.chat.emit(.reply("First sentence. Seco"))
        XCTAssertEqual(h.speaker.spoken, ["First sentence."])
        XCTAssertEqual(h.controller.state, .speaking)
        h.chat.emit(.reply("First sentence. Second one. Thi"))
        XCTAssertEqual(h.speaker.spoken, ["First sentence.", "Second one."])
        h.chat.isTurnBusy = false
        h.chat.emit(.turnFinished(finalReply: "First sentence. Second one. Third."))
        XCTAssertEqual(h.speaker.spoken, ["First sentence.", "Second one.", "Third."])
        XCTAssertEqual(h.controller.state, .speaking, "still speaking until the queue drains")
        h.speaker.drain()
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertNotNil(h.live, "recognition resumes once the turn is over")
        XCTAssertEqual(h.controller.captions.last?.speaker, .reply)
    }

    func testCodeIsNotReadAloud() async {
        let h = CallHarness()
        await h.startCall()
        h.chat.isTurnBusy = true
        h.chat.emit(.turnStarted)
        h.chat.isTurnBusy = false
        h.chat.emit(.turnFinished(finalReply: "Run this:\n```sh\nrm -rf /\n```\nDone.\n"))
        XCTAssertEqual(h.speaker.spoken, ["Run this:", "Code block skipped.", "Done."])
    }

    func testAnEmptyReplyReturnsToListeningWithANote() async {
        let h = CallHarness()
        await h.startCall()
        h.chat.isTurnBusy = true
        h.chat.emit(.turnStarted)
        h.chat.isTurnBusy = false
        h.chat.emit(.turnFinished(finalReply: nil))
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertEqual(h.controller.captions.last?.speaker, .note)
        XCTAssertTrue(h.speaker.spoken.isEmpty)
    }

    func testSpeechWhileThinkingIsDroppedNotQueued() async {
        let h = CallHarness()
        await h.startCall()
        h.chat.isTurnBusy = true
        h.chat.emit(.turnStarted)
        let made = h.transcribers.count
        h.talk(0.5); h.silence(0.8)
        await h.settle()
        XCTAssertEqual(h.controller.state, .thinking)
        XCTAssertEqual(h.chat.draft, "")
        XCTAssertNotNil(h.controller.notice)
        XCTAssertEqual(h.transcribers.count, made, "no recognizer while a turn runs")
    }

    // MARK: Barge-in

    func testBargeInStopsPlaybackAndReturnsToListening() async {
        let h = CallHarness()
        await h.reachSpeaking()
        h.chat.isTurnBusy = false
        h.talk(0.4, amplitude: 0.2)
        XCTAssertEqual(h.speaker.stops, 1)
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertNotNil(h.live, "no turn is busy, so recognition starts at once")
    }

    func testSpeakerBleedBelowTheBargeInBarDoesNotInterrupt() async {
        let h = CallHarness()
        await h.reachSpeaking()
        h.talk(1.0, amplitude: 0.04)
        XCTAssertEqual(h.speaker.stops, 0)
        XCTAssertEqual(h.controller.state, .speaking)
    }

    func testBargeInDuringAStillRunningTurnWaitsForTheLaneAndKeepsTheFirstWords() async {
        let h = CallHarness()
        await h.reachSpeaking()
        let made = h.transcribers.count
        h.chat.isTurnBusy = true
        h.talk(0.4)
        XCTAssertEqual(h.speaker.stops, 1)
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertEqual(h.transcribers.count, made, "lane: no recognizer beside a running turn")
        // The rest of the interrupted reply is not read.
        h.chat.emit(.reply("Here is the answer. And more. Even more here."))
        XCTAssertEqual(h.speaker.spoken, ["Here is the answer."])
        h.talk(0.3)
        h.silence(0.8)
        await h.settle()
        h.chat.isTurnBusy = false
        h.chat.emit(.turnFinished(finalReply: nil))
        await h.settle()
        // One recognizer for the held utterance, then a fresh one for the next.
        XCTAssertEqual(h.transcribers.count, made + 2)
        guard h.transcribers.count > made else { return XCTFail("no recognizer opened after the turn") }
        XCTAssertGreaterThan(h.transcribers[made].fed.count, 16_000 * 6 / 10, "the audio kept meanwhile reaches the recognizer")
        XCTAssertEqual(h.transcribers[made].finishCount, 1)
        XCTAssertEqual(h.chat.draft, "hello there")
    }

    func testStopSpeakingButtonCutsPlaybackWhateverTheMic() async {
        let h = CallHarness()
        await h.reachSpeaking()
        h.controller.stopSpeaking()
        XCTAssertEqual(h.speaker.stops, 1)
        XCTAssertEqual(h.controller.state, .thinking, "the turn is still running, so recognition stays closed")
        h.chat.isTurnBusy = false
        h.chat.emit(.turnFinished(finalReply: "Here is the answer. And more"))
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertEqual(h.speaker.spoken, ["Here is the answer."], "nothing more is read after a stop")
    }

    // MARK: Lane

    func testATurnStartingBeneathAnOpenRecognizerClosesItFirst() async {
        let h = CallHarness()
        await h.startCall()
        h.talk(0.5) // half an utterance, typed Send arrives
        let open = h.live
        XCTAssertNotNil(open)
        h.chat.isTurnBusy = true
        h.chat.emit(.turnStarted)
        XCTAssertTrue(open?.cancelled == true)
        XCTAssertNil(h.live)
        XCTAssertEqual(h.controller.state, .thinking)
        XCTAssertEqual(open?.finishCount, 0)
    }

    func testRecognitionNeverOpensWhileATurnIsBusy() async {
        let h = CallHarness()
        await h.startCall()
        h.chat.isTurnBusy = true
        h.chat.emit(.turnStarted)
        h.chat.emit(.reply("One. Two."))
        h.speaker.drain()
        XCTAssertEqual(h.controller.state, .thinking)
        XCTAssertNil(h.live)
        h.chat.isTurnBusy = false
        h.chat.emit(.turnFinished(finalReply: "One. Two."))
        h.speaker.drain()
        XCTAssertNotNil(h.live)
    }

    // MARK: Mute, approval

    func testMuteReleasesTheMicrophoneAndIgnoresAudio() async {
        let h = CallHarness()
        await h.startCall()
        let first = h.live
        h.controller.mute()
        await h.settle()
        XCTAssertEqual(h.controller.state, .paused)
        XCTAssertEqual(h.audio.stops, 1, "muting releases the microphone, not just the samples")
        XCTAssertTrue(first?.cancelled == true)
        h.talk(1)
        XCTAssertEqual(first?.fed.count, 0)
        await h.controller.unmute()
        XCTAssertEqual(h.audio.starts, 2)
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertNotNil(h.live)
    }

    func testAnApprovalPausesTheCallStopsSpeakingAndOnlyUnmuteResumes() async {
        let h = CallHarness()
        await h.reachSpeaking()
        h.chat.hasPendingApproval = true
        h.chat.emit(.approvalRaised)
        await h.settle()
        XCTAssertEqual(h.controller.state, .paused)
        XCTAssertGreaterThanOrEqual(h.speaker.stops, 1)
        XCTAssertEqual(h.audio.stops, 1)
        XCTAssertNotNil(h.controller.notice)
        // The turn ending does not resume a paused call.
        h.chat.emit(.turnFinished(finalReply: "ignored."))
        XCTAssertEqual(h.controller.state, .paused)
        // Unmute refuses while the card is still up.
        let starts = h.audio.starts
        await h.controller.unmute()
        XCTAssertEqual(h.controller.state, .paused)
        XCTAssertEqual(h.audio.starts, starts)
        h.chat.hasPendingApproval = false
        h.chat.isTurnBusy = false
        await h.controller.unmute()
        XCTAssertEqual(h.controller.state, .listening)
    }

    // MARK: Start refusals and failures

    func testADeclinedMicrophoneNeverStartsAnything() async {
        let h = CallHarness()
        h.microphoneAllowed = false
        await h.controller.start()
        await h.settle()
        XCTAssertEqual(h.controller.state, .idle)
        XCTAssertEqual(h.controller.refusal, .microphone)
        XCTAssertEqual(h.audio.starts, 0)
        XCTAssertTrue(h.transcribers.isEmpty)
        // Everything else still works: the draft is writable and a later
        // grant starts a call from the same controller.
        h.chat.draft = "typing still works"
        h.microphoneAllowed = true
        await h.startCall()
        XCTAssertEqual(h.controller.state, .listening)
        XCTAssertNil(h.controller.refusal)
    }

    func testRefusalsLeaveTheCallIdle() async {
        let busy = CallHarness()
        busy.chat.isTurnBusy = true
        await busy.controller.start()
        XCTAssertEqual(busy.controller.refusal, .turnRunning)
        XCTAssertEqual(busy.controller.state, .idle)
        XCTAssertEqual(busy.audio.starts, 0)

        let recording = CallHarness()
        recording.recordingActive = true
        await recording.controller.start()
        XCTAssertEqual(recording.controller.refusal, .recordingActive)
        XCTAssertEqual(recording.audio.starts, 0)

        let missing = CallHarness()
        missing.model = .unavailable(reason: "Download it", canDownload: true)
        await missing.controller.start()
        XCTAssertEqual(missing.controller.refusal, .noSpeechModel(reason: "Download it", canDownload: true))
        XCTAssertEqual(missing.controller.state, .idle)
        XCTAssertEqual(missing.audio.starts, 0)
        XCTAssertTrue(missing.transcribers.isEmpty)
    }

    func testRevokedAccessEndsALiveCallAndReleasesTheMicrophone() async {
        let h = CallHarness()
        await h.startCall()
        h.controller.microphoneAccessChanged(allowed: false)
        await h.settle()
        guard case .failed = h.controller.state else { return XCTFail("\(h.controller.state)") }
        XCTAssertEqual(h.audio.stops, 1)
        XCTAssertNil(h.live)
    }

    func testRevokedAccessWhileMutedDoesNotFailACallWithNoMicrophoneOpen() async {
        let h = CallHarness()
        await h.startCall()
        h.controller.mute()
        h.controller.microphoneAccessChanged(allowed: false)
        XCTAssertEqual(h.controller.state, .paused)
    }

    func testACaptureFailureEndsTheCallCleanlyAndNeverRetries() async {
        let h = CallHarness()
        await h.startCall()
        h.audio.onFailure?("device changed")
        await h.settle()
        XCTAssertEqual(h.controller.state, .failed("device changed"))
        XCTAssertEqual(h.audio.starts, 1)
        XCTAssertEqual(h.audio.stops, 1)
        XCTAssertNil(h.live)
        h.controller.end()
        XCTAssertEqual(h.controller.state, .idle)
    }

    func testAMicrophoneThatCannotOpenFailsTheCallWithItsReason() async {
        let h = CallHarness()
        struct Boom: LocalizedError { var errorDescription: String? { "no input" } }
        h.audio.startError = Boom()
        await h.startCall()
        XCTAssertEqual(h.controller.state, .failed("no input"))
        XCTAssertNil(h.live)
    }

    func testARecognizerThatStopsEndsTheCallWithItsReason() async {
        let h = CallHarness()
        await h.startCall()
        h.transcribers[0].emit?(.stopped("model unloaded"))
        XCTAssertEqual(h.controller.state, .failed("model unloaded"))
    }

    func testEndReleasesEverythingAndLateAudioIsIgnored() async {
        let h = CallHarness()
        await h.startCall()
        let late = h.audio.onSamples
        let open = h.live
        h.controller.end()
        await h.settle()
        XCTAssertEqual(h.controller.state, .idle)
        XCTAssertEqual(h.audio.stops, 1)
        XCTAssertTrue(open?.cancelled == true)
        late?([Float](repeating: 0.2, count: 16_000))
        await h.settle()
        XCTAssertEqual(open?.fed.count, 0)
        XCTAssertEqual(h.controller.state, .idle)
    }

    func testSwitchingChatsEndsTheCall() async {
        let h = CallHarness()
        await h.startCall()
        h.chat.selectedChatID = UUID()
        h.chat.emit(.chatChanged)
        XCTAssertEqual(h.controller.state, .idle)
    }

    func testATranscriptIsNeverWrittenIntoAnotherChat() async {
        let h = CallHarness()
        await h.startCall()
        h.talk(0.5)
        h.chat.selectedChatID = UUID() // switched without the event
        h.silence(0.8)
        await h.settle()
        XCTAssertEqual(h.chat.draft, "")
        XCTAssertEqual(h.controller.state, .idle)
    }

    // MARK: Structure

    /// The call has no way to start a turn: the chat port offers a draft and
    /// state, and the controller never calls a send entry point.
    func testTheCallCannotSubmitATurn() throws {
        let sources = VoiceCallSources.directory().appendingPathComponent("Audio")
        let seams = try String(contentsOf: sources.appendingPathComponent("VoiceCallSeams.swift"), encoding: .utf8)
        let controller = try String(contentsOf: sources.appendingPathComponent("VoiceCallController.swift"), encoding: .utf8)
        let port = try XCTUnwrap(seams.range(of: "protocol VoiceCallChatPort")).lowerBound
        let portBody = String(seams[port...].prefix(while: { _ in true }))
        for forbidden in ["func send", "func run", "func submit", "func generate"] {
            XCTAssertFalse(portBody.contains(forbidden), "the chat port offers \(forbidden)")
        }
        for forbidden in [".run()", ".send(", ".submit(", "generate("] {
            XCTAssertFalse(controller.contains(forbidden), "the controller calls \(forbidden)")
        }
    }

}
