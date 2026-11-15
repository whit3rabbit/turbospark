import Foundation

/// OC-08: a local voice call bound to one chat.
///
/// Listen, transcribe, draft, answer, speak, barge in. Everything composes
/// what the app already owns: microphone capture, the streaming recognizer
/// (`AudioLiveTranscriber`), the system voice and the ordinary chat turn. The
/// controller owns only the turn-taking.
///
/// The rules it enforces, each one pinned by a test:
///
/// - **Draft-first.** A final utterance lands in the composer draft and the
///   user presses Send. The controller has no way to start a turn, so
///   misheard or echoed speech never reaches an agent that can run tools.
/// - **Lane exclusion.** The app has one turn lane (ZP-23) and one engine
///   behind a mutex. A recognizer is only ever open while no turn is busy: it
///   is closed before the user's Send, cancelled if a turn starts beneath it,
///   and not reopened until the turn ends.
/// - **Half-duplex barge-in.** While the reply speaks, mic onset must clear a
///   higher threshold than in a quiet room. A button and Escape always stop
///   playback.
/// - **No silent retry.** A device change, a recognizer failure or revoked
///   access ends the call, releases the microphone and says why.
/// - **Access only through the gate.** The controller checks status; the
///   access sheet (`RecordingPermissions`) is the only thing that prompts.
@MainActor
final class VoiceCallController: ObservableObject {
    @Published private(set) var state: VoiceCallState = .idle
    @Published private(set) var captions: [VoiceCallCaption] = []
    /// One line under the state: why the call paused, what was dropped.
    @Published private(set) var notice: String?
    /// Why the last Call press did not start a call. The host shows it.
    @Published private(set) var refusal: VoiceCallRefusal?
    /// A transcript is waiting in the composer for the user to send.
    @Published private(set) var draftReady = false

    private let chat: VoiceCallChatPort
    private let audio: VoiceCallAudioSource
    private let speaker: VoiceCallSpeaking
    private let environment: VoiceCallEnvironment
    private let makeTranscriber: VoiceCallTranscriberFactory
    /// Auto-send (OC-08). nil means draft-first only.
    private let submitDecider: VoiceCallSubmitDeciding?
    private let autoSendEnabled: @MainActor () -> Bool
    private let decisionTimeout: Duration
    private let decisionSleep: VoiceCallDecisionRace.Sleep
    /// The user's explicit Talk signal. Auto-send considers only speech that
    /// overlapped it (push-to-talk addressing); everything else is draft-first.
    let talk: VoiceCallTalkSwitch
    /// The open utterance overlapped Talk. False by default: only an event
    /// (Talk on at speech onset, or Talk starting mid-utterance) sets it.
    private var utteranceAddressed = false
    /// The decision in flight is for an addressed utterance. Lives and dies
    /// with `decisionToken`, and a forced Talk clear revokes it.
    private var decisionAddressed = false
    /// Identifies the decision in flight; nil when none. Clearing it is how
    /// a decision is cancelled, so a late result is dropped.
    private var decisionToken: UUID? {
        didSet {
            // A decision that is dropped (End, Mute, an approval, a turn)
            // must also stop a dedicated model that is still generating.
            if decisionToken == nil { decisionAddressed = false }
            if decisionToken == nil, oldValue != nil { onDecisionCancelled?() }
        }
    }
    /// The budget for the next decision, read per decision so a dedicated
    /// model can have its own. nil means `decisionTimeout`.
    var decisionTimeoutProvider: (@MainActor () -> Duration?)?
    var onDecisionCancelled: (@MainActor () -> Void)?
    private var detector: VoiceCallSpeechDetecting
    private var planner: VoiceCallSpeechPlanner

    private var transcriber: VoiceCallTranscribing?
    private var transcriberToken: UUID?
    private var transcriberReady = false
    /// Bumped by End, so a late audio block or a finishing recognizer from
    /// an older call is ignored.
    private var epoch = 0
    private var boundChatID: UUID?

    private enum Utterance { case none, active, ended }
    private var utterance: Utterance = .none
    private var finishInFlight = false
    /// Recent 16 kHz audio while no recognizer is open (see `VoiceCallTuning`).
    private var rollingBuffer: [Float] = []

    private var turnRunning = false
    /// The running turn started while the call was listening, so its reply
    /// is read aloud. A turn that started while muted is not.
    private var callTurn = false
    private var replySuppressed = false
    private var spokeThisTurn = false
    private var liveReply = ""
    private var replyCaptionID: UUID?

    init(chat: VoiceCallChatPort, audio: VoiceCallAudioSource, speaker: VoiceCallSpeaking,
         environment: VoiceCallEnvironment, makeTranscriber: @escaping VoiceCallTranscriberFactory,
         detector: VoiceCallSpeechDetecting = VoiceCallEnergyDetector(),
         placeholders: VoiceCallSpeechPlanner.Placeholders = .localized,
         submitDecider: VoiceCallSubmitDeciding? = nil,
         autoSendEnabled: @escaping @MainActor () -> Bool = { false },
         decisionTimeout: Duration = VoiceCallTuning.decisionTimeout,
         decisionSleep: @escaping VoiceCallDecisionRace.Sleep = { try await Task.sleep(for: $0) },
         talk: VoiceCallTalkSwitch? = nil) {
        let talk = talk ?? VoiceCallTalkSwitch()
        self.talk = talk
        self.submitDecider = submitDecider
        self.autoSendEnabled = autoSendEnabled
        self.decisionTimeout = decisionTimeout
        self.decisionSleep = decisionSleep
        self.chat = chat
        self.audio = audio
        self.speaker = speaker
        self.environment = environment
        self.makeTranscriber = makeTranscriber
        self.detector = detector
        self.planner = VoiceCallSpeechPlanner(placeholders: placeholders)
        chat.onEvent = { [weak self] event in self?.handle(event) }
        speaker.onIdle = { [weak self] in self?.speakerIdle() }
        talk.canTalk = { [weak self] in self?.talkAllowed ?? false }
        talk.onStarted = { [weak self] in self?.talkStarted() }
        talk.onCleared = { [weak self] in self?.addressingRevoked() }
    }

    /// Talk can only be on while the call can hear the user and auto-send is
    /// on (it has no other effect). Never while muted, paused, thinking,
    /// failed or ended, so it cannot linger across those.
    private var talkAllowed: Bool {
        switch state {
        case .preparing, .listening, .speaking: return autoSendEnabled()
        case .idle, .thinking, .paused, .failed: return false
        }
    }

    /// Talk began. An utterance already open overlaps it, so it counts. One
    /// that already ended (waiting to be transcribed) does not.
    private func talkStarted() {
        if utterance == .active { utteranceAddressed = true }
    }

    /// A forced Talk clear drops addressing captured so far.
    private func addressingRevoked() {
        utteranceAddressed = false
        decisionAddressed = false
    }

    var isActive: Bool { state.isActive }

    // MARK: Start and end

    /// Starts a call on the selected chat. Call only after the access gate
    /// let the user through; the checks here are the controller's own.
    func start() async {
        guard !state.isActive else { return }
        refusal = nil
        notice = nil
        guard environment.microphoneAllowed() else { refusal = .microphone; return }
        guard !chat.isTurnBusy, !chat.hasPendingApproval else { refusal = .turnRunning; return }
        // The Record page and a call must not hold the microphone together.
        guard !environment.recordingActive() else { refusal = .recordingActive; return }

        epoch += 1
        let mine = epoch
        // While the access sheet shows the state is idle; from here the call exists.
        setState(.preparing)
        let status = await environment.speechModel()
        guard epoch == mine, state == .preparing else { return }
        if case .unavailable(let reason, let canDownload) = status {
            setState(.idle)
            refusal = .noSpeechModel(reason: reason, canDownload: canDownload)
            return
        }
        // A turn may have started while the model status was read.
        guard !chat.isTurnBusy, !chat.hasPendingApproval else {
            setState(.idle)
            refusal = .turnRunning
            return
        }
        boundChatID = chat.selectedChatID
        resetSession()
        notice = AppStrings.localized("Preparing live text. The first time on this Mac can take a few minutes.")
        openTranscriber()
        do {
            try await audio.start(
                onSamples: { [weak self] samples in
                    Task { @MainActor [weak self] in self?.ingest(samples, epoch: mine) }
                },
                onFailure: { [weak self] message in
                    Task { @MainActor [weak self] in self?.captureFailed(message, epoch: mine) }
                })
        } catch {
            guard epoch == mine, state.isActive else { return }
            fail(error.localizedDescription)
            return
        }
        // End was pressed while the microphone was opening.
        if epoch != mine { await audio.stop() }
    }

    /// Ends the call from any state and releases the microphone.
    func end() {
        guard state.isActive else { refusal = nil; return }
        epoch += 1
        speaker.stop()
        cancelTranscriber()
        releaseMicrophone()
        resetSession()
        notice = nil
        setState(.idle)
    }

    func dismissRefusal() { refusal = nil }

    private func resetSession() {
        talk.clear()
        decisionToken = nil
        utterance = .none
        finishInFlight = false
        rollingBuffer.removeAll()
        turnRunning = false
        callTurn = false
        replySuppressed = false
        spokeThisTurn = false
        liveReply = ""
        replyCaptionID = nil
        draftReady = false
        captions.removeAll()
        planner.reset()
        detector.reset()
    }

    private func fail(_ reason: String) {
        guard state.isActive else { return }
        talk.clear()
        decisionToken = nil
        speaker.stop()
        cancelTranscriber()
        releaseMicrophone()
        utterance = .none
        rollingBuffer.removeAll()
        callTurn = false
        notice = reason
        setState(.failed(reason))
    }

    // MARK: Mute and approval pause

    /// Mute: the microphone is released, not just ignored, so the system's
    /// recording indicator goes away. Only from listening.
    func mute() {
        guard state == .listening || state == .preparing else { return }
        talk.clear()
        decisionToken = nil
        cancelTranscriber()
        utterance = .none
        rollingBuffer.removeAll()
        notice = nil
        setState(.paused)
        releaseMicrophone()
    }

    func unmute() async {
        guard state == .paused else { return }
        guard !chat.hasPendingApproval else {
            notice = Self.approvalNotice
            return
        }
        let mine = epoch
        do {
            try await audio.start(
                onSamples: { [weak self] samples in
                    Task { @MainActor [weak self] in self?.ingest(samples, epoch: mine) }
                },
                onFailure: { [weak self] message in
                    Task { @MainActor [weak self] in self?.captureFailed(message, epoch: mine) }
                })
        } catch {
            guard epoch == mine, state == .paused else { return }
            fail(error.localizedDescription)
            return
        }
        guard epoch == mine, state == .paused else { await audio.stop(); return }
        notice = nil
        if chat.isTurnBusy {
            turnRunning = true
            callTurn = !replySuppressed
            setState(.thinking)
        } else {
            turnRunning = false
            setState(.listening)
            openTranscriber()
        }
    }

    func toggleMute() async {
        if state == .paused { await unmute() } else { mute() }
    }

    private static var approvalNotice: String {
        AppStrings.localized("A tool needs your approval. The call is paused and the microphone is off.")
    }

    private func pauseForApproval() {
        guard state != .paused else { return }
        talk.clear()
        decisionToken = nil
        speaker.stop()
        cancelTranscriber()
        utterance = .none
        rollingBuffer.removeAll()
        callTurn = false
        notice = Self.approvalNotice
        setState(.paused)
        releaseMicrophone()
    }

    /// From `RecordingPermissions`: access was turned off. A paused call has
    /// no microphone open, so only a live one ends.
    func microphoneAccessChanged(allowed: Bool) {
        guard !allowed, state.isActive, state != .paused else { return }
        if case .failed = state { return }
        fail(AppStrings.localized("Microphone access was turned off. The call ended."))
    }

    private func captureFailed(_ message: String, epoch mine: Int) {
        guard epoch == mine, state.isActive, state != .paused else { return }
        fail(message)
    }

    // MARK: Audio in

    private func ingest(_ samples: [Float], epoch mine: Int) {
        guard epoch == mine else { return }
        ingest(samples)
    }

    /// One block of 16 kHz mono audio from the microphone.
    func ingest(_ samples: [Float]) {
        switch state {
        case .idle, .paused, .failed: return
        case .preparing, .listening, .thinking, .speaking: break
        }
        let scale: Float = state == .speaking ? VoiceCallTuning.bargeInScale : 1
        let events = detector.process(samples, onsetScale: scale)
        if let transcriber { transcriber.feed(samples) } else { appendRolling(samples) }
        for event in events { handle(event) }
    }

    private func appendRolling(_ samples: [Float]) {
        rollingBuffer.append(contentsOf: samples)
        let seconds = utterance == .none ? VoiceCallTuning.preRollSeconds : VoiceCallTuning.deferredUtteranceSeconds
        let limit = Int(seconds * 16_000)
        if rollingBuffer.count > limit { rollingBuffer.removeFirst(rollingBuffer.count - limit) }
    }

    private func handle(_ event: VoiceCallSpeechEvent) {
        switch event {
        case .speechStarted:
            switch state {
            case .preparing, .listening:
                // Addressed if Talk is on now, or this merges into an
                // utterance that already was (never stale: `.none` means new).
                utteranceAddressed = talk.isTalking || (utterance != .none && utteranceAddressed)
                utterance = .active
            case .thinking:
                // Speech while a turn runs is dropped, never queued.
                notice = AppStrings.localized("Speech while thinking is not sent.")
            case .speaking:
                bargeIn()
            case .idle, .paused, .failed:
                break
            }
        case .speechEnded:
            guard utterance == .active else { return }
            utterance = .ended
            finishUtteranceIfPossible()
        }
    }

    /// The user spoke over the reply. Playback stops at once and the rest of
    /// this reply is not read. Recognition starts now when no turn is busy,
    /// and otherwise when the turn finishes, from the audio kept meanwhile.
    private func bargeIn() {
        speaker.stop()
        replySuppressed = true
        utteranceAddressed = talk.isTalking
        utterance = .active
        setState(.listening, resetDetector: false)
        openTranscriber()
    }

    // MARK: Recognizer

    /// Opens the per-turn recognizer. The lane rule lives here: nothing is
    /// opened while a turn is busy or an approval is up.
    private func openTranscriber() {
        guard transcriber == nil, state == .listening || state == .preparing,
              !chat.isTurnBusy, !chat.hasPendingApproval else { return }
        let token = UUID()
        transcriberToken = token
        transcriberReady = false
        let created = makeTranscriber { [weak self] event in self?.transcriberEvent(event, token: token) }
        transcriber = created
        if utterance != .none, !rollingBuffer.isEmpty { created.feed(rollingBuffer) }
        rollingBuffer.removeAll()
        // An utterance that ended while no recognizer could open is finished
        // now, from the audio just fed (a recognizer that reports ready at
        // once never gets a second chance otherwise).
        finishUtteranceIfPossible()
    }

    private func cancelTranscriber() {
        transcriber?.cancel()
        transcriber = nil
        transcriberToken = nil
        transcriberReady = false
    }

    private func transcriberEvent(_ event: VoiceCallTranscriberEvent, token: UUID) {
        guard token == transcriberToken else { return }
        switch event {
        case .ready:
            transcriberReady = true
            if state == .preparing {
                notice = nil
                setState(.listening)
            }
            finishUtteranceIfPossible()
        case .stopped(let reason):
            fail(reason)
        }
    }

    private func finishUtteranceIfPossible() {
        guard utterance == .ended, let recognizer = transcriber, transcriberReady, !finishInFlight else { return }
        finishInFlight = true
        transcriber = nil
        transcriberToken = nil
        transcriberReady = false
        let mine = epoch
        // Captured now: a new utterance may open while the recognizer closes.
        let addressed = utteranceAddressed
        utteranceAddressed = false
        Task { [weak self] in
            let text = await recognizer.finish()
            self?.utteranceFinished(text, addressed: addressed, epoch: mine)
        }
    }

    private func utteranceFinished(_ text: String, addressed: Bool, epoch mine: Int) {
        finishInFlight = false
        guard epoch == mine else { return }
        if utterance == .ended { utterance = .none }
        // Mute, a turn or a failure can land while the recognizer closes.
        guard state == .listening || state == .preparing else { return }
        let candidate = deliver(text)
        if let candidate, beginDecision(candidate, addressed: addressed, epoch: mine) { return }
        openTranscriber()
    }

    /// The transcript goes to the composer draft and the captions. The user
    /// sends it; nothing here starts a turn.
    /// Returns the text only when the composer was empty before it, which is
    /// the one case auto-send may consider (it never sends text the user
    /// typed).
    @discardableResult
    private func deliver(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // A call writes into the chat it was bound to, never another.
        guard chat.selectedChatID == boundChatID else { end(); return nil }
        addCaption(.you, trimmed)
        let existing = chat.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        chat.draft = existing.isEmpty ? trimmed : existing + " " + trimmed
        draftReady = true
        return existing.isEmpty ? trimmed : nil
    }

    // MARK: Auto-send

    /// Starts a decision when auto-send is on and nothing blocks it. Returns
    /// whether one is in flight; the recognizer stays closed until it ends,
    /// so the model and the recognizer never overlap.
    private func beginDecision(_ text: String, addressed: Bool, epoch mine: Int) -> Bool {
        // Push-to-talk: speech said while Talk was off never reaches the
        // decider. It stays in the composer, as with auto-send off.
        guard addressed, let decider = submitDecider, autoSendEnabled(), state == .listening,
              !chat.isTurnBusy, !chat.hasPendingApproval else { return false }
        let token = UUID()
        decisionToken = token
        decisionAddressed = true
        let timeout = decisionTimeoutProvider?() ?? decisionTimeout
        let sleep = decisionSleep
        Task { [weak self] in
            let decision = await VoiceCallDecisionRace.run(
                decider: decider, utterance: text, timeout: timeout, sleep: sleep)
            self?.decisionFinished(decision, text: text, token: token, epoch: mine)
        }
        return true
    }

    private func decisionFinished(_ decision: VoiceCallSubmitDecision, text: String, token: UUID, epoch mine: Int) {
        // Cancelled by End, Mute, an approval, a failure or a turn.
        guard decisionToken == token else { return }
        // Read before the token clears it: a forced Talk clear during the
        // decision has already dropped it.
        let addressed = decisionAddressed
        decisionToken = nil
        guard epoch == mine, state == .listening else { return }
        // Every other result leaves the text in the composer, as before.
        if decision == .allow, addressed, autoSendEnabled(), !chat.isTurnBusy, !chat.hasPendingApproval,
           chat.selectedChatID == boundChatID,
           chat.draft.trimmingCharacters(in: .whitespacesAndNewlines) == text,
           chat.submitAutoVoiceUtterance(
               text, chatID: boundChatID ?? UUID(), decision: decision, addressed: addressed) {
            draftReady = false
        }
        // A sent turn moved the state to thinking; otherwise listening resumes.
        openTranscriber()
        finishUtteranceIfPossible()
    }

    // MARK: Chat events

    private func handle(_ event: VoiceCallChatEvent) {
        guard state.isActive else { return }
        switch event {
        case .turnStarted: turnStarted()
        case .reply(let text):
            guard !text.isEmpty else { return }
            liveReply = text
            speakReply(text, final: false)
        case .turnFinished(let finalReply): turnFinished(finalReply)
        case .approvalRaised: pauseForApproval()
        case .chatChanged:
            if chat.selectedChatID != boundChatID { end() }
        }
    }

    private func turnStarted() {
        turnRunning = true
        // The turn consumed whatever Talk was for; the next utterance needs a
        // fresh Talk. And a turn from any source beats a decision in flight:
        // never two sends.
        talk.clear()
        decisionToken = nil
        draftReady = false
        liveReply = ""
        planner.reset()
        replySuppressed = false
        spokeThisTurn = false
        replyCaptionID = nil
        switch state {
        case .preparing, .listening:
            // The lane rule: a turn is starting, so recognition ends first.
            // A typed Send beside a half-spoken utterance discards it.
            cancelTranscriber()
            utterance = .none
            rollingBuffer.removeAll()
            callTurn = true
            setState(.thinking)
        case .paused:
            callTurn = false
        case .thinking, .speaking, .idle, .failed:
            break
        }
    }

    private func speakReply(_ text: String, final: Bool) {
        guard callTurn, !replySuppressed, state == .thinking || state == .speaking else { return }
        let sentences = planner.ingest(text, final: final)
        for sentence in sentences {
            speaker.speak(sentence)
            spokeThisTurn = true
            appendReplyCaption(sentence)
        }
        if !sentences.isEmpty { setState(.speaking) }
    }

    private func turnFinished(_ finalReply: String?) {
        turnRunning = false
        if state == .paused { callTurn = false; return }
        if callTurn, !replySuppressed {
            let tail: String
            if liveReply.isEmpty { tail = finalReply ?? "" }
            else if let finalReply, finalReply.hasPrefix(liveReply) { tail = finalReply }
            else { tail = liveReply }
            if !tail.isEmpty { speakReply(tail, final: true) }
            if !spokeThisTurn {
                addCaption(.note, AppStrings.localized("The reply had nothing to read aloud."))
            }
        }
        if state == .speaking, speaker.isSpeaking { return }
        returnToListening()
    }

    private func speakerIdle() {
        guard state == .speaking else { return }
        if turnRunning { setState(.thinking) } else { returnToListening() }
    }

    private func returnToListening() {
        callTurn = false
        replySuppressed = false
        if state == .listening {
            // A barge-in whose recognition waited for this turn to end.
            openTranscriber()
            finishUtteranceIfPossible()
            return
        }
        utterance = .none
        rollingBuffer.removeAll()
        setState(.listening)
        openTranscriber()
    }

    // MARK: Output

    /// Stops playback and returns to listening; the button and Escape.
    func stopSpeaking() {
        guard state == .speaking else { return }
        speaker.stop()
        replySuppressed = true
        if turnRunning { setState(.thinking) } else { returnToListening() }
    }

    private func setState(_ new: VoiceCallState, resetDetector: Bool = true) {
        guard state != new else { return }
        state = new
        if resetDetector, new == .speaking || new == .listening { detector.reset() }
    }

    private func releaseMicrophone() {
        let source = audio
        Task { await source.stop() }
    }

    private func addCaption(_ speaker: VoiceCallCaption.Speaker, _ text: String) {
        captions.append(VoiceCallCaption(id: UUID(), speaker: speaker, text: text))
        trimCaptions()
    }

    private func appendReplyCaption(_ sentence: String) {
        if let id = replyCaptionID, let index = captions.firstIndex(where: { $0.id == id }) {
            captions[index].text += " " + sentence
        } else {
            let caption = VoiceCallCaption(id: UUID(), speaker: .reply, text: sentence)
            replyCaptionID = caption.id
            captions.append(caption)
            trimCaptions()
        }
    }

    private func trimCaptions() {
        if captions.count > VoiceCallTuning.captionLimit {
            captions.removeFirst(captions.count - VoiceCallTuning.captionLimit)
        }
    }
}
