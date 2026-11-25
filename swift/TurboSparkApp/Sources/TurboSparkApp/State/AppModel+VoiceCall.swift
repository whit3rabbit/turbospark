import Combine
import Foundation
import TurboSpark

/// OC-08: wires the voice call controller to the app.
///
/// The call reads the selected chat's turn state and writes its composer
/// draft, nothing more. The user's own press of Send (or Return) is what
/// starts a turn, through `run()`, so steering, permissions, hooks and usage
/// accounting are exactly the composer's.
@MainActor
final class AppModelVoiceCallPort: VoiceCallChatPort {
    private unowned let model: AppModel
    var onEvent: (@MainActor (VoiceCallChatEvent) -> Void)?
    private var cancellables = Set<AnyCancellable>()
    private var busy = false

    init(model: AppModel) {
        self.model = model
        busy = Self.isBusy(model)

        // The same three cards as `hasPendingVoiceCallApproval`: a tool
        // approval, a question, and a plan parked in the ExitPlanMode gate.
        let approvalOpen = Publishers.CombineLatest3(
            model.$pendingToolCall.map { $0 != nil }.removeDuplicates(),
            model.$pendingUserQuestions.map { $0 != nil }.removeDuplicates(),
            model.$pendingPlanApproval.map { $0 != nil }.removeDuplicates()
        )
        .map { $0 || $1 || $2 }
        .removeDuplicates()

        // A turn is busy while it generates, while a submission awaits its
        // hook, and while an approval card is open (`generating` is lowered
        // on purpose while a call waits for a human).
        Publishers.CombineLatest3(model.$generating, model.$submitting, approvalOpen)
        .map { $0 || $1 || $2 }
        .removeDuplicates()
        .dropFirst()
        .sink { [weak self] nowBusy in self?.busyChanged(nowBusy) }
        .store(in: &cancellables)

        approvalOpen
        .dropFirst()
        .sink { [weak self] raised in
            if raised { self?.onEvent?(.approvalRaised) }
        }
        .store(in: &cancellables)

        model.$outputText
            .dropFirst()
            .sink { [weak self] text in
                guard let self, self.model.liveOutputBelongsToSelectedChat else { return }
                self.onEvent?(.reply(text))
            }
            .store(in: &cancellables)

        model.$selectedChatID
            .dropFirst()
            .removeDuplicates()
            .sink { [weak self] _ in self?.onEvent?(.chatChanged) }
            .store(in: &cancellables)
    }

    private static func isBusy(_ model: AppModel) -> Bool {
        model.generating || model.submitting || model.hasPendingVoiceCallApproval
    }

    var selectedChatID: UUID { model.selectedChatID }
    var isTurnBusy: Bool { Self.isBusy(model) }
    var hasPendingApproval: Bool { model.hasPendingVoiceCallApproval }
    var draft: String {
        get { model.promptText }
        set { model.promptText = newValue }
    }

    func submitAutoVoiceUtterance(
        _ text: String, chatID: UUID, decision: VoiceCallSubmitDecision, addressed: Bool
    ) -> Bool {
        model.submitAutoVoiceUtterance(text, chatID: chatID, decision: decision, addressed: addressed)
    }

    private func busyChanged(_ nowBusy: Bool) {
        if nowBusy {
            guard !busy else { return }
            busy = true
            onEvent?(.turnStarted)
            return
        }
        // The flags drop in separate steps and the agent loop hands over
        // between steps: report the end only if the turn is still over one
        // main-queue hop later.
        DispatchQueue.main.async { [weak self] in
            MainActor.assumeIsolated {
                guard let self, self.busy, !self.isTurnBusy else { return }
                self.busy = false
                self.onEvent?(.turnFinished(finalReply: self.lastAssistantReply()))
            }
        }
    }

    private func lastAssistantReply() -> String? {
        model.chats.first(where: { $0.id == model.selectedChatID })?
            .messages.last(where: { $0.role == .assistant })?.content
    }
}

extension AppModel {
    func makeVoiceCallController() -> VoiceCallController {
        voiceCallBuilt = true
        let port = AppModelVoiceCallPort(model: self)
        voiceCallPort = port
        let environment = VoiceCallEnvironment(
            microphoneAllowed: { AudioCapture.microphoneAccessAllowed() },
            // Reading the lazy workspace builds it, and a call must not.
            recordingActive: { [weak self] in
                guard let self, self.audioWorkspaceCreated else { return false }
                return self.audioWorkspace.hasRecordingActivity
            },
            speechModel: { [weak self] in
                guard let self else { return .unavailable(reason: "", canDownload: false) }
                return await self.voiceCallSpeechModelStatus()
            })
        let controller = makeVoiceCallControllerCore(port: port, environment: environment)
        voiceCallStateObserver = controller.$state
            .map(\.isActive)
            .removeDuplicates()
            .sink { [weak self] active in
                self?.voiceCallActive = active
                self?.syncVoiceCallDecider()
            }
        return controller
    }

    private func makeVoiceCallControllerCore(
        port: AppModelVoiceCallPort, environment: VoiceCallEnvironment
    ) -> VoiceCallController {
        let controller = VoiceCallController(
            chat: port,
            audio: AudioCaptureVoiceSource(microphoneID: { [weak self] in self?.voiceCallMicrophoneResolution().deviceID }),
            speaker: VoiceCallSpeaker(),
            environment: environment,
            makeTranscriber: { [weak self] onEvent in
                let path = self?.voiceCallSpeechModelPath()
                return VoiceCallLiveTranscriber(
                    open: {
                        guard let path else {
                            throw TurboSparkError(code: .invalidArgument, message: "no speech model")
                        }
                        return try AudioStreamSession(modelPath: path)
                    },
                    onEvent: onEvent)
            },
            submitDecider: voiceCallSubmitDecider(),
            autoSendEnabled: { [weak self] in
                guard let self else { return false }
                return self.voiceCallEnabled && self.voiceCallAutoSendEnabled
            })
        // Per decision: a dedicated model has its own budget, and a dropped
        // decision (turn, End, Mute, approval) stops its generation.
        controller.decisionTimeoutProvider = { [weak self] in
            self?.voiceCallDeciderResolution.dedicatedCandidate.map {
                VoiceCallDeciderBudget.decisionTimeout(modelBytes: $0.bytes)
            }
        }
        controller.onDecisionCancelled = { [weak self] in self?.voiceCallDeciderHost.cancelInFlight() }
        return controller
    }

    /// The decider runs on the local model that is already loaded: no
    /// network, no new credential, no download. With no model loaded it
    /// throws, which the race turns into draft-first.
    private func voiceCallSubmitDecider() -> VoiceCallSubmitDeciding {
        let dedicated = VoiceCallLocalDecider { [weak self] messages, options in
            guard let host = await self?.voiceCallDeciderHost else { throw VoiceCallDeciderError.notWanted }
            return try await host.complete(messages, options: options)
        }
        let automatic = VoiceCallLocalDecider { [weak self] messages, options in
            guard let session = await self?.session else {
                throw TurboSparkError(code: .invalidArgument, message: "no local model loaded")
            }
            var response = ""
            for try await event in session.generate(messages, options: options) {
                try Task.checkCancellation()
                if case .content(let chunk) = event { response += chunk }
            }
            return response
        }
        return VoiceCallRoutedDecider(
            automatic: automatic, dedicated: dedicated,
            usesDedicated: { [weak self] in
                await self?.voiceCallDeciderResolution.dedicatedCandidate != nil
            })
    }

    // MARK: Dedicated decider model

    /// Installed text models that can answer the check: loadable now, a chat
    /// model (not a vision sidecar). Nothing is downloaded or loaded to list them.
    var voiceCallDeciderCandidates: [VoiceCallDeciderCandidate] {
        loadableModels
            .filter { $0.modality == "text" && $0.kind == nil }
            .map {
                VoiceCallDeciderCandidate(
                    installId: $0.installId, alias: $0.alias, path: $0.path, bytes: $0.installBytes)
            }
    }

    var voiceCallDeciderResolution: VoiceCallDeciderResolution {
        VoiceCallDeciderResolution.resolve(
            savedID: voiceCallDeciderModelID, available: voiceCallDeciderCandidates,
            loadedChatPath: loadedModelPath)
    }

    func setVoiceCallDeciderModelID(_ id: String?) {
        guard voiceCallDeciderModelID != id else { return }
        voiceCallDeciderModelID = id
        persistSettings()
        syncVoiceCallDecider()
    }

    /// A card is waiting for the user: a tool approval, a question, or a
    /// plan parked in the blocking ExitPlanMode gate. The call pauses on any
    /// of them and never auto-sends over one.
    var hasPendingVoiceCallApproval: Bool {
        pendingToolCall != nil || pendingUserQuestions != nil || pendingPlanApproval != nil
    }

    /// Brings the second session in line with the setting, the call and the
    /// choice. It is wanted only while a call is active with auto-send on.
    func syncVoiceCallDecider() {
        let resolved = voiceCallDeciderResolution.dedicatedCandidate
        voiceCallDeciderHost.configure(candidate: resolved)
        voiceCallDeciderHost.setWanted(
            resolved != nil && voiceCallActive && voiceCallEnabled && voiceCallAutoSendEnabled)
    }

    func makeVoiceCallDeciderHost() -> VoiceCallDeciderHost {
        let host = VoiceCallDeciderHost()
        host.turnBusy = { [weak self] in
            guard let self else { return true }
            return self.generating || self.submitting || self.hasPendingVoiceCallApproval
        }
        host.open = { [weak self] path in
            // Same resource policy as the chat session and Humanize, read on
            // the main actor at open time.
            let (options, moving): (OpenOptions, Bool) = await MainActor.run { [weak self] in
                var options = OpenOptions()
                guard let self else { return (options, true) }
                options.powerProfile = self.runtimeOptions.powerProfile.powerProfile
                options.loadGuard = self.activeLoadGuard
                // The rubric and one utterance fit in a small window.
                options.maxContext = .fixed(2_048)
                return (options, self.isMoving(path: path))
            }
            guard !moving else { throw VoiceCallDeciderError.notReady }
            return VoiceCallSessionEngine(session: try await TurboSparkSession(modelPath: path, options: options))
        }
        return host
    }

    /// OC-08: the ONE path that sends a voice utterance without the user's
    /// press of Send. Every guard is re-checked here, at the moment of
    /// sending, whatever the caller believes. A refusal changes nothing and
    /// the text stays in the composer.
    func submitAutoVoiceUtterance(
        _ text: String, chatID: UUID, decision: VoiceCallSubmitDecision, addressed: Bool
    ) -> Bool {
        let permitted = VoiceCallAutoSubmitGate.permits(.init(
            settingOn: voiceCallAutoSendEnabled,
            callEnabled: voiceCallEnabled,
            decision: decision,
            sameChat: selectedChatID == chatID,
            approvalPending: hasPendingVoiceCallApproval,
            turnBusy: generating || submitting,
            canRun: canRun && !imageModeEnabled,
            text: text,
            composerText: promptText,
            addressed: addressed))
        guard permitted else { return false }
        run()
        return true
    }

    func setVoiceCallAutoSendEnabled(_ enabled: Bool) {
        guard voiceCallAutoSendEnabled != enabled else { return }
        voiceCallAutoSendEnabled = enabled
        persistSettings()
        syncVoiceCallDecider()
    }

    /// The call uses the live text model the Record page uses.
    func voiceCallSpeechModelPath() -> String? {
        guard audioWorkspaceCreated else { return nil }
        return audioWorkspace.liveTextProfile(for: nil).flatMap { audioWorkspace.installedPaths[$0.identity.alias] }
    }

    func voiceCallSpeechModelStatus() async -> VoiceCallSpeechModelStatus {
        let workspace = audioWorkspace
        // A workspace built just now is still reading its install receipts.
        var waits = 0
        while workspace.isRefreshingModels, waits < 100 {
            try? await Task.sleep(nanoseconds: 50_000_000)
            waits += 1
        }
        let profile = workspace.liveTextProfile(for: nil)
        return VoiceCallSpeechModel.status(
            profileListed: profile != nil,
            canRun: profile?.capabilities.canRun ?? false,
            unavailableReason: profile?.capabilities.unavailableReason,
            installed: profile.flatMap { workspace.installedPaths[$0.identity.alias] } != nil,
            english: AudioWorkspaceController.isEnglish("auto"))
    }

    /// The Download offer in the call panel.
    func downloadVoiceCallSpeechModel() {
        audioWorkspace.installLiveTextModel()
    }

    // MARK: Microphone

    /// Input devices from the Core Audio device list. Listing reads HAL
    /// properties only: it opens no input and never shows the macOS prompt.
    func voiceCallMicrophones() -> [MicrophoneSelection.Device] {
        ((try? AudioCapture.microphones()) ?? [])
            .filter { !$0.uid.isEmpty }
            .map { MicrophoneSelection.Device(id: $0.id, uid: $0.uid, name: $0.name) }
    }

    func voiceCallMicrophoneResolution(
        available: [MicrophoneSelection.Device]? = nil
    ) -> MicrophoneSelection.Resolution {
        MicrophoneSelection.resolve(savedUID: voiceCallMicrophoneUID, available: available ?? voiceCallMicrophones())
    }

    func setVoiceCallMicrophoneUID(_ uid: String?) {
        guard voiceCallMicrophoneUID != uid else { return }
        voiceCallMicrophoneUID = uid
        persistSettings()
    }

    func setVoiceCallEnabled(_ enabled: Bool) {
        guard voiceCallEnabled != enabled else { return }
        voiceCallEnabled = enabled
        persistSettings()
    }
}

extension Notification.Name {
    /// A menu item or button asks the conversation pane to start (or end) a
    /// call. The pane owns the access gate, so every entry point shares it.
    static let toggleVoiceCall = Notification.Name("TurboSpark.toggleVoiceCall")
    /// The Toggle Talk chord asks the call panel to flip Talk (push to talk).
    static let toggleVoiceTalk = Notification.Name("TurboSpark.toggleVoiceTalk")
}
