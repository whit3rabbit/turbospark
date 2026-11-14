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

        // A turn is busy while it generates, while a submission awaits its
        // hook, and while an approval card is open (`generating` is lowered
        // on purpose while a call waits for a human).
        Publishers.CombineLatest4(
            model.$generating, model.$submitting,
            model.$pendingToolCall.map { $0 != nil }.removeDuplicates(),
            model.$pendingUserQuestions.map { $0 != nil }.removeDuplicates()
        )
        .map { $0 || $1 || $2 || $3 }
        .removeDuplicates()
        .dropFirst()
        .sink { [weak self] nowBusy in self?.busyChanged(nowBusy) }
        .store(in: &cancellables)

        Publishers.CombineLatest(
            model.$pendingToolCall.map { $0 != nil }.removeDuplicates(),
            model.$pendingUserQuestions.map { $0 != nil }.removeDuplicates()
        )
        .map { $0 || $1 }
        .removeDuplicates()
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
        model.generating || model.submitting || model.pendingToolCall != nil || model.pendingUserQuestions != nil
    }

    var selectedChatID: UUID { model.selectedChatID }
    var isTurnBusy: Bool { Self.isBusy(model) }
    var hasPendingApproval: Bool { model.pendingToolCall != nil || model.pendingUserQuestions != nil }
    var draft: String {
        get { model.promptText }
        set { model.promptText = newValue }
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
            .sink { [weak self] active in self?.voiceCallActive = active }
        return controller
    }

    private func makeVoiceCallControllerCore(
        port: AppModelVoiceCallPort, environment: VoiceCallEnvironment
    ) -> VoiceCallController {
        VoiceCallController(
            chat: port, audio: AudioCaptureVoiceSource(), speaker: VoiceCallSpeaker(),
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
            })
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
}
