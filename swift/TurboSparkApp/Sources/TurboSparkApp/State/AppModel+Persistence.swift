import Foundation
import TurboSpark

extension AppModel {
    /// Clamps a persisted integer that later becomes a `UInt32`.
    ///
    /// **A NEGATIVE OR OVERSIZED VALUE IN `settings.json` IS A TRAP, NOT AN
    /// ERROR.** `UInt32(maxContextTokens)`, `UInt32(expertCacheSlots)` and
    /// `UInt32(topK)` are all assigned verbatim from disk and all of them
    /// crash the app on a value outside the range -- at `open()` for the
    /// first two and at the first generate for the third. `maxNewTokens` was
    /// the only one already guarded (`max(1, ...)` at its use site). The file
    /// is user-editable and survives across versions that changed what a
    /// field means, so "nothing writes a bad one today" is not the question.
    private static func clampedSetting(_ value: Int, upperBound: Int) -> Int {
        min(max(0, value), upperBound)
    }

    /// Loads persisted generation parameters, execution options, and model
    /// paths from disk.
    func loadSettings() {
        let settings = MacAppSettingsFileStore.load()
        self.maxContextTokens = Self.clampedSetting(
            settings.contextTokens, upperBound: Int(UInt32.max))
        // Not merely clamped: an out-of-set slot count is refused by
        // `ts_session_open` and, before that guard existed, panicked the
        // process. Anything unrecognized falls back to automatic sizing.
        self.runtimeOptions.expertCacheSlots =
            AppRuntimeOptions.allowedSlotCounts.contains(settings.expertCacheSlots)
            ? settings.expertCacheSlots : 0
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = Self.clampedSetting(settings.topK, upperBound: Int(UInt32.max))
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.runtimeOptions.prefillEnabled = settings.prefillEnabled
        self.reasoning = GenerateOptions.Reasoning(rawValue: settings.reasoning) ?? .off
        self.maxNewTokens = settings.maxNewTokens
        self.repetitionPenaltyEnabled = settings.repetitionPenaltyEnabled
        self.repetitionPenalty = settings.repetitionPenalty
        self.seedEnabled = settings.seedEnabled
        self.seed = settings.seed
        self.stopSequences = settings.stopSequences
        self.runtimeOptions.powerProfile = AppPowerProfileOption(rawValue: settings.powerProfile) ?? .auto
        self.runtimeOptions.loadGuard = AppLoadGuardOption(rawValue: settings.loadGuard) ?? .relaxed
        self.runtimeOptions.loadGuardCustomBytes = settings.loadGuardCustomBytes
        self.runtimeOptions.minAutoContextTokens = settings.minAutoContextTokens
        self.runtimeOptions.speculation = AppSpeculationOption(rawValue: settings.speculation) ?? .auto
        self.runtimeOptions.speculativeDrafter = AppSpeculativeDrafterOption(rawValue: settings.speculativeDrafter) ?? .auto
        self.runtimeOptions.maxTokensPerSec = settings.maxTokensPerSec
        self.runtimeOptions.steeringPath = settings.steeringPath.isEmpty ? nil : settings.steeringPath
        self.runtimeOptions.steeringMode = AppSteeringModeOption(rawValue: settings.steeringMode) ?? .ablate
        self.runtimeOptions.steeringScale = settings.steeringScale
        self.runtimeOptions.steeringLayers = settings.steeringLayers
        self.runtimeOptions.steeringTarget = settings.steeringTarget
        self.runtimeOptions.steeringGate = settings.steeringGate
        self.steeringPath = self.runtimeOptions.steeringPath
        self.modelsDirectory = settings.modelsDirectory
        self.enableLMStudioDetection = settings.enableLMStudioDetection
        self.lmStudioDirectory = settings.lmStudioDirectory
        self.customModelDirectories = settings.customModelDirectories
        self.guardrailsMode = AppGuardrailsMode(rawValue: settings.guardrailsMode) ?? .select
        self.modelReasoningDefaults = settings.modelReasoningDefaults
        self.interactionMode = AppInteractionMode(rawValue: settings.interactionMode) ?? .chat
        // `ToolRiskClassifier` is a static surface reached from the agent loop
        // with no AppModel in hand, so the flag lives on the gate rather than
        // being threaded through `assessTerminalCommand`. Set it here, once,
        // where settings are already being applied.
        CommandGate.vetoEnabled = settings.commandAdvisoryVeto
    }

    /// Persists after a short quiet period, collapsing a burst of mutations
    /// (a slider drag firing on every intermediate value) into one write.
    /// See `persistChatsDebounced()`, the same pattern for the chat archive.
    public func persistSettingsDebounced() {
        settingsPersistDebounceTask?.cancel()
        settingsPersistDebounceTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled, let self else { return }
            self.settingsPersistDebounceTask = nil
            self.persistSettings()
        }
    }

    /// Persists current runtime options, steering parameters, and directory paths to disk.
    public func persistSettings() {
        settingsPersistDebounceTask?.cancel()
        settingsPersistDebounceTask = nil
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled,
            reasoning: reasoning.rawValue,
            maxNewTokens: maxNewTokens,
            repetitionPenaltyEnabled: repetitionPenaltyEnabled,
            repetitionPenalty: repetitionPenalty,
            seedEnabled: seedEnabled,
            seed: seed,
            stopSequences: stopSequences,
            powerProfile: runtimeOptions.powerProfile.rawValue,
            loadGuard: runtimeOptions.loadGuard.rawValue,
            loadGuardCustomBytes: runtimeOptions.loadGuardCustomBytes,
            minAutoContextTokens: runtimeOptions.minAutoContextTokens,
            speculation: runtimeOptions.speculation.rawValue,
            speculativeDrafter: runtimeOptions.speculativeDrafter.rawValue,
            maxTokensPerSec: runtimeOptions.maxTokensPerSec,
            steeringPath: runtimeOptions.steeringPath ?? "",
            steeringMode: runtimeOptions.steeringMode.rawValue,
            steeringScale: runtimeOptions.steeringScale,
            steeringLayers: runtimeOptions.steeringLayers,
            steeringTarget: runtimeOptions.steeringTarget,
            steeringGate: runtimeOptions.steeringGate,
            modelsDirectory: modelsDirectory,
            enableLMStudioDetection: enableLMStudioDetection,
            lmStudioDirectory: lmStudioDirectory,
            customModelDirectories: customModelDirectories,
            commandAdvisoryVeto: CommandGate.vetoEnabled,
            guardrailsMode: guardrailsMode.rawValue,
            modelReasoningDefaults: modelReasoningDefaults,
            interactionMode: interactionMode.rawValue
        )
        MacAppSettingsFileStore.save(settings)
    }

    /// Loads persisted chat threads and the active selected conversation ID.
    ///
    /// Validates `selectedChatID` against the loaded chats rather than
    /// trusting it outright: an archive whose selection points at nothing
    /// (corrupt state, or a chat removed by some other path) used to be
    /// silently repaired later by `selectedChat`'s getter mutating
    /// `@Published` state on read (state#7). Fixing it up front, once, at
    /// the one place `chats` is genuinely being (re)established keeps that
    /// getter pure.
    func loadChats() {
        let archive = AppChatFileStore.load()
        self.chats = archive.chats
        if archive.chats.isEmpty || archive.chats.contains(where: { $0.id == archive.selectedChatID }) {
            self.selectedChatID = archive.selectedChatID
        } else {
            self.selectedChatID = archive.chats[0].id
        }
    }

    /// Persists all conversation threads and active selection to disk.
    public func persistChats() {
        chatPersistDebounceTask?.cancel()
        chatPersistDebounceTask = nil
        let archive = AppChatArchive(selectedChatID: selectedChatID, chats: chats)
        AppChatFileStore.save(archive)
    }

    /// Persists after a short quiet period, collapsing a burst of mutations
    /// into one write.
    ///
    /// For the DRAFT path only. `promptText`'s setter calls into persistence
    /// on every keystroke, and the store writes the entire archive whole with
    /// `.atomic` on each call, so typing a sentence re-encoded and rewrote
    /// every chat in the file sixty times. Anything that must survive a crash
    /// (an appended message, a completed turn) still calls `persistChats()`
    /// directly and writes immediately; a half-typed draft does not need
    /// that guarantee.
    ///
    /// Cancelling in `persistChats()` above is what keeps the two orderings
    /// from racing: a pending debounced write must never land AFTER an
    /// immediate one and reinstate older state.
    public func persistChatsDebounced() {
        chatPersistDebounceTask?.cancel()
        chatPersistDebounceTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled, let self else { return }
            self.chatPersistDebounceTask = nil
            let archive = AppChatArchive(selectedChatID: self.selectedChatID, chats: self.chats)
            AppChatFileStore.save(archive)
        }
    }

    /// Loads persisted project configurations and selected project ID.
    ///
    /// **THE RESTORED ID IS VALIDATED AND THE WORKTREE REBUILT.** It used to
    /// be assigned verbatim, which cost two things on every relaunch: an id
    /// no longer in `projects` filtered the chat list down to nothing with no
    /// way to see why, and even a VALID id left `worktree` nil -- so the git
    /// pane came up empty until the user re-picked the project they were
    /// already in.
    func loadProjects() {
        let archive = AppProjectFileStore.load()
        self.projects = archive.projects
        guard let restored = archive.selectedProjectID,
            let project = archive.projects.first(where: { $0.id == restored })
        else {
            self.selectedProjectID = nil
            self.worktree = nil
            return
        }
        self.selectedProjectID = restored
        if let path = project.rootDirectoryPath, !path.isEmpty {
            self.worktree = WorktreeModel(rootDirectoryPath: path)
        } else {
            self.worktree = nil
        }
    }

    /// Persists all project workspaces and active project selection.
    public func persistProjects() {
        let archive = AppProjectArchive(selectedProjectID: selectedProjectID, projects: projects)
        AppProjectFileStore.save(archive)
    }

    /// Loads global Model Context Protocol server configurations.
    func loadGlobalMcpServers() {
        let archive = GlobalMcpFileStore.load()
        self.globalMcpServers = archive.servers
    }

    /// Persists global MCP servers to application support directory.
    public func persistGlobalMcpServers() {
        let archive = GlobalMcpArchive(servers: globalMcpServers)
        GlobalMcpFileStore.save(archive)
    }
}

/// Ordered shutdown, and the hand-off that lets the app delegate reach it.
///
/// **THE QUIT FLUSH USED TO LIVE IN A VIEW MODIFIER** (state#25). `RootView` observed
/// `willTerminateNotification` and called `unloadModel` / `persistChats` /
/// `persistSettings`. Three things were wrong with that.
///
/// `applicationShouldTerminateAfterLastWindowClosed` is true, so the view tree
/// can already be gone when the notification arrives -- and with it the
/// 400 ms draft debounce, which is the most recent thing the user typed.
///
/// `unloadModel()` bails on `guard !generating`, so quitting mid-turn skipped
/// the whole flush including both persists.
///
/// And `stopServer()` was never called at all, so `ts_server_stop` never ran
/// and `serverPollTimer` was never invalidated.
@MainActor
public final class AppShutdownCoordinator {
    public static let shared = AppShutdownCoordinator()
    /// Set once by `RootView`; the delegate cannot see the `@StateObject`.
    public var onTerminate: (() -> Void)?
    private init() {}
}

extension AppModel {
    /// Everything that must happen before the process exits, in order.
    public func shutdown() {
        // First, so the engine stops answering requests for a model that is
        // about to be released (`swift/CLAUDE.md` Gotcha 26).
        stopServer()
        stopServerPolling()

        // Deliberately NOT `unloadModel()`: that refuses while `generating`,
        // which is exactly the case where the flush below matters most.
        cancel()
        detachChatSessionFromServer()
        session = nil

        // Both force the pending debounces through rather than waiting on
        // them, so the last keystroke and the last setting reach disk.
        persistChats()
        persistSettings()
    }
}
