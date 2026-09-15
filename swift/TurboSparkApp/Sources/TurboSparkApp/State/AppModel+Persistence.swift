import Foundation
import TurboSpark

extension AppModel {
    /// Clamps a persisted integer that later becomes a `UInt32`.
    ///
    /// **A NEGATIVE OR OVERSIZED VALUE IN `settings.json` IS A TRAP, NOT AN
    /// ERROR.** `UInt32(maxContextTokens)`, `UInt32(expertCacheSlots)` and
    /// `UInt32(topK)` are all assigned verbatim from disk and all of them
    /// crash the app on a value outside the range -- at `open()` for the
    /// first two and at the first generate for the third. The file is
    /// user-editable and survives across versions that changed what a field
    /// means, so "nothing writes a bad one today" is not the question.
    ///
    /// **`maxNewTokens` IS THE FOURTH, AND THIS DOC USED TO CLAIM IT WAS THE
    /// ONE ALREADY GUARDED** (state#35). Its use site read
    /// `UInt32(max(1, maxNewTokens))`, and `max(1,)` guards the LOWER bound
    /// only: `UInt32(_:)` still traps on anything above `UInt32.max`, which
    /// is exactly the shape the other three were fixed for. A comment
    /// asserting a guard is not a guard.
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
        self.reasoning = GenerateOptions.Reasoning(rawValue: settings.reasoning) ?? .off
        self.maxNewTokens = Self.clampedSetting(
            settings.maxNewTokens, upperBound: Int(UInt32.max))
        self.repetitionPenaltyEnabled = settings.repetitionPenaltyEnabled
        self.repetitionPenalty = settings.repetitionPenalty
        self.seedEnabled = settings.seedEnabled
        self.seed = settings.seed
        self.stopSequences = settings.stopSequences
        self.samplingPresets = settings.samplingPresets
        self.defaultSystemPrompt = settings.defaultSystemPrompt
        self.systemPrompts = settings.systemPrompts
        self.selectedSystemPromptID = UUID(uuidString: settings.activeSystemPromptID)
            .flatMap { id in settings.systemPrompts.contains(where: { $0.id == id }) ? id : nil }
        self.personalities = settings.personalities
        self.selectedPersonalityID = UUID(uuidString: settings.activePersonalityID)
            .flatMap { id in settings.personalities.contains(where: { $0.id == id }) ? id : nil }
        self.soulPrompt = settings.soulPrompt
        self.pluginEnableState = settings.enabledPlugins
        self.runtimeOptions.powerProfile = AppPowerProfileOption(rawValue: settings.powerProfile) ?? .auto
        self.runtimeOptions.loadGuard = AppLoadGuardOption(rawValue: settings.loadGuard) ?? .relaxed
        self.runtimeOptions.loadGuardCustomBytes = settings.loadGuardCustomBytes
        self.runtimeOptions.minAutoContextTokens = settings.minAutoContextTokens
        self.runtimeOptions.speculation = AppSpeculationOption(rawValue: settings.speculation) ?? .auto
        self.runtimeOptions.speculativeDrafter = AppSpeculativeDrafterOption(rawValue: settings.speculativeDrafter) ?? .auto
        self.runtimeOptions.kvBits = AppKvBitsOption(rawValue: settings.kvBits) ?? .auto
        self.runtimeOptions.maxTokensPerSec = settings.maxTokensPerSec
        self.runtimeOptions.steeringPath = settings.steeringPath.isEmpty ? nil : settings.steeringPath
        self.runtimeOptions.steeringMode = AppSteeringModeOption(rawValue: settings.steeringMode) ?? .ablate
        self.runtimeOptions.steeringScale = settings.steeringScale
        self.runtimeOptions.steeringLayers = settings.steeringLayers
        self.runtimeOptions.steeringTarget = settings.steeringTarget
        self.runtimeOptions.steeringGate = settings.steeringGate
        self.steeringPath = self.runtimeOptions.steeringPath
        self.steeringPresets = settings.steeringPresets
        // A stored id naming a preset that is gone resolves to nil rather
        // than to the first row: silently steering with a DIFFERENT direction
        // than the one selected is the failure this whole surface exists to
        // avoid, and `installed.first` was exactly that mistake one file over
        // (state#97).
        self.activeSteeringPresetID = UUID(uuidString: settings.activeSteeringPresetID)
            .flatMap { id in settings.steeringPresets.contains(where: { $0.id == id }) ? id : nil }
        self.steeringEnabled = settings.steeringEnabled
        self.enableLMStudioDetection = settings.enableLMStudioDetection
        self.lmStudioDirectory = settings.lmStudioDirectory
        self.customModelDirectories = settings.customModelDirectories
        self.guardrailsMode = AppGuardrailsMode(rawValue: settings.guardrailsMode) ?? .select
        self.modelReasoningDefaults = settings.modelReasoningDefaults
        self.interactionMode = AppInteractionMode(rawValue: settings.interactionMode) ?? .chat
        self.alwaysStartInGhostMode = settings.alwaysStartInGhostMode
        self.autoCompactEnabled = settings.autoCompact
        self.actionFusionEnabled = settings.actionFusion
        self.observationPackEnabled = settings.observationPack
        self.evidenceReducerEnabled = settings.evidenceReducer
        self.todoBoundaryCompactionEnabled = settings.todoBoundaryCompaction
        self.memoryEnabled = settings.memoryEnabled
        self.memoryEmbeddingModel = settings.memoryEmbeddingModel
        self.syntextIndexingEnabled = settings.syntextIndexingEnabled
        AppToolRegistry.syntextIndexingEnabled = settings.syntextIndexingEnabled
        self.compactionKeepRecentTurns = AppChatCompaction.clampKeepRecent(
            settings.compactionKeepRecentTurns)
        // The pinned port is a plain preference; the server API key is a
        // credential and comes from the Keychain instead (ServerKeychain).
        self.serverHost = settings.serverHost
        self.serverFavorites = settings.serverFavorites
        self.serverPinnedPort = settings.serverPinnedPort
        self.serverAPIKeyInput = ServerKeychain.loadKey() ?? ""
        self.serverEmbeddingModelInput = settings.serverEmbeddingModel
        self.hfEndpointInput = settings.hfEndpoint
        if !settings.hfEndpoint.isEmpty {
            try? TurboSparkCatalog.setHfEndpoint(settings.hfEndpoint)
        }
        self.showMenuBarItem = settings.showMenuBarItem
        self.keepFansPinnedOnQuit = settings.keepFansPinnedOnQuit
        // The quit restore reads the controller's live copy: the settings
        // write is debounced, so a toggle-and-quick-quit must not consult
        // disk and find the previous value there.
        FanController.shared.keepFansPinnedOnQuit = settings.keepFansPinnedOnQuit
        self.serverAutoStartOnLaunch = settings.serverAutoStartOnLaunch
        self.keepServerRunningInBackground = settings.keepServerRunningInBackground
        // `ToolRiskClassifier` is a static surface reached from the agent loop
        // with no AppModel in hand, so the flag lives on the gate rather than
        // being threaded through `assessTerminalCommand`. Set it here, once,
        // where settings are already being applied.
        CommandGate.vetoEnabled = settings.commandAdvisoryVeto
        self.agentModeHints = settings.agentModeHints

        if settings.serverAutoStartOnLaunch && server == nil {
            startServer()
        }
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
            reasoning: reasoning.rawValue,
            maxNewTokens: maxNewTokens,
            repetitionPenaltyEnabled: repetitionPenaltyEnabled,
            repetitionPenalty: repetitionPenalty,
            seedEnabled: seedEnabled,
            seed: seed,
            stopSequences: stopSequences,
            samplingPresets: samplingPresets,
            powerProfile: runtimeOptions.powerProfile.rawValue,
            loadGuard: runtimeOptions.loadGuard.rawValue,
            loadGuardCustomBytes: runtimeOptions.loadGuardCustomBytes,
            minAutoContextTokens: runtimeOptions.minAutoContextTokens,
            speculation: runtimeOptions.speculation.rawValue,
            speculativeDrafter: runtimeOptions.speculativeDrafter.rawValue,
            kvBits: runtimeOptions.kvBits.rawValue,
            maxTokensPerSec: runtimeOptions.maxTokensPerSec,
            steeringPath: runtimeOptions.steeringPath ?? "",
            steeringMode: runtimeOptions.steeringMode.rawValue,
            steeringScale: runtimeOptions.steeringScale,
            steeringLayers: runtimeOptions.steeringLayers,
            steeringTarget: runtimeOptions.steeringTarget,
            steeringGate: runtimeOptions.steeringGate,
            steeringPresets: steeringPresets,
            activeSteeringPresetID: activeSteeringPresetID?.uuidString ?? "",
            steeringEnabled: steeringEnabled,
            enableLMStudioDetection: enableLMStudioDetection,
            lmStudioDirectory: lmStudioDirectory,
            customModelDirectories: customModelDirectories,
            commandAdvisoryVeto: CommandGate.vetoEnabled,
            guardrailsMode: guardrailsMode.rawValue,
            modelReasoningDefaults: modelReasoningDefaults,
            interactionMode: interactionMode.rawValue,
            alwaysStartInGhostMode: alwaysStartInGhostMode,
            autoCompact: autoCompactEnabled,
            actionFusion: actionFusionEnabled,
            observationPack: observationPackEnabled,
            evidenceReducer: evidenceReducerEnabled,
            todoBoundaryCompaction: todoBoundaryCompactionEnabled,
            compactionKeepRecentTurns: compactionKeepRecentTurns,
            serverHost: serverHost,
            serverFavorites: serverFavorites,
            serverPinnedPort: serverPinnedPort,
            defaultSystemPrompt: defaultSystemPrompt,
            systemPrompts: systemPrompts,
            activeSystemPromptID: selectedSystemPromptID?.uuidString ?? "",
            personalities: personalities,
            activePersonalityID: selectedPersonalityID?.uuidString ?? "",
            soulPrompt: soulPrompt,
            enabledPlugins: pluginEnableState,
            showMenuBarItem: showMenuBarItem,
            keepFansPinnedOnQuit: keepFansPinnedOnQuit,
            serverAutoStartOnLaunch: serverAutoStartOnLaunch,
            keepServerRunningInBackground: keepServerRunningInBackground,
            serverEmbeddingModel: serverEmbeddingModelInput,
            hfEndpoint: hfEndpointInput,
            memoryEnabled: memoryEnabled,
            memoryEmbeddingModel: memoryEmbeddingModel,
            agentModeHints: agentModeHints,
            syntextIndexingEnabled: syntextIndexingEnabled
        )
        // The API key follows its own storage: Keychain, written only when
        // the field changed, so a persist of unrelated settings does not
        // touch the item.
        if serverAPIKeyInput != (ServerKeychain.loadKey() ?? "") {
            ServerKeychain.saveKey(serverAPIKeyInput)
        }
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
    /// Repairs tool calls left mid-flight by an unclean exit, before the
    /// loaded chats reach the UI.
    ///
    /// **A CRASH MID-TURN PERSISTS A CALL THAT NEVER FINISHED.**
    /// `persistChats()` writes at message-append and status-change points, so
    /// a force-quit while a call executes or waits for approval lands an
    /// archive whose row reads `.running` or `.pendingApproval` -- and
    /// nothing in-memory resets either on the next launch: the transcript
    /// shows a spinner that never stops or an approval affordance no button
    /// can resolve, and the prompt assembly replays a `toolCalls` row with
    /// no matching result (the result is appended only on completion). The
    /// repair gives each state its live-path terminal twin -- `.failed` and
    /// `.denied` respectively, with the synthetic result row the live path
    /// would have recorded -- so the next turn shows the model an honest
    /// record instead of a dangling call. Alternates are repaired too: an
    /// edited or retried row carries its own stuck versions.
    ///
    /// Load-time only, deliberately: mid-turn `.running` rows on disk are the
    /// CORRECT live state, and the repair is for the state they freeze into
    /// when no process is left to finish them.
    nonisolated static func reconcilingOrphanedToolCalls(_ chats: [AppChat]) -> [AppChat] {
        func repair(_ message: AppChatMessage) -> AppChatMessage {
            var message = message
            message.alternates = message.alternates.map(repair)
            var results = message.toolResults
            for index in message.toolCalls.indices {
                switch message.toolCalls[index].status {
                case .running:
                    message.toolCalls[index].status = .failed
                    if !results.contains(where: { $0.callID == message.toolCalls[index].id }) {
                        results.append(AppToolResult(
                            callID: message.toolCalls[index].id,
                            output: "Interrupted: the app quit before this call finished, "
                                + "and no result was recorded.",
                            isError: true))
                    }
                case .pendingApproval:
                    message.toolCalls[index].status = .denied
                    if !results.contains(where: { $0.callID == message.toolCalls[index].id }) {
                        results.append(AppToolResult(
                            callID: message.toolCalls[index].id,
                            output: "Tool execution denied: the app quit before this call was approved.",
                            isError: true))
                    }
                case .completed, .denied, .failed:
                    break
                }
            }
            message.toolResults = results
            return message
        }
        return chats.map { chat in
            var chat = chat
            chat.messages = chat.messages.map(repair)
            return chat
        }
    }

    func loadChats() {
        let archive = AppChatFileStore.load()
        self.chats = Self.reconcilingOrphanedToolCalls(archive.chats)
        if archive.chats.isEmpty || archive.chats.contains(where: { $0.id == archive.selectedChatID }) {
            self.selectedChatID = archive.selectedChatID
        } else {
            self.selectedChatID = archive.chats[0].id
        }
        // Restore active goals AFTER the rows are in: the mirror hydrates
        // through `restoredForRelaunch()`, which keeps only the condition
        // and the set time (CC's resume rule), so counters and timers do
        // not come back.
        restoreGoalsFromRows()
    }

    /// The chats that may reach disk: ghost chats live only in memory.
    var archivableChats: [AppChat] {
        chats.filter { !$0.isGhost }
    }

    /// Archive construction shared by the immediate and debounced writers.
    ///
    /// **THIS FILTER IS THE WHOLE GHOST GUARANTEE.** Every one of the ~30
    /// save sites funnels through `persistChats()` or `persistChatsDebounced()`
    /// below, and these are the only two places an `AppChatArchive` is built,
    /// so excluding `isGhost` rows here excludes them from disk -- quit,
    /// crash, profile switch and all. A selection pointing at a ghost (or at
    /// nothing archivable) is rewritten to the newest archivable chat, so
    /// the file never names a chat it does not contain; `loadChats()` would
    /// repair a stale id anyway, but a file that references an absent row is
    /// its own trap.
    func makeChatArchive() -> AppChatArchive {
        let archivable = archivableChats
        assert(
            chats.allSatisfy { chat in
                !chat.isGhost || (chat.messages.isEmpty && chat.todos.isEmpty
                    && chat.draft.isEmpty && chat.contextSummary == nil
                    && chat.skillState == nil)
            },
            "ghost chat carrying plaintext conversation content on its row")
        let selection = archivable.contains(where: { $0.id == selectedChatID })
            ? selectedChatID
            : (archivable.first?.id ?? selectedChatID)
        return AppChatArchive(selectedChatID: selection, chats: archivable)
    }

    /// Persists all conversation threads and active selection to disk.
    public func persistChats() {
        chatPersistDebounceTask?.cancel()
        chatPersistDebounceTask = nil
        AppChatFileStore.save(makeChatArchive())
        surfaceStorageIssues()
    }

    /// Shows a storage failure the JSON store recorded, once, and clears it.
    ///
    /// **`lastWriteError` HAD NO PRODUCTION READER** (state#42). It exists so
    /// a failed write stops being invisible, and nothing ever read it -- so a
    /// full disk lost the session exactly as silently as before the field was
    /// added, with the mechanism in place and unwired. `lastReadError` is its
    /// twin on the load side. Reported as an error toast rather than thrown:
    /// both are recorded on `didSet`-shaped paths that cannot propagate.
    /// **THE LATCH IS WHAT COVERS THE STORES THAT CANNOT CALL THIS**
    /// (state#103). `ModelOrganizationStore.save` and
    /// `DisabledItemStore.setNames` are not on `AppModel` and have no toast
    /// to raise, but `AppJSONStore` HOLDS the error until it is read -- so a
    /// failure there is reported at the next chat, project or MCP write
    /// rather than lost. Delayed, not dropped; the alternative is a second
    /// reporting path for two stores that write kilobytes.
    func surfaceStorageIssues() {
        if let readError = AppJSONStore.lastReadError {
            AppJSONStore.clearLastReadError()
            showToast(readError, style: .error, duration: 10)
        }
        if let writeError = AppJSONStore.lastWriteError {
            AppJSONStore.clearLastWriteError()
            showToast(writeError, style: .error, duration: 10)
        }
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
            AppChatFileStore.save(self.makeChatArchive())
            // **THE DEBOUNCED WRITE REPORTS TOO** (state#103). state#42 gave
            // `lastWriteError` a reader and put it on `persistChats()`; this
            // is the path a DRAFT takes, and a disk that has gone read-only
            // fails here first and silently. Four other writers were in the
            // same position.
            self.surfaceStorageIssues()
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
        surfaceStorageIssues()
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
        surfaceStorageIssues()
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
        // Background work is NOT reachable by `cancel()`: shells are
        // separate processes and agent tasks are unstructured, so both would
        // otherwise outlive the app (a shell's children keep running after
        // the process exits). Agents are cancelled first so a completion
        // cannot race the persists below, then the shells' whole trees die.
        stopAllBackgroundWorkForShutdown()
        detachChatSessionFromServer()
        session = nil

        // A draft written before any chat existed lives outside `chats`, and
        // `persistChats` walks `chats` (state#63). The two writers promote it
        // as it gains content; this is the backstop for a path that set it
        // some other way.
        materializeDraftChatIfNeeded()

        // Ghost chats die with the process anyway (nothing persisted them),
        // so this drops the sealed payloads and replaces the vault key; the
        // filtered persist below could not write them if it tried.
        ghostVault.wipeAll()

        // Both force the pending debounces through rather than waiting on
        // them, so the last keystroke and the last setting reach disk.
        persistChats()
        persistSettings()
    }
}
