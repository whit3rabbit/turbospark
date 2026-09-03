import AppKit
import Foundation
import SwiftUI
import TurboSpark

/// Primary application model and state coordinator for the TurboSpark macOS application.
@MainActor
public final class AppModel: ObservableObject {
    /// The current execution phase of text generation.
    public enum GenerationPhase: Equatable, Sendable {
        /// No active generation.
        case idle
        /// Evaluating prompt tokens into key-value cache.
        case prefill
        /// Generating new output tokens sequentially or speculatively.
        case decode
    }

    /// Primary top-level navigation destination in the application.
    public enum AppNavigationSection: String, CaseIterable, Identifiable, Sendable {
        case chat
        case files
        case modelManager
        case modelHub
        case server

        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .chat: return "Chat"
            case .files: return "Files"
            case .modelManager: return "Installed"
            case .modelHub: return "Discover"
            case .server: return "Server"
            }
        }
        public var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right"
            case .files: return "folder"
            case .modelManager: return "internaldrive"
            case .modelHub: return "shippingbox"
            case .server: return "server.rack"
            }
        }
        /// Filled variant used when the section is the active one.
        public var selectedSystemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right.fill"
            case .files: return "folder.fill"
            case .modelManager: return "internaldrive.fill"
            case .modelHub: return "shippingbox.fill"
            case .server: return "server.rack"
            }
        }
        /// Keyboard shortcut character shown in the rail tooltip.
        public var shortcutKey: Character {
            switch self {
            case .chat: return "1"
            case .files: return "2"
            case .modelManager: return "3"
            case .modelHub: return "4"
            case .server: return "5"
            }
        }
    }

    /// Currently active navigation section in the main window.
    @Published public var activeSection: AppNavigationSection = .chat

    /// Attachment currently shown in the right-hand preview pane, if any.
    ///
    /// Non-nil is what makes the preview pane visible; there is no second
    /// visibility flag that could disagree with it.
    @Published public var previewAttachmentID: UUID? = nil

    // Model management
    /// Models currently installed locally on disk.
    @Published public var installed: [InstalledModel] = []
    /// Catalog entries available for download.
    @Published public var catalog: [CatalogEntry] = []
    /// Currently selected installed model.
    @Published public var selected: InstalledModel?
    /// Active TurboSpark session when a model is loaded.
    @Published public var session: TurboSparkSession?
    /// Whether a model is currently being loaded into memory.
    @Published public var opening: Bool = false
    /// Custom filesystem path entered for manual model loading.
    @Published public var modelPathText: String = ""
    /// The in-process HTTP server, when one is running.
    ///
    /// It serves ALREADY-OPEN models rather than opening its own, and holds
    /// each one alive independently of whoever else has a reference. So a
    /// server can outlive `session`, and every site that clears `session`
    /// calls `detachChatSessionFromServer()` first -- otherwise the model
    /// stays resident and served with nothing in the Chat pane still showing
    /// it as loaded. That is `swift/CLAUDE.md` Gotcha 26 in its multi-model
    /// form: it used to be `stopServer()`, which is now the wrong tool
    /// (stopping a whole server to unload one of its models).
    @Published public var server: TurboSparkServer?
    /// Whether `startServer()`/`stopServer()` is in flight.
    @Published public var serverBusy: Bool = false
    /// Bearer / `x-api-key` value to require on the server, or empty for no
    /// auth. Read at `startServer()` time, not persisted: a key typed for
    /// one session sharing a machine is not something to write to disk by
    /// default.
    @Published public var serverAPIKeyInput: String = ""
    /// The port to ask for, or 0 to let the OS choose.
    ///
    /// 0 is the default because nothing needs a fixed one to work: the pane
    /// shows the bound address and it is one click to copy. A user pins one
    /// when something ELSE holds the number -- a config file, a shell
    /// profile, a teammate's notes -- and then a changing port is the bug.
    @Published public var serverPinnedPort: UInt16 = 0
    /// What the running server last reported: bound host and port, attached
    /// models, auth state, uptime.
    ///
    /// **PUBLISHED RATHER THAN READ PER BODY.** `TurboSparkServer.info()`
    /// takes a lock, crosses the C ABI and decodes JSON; a SwiftUI body runs
    /// far more often than a server changes. `refreshServerInfo()` is the
    /// only writer, called when something changed it and once per poll tick
    /// for the uptime.
    @Published public var serverInfo: ServerInfo?
    /// Sessions the server is holding, by the model id it serves them under.
    ///
    /// A model attached from the Server pane has its session here and
    /// NOWHERE else in the app, so this is what keeps it alive -- and
    /// removing an entry is half of what frees it, the server's own detach
    /// being the other half. The chat model appears here too when it is
    /// being served, and is the one entry `session` also holds.
    @Published public var serverAttachedSessions: [String: TurboSparkSession] = [:]
    /// The rolling window behind the Server pane's charts.
    @Published public var serverMetrics = ServerMetricsStore()
    /// The console's own buffer, bounded.
    @Published public var serverEventLog: [ServerEvent] = []
    /// Drains the server's event ring while one is running. Not `@Published`:
    /// nothing draws it, and republishing on every tick would re-render the
    /// pane for a timer identity nobody reads.
    var serverPollTimer: Timer?

    /// Primary user interface interaction mode.
    public enum AppInteractionMode: String, Codable, CaseIterable, Identifiable, Sendable {
        case chat
        case projects

        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .chat: return "Chat"
            case .projects: return "Projects"
            }
        }
        public var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right"
            case .projects: return "chevron.left.forwardslash.chevron.right"
            }
        }
    }

    /// Current UI interaction mode (Chat vs Projects / Coding).
    @Published public var interactionMode: AppInteractionMode = .chat

    // Project and Agent State
    /// All configured codebase projects.
    @Published public var projects: [AppProject] = []
    /// Currently active project filter (nil = all chats).
    @Published public var selectedProjectID: UUID? = nil
    /// Working tree model for git status and diffs when a project is selected.
    @Published public var worktree: WorktreeModel? = nil
    /// Global application-level MCP server configurations.
    @Published public var globalMcpServers: [McpServerConfig] = []
    /// Currently pending tool call requiring user approval.
    @Published public var pendingToolCall: AppToolCall? = nil
    /// The chat the pending tool call was proposed in, captured at proposal
    /// time. Approving/denying appends to THIS chat, never to whatever
    /// `selectedChatID` happens to be when the user responds -- `generating`
    /// goes false as soon as the call is proposed (state#9), so a chat
    /// switch in between is possible and must not misroute the result.
    @Published public var pendingToolCallChatID: UUID? = nil
    /// The agent-loop step the pending tool call was proposed at (state#6):
    /// resuming after approval continues from HERE, not from step 1, or
    /// `maxAutonomousSteps` resets every time a call needs confirmation.
    @Published public var pendingToolCallStep: Int = 0
    /// The project the pending tool call was EVALUATED against, captured at
    /// proposal time. `AppToolPermissionEngine` computed its `.allow` from
    /// this project's permissions and this project's root, so running the
    /// call against whatever `selectedProject` resolves to at approval time
    /// executes it under a policy nothing ever checked -- and `selectProject`
    /// guards only on `!generating`, which is false while a call waits
    /// (state#9). Not `@Published`: nothing draws it.
    var pendingToolCallProject: AppProject?
    /// Why the last SKILL.state patch was rejected, or nil if the last one
    /// merged. Surfaced rather than swallowed: a dropped patch means the run
    /// lost a step's bookkeeping, which is invisible in the transcript.
    @Published public var skillStateLastError: String? = nil

    // Skills State
    /// User-scoped skills (~/.turbospark/skills and user agent directories).
    @Published public var userSkills: [AppSkill] = []
    /// Project-scoped skills for the currently selected project.
    @Published public var projectSkills: [AppSkill] = []

    // Agents State
    /// All discovered agents (built-in, user, project).
    @Published public var discoveredAgents: [AppAgentDefinition] = []

    // Multi-chat State
    /// All user chat conversations.
    @Published public var chats: [AppChat] = []
    /// Unique identifier of the currently active chat conversation.
    ///
    /// `didSet` keeps `activeDraftChat`'s identity in step with every
    /// assignment (load, select, create, delete), which is what lets
    /// `selectedChat` below be a PURE getter: the repair used to happen
    /// lazily inside that getter, mutating `@Published` state while a
    /// SwiftUI view was reading it (state#7).
    @Published public var selectedChatID: UUID = UUID() {
        didSet {
            if activeDraftChat.id != selectedChatID {
                activeDraftChat = AppChat(id: selectedChatID)
            }
        }
    }
    /// Transient active chat draft when chats list is empty.
    private var activeDraftChat = AppChat()

    // Live Generation State
    /// Whether token generation is currently running.
    @Published public var generating: Bool = false
    /// Whether `run()` has accepted a submission and not yet reached
    /// `executeGenerationTurn`.
    ///
    /// `run()` awaits `UserPromptSubmit` hooks before it appends anything, so
    /// `generating` stays false across that await and `canRun` stays true --
    /// a second Return with any such hook installed appends the prompt twice
    /// and overwrites `runTask`. This flag is set SYNCHRONOUSLY, before the
    /// `Task`, which is the only place a second `run()` can be refused.
    @Published public var submitting: Bool = false
    /// Current phase of the generation runner.
    @Published public var phase: GenerationPhase = .idle
    /// Number of prompt tokens processed so far during prefill.
    @Published public var livePrefillDone: Int = 0
    /// Total number of prompt tokens to process during prefill.
    @Published public var livePrefillTotal: Int = 0
    /// Number of tokens emitted so far during decode.
    @Published public var liveTokenCount: Int = 0
    /// Elapsed wall-clock decode time in seconds.
    @Published public var liveElapsedDecodeSeconds: Double = 0
    /// Whether cancellation has been requested by the user.
    @Published public var isCancellationPending: Bool = false
    /// Prompt text submitted for the active turn.
    @Published public var outputPromptText: String = ""
    /// Streamed assistant output text for the active turn.
    @Published public var outputText: String = ""
    /// Streamed reasoning text for the active turn.
    @Published public var outputReasoningText: String = ""

    // Status & Diagnostics
    /// Diagnostics summary from the last generation run.
    @Published public var diagnostics: AppDiagnostics?
    /// Estimated token count of the current prompt draft.
    @Published public var estimatedPromptTokens: Int = 0
    /// System thermal and memory telemetry readings.
    @Published public var telemetry: SystemTelemetry?
    /// Active error banner message, if any.
    @Published public var error: String?
    /// Active toast notification displayed on screen and announced to assistive technologies.
    @Published public var activeToast: AppToast? = nil


    // Runtime Settings
    /// Custom maximum context tokens (0 for automatic model default).
    @Published public var maxContextTokens: Int = 0
    /// High-level runtime options such as speculative decoding and steering.
    @Published public var runtimeOptions = AppRuntimeOptions()
    /// Sampling temperature.
    @Published public var temperature: Double = 0.2
    /// Whether top-k filtering is enabled.
    @Published public var topKEnabled: Bool = true
    /// Maximum candidate tokens considered during top-k sampling.
    @Published public var topK: Int = 64
    /// Whether nucleus (top-p) filtering is enabled.
    @Published public var topPEnabled: Bool = true
    /// Cumulative probability threshold for nucleus sampling.
    @Published public var topP: Double = 0.95
    /// Reasoning effort level for reasoning-capable models.
    @Published public var reasoning: GenerateOptions.Reasoning = .off
    /// Remembered reasoning default preferences per model alias or path.
    @Published public var modelReasoningDefaults: [String: String] = [:]
    /// Maximum new tokens to generate per response.
    @Published public var maxNewTokens: Int = 2048
    /// Whether repetition penalty is enabled.
    @Published public var repetitionPenaltyEnabled: Bool = false
    /// Repetition penalty multiplier applied to generated tokens.
    @Published public var repetitionPenalty: Double = 1.0
    /// Whether a fixed random seed is specified for deterministic sampling.
    @Published public var seedEnabled: Bool = false
    /// Fixed random seed value.
    @Published public var seed: UInt64 = 0
    /// Comma-separated list of custom stop sequences.
    @Published public var stopSequences: String = ""
    /// Path to activation steering vectors file.
    @Published public var steeringPath: String? = nil
    /// Forge Tool-Call Guardrails global mode ("alwaysOn", "alwaysOff", "select").
    @Published public var guardrailsMode: AppGuardrailsMode = .select
    /// Per-session manual override for guardrails when not in a project workspace.
    @Published public var composerGuardrailsOverride: Bool? = nil

    // Storage and Model Discovery Settings
    /// Custom TurboSpark primary models directory (empty uses default ~/.turbospark/models).
    @Published public var modelsDirectory: String = ""
    /// Whether to automatically scan and include models from LM Studio library.
    @Published public var enableLMStudioDetection: Bool = true
    /// Custom LM Studio models directory (empty uses default ~/.lmstudio/models).
    @Published public var lmStudioDirectory: String = ""
    /// Additional custom directories to scan for models without copying.
    @Published public var customModelDirectories: [String] = []

    // Installation State
    /// Whether a model download and installation task is currently running.
    @Published public var isInstallingModel: Bool = false
    /// Description of the current installation stage.
    @Published public var installStageText: String? = nil
    /// Download progress fraction between 0.0 and 1.0.
    @Published public var installProgressFraction: Double? = nil
    /// Number of bytes downloaded so far.
    @Published public var installDownloadedBytes: UInt64? = nil
    /// Total expected download size in bytes.
    @Published public var installTotalBytes: UInt64? = nil
    /// Human-readable estimated time remaining for download.
    @Published public var installETAText: String? = nil
    /// The alias currently installing, if any.
    @Published public var installingAlias: String? = nil
    /// Aliases whose install was abandoned by `cancelInstall()`.
    ///
    /// The engine exposes no install-cancel call, so dropping the consumer
    /// ends DELIVERY while `ts_install` keeps streaming the checkpoint to
    /// that directory. There is no way to learn when it finishes, so a
    /// second install of the same alias is refused for the rest of the
    /// process rather than raced against the first.
    @Published public var abandonedInstallAliases: Set<String> = []

    var runTask: Task<Void, Never>?
    /// Work spawned OUTSIDE `runTask`: an approved or denied pending call,
    /// which runs a tool and then re-enters the loop. `runTask` cannot reach
    /// it (that turn's stream ended when the call was proposed), so `cancel()`
    /// cancels this too or Stop cannot stop a shell command started from the
    /// approval card.
    var toolExecutionTask: Task<Void, Never>?
    var installTask: Task<Void, Never>?
    /// The off-main-actor scan of the LM Studio and custom model directories.
    /// Cancelled and restarted per `refreshModels()`, which several views call
    /// in quick succession.
    var modelScanTask: Task<Void, Never>?
    var tokenEstimateTask: Task<Void, Never>?
    /// Pending debounced archive write; see `persistChatsDebounced()`.
    var chatPersistDebounceTask: Task<Void, Never>?
    /// Pending debounced settings write; see `persistSettingsDebounced()`.
    var settingsPersistDebounceTask: Task<Void, Never>?
    var decodeStartTime: Date?

    /// Bumped once per `executeGenerationTurn` call. A turn's own
    /// `runTask` compares its captured value against this at its tail
    /// before resetting `generating`/`phase`/`runTask`; if a NEWER turn has
    /// already started (the tool-call-continuation trampoline reenters
    /// `executeGenerationTurn` while the previous turn's tail is still
    /// pending), the stale tail is a no-op instead of clobbering the new
    /// turn's state out from under it (state#10).
    var generationEpoch: Int = 0

    /// Same shape as `generationEpoch`, for installs. `cancelInstall()`
    /// cancels the running `Task` cooperatively -- the task keeps running
    /// until its next suspension point notices -- so a user who cancels and
    /// immediately starts a NEW install can have the OLD task's delayed
    /// `CancellationError` tail run AFTER the new install's own `installTask`
    /// is already in flight. Without this guard that tail unconditionally
    /// reset `isInstallingModel`/`installTask` to nil, silently clobbering
    /// the new install's state and dropping the only reference that could
    /// cancel IT (state#15).
    var installEpoch: Int = 0

    /// Consecutive times a `Stop` hook has blocked and re-entered the
    /// current turn (`AppModel+Generation.swift`'s
    /// `dispatchStopAndContinueIfBlocked`). Reset when a turn starts and
    /// when a `Stop` dispatch is not blocked; capped at 8 (Claude Code's
    /// own cap) so a hook that always blocks cannot loop forever.
    var stopHookReentryCount: Int = 0

    /// Creates and initializes the application model, restoring saved settings and chats.
    public init() {
        loadSettings()
        loadProjects()
        loadChats()
        loadGlobalMcpServers()
        reloadSkills()
        reloadAgents()
        refreshModels()
        AppToolRegistry.activeSessionProvider = { [weak self] in
            self?.session
        }
        TodoWriteExecutor.onTodosUpdated = { [weak self] targetChatID, newTodos in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                let chatID = targetChatID ?? self.selectedChatID
                self.updateTodos(for: chatID, todos: newTodos)
            }
        }
        AppHookStore.shared.refresh(projectDirectory: selectedProject?.rootDirectoryPath)
        Task {
            _ = await self.dispatchLifecycleHook(event: .sessionStart, source: "startup")
        }
    }

    /// Combined list of all currently active (enabled) MCP servers from global settings and active project.
    public var activeMcpServers: [McpServerConfig] {
        var list = globalMcpServers.filter { $0.isEnabled }
        if let proj = selectedProject {
            list.append(contentsOf: proj.mcpServers.filter { $0.isEnabled })
        }
        return list
    }

    /// Currently selected project if any.
    public var selectedProject: AppProject? {
        guard let id = selectedProjectID else { return nil }
        return projects.first { $0.id == id }
    }

    /// Active agent profile for the current context.
    public var activeAgentType: AppAgentType {
        selectedProject?.agentType ?? .coder
    }

    /// Index of the currently selected chat in `chats`.
    public var selectedChatIndex: Int? {
        chats.firstIndex { $0.id == selectedChatID }
    }

    /// The currently selected chat conversation.
    ///
    /// Pure: no mutation of `@Published` state on read (state#7 -- "Publishing
    /// changes from within view updates" is undefined behavior, and this
    /// getter is read from SwiftUI view bodies). `selectedChatID` is kept
    /// valid at the points where `chats` actually changes -- `loadChats()`,
    /// `selectChat`, `createChat`, `deleteChat` -- rather than patched
    /// lazily here; `activeDraftChat`'s identity is kept in sync by
    /// `selectedChatID`'s own `didSet` above.
    public var selectedChat: AppChat {
        if let index = selectedChatIndex {
            return chats[index]
        }
        return activeDraftChat
    }

    /// Active task checklist for the currently selected chat.
    public var currentTodos: [TodoItem] {
        selectedChat.todos
    }

    /// Present continuous description of the active `in_progress` task, if any.
    public var activeTaskDescription: String? {
        if let inProgress = selectedChat.todos.first(where: { $0.isInProgress }) {
            return inProgress.activeForm.isEmpty ? inProgress.content : inProgress.activeForm
        }
        return nil
    }

    /// Updates the checklist items for a given chat and persists the change.
    public func updateTodos(for chatID: UUID, todos: [TodoItem]) {
        if let index = chats.firstIndex(where: { $0.id == chatID }) {
            chats[index].todos = todos
            chats[index].updatedAt = Date()
            persistChats()
        } else if activeDraftChat.id == chatID {
            activeDraftChat.todos = todos
            activeDraftChat.updatedAt = Date()
        }
    }

    /// Draft prompt text for the currently selected chat.
    public var promptText: String {
        get { selectedChat.draft }
        set {
            if let index = selectedChatIndex {
                chats[index].draft = newValue
                chats[index].updatedAt = Date()
                // Debounced: this is one keystroke, and the store rewrites
                // every chat in the archive on each call.
                persistChatsDebounced()
            } else {
                activeDraftChat.draft = newValue
                activeDraftChat.updatedAt = Date()
            }
            updateTokenEstimate()
        }
    }

    /// Document attachments attached to the current prompt draft.
    public var promptAttachments: [AppPromptAttachment] {
        selectedChat.draftAttachments
    }

    /// Metadata and capabilities for the currently loaded model session.
    public var info: SessionInfo? { session?.info }

    /// Whether generation is currently running.
    public var isRunning: Bool { generating }

    /// Whether at least one model is installed locally.
    public var isModelInstalled: Bool { !installed.isEmpty }

    /// Whether no models are installed and initial installation is required.
    public var requiresModelInstallation: Bool { installed.isEmpty }

    /// Whether a model session is currently loaded into memory.
    public var isModelAvailable: Bool { session != nil }

    /// Whether conditions allow starting a new generation run.
    public var canRun: Bool {
        !generating && !submitting && !opening && session != nil && (!promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !promptAttachments.isEmpty)
    }

    /// Whether active generation can be cancelled.
    public var canCancel: Bool { generating && !isCancellationPending }

    /// Whether an active model download can be cancelled.
    public var canCancelInstall: Bool { isInstallingModel }

    /// Whether the selected model can be loaded.
    public var canLoadModel: Bool { !generating && !opening && session == nil && selected != nil }

    /// Whether the active model session can be reloaded.
    public var canReloadModel: Bool { !generating && !opening && session != nil }

    /// Whether the active model session can be unloaded.
    public var canUnloadModel: Bool { !generating && !opening && session != nil }

    // `reasoningAvailable` was here and is gone: it had no callers, and once
    // `isReasoningSupported` stopped guessing from the family the two bodies
    // were byte-identical. Two names for one fact is a drift hazard, and the
    // one that drifts is the one nothing exercises.

    /// Whether the LOADED checkpoint's template can express a reasoning
    /// level at all.
    ///
    /// **There is deliberately no answer for an unloaded model.** This used
    /// to fall back to a hardcoded `reasoningFamilies` set when `info` was
    /// nil, which is the per-family table root Gotcha 56 exists to refuse:
    /// it guessed for a checkpoint whose template nobody had read, and every
    /// family it named has releases on both sides of the question. The
    /// levels arrive with the session, so the control waits for the session
    /// (see `reasoningPickerEnabled`).
    public var isReasoningSupported: Bool {
        info?.reasoningSupport != SessionInfo.ReasoningSupport.none
    }

    /// Whether the reasoning picker should accept input.
    ///
    /// One accessor rather than three view-local spellings, the same reason
    /// `activeLoadGuard` and `visionIsActive` are one each: a view assembling
    /// its own would drift, and the one that drifted would offer a level the
    /// turn then refuses.
    public var reasoningPickerEnabled: Bool {
        session != nil && isReasoningSupported
    }

    /// The reasoning levels worth offering for the loaded checkpoint.
    ///
    /// **THE SET COMES OFF THE CHECKPOINT'S OWN TEMPLATE**, probed at open
    /// and reported as `SessionInfo.reasoningEfforts`. It is not derivable
    /// from the family: Qwen 3.8 answers `[.off, .low, .medium, .xhigh]` and
    /// RAISES on `.high`, where gpt-oss and Muse Glimmer answer
    /// `[.off, .low, .medium, .high]`. This returned `allCases` for every
    /// `.level` checkpoint until 2026-08-31, so the menu carried an entry
    /// that failed the turn with a template error.
    ///
    /// `.toggleOnly` is the one case still decided here rather than by the
    /// engine. Its on-levels all render the same prompt, so the engine
    /// collapses them to one and reports `.low` BY POSITION; `.medium` is
    /// the friendlier middle-of-the-road spelling for a control that is
    /// really an on/off switch, and the views label it accordingly.
    /// The decision itself lives in `ReasoningLevelPolicy`, which is pure and
    /// therefore testable without a model, a Metal device and a 13 GB
    /// install. This is the binding of arguments to it and nothing else.
    public var availableReasoningLevels: [GenerateOptions.Reasoning] {
        ReasoningLevelPolicy.offered(
            support: info?.reasoningSupport,
            efforts: info?.reasoningEfforts ?? []
        )
    }

    /// What to call a level in this checkpoint's picker. See
    /// `ReasoningLevelPolicy.label(for:support:)`.
    public func reasoningLabel(for level: GenerateOptions.Reasoning) -> String {
        ReasoningLevelPolicy.label(for: level, support: info?.reasoningSupport)
    }

    /// The one-line description under a level in this checkpoint's picker.
    public func reasoningDescription(for level: GenerateOptions.Reasoning) -> String {
        ReasoningLevelPolicy.description(for: level, support: info?.reasoningSupport)
    }

    /// The offered level closest to `wanted`, used when a preference restored
    /// from another model does not carry over. See
    /// `ReasoningLevelPolicy.nearest(to:in:)`.
    public func nearestAvailableReasoning(
        to wanted: GenerateOptions.Reasoning
    ) -> GenerateOptions.Reasoning {
        ReasoningLevelPolicy.nearest(to: wanted, in: availableReasoningLevels)
    }

    /// Updates the current reasoning effort level and saves it as the preferred default for the active model.
    public func setReasoning(_ level: GenerateOptions.Reasoning) {
        self.reasoning = level
        if let key = selected?.alias ?? (selected?.path.isEmpty == false ? selected?.path : nil) {
            modelReasoningDefaults[key] = level.rawValue
        }
        persistSettings()
        updateTokenEstimate()
    }

    /// Switches the primary interaction mode and saves it as the default for future launches.
    public func setInteractionMode(_ mode: AppInteractionMode) {
        self.interactionMode = mode
        persistSettings()
    }

    /// Whether the active model supports tool calling and structured function invocation.
    public var isToolCallingSupported: Bool {
        guard let selectedModel = selected ?? installed.first else {
            return true
        }
        let family = selectedModel.family.lowercased()
        let dialect = info?.dialect.lowercased() ?? ""
        let toolFamilies: Set<String> = ["gemma4", "qwen36", "qwen3moe", "qwen35", "gptoss", "llama"]
        if toolFamilies.contains(family) {
            return true
        }
        if dialect.contains("chatml") || dialect.contains("harmony") || dialect.contains("llama") || dialect.contains("gemma") || dialect.contains("mistral") {
            return true
        }
        return false
    }

    /// Effective resolution of whether Forge Guardrails is active for the current context.
    public var effectiveForgeGuardrailsEnabled: Bool {
        switch guardrailsMode {
        case .alwaysOn:
            return true
        case .alwaysOff:
            return false
        case .select:
            if let projectOverride = selectedProject?.forgeGuardrailsEnabled {
                return projectOverride
            }
            if let composerOverride = composerGuardrailsOverride {
                return composerOverride
            }
            return isToolCallingSupported
        }
    }

    /// Toggles or sets the Forge Guardrails active state for the current project or draft.
    public func setForgeGuardrailsEnabled(_ enabled: Bool) {
        if var proj = selectedProject {
            proj.forgeGuardrailsEnabled = enabled
            proj.updatedAt = Date()
            updateProject(proj)
        } else {
            composerGuardrailsOverride = enabled
        }
    }

    /// Opens the application Settings window and navigates to the requested tab.
    public func openSettings(tab: AppSettingsView.SettingsTab = .engine) {
        NotificationCenter.default.post(name: .openSettingsTab, object: tab)
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
    }

    /// Displays a toast notification and posts an accessible VoiceOver announcement.
    public func showToast(_ message: String, style: AppToast.Style = .info, duration: TimeInterval = 3.0) {
        activeToast = AppToast(message: message, style: style, duration: duration)
        _ = AccessibilityNotification.Announcement.post(.init(message))
    }

    /// Dismisses the active toast notification immediately.
    public func dismissToast() {
        activeToast = nil
    }
}
