import AppKit
import Foundation
import SwiftUI
import TurboSpark

/// Primary application model and state coordinator for the TurboSpark macOS
/// application.
///
/// **THIS FILE IS PUBLISHED STATE AND THE LIFECYCLE THAT OWNS IT, AND
/// NOTHING ELSE.** That is the package's own convention (`swift/CLAUDE.md`),
/// and the base file had drifted to 863 lines by growing accessors rather
/// than extensions. The nested vocabulary is in `AppModel+Types`, the
/// `can*` predicates in `AppModel+Gating`, the reasoning-level accessors in
/// `AppModel+Reasoning`, the guardrails resolution in `AppModel+Guardrails`,
/// and the selected-chat accessors moved into `AppModel+Chat` beside the
/// chat operations that call them. What stays here is what an extension
/// cannot hold anyway: stored properties.
@MainActor
public final class AppModel: ObservableObject {
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
    /// A stop pressed while a start was still binding (state#28).
    ///
    /// `server` is published only after the awaited bind, so `stopServer()`
    /// found nil and returned -- leaving a server listening that the UI
    /// showed as stopped. The start path checks this before publishing.
    var serverStopRequested = false
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
    /// Whether the last `info()` failure has already been reported (state#86).
    ///
    /// The poll runs at 2 Hz, so a server that has genuinely gone away would
    /// otherwise raise the same banner twice a second for as long as the pane
    /// is open.
    var serverInfoErrorReported = false
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
    /// proposal time (state#19). `AppToolPermissionEngine` computed its `.allow` from
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
                // **STAMPED WITH THE PROJECT** (state#79). `run()`'s lazy
                // create passes `selectedProjectID` and this did not, so
                // after deleting the last chat under a project, typing built
                // a chat the sidebar filters OUT -- and one whose turns get
                // no system prompt, no tools and no workspace root, because
                // every one of those is resolved from the chat's project.
                activeDraftChat = AppChat(id: selectedChatID, projectID: selectedProjectID)
            }
        }
    }
    /// Transient active chat draft when the chats list is empty.
    ///
    /// **NEITHER PUBLISHED NOR PERSISTED** until state#63. Every write to it
    /// went to a plain stored property, so no view re-rendered on it and the
    /// quit flush -- which walks `chats` -- never saw it: draft text and a
    /// whole checklist written before any chat existed were gone on relaunch.
    /// `materializeDraftChatIfNeeded()` promotes it into `chats` as soon as
    /// it carries anything worth losing, after which every existing path
    /// (publication, the debounce, the flush) applies to it unchanged.
    var activeDraftChat = AppChat()

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
    /// Named steering directions the operator registered. Empty until one is
    /// added: nothing ships a direction.
    @Published public var steeringPresets: [AppSteeringPreset] = []
    /// `id` of the selected preset, or nil for the Inspector's raw knobs.
    @Published public var activeSteeringPresetID: UUID? = nil
    /// Whether steering is applied at the next model open. **Not derived from
    /// "a path is set"**: a path is configuration, running the edit is a
    /// decision, and separating them is what makes the A/B one click.
    @Published public var steeringEnabled: Bool = false
    /// Forge Tool-Call Guardrails global mode ("alwaysOn", "alwaysOff", "select").
    @Published public var guardrailsMode: AppGuardrailsMode = .select
    /// What the RUNNING server was started with, or nil when none is.
    ///
    /// Not persisted and not derived from `guardrailsMode`: a server resolves
    /// its guardrails once at start and keeps them, so changing the setting
    /// afterwards does not move what is already serving. Reporting the
    /// setting instead of the start value would tell an operator the server
    /// changed when it did not.
    @Published public var serverStartedGuardrails: ServerOptions.Guardrails? = nil
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
    /// `run()`'s pre-append work: the awaited `UserPromptSubmit` hook and the
    /// agent slash-command dispatch (state#33).
    ///
    /// Stored so `cancel()` can reach it. The hook is awaited with a 120 s
    /// budget, and while it runs `submitting` is true -- so Send is refused
    /// by `canRun` and Stop was refused by `canCancel`, leaving a wedged hook
    /// with no exit at all and nothing for `cancel()` to cancel.
    var submissionTask: Task<Void, Never>?
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
        TaskManager.shared.onTasksUpdated = { [weak self] targetChatID, tasks in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                // Trigger view update
                self.objectWillChange.send()
            }
        }
        SendUserFileExecutor.onFileSent = { [weak self] targetChatID, fileURL, note in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                NSWorkspace.shared.activateFileViewerSelecting([fileURL])
                self.activeToast = AppToast(message: "Presented file: \(fileURL.lastPathComponent)", style: .info)
            }
        }
        PushNotificationExecutor.onNotificationPushed = { [weak self] title, message in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                self.activeToast = AppToast(message: "\(title): \(message)", style: .info)
            }
        }
        AppHookStore.shared.refresh(projectDirectory: selectedProject?.rootDirectoryPath)
        // A quarantined settings, chat or project file is the one thing the
        // user must be told about at launch: the app comes up looking EMPTY,
        // which reads as lost data rather than as a file set aside (state#42).
        surfaceStorageIssues()
        Task {
            _ = await self.dispatchLifecycleHook(
                event: .sessionStart, chatID: self.selectedChatID, project: self.selectedProject,
                source: "startup")
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

    /// Metadata and capabilities for the currently loaded model session.
    public var info: SessionInfo? { session?.info }

    /// Switches the primary interaction mode and saves it as the default for future launches.
    public func setInteractionMode(_ mode: AppInteractionMode) {
        self.interactionMode = mode
        persistSettings()
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
