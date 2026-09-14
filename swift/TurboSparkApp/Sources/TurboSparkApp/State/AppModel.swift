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

    /// Artifact currently shown in the right-hand artifact panel, if any.
    ///
    /// Same one-flag rule as `previewAttachmentID`. The three right-column
    /// claimants clear each other in their setters
    /// (`AppModel+ArtifactPanel`), never by ordering in a view.
    @Published public var openArtifactID: UUID? = nil

    /// Inline HTML preview from a chat fence's "Preview" click.
    ///
    /// Bytes in MEMORY and never persisted: the artifact archive stores paths
    /// and metadata rather than bytes (`AppArtifact`), and a fence is not a
    /// file on disk at all. Ghost chats keep their ghosting for free -- there
    /// is nothing on disk to seal or leak. Mutate only through
    /// `openHTMLPreview`/`dismissHTMLPreview` so the claimant invariant
    /// holds.
    @Published public var htmlPreview: ArtifactHTMLPreview? = nil

    /// Network grants for the sandboxed web panel, by grant key.
    ///
    /// In-memory on purpose: a grant survives a panel close within the
    /// session but a relaunch asks again, which is the safe direction for a
    /// switch whose whole job is to be deliberate. Keys are namespaced by
    /// source (`artifact:` + contentKey, `html:` + content hash) so a rewritten
    /// file or an edited fence never inherits the old grant.
    ///
    /// `@Published` is load-bearing: the panel banner and webview read this
    /// through the observed model, so a grant written without a publish
    /// would leave the banner up and the page offline until an unrelated
    /// property happened to change.
    @Published var artifactNetworkGrants: [String: Bool] = [:]

    /// Artifacts whose panel has already auto-opened.
    ///
    /// `AppArtifact.upsert` keeps ids stable across rewrites precisely so
    /// this set stays honest: a file rewritten three times pops the panel
    /// once, not once per write.
    var autoOpenedArtifactIDs: Set<UUID> = []

    /// New-chat suggestion dismissals, per chat, in memory.
    ///
    /// On the model rather than the banner's own `@State` because the
    /// banner sits in the transcript's LazyVStack: row state is discarded
    /// when it scrolls away, and a dismissal that un-dismisses on the way
    /// back down reads as a banner the user cannot get rid of.
    @Published var dismissedNewChatSuggestionChatIDs: Set<UUID> = []

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
    @Published public var serverHost = "127.0.0.1"
    @Published public var serverCaptureText = false
    @Published public var serverPortIsValid = true
    @Published public var serverFavorites: [ServerFavorite] = []
    @Published public var serverLive = ServerLiveHistory()
    @Published public var serverPinnedPort: UInt16 = 0
    /// Optional embedding model (.safetensors directory or alias) attached to the server.
    @Published public var serverEmbeddingModelInput: String = ""
    /// Hugging Face mirror endpoint override ($HF_ENDPOINT), e.g. https://hf-mirror.com.
    @Published public var hfEndpointInput: String = ""
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
    /// Whether to display the menu bar status extra icon in macOS menu bar.
    @Published public var showMenuBarItem: Bool = true
    /// Whether fans pinned through the status bar's ThermalForge control
    /// stay pinned when the app quits. Mirrored onto
    /// `FanController.keepFansPinnedOnQuit` at load, because the quit path
    /// cannot consult the debounced settings write.
    @Published public var keepFansPinnedOnQuit: Bool = false
    /// Whether the server automatically starts when the app launches.
    @Published public var serverAutoStartOnLaunch: Bool = false
    /// Whether the server keeps running in the background when the main window is closed.
    @Published public var keepServerRunningInBackground: Bool = true

    /// Current UI interaction mode (Chat vs Projects / Coding).
    @Published public var interactionMode: AppInteractionMode = .chat
    /// Whether the app opens a temporary (Ghost Mode) chat on launch.
    @Published public var alwaysStartInGhostMode: Bool = false
    /// Whether older turns are summarized automatically as the prompt
    /// approaches the context window (context compaction).
    @Published public var autoCompactEnabled: Bool = true
    /// Native action fusion, observation packing, verified reduction, and
    /// completed-todo compaction are independent reversible defaults.
    @Published public var actionFusionEnabled: Bool = true
    @Published public var observationPackEnabled: Bool = true
    @Published public var evidenceReducerEnabled: Bool = true
    @Published public var todoBoundaryCompactionEnabled: Bool = true
    /// Whether the model sees the auto-memory section and the `memory` tool
    /// (swift/docs/SWIFT_MEMORY.md). The `didSet` mirrors the value into
    /// `MemoryStore.shared`, the static surface `AppToolCatalog` and
    /// `SubagentRunner` read -- neither holds an `AppModel`.
    @Published public var memoryEnabled: Bool = false {
        didSet { MemoryStore.shared.isModelEnabled = memoryEnabled }
    }
    /// Profile-local encoder used for semantic memory recall.
    @Published public var memoryEmbeddingModel: String = ""
    /// Whether Syntext code search and project indexing is enabled globally.
    /// Mirrored to `AppToolRegistry.syntextIndexingEnabled` so tool execution and background
    /// indexers can check it without holding an `AppModel`.
    @Published public var syntextIndexingEnabled: Bool = true {
        didSet { AppToolRegistry.syntextIndexingEnabled = syntextIndexingEnabled }
    }
    /// Trailing message rows that stay verbatim after a compaction.
    @Published public var compactionKeepRecentTurns: Int = 2
    /// Whether a compaction summarizer is running right now, auto or manual.
    /// What the status bar's "Compacting conversation" state keys on; set and
    /// cleared by `performCompaction` around the generate loop.
    @Published public var isCompacting: Bool = false

    // Project and Agent State
    /// All configured codebase projects.
    @Published public var projects: [AppProject] = []
    /// Currently active project filter (nil = all chats).
    @Published public var selectedProjectID: UUID? = nil
    /// Working tree model for git status and diffs when a project is selected.
    @Published public var worktree: WorktreeModel? = nil
    /// Global application-level MCP server configurations.
    @Published public var globalMcpServers: [McpServerConfig] = []
    /// MCP servers detected in the selected project's config files that are
    /// awaiting an approve/reject decision (`AppModel+Mcp`). Drives the
    /// project approval sheet.
    @Published public var pendingMcpApprovals: [PendingMcpServerApproval] = []
    /// Global / rootless chat permission mode (defaults to .auto / Approve for me).
    @Published public var activePermissionMode: AppPermissionMode = .auto
    /// Web search tool toggle state in the chat bar.
    @Published public var webSearchEnabled: Bool = false

    /// Effective permission mode for current chat or project.
    public var effectivePermissionMode: AppPermissionMode {
        selectedProject?.permissions.mode ?? activePermissionMode
    }

    /// Updates permission mode for current project or global chat.
    public func setEffectivePermissionMode(_ mode: AppPermissionMode) {
        // Entering Agent mode starts its fallback counters fresh: a streak
        // recorded under an earlier selection is not this selection's
        // history, and `AgentModeGate.reset` also lifts a session
        // suspension, which re-selecting the mode is the documented way to
        // end (`swift/docs/SWIFT_AGENT_MODE.md`).
        if mode == .agentAuto {
            let sessionID = selectedChatID.uuidString
            Task { await AgentModeGate.shared.reset(sessionID: sessionID) }
        }
        if let project = selectedProject {
            var updated = project
            updated.permissions = AppProjectPermissions.preset(for: mode)
            updated.permissions.mode = mode
            updateProject(updated)
        } else {
            activePermissionMode = mode
        }
    }

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
    /// The full agent batch parked behind `pendingToolCall` (nil for a solo
    /// call). A parallel subagent launch under an asking project shows ONE
    /// card; approving runs every call of the batch concurrently, denying
    /// refuses all. `pendingToolCall` remains the call whose verdict the
    /// card renders.
    var pendingBatchCalls: [AppToolCall]? = nil
    /// Synthetic terminal leg already checked alongside the pending mutation.
    var pendingValidationCall: AppToolCall? = nil
    /// Why Agent mode (`swift/docs/SWIFT_AGENT_MODE.md`) punted this call to
    /// the card: classifier unavailable, or the skip thresholds tripped. Nil
    /// for an ordinary ask, so the card renders exactly as before.
    @Published public var pendingToolCallClassifierNotice: String? = nil
    /// Natural-language classifier steering (allow / softDeny / hardDeny /
    /// environment), mirrored from settings like every other preference.
    /// Read at classification time by the router; edited in the Permissions
    /// settings pane.
    @Published public var agentModeHints: AgentModeHints = AgentModeHints()
    /// Test seam for `resolveAskUnderAgentMode`: when set, it classifies
    /// instead of `LocalModelToolClassifier`, so verdict routing is
    /// testable with no model loaded. Not published; nothing draws it.
    var agentModeClassifierOverride: ToolCallClassifying? = nil
    /// Why the last SKILL.state patch was rejected, or nil if the last one
    /// merged. Surfaced rather than swallowed: a dropped patch means the run
    /// lost a step's bookkeeping, which is invisible in the transcript.
    @Published public var skillStateLastError: String? = nil
    /// The questions an `AskUserQuestion` tool call is parked on (the
    /// qwen-code inline question card). Non-nil while the model's turn is
    /// suspended inside the tool waiting for the user's pick; the
    /// transcript card renders tappable options while it is set.
    @Published public var pendingUserQuestions: PendingUserQuestions? = nil
    /// Whether the git info sheet (`/diff`, `/log`, `/prs`) is showing.
    @Published public var showGitSheet: Bool = false
    @Published public var gitInfoTab: GitInfoTab = .diff
    /// The full unified diff behind `/diff`, fetched once per presentation.
    @Published public var gitDiffText: String = ""
    /// The commits behind `/log`, newest first.
    @Published public var gitCommits: [WorktreeCommit] = []
    /// The pull requests behind `/prs`, newest first.
    @Published public var gitPullRequests: [GitPullRequestRow] = []
    @Published public var isLoadingGitInfo: Bool = false
    /// A load error, shown inside the sheet above the stale rows.
    @Published public var gitInfoError: String? = nil
    /// The chats shown as secondary split panes (qwen-code split-view
    /// parity), in pane order. Restored from defaults so the pane set
    /// survives a relaunch, the way the web shell's `?split=` URL does.
    @Published public var splitChatIDs: [UUID] = AppModel.loadSplitPaneIDs()

    // Skills State
    /// User-scoped skills (~/.turbospark/skills and user agent directories).
    @Published public var userSkills: [AppSkill] = []
    /// Project-scoped skills for the currently selected project.
    @Published public var projectSkills: [AppSkill] = []

    // Plugins State
    /// Every plugin discovered across both roots, precedence-ordered,
    /// including disabled ones. Refreshed by `reloadPlugins()`.
    @Published public var installedPlugins: [LoadedPlugin] = []
    /// One line per plugin that failed to load, plus shadowed-name notes.
    @Published public var pluginLoadDiagnostics: [String] = []
    /// User-scope plugin enable state, keyed `<plugin>@<origin>`. Persisted
    /// inside `settings.json` via `persistSettings`, NOT by `PluginManager`
    /// -- a direct write there would be clobbered by this model's own
    /// debounced settings save.
    @Published public var pluginEnableState: [String: Bool] = [:]

    // Profiles State
    /// The ADDITIONAL users this installation knows about. The Default user
    /// is implicit and never in this list; `currentProfile` resolves it.
    /// Which user a run belongs to was fixed before any store opened
    /// (`UserProfileStore.active`); this list is display and management.
    @Published public var profiles: [UserProfile] = []

    // Agents State
    /// All discovered agents (built-in, user, project).
    @Published public var discoveredAgents: [AppAgentDefinition] = []

    // Subagent State
    /// Live FOREGROUND subagent runs, keyed by the tool-call UUID string
    /// that started them. A run is removed when its `finished` event lands,
    /// which is within moments of the transcript's own tool card appearing
    /// -- the card is the durable record, this dict is the live view.
    @Published public var liveSubagentRuns: [String: SubagentRunState] = [:]
    /// Background subagent runs, keyed `bga_N`. Unlike the foreground dict,
    /// finished runs STAY here until pruned: nothing else in the transcript
    /// represents a background run, so its card (with its result) is the
    /// only record the chat has.
    @Published public var backgroundAgentRuns: [String: SubagentRunState] = [:]
    /// User-role `<task-notification>` turns a background agent finished
    /// while its chat was busy, parked per chat until a turn tail can inject
    /// them (Claude Code's pending-notification queue).
    var pendingTaskNotifications: [UUID: [String]] = [:]
    /// Prompts submitted while their chat was busy, parked per chat and
    /// sent from a turn tail (Claude Code's message queue). Published for
    /// the composer's queued-count pill; in-memory only, never persisted.
    @Published public var pendingUserMessages: [UUID: [QueuedUserPrompt]] = [:]
    /// The spawned `Task` per background agent id. Deliberately NOT a child
    /// of `runTask`: an unstructured `Task` inherits no cancellation, which
    /// is the whole point -- chat Stop must not kill an agent the model was
    /// told runs independently.
    var backgroundAgentTasks: [String: Task<SubagentRunResult, Never>] = [:]
    /// Ids the user or the model asked to stop, so their completion reads
    /// `killed` rather than `cancelled`.
    var killedBackgroundAgentIDs: Set<String> = []
    var nextBackgroundAgentID = 1

    /// Running background shells, for the kill strip. Value-type snapshots
    /// rebuilt off the registry's change hook; the strip filters by chat.
    @Published public var backgroundShellSummaries: [BackgroundShellSummary] = []

    // Goal state (swift/docs/SWIFT_GOALS.md). The row (or ghost payload)
    // is the persisted source; this mirror is what SwiftUI and the stop
    // seam read, kept in step by `AppModel+Goal.updateGoal`.
    /// Each chat's active `/goal`, keyed by chat id. Published for the
    /// transcript banner.
    @Published public var activeGoals: [UUID: ChatGoalState] = [:]
    /// The idle check-in timer per chat, cancelled on clear, teardown and
    /// re-arm. The pipeline's first recurring timer: it exists so a
    /// deferral that goes QUIET (nothing left to start a turn) still
    /// surfaces its check-in.
    var goalIdleTimerTasks: [UUID: Task<Void, Never>] = [:]
    /// Transcript message count at the goal's set point / last evaluation:
    /// what the evaluator's slice and the stall detector read since.
    var goalEvalMessageCounts: [UUID: Int] = [:]
    /// Consecutive tool-free evaluation rounds, per chat (the stall
    /// counter).
    var goalToolFreeEvals: [UUID: Int] = [:]
    /// Whether the goal evaluator's side query is running right now.
    @Published public var isEvaluatingGoal = false

    // Message editing state
    /// The transcript row currently open in the in-place edit composer, or
    /// nil. Pure UI state: set by `beginEdit`, cleared on commit and cancel,
    /// never persisted.
    @Published public var editingMessageID: UUID? = nil
    /// Response variants parked by Retry and Edit, keyed by chat. The
    /// replaced prose response waits here and is seeded as the NEXT
    /// assistant message's `alternates` the moment one commits
    /// (`finishProseTurn`), or onto a cancelled turn's partial append
    /// (`finishCancelled`). Both consumers REMOVE their entry; a turn that
    /// came back as tool calls drops it (variants describe a prose reply,
    /// and seeding one onto a chain of tool rows would swap content under
    /// live tool cards); an ordinary submission clears it up front. Without
    /// those three exits a stale entry would graft old text onto a turn it
    /// was never written for.
    var pendingResponseVariants: [UUID: [AppChatMessage]] = [:]

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
    /// AES-GCM vault holding the conversation contents of every ghost chat.
    ///
    /// A ghost row in `chats` keeps its messages, todos, draft, context
    /// summary and skill state EMPTY; they are sealed in here under a
    /// per-launch key. See `GhostChatVault` and `AppModel+Ghost.swift`.
    var ghostVault = GhostChatVault()
    /// App-wide prompt history behind the composer's Up/Down recall
    /// (`InputHistoryStore`). Loads its own file at init, records at
    /// submission, and is shared across chats: recall is a habit of the
    /// keyboard, not of the conversation.
    let promptHistory = InputHistoryStore()

    // Live Generation State
    /// Whether token generation is currently running.
    @Published public var generating: Bool = false {
        didSet {
            if !generating && oldValue {
                onGenerationFinished()
            }
        }
    }
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
    /// Per-component breakdown of what the context window holds. Nil until
    /// the first estimate after a session loads; refreshed by the same
    /// debounced task that writes `estimatedPromptTokens`.
    @Published public var contextUsageSummary: ContextUsageSummary?
    /// System thermal and memory telemetry readings.
    @Published public var telemetry: SystemTelemetry?
    /// Active error banner message, if any.
    @Published public var error: String?
    /// Active toast notification displayed on screen and announced to assistive technologies.
    @Published public var activeToast: AppToast? = nil

    // Local command surfaces (qwen-code /stats and /help parity). Pure UI
    // state; sheets on the chat pane present while these are true.
    /// Whether the session stats sheet (`/stats`) is showing.
    @Published public var showSessionStats: Bool = false
    /// Whether the help sheet (`/help`: commands and shortcuts) is showing.
    @Published public var showHelpSheet: Bool = false
    /// Whether the context usage sheet (`/context`) is showing.
    @Published public var showContextSheet: Bool = false
    /// Whether the background tasks sheet (`/tasks`) is showing.
    @Published public var showTasksSheet: Bool = false
    /// Whether the tool catalog sheet (`/tools`) is showing.
    @Published public var showToolsSheet: Bool = false
    /// Whether the status sheet (`/status`) is showing.
    @Published public var showStatusSheet: Bool = false
    /// Whether the rewind sheet (`/rewind`) is showing.
    @Published public var showRewindSheet: Bool = false
    /// Whether the recap sheet (`/recap`) is showing, with the recap text
    /// beside it. The text is transient: it dies with the sheet, unlike a
    /// compaction summary which replaces transcript in the prompt.
    @Published public var showRecapSheet: Bool = false
    @Published public var recapText: String = ""
    @Published public var isRecapping: Bool = false
    /// Arms the `/delete` confirmation alert: the command refuses to delete
    /// without one, exactly like the sidebar's own Delete action.
    @Published public var confirmDeleteChat: Bool = false
    /// Transcript collapse (qwen-code turn folding): anchors of the turns
    /// currently folded to prompt + final answer. Session-only state, like
    /// turn navigation; there is no persisted "was collapsed" to restore.
    @Published public var collapsedTurnAnchors: Set<UUID> = []
    /// Two-stage Esc cancel (qwen-code `escapeIntent` parity): the first
    /// Escape while generating ARMS the cancel and the footer says so; the
    /// second within the window stops the turn. Auto-disarms.
    @Published public var isEscCancelArmed: Bool = false
    var escCancelArmTask: Task<Void, Never>? = nil

    // Turn navigation (qwen-code turn-jump parity). `turnNavigationToken`
    // is what the transcript's ScrollViewReader observes; the target it
    // points at travels beside it. Both die with the view: no persistence.
    /// Message id the transcript should scroll to, per `jumpTurn`.
    @Published public var turnNavigationTargetID: UUID? = nil
    /// Bumped on every jump so repeating a jump to the SAME row still
    /// scrolls (an onChange keyed on the id alone would not).
    @Published public var turnNavigationToken: Int = 0
    /// The cron poll timer, installed by `startCronScheduler`; held so
    /// shutdown can invalidate it.
    var cronPollTimer: Timer?


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
    /// Named sampling snapshots saved from the Inspector's Generation
    /// Sampling section, applied into whichever scope is being edited. See
    /// `AppSamplingSettings`.
    @Published public var samplingPresets: [AppSamplingPreset] = []
    /// Compatibility mirror of the selected reusable system prompt. New UI
    /// writes go through `systemPrompts`; this remains the direct source used
    /// by legacy callers and per-chat fallback tests.
    @Published public var defaultSystemPrompt: String = AppSystemPrompt.builtIns[0].instructions

    /// Reusable app-wide prompts. The first built-in is selected by default,
    /// while an empty selection explicitly sends no app-wide default.
    @Published public var systemPrompts: [AppSystemPrompt] = AppSystemPrompt.builtIns

    /// `nil` means no app-wide system prompt is selected.
    @Published public var selectedSystemPromptID: UUID? = AppSystemPrompt.builtIns[0].id

    /// App-wide response styles, including the compact starter library.
    @Published public var personalities: [AppPersonality] = AppPersonality.builtIns

    /// `nil` means no personality is added to a turn's system prompt.
    @Published public var selectedPersonalityID: UUID? = nil
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
    /// Refuses a re-install of `alias` while a cancelled walk may still be
    /// writing its directory. `cancelInstall()` inserts the alias and
    /// `watchCancelledWalkExit` lifts the refusal once `installsFinished()`
    /// shows the walk exited; if the walk never notices the flag, the entry
    /// stays until app restart rather than racing a live writer.
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
        loadProfiles()
        loadProjects()
        loadChats()
        installBackgroundShellObserver()
        // Ghost Mode opt-in, AFTER `loadChats`: the restored archive is
        // already in place and the temporary chat sits on top of it, selected.
        if alwaysStartInGhostMode {
            enterGhostChat()
        }
        loadGlobalMcpServers()
        reloadSkills()
        reloadAgents()
        reloadPlugins()
        refreshModels()
        startCronScheduler()
        AppToolRegistry.activeSessionProvider = { [weak self] in
            self?.session
        }
        AppToolRegistry.userSystemPromptProvider = { [weak self] in
            self?.appWideSystemPrompt ?? ""
        }
        AppToolRegistry.subagentSamplingOptionsProvider = { [weak self] in
            self?.samplingOptions() ?? GenerateOptions()
        }
        AppToolRegistry.webToolsEnabledProvider = { [weak self] in
            self?.webSearchEnabled ?? true
        }
        AppToolRegistry.subagentProgressSink = { [weak self] key, event in
            await self?.applySubagentEvent(key, event)
        }
        AppToolRegistry.backgroundAgentLauncher = { [weak self] launch in
            guard let self else {
                throw NSError(domain: "TurboSparkTool", code: 26, userInfo: [
                    NSLocalizedDescriptionKey: "Background subagents are unavailable: the app model is gone."
                ])
            }
            return try await self.launchBackgroundAgent(launch)
        }
        AppToolRegistry.backgroundAgentStopper = { [weak self] id in
            guard let self else {
                throw NSError(domain: "TurboSparkTool", code: 26, userInfo: [
                    NSLocalizedDescriptionKey: "Background subagents are unavailable: the app model is gone."
                ])
            }
            return try await self.stopBackgroundAgent(id)
        }
        AppToolRegistry.observationRecaller = { [weak self] chatID, id, offset, maximum in
            guard let self else {
                throw ToolObservationStore.ObservationError.invalidRange
            }
            return try await self.recallToolOutput(
                chatID: chatID, observationID: id, offsetBytes: offset, maxBytes: maximum)
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
        ArtifactRegistrar.onArtifactsProduced = { [weak self] targetChatID, files in
            Task { @MainActor [weak self] in
                self?.registerProducedArtifacts(chatID: targetChatID, files: files)
            }
        }
        PushNotificationExecutor.onNotificationPushed = { [weak self] title, message in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                self.activeToast = AppToast(message: "\(title): \(message)", style: .info)
            }
        }
        // The interactive AskUserQuestion surface (qwen-code parity): the
        // registry routes question calls through this waiter, the transcript
        // card answers them, and the tool result carries the reply back.
        AskUserQuestionExecutor.answerWaiter = { [weak self] chatID, items, toolCallID in
            guard let self else {
                return "The user interface is not available to answer questions."
            }
            return await self.waitForUserAnswers(
                chatID: chatID, items: items, toolCallID: toolCallID)
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

    deinit {
        // Releasing our reference alone leaves the run loop polling shared state.
        cronPollTimer?.invalidate()
    }

    /// Combined list of all currently active (enabled) MCP servers from global settings and active project.
    public var activeMcpServers: [McpServerConfig] {
        AppToolCatalogMcp.visibleServers(global: globalMcpServers, project: selectedProject)
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

    /// Brings the application to the foreground and presents the main window,
    /// optionally navigating to a specific application section.
    public func showMainWindow(navigatingTo section: AppNavigationSection? = nil) {
        if let section {
            activeSection = section
        }
        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.windows.first(where: { $0.canBecomeMain }) {
            window.makeKeyAndOrderFront(nil)
        }
    }

    /// Background synchronization hook invoked when a generation turn completes.
    private func onGenerationFinished() {
        if syntextIndexingEnabled,
           selectedProject?.syntextIndexEnabled == true,
           let rootURL = selectedProject?.rootDirectoryURL {
            Task.detached(priority: .utility) {
                await SyntextIndexManager.shared.syncQuietly(for: rootURL)
            }
        }
    }
}
