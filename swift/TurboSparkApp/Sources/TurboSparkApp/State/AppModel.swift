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
        case modelHub

        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .chat: return "Chat"
            case .files: return "Files"
            case .modelHub: return "Models"
            }
        }
        public var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right"
            case .files: return "folder"
            case .modelHub: return "shippingbox"
            }
        }
        /// Filled variant used when the section is the active one.
        public var selectedSystemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right.fill"
            case .files: return "folder.fill"
            case .modelHub: return "shippingbox.fill"
            }
        }
        /// Keyboard shortcut character shown in the rail tooltip.
        public var shortcutKey: Character {
            switch self {
            case .chat: return "1"
            case .files: return "2"
            case .modelHub: return "3"
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
    /// The in-process HTTP server, when one is running against `session`.
    /// It shares `session`'s engine rather than opening a second one, and
    /// OUTLIVES `session` if `session` is cleared without also calling
    /// `stopServer()` (`TurboSparkServer`'s own doc) -- `unloadModel()` and
    /// `setModelURL(_:)` both stop it first for exactly that reason, so a
    /// user clicking "Unload" actually releases the resident model rather
    /// than leaving it pinned by a server nothing in the UI still shows.
    @Published public var server: TurboSparkServer?
    /// Whether `startServer()`/`stopServer()` is in flight.
    @Published public var serverBusy: Bool = false
    /// Bearer / `x-api-key` value to require on the server, or empty for no
    /// auth. Read at `startServer()` time, not persisted: a key typed for
    /// one session sharing a machine is not something to write to disk by
    /// default.
    @Published public var serverAPIKeyInput: String = ""

    // Project and Agent State
    /// All configured codebase projects.
    @Published public var projects: [AppProject] = []
    /// Currently active project filter (nil = all chats).
    @Published public var selectedProjectID: UUID? = nil
    /// Global application-level MCP server configurations.
    @Published public var globalMcpServers: [McpServerConfig] = []
    /// Currently pending tool call requiring user approval.
    @Published public var pendingToolCall: AppToolCall? = nil
    /// Live tool calls executed during the active generation turn.
    @Published public var liveToolCalls: [AppToolCall] = []

    // Skills State
    /// User-scoped skills (~/.turbospark/skills and user agent directories).
    @Published public var userSkills: [AppSkill] = []
    /// Project-scoped skills for the currently selected project.
    @Published public var projectSkills: [AppSkill] = []

    // Multi-chat State
    /// All user chat conversations.
    @Published public var chats: [AppChat] = []
    /// Unique identifier of the currently active chat conversation.
    @Published public var selectedChatID: UUID = UUID()
    /// Transient active chat draft when chats list is empty.
    private var activeDraftChat = AppChat()

    // Live Generation State
    /// Whether token generation is currently running.
    @Published public var generating: Bool = false
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

    var runTask: Task<Void, Never>?
    var installTask: Task<Void, Never>?
    var tokenEstimateTask: Task<Void, Never>?
    var decodeStartTime: Date?

    /// Creates and initializes the application model, restoring saved settings and chats.
    public init() {
        loadSettings()
        loadProjects()
        loadChats()
        loadGlobalMcpServers()
        reloadSkills()
        refreshModels()
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
    public var selectedChat: AppChat {
        if let index = selectedChatIndex {
            return chats[index]
        }
        if let first = chats.first {
            selectedChatID = first.id
            return first
        }
        if activeDraftChat.id != selectedChatID {
            activeDraftChat.id = selectedChatID
        }
        return activeDraftChat
    }

    /// Draft prompt text for the currently selected chat.
    public var promptText: String {
        get { selectedChat.draft }
        set {
            if let index = selectedChatIndex {
                chats[index].draft = newValue
                chats[index].updatedAt = Date()
                persistChats()
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
        !generating && !opening && session != nil && (!promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !promptAttachments.isEmpty)
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

    /// Whether the loaded model supports reasoning output.
    public var reasoningAvailable: Bool {
        info?.reasoningSupport != SessionInfo.ReasoningSupport.none
    }

    /// Whether the active model or currently selected model supports reasoning output.
    public var isReasoningSupported: Bool {
        if let info = info {
            return info.reasoningSupport != SessionInfo.ReasoningSupport.none
        }
        guard let selectedModel = selected ?? installed.first else {
            return false
        }
        let family = selectedModel.family.lowercased()
        let reasoningFamilies: Set<String> = ["gptoss", "museglimmer", "qwen38", "qwen36", "qwen3moe", "qwen35", "gemma4"]
        return reasoningFamilies.contains(family) || family.contains("reason")
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
