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
        case modelHub

        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .chat: return "Chat"
            case .modelHub: return "Model hub"
            }
        }
        public var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right"
            case .modelHub: return "square.grid.2x2"
            }
        }
    }

    /// Currently active navigation section in the main window.
    @Published public var activeSection: AppNavigationSection = .chat

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

    // Project and Agent State
    /// All configured codebase projects.
    @Published public var projects: [AppProject] = []
    /// Currently active project filter (nil = all chats).
    @Published public var selectedProjectID: UUID? = nil
    /// Currently pending tool call requiring user approval.
    @Published public var pendingToolCall: AppToolCall? = nil
    /// Live tool calls executed during the active generation turn.
    @Published public var liveToolCalls: [AppToolCall] = []

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
        refreshModels()
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

    /// Live decode throughput in tokens per second.
    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    /// Peak resident process memory footprint in bytes.
    public var currentProcessMemoryBytes: UInt64? {
        TurboSparkSession.peakFootprintBytes
    }

    /// Whether there is any conversation history or live output to display.
    public var hasOutputTranscript: Bool {
        !selectedChat.messages.isEmpty || !outputText.isEmpty || !outputReasoningText.isEmpty
    }

    /// Resolved context token limit when in automatic mode.
    public var resolvedContextTokens: Int {
        if let info = info {
            return Int(info.maxContext)
        }
        return 4096
    }

    /// Whether starter prompt examples should be displayed in place of transcript.
    public var showsPromptExamples: Bool {
        promptText.isEmpty && promptAttachments.isEmpty && !hasOutputTranscript
    }

    /// Plain text of the latest assistant output.
    public var outputResponsePlainText: String {
        if !outputText.isEmpty {
            return outputText
        }
        return selectedChat.messages.last(where: { $0.role == .assistant })?.content ?? ""
    }

    /// Full plain text transcript of the active chat conversation.
    public var outputConversationPlainText: String {
        var transcriptLines: [String] = []
        for message in selectedChat.messages {
            let label = message.role == .user ? "You" : "Assistant"
            transcriptLines.append("\(label):\n\(message.content)")
        }
        if !outputText.isEmpty {
            transcriptLines.append("Assistant:\n\(outputText)")
        }
        return transcriptLines.joined(separator: "\n\n")
    }

    /// History messages in the active chat conversation.
    public var transcriptBaseMessages: [AppChatMessage] {
        selectedChat.messages
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

