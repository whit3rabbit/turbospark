import Foundation
import TurboSpark

/// Functional category of a tool action for permission checks.
public enum AppToolCategory: String, Codable, CaseIterable, Identifiable, Sendable {
    case fileRead
    case fileWrite
    case terminal
    case web
    case mcp
    case automation

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .fileRead: return "File & Codebase Reading"
        case .fileWrite: return "File Editing & Creation"
        case .terminal: return "Terminal & Shell Execution"
        case .web: return "Web & Network Requests"
        case .mcp: return "MCP External Tools"
        case .automation: return "Automation & Cron Tasks"
        }
    }

    public var systemImage: String {
        switch self {
        case .fileRead: return "doc.text.magnifyingglass"
        case .fileWrite: return "square.and.pencil"
        case .terminal: return "terminal"
        case .web: return "globe"
        case .mcp: return "server.rack"
        case .automation: return "clock.arrow.2.circlepath"
        }
    }
}

/// Constant rejection message fed back to the model on denial (Unsloth Studio parity).
public let TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."

/// Execution status of an individual tool call.
public enum AppToolCallStatus: String, Codable, Sendable {
    case pendingApproval
    case running
    case completed
    case denied
    case failed
}

/// A parsed tool call requested by the model.
public struct AppToolCall: Identifiable, Codable, Equatable, Sendable {
    public var id = UUID()
    /// Unique approval identifier for tracking pending decisions.
    public var approvalID: String
    /// Canonical tool name (e.g. read_file, write_file, run_command).
    public var name: String
    /// Parsed parameter dictionary.
    public var arguments: [String: String]
    /// Raw unparsed invocation string from model output.
    public var rawInvocation: String
    /// Execution status.
    public var status: AppToolCallStatus
    /// Associated tool category.
    public var category: AppToolCategory
    /// Evaluated security risk assessment.
    public var riskAssessment: ToolRiskAssessment?
    /// "classifier" when Agent mode's classifier approved the call
    /// (`swift/docs/SWIFT_AGENT_MODE.md`); nil for everything else, which
    /// covers every archive written before the field existed.
    public var autoApprovedBy: String?
    /// Timestamp when invocation was requested.
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        approvalID: String = UUID().uuidString.prefix(16).lowercased(),
        name: String,
        arguments: [String: String] = [:],
        rawInvocation: String = "",
        status: AppToolCallStatus = .pendingApproval,
        category: AppToolCategory = .fileRead,
        riskAssessment: ToolRiskAssessment? = nil,
        autoApprovedBy: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.approvalID = approvalID
        self.name = name
        self.arguments = arguments
        self.rawInvocation = rawInvocation
        self.status = status
        self.category = category
        self.riskAssessment = riskAssessment
        self.autoApprovedBy = autoApprovedBy
        self.createdAt = createdAt
    }

    /// Tolerant decode: every field is read with `decodeIfPresent` and a
    /// default, for the same reason `AppChatMessage.init(from:)` is
    /// hand-written (`swift/CLAUDE.md` Gotcha 13). The synthesized decoder
    /// this replaced required every field, so ONE `AppToolCall` written
    /// before a field like `category` or `riskAssessment` existed threw
    /// while decoding the `toolCalls` ARRAY inside a message -- and because
    /// that throw propagates up through `AppChatMessage`'s own tolerant
    /// `decodeIfPresent(forKey: .toolCalls)`, the whole chat still failed to
    /// decode. A tolerant decoder one level down is what a tolerant
    /// container one level up actually needs to be tolerant.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        approvalID = try container.decodeIfPresent(String.self, forKey: .approvalID)
            ?? UUID().uuidString.prefix(16).lowercased()
        name = try container.decodeIfPresent(String.self, forKey: .name) ?? ""
        arguments = try container.decodeIfPresent([String: String].self, forKey: .arguments) ?? [:]
        rawInvocation = try container.decodeIfPresent(String.self, forKey: .rawInvocation) ?? ""
        status = try container.decodeIfPresent(AppToolCallStatus.self, forKey: .status) ?? .completed
        category = try container.decodeIfPresent(AppToolCategory.self, forKey: .category) ?? .fileRead
        riskAssessment = try container.decodeIfPresent(ToolRiskAssessment.self, forKey: .riskAssessment)
        autoApprovedBy = try container.decodeIfPresent(String.self, forKey: .autoApprovedBy)
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
    }

    /// Single line summary of call arguments for display.
    public var argumentsSummary: String {
        if let path = arguments["path"] ?? arguments["file_path"] {
            return path
        }
        if let cmd = shellCommand {
            return cmd
        }
        if let q = arguments["query"] ?? arguments["pattern"] {
            return "\"\(q)\""
        }
        return arguments.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
    }

    /// The command accepted by the terminal executor. Keep every security
    /// consumer on this accessor so tolerant wire aliases cannot bypass a
    /// gate while still reaching the shell.
    var shellCommand: String? {
        Self.shellCommand(in: arguments)
    }

    static func shellCommand(in arguments: [String: String]) -> String? {
        arguments["command"]
            ?? arguments["cmd"]
            ?? arguments["CommandLine"]
            ?? arguments["code"]
    }
}

/// The result returned from executing a tool call.
public struct AppToolResult: Identifiable, Codable, Equatable, Sendable {
    public var id = UUID()
    public var callID: UUID
    public var output: String
    public var isError: Bool
    public var durationSeconds: Double
    /// The compact representation assembled into the next model prompt. The
    /// ordinary `output` remains the transcript, export, and UI value.
    public var promptProjection: String?
    /// Exact oversized source retained outside a normal-chat transcript.
    public var observation: ToolObservationRef?
    /// Small, user-visible outcomes from native efficiency features.
    public var efficiency: ToolEfficiencyOutcome?
    /// Number of actual prompt sends that included the complete observation.
    public var fullPromptSendCount: Int
    /// Full source available only while this result is in memory. It is never
    /// encoded into the normal chat archive; `ToolObservationStore` receives
    /// it before persistence.
    public var archivalOutput: String?

    public init(
        id: UUID = UUID(),
        callID: UUID,
        output: String,
        isError: Bool = false,
        durationSeconds: Double = 0.0,
        promptProjection: String? = nil,
        observation: ToolObservationRef? = nil,
        efficiency: ToolEfficiencyOutcome? = nil,
        fullPromptSendCount: Int = 0,
        archivalOutput: String? = nil
    ) {
        self.id = id
        self.callID = callID
        self.output = output
        self.isError = isError
        self.durationSeconds = durationSeconds
        self.promptProjection = promptProjection
        self.observation = observation
        self.efficiency = efficiency
        self.fullPromptSendCount = fullPromptSendCount
        self.archivalOutput = archivalOutput
    }

    /// Spelled out because this type now has BOTH a custom `init(from:)` and
    /// a custom `encode(to:)`, and Swift only synthesizes `CodingKeys` while
    /// it is synthesizing one of the two. The names must stay byte-identical
    /// to the property names or every archive written before this change
    /// decodes its fields as absent.
    enum CodingKeys: String, CodingKey {
        case id, callID, output, isError, durationSeconds, promptProjection, observation, efficiency, fullPromptSendCount
    }

    /// Largest tool output written to the chat archive.
    ///
    /// `ProcessExecutor` caps a single command's output at 1 MB, which bounds
    /// one call and bounds nothing about the file: results accumulate in
    /// `AppChatMessage.toolResults`, the whole archive is re-encoded on every
    /// mutation, and `promptText`'s setter makes one of those per keystroke.
    /// A few `run_command` calls against a verbose build therefore turn
    /// typing into a multi-megabyte JSON encode per character.
    public static let maximumPersistedOutputBytes = 64 * 1_024

    /// Truncates at ENCODE time rather than at the call site, so no path into
    /// the archive can bypass it. The in-memory value stays whole for the
    /// turn that produced it, which is what the model is shown.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(callID, forKey: .callID)
        try container.encode(isError, forKey: .isError)
        try container.encode(durationSeconds, forKey: .durationSeconds)
        if let promptProjection, promptProjection.utf8.count <= Self.maximumPersistedOutputBytes {
            try container.encode(promptProjection, forKey: .promptProjection)
        }
        try container.encodeIfPresent(observation, forKey: .observation)
        try container.encodeIfPresent(efficiency, forKey: .efficiency)
        try container.encode(fullPromptSendCount, forKey: .fullPromptSendCount)

        if output.utf8.count > Self.maximumPersistedOutputBytes {
            // Head-plus-tail with the omission marker in the middle, rather
            // than head-only: the END of an oversized output is usually the
            // part worth reading (a build summary, the error a long log was
            // building toward), and a head-only cut rendered every verbose
            // log as the same opening lines. The total stays inside the
            // budget, so the archive bound this encode exists for is
            // unchanged. Slicing on utf8 and decoding lossily is safe at a
            // partial character.
            let budget = Self.maximumPersistedOutputBytes
            let head = String(decoding: output.utf8.prefix(budget * 3 / 4), as: UTF8.self)
            let tail = String(decoding: output.utf8.suffix(budget / 4), as: UTF8.self)
            let omittedKB = (output.utf8.count - budget) / 1_024
            try container.encode(
                head
                    + "\n... (tool output truncated in the saved transcript: "
                    + "\(max(omittedKB, 1)) KB omitted here; the transcript keeps the head and the tail only)...\n"
                    + tail,
                forKey: .output)
        } else {
            try container.encode(output, forKey: .output)
        }
    }

    /// Tolerant decode, for the same reason as `AppToolCall.init(from:)`
    /// above: this struct is stored inside `AppChatMessage.toolResults`, and
    /// a required field failing to decode there would take the whole chat
    /// archive down with it.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        callID = try container.decodeIfPresent(UUID.self, forKey: .callID) ?? UUID()
        output = try container.decodeIfPresent(String.self, forKey: .output) ?? ""
        isError = try container.decodeIfPresent(Bool.self, forKey: .isError) ?? false
        durationSeconds = try container.decodeIfPresent(Double.self, forKey: .durationSeconds) ?? 0.0
        promptProjection = try container.decodeIfPresent(String.self, forKey: .promptProjection)
        observation = try container.decodeIfPresent(ToolObservationRef.self, forKey: .observation)
        efficiency = try container.decodeIfPresent(ToolEfficiencyOutcome.self, forKey: .efficiency)
        fullPromptSendCount = try container.decodeIfPresent(Int.self, forKey: .fullPromptSendCount) ?? 0
        archivalOutput = nil
    }

    /// The only result representation model prompt assembly may use.
    public var modelOutput: String { promptProjection ?? output }
}

/// Description of an available tool for prompt construction.
public struct AppToolDefinition: Sendable {
    public let name: String
    public let category: AppToolCategory
    public let description: String
    public let usageExample: String
}

/// Built-in tools and executor implementations for codebase operations.
