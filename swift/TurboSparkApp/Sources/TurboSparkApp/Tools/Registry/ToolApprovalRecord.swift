import Foundation

/// What the user chose on a tool approval card. Display only: the grant each
/// choice makes is applied by the approve and deny paths, never read back
/// from this value.
public enum ToolApprovalDecision: String, Codable, Sendable {
    case allowedOnce
    /// The session grant ("Always Allow" without a persisted rule).
    case allowedForChat
    /// A rule persisted beyond the chat (MCP project rule, network domain,
    /// browser origin).
    case allowedAlways
    case denied
    /// The user replaced the proposed command on the card. The original did
    /// not run; the edit became a new call with its own card and record.
    /// Older builds drop this record (the `try?` decode on `AppToolCall`).
    case editedByUser
}

/// Who resolved the card. `.policy` is the parked-approval replay refusing a
/// call under the current permission settings: the user did not say no.
public enum ToolApprovalActor: String, Codable, Sendable {
    case user
    case policy
}

/// The resolved approval kept on the call so the transcript can show how a
/// card was answered after it closes. Optional on `AppToolCall`, so every
/// archive written before it existed decodes with nil.
public struct ToolApprovalRecord: Codable, Equatable, Sendable {
    public var decision: ToolApprovalDecision
    public var actor: ToolApprovalActor
    public var decidedAt: Date

    public init(decision: ToolApprovalDecision, actor: ToolApprovalActor, decidedAt: Date = Date()) {
        self.decision = decision
        self.actor = actor
        self.decidedAt = decidedAt
    }
}

/// The one-line transcript record for a resolved card: a catalog key for
/// the words and whether it reads as allowed. The view adds glyph and time.
struct ToolApprovalRecordLine: Equatable {
    let labelKey: String
    let isAllowed: Bool
    /// The superseded call of an edit: neither allowed nor refused, so it
    /// gets its own glyph rather than a check or a cross.
    let isEdited: Bool

    init(_ record: ToolApprovalRecord) {
        switch (record.decision, record.actor) {
        case (.allowedOnce, _): labelKey = "Allowed once by you"
        case (.allowedForChat, _): labelKey = "Allowed for this chat by you"
        case (.allowedAlways, _): labelKey = "Always allowed by you"
        case (.denied, .user): labelKey = "Denied by you"
        case (.denied, .policy): labelKey = "Denied by permission settings"
        case (.editedByUser, _): labelKey = "Edited by you, not run"
        }
        isAllowed = record.decision != .denied && record.decision != .editedByUser
        isEdited = record.decision == .editedByUser
    }
}

/// Context lines for a pending approval card: the model's stated reason and
/// the directory a shell command starts in. Pure so the rules are testable.
enum ToolApprovalContext {
    static let maxReasonLength = 240

    /// The model's own explanation for the call, or nil. A shell tool's
    /// `description` argument is its schema's "what the command does" field;
    /// other tools count only an explicit `reason` or `justification`, so an
    /// agent's task description is never relabelled as a justification.
    static func reason(for call: AppToolCall) -> String? {
        var raw = call.arguments["reason"] ?? call.arguments["justification"]
        if raw == nil, call.shellCommand != nil, call.category == .terminal {
            raw = call.arguments["description"]
        }
        guard let text = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            return nil
        }
        // A model can put a paragraph here; the card is a glance, not a log.
        if text.count > maxReasonLength {
            return String(text.prefix(maxReasonLength)) + "..."
        }
        return text
    }

    /// `dir` with the home directory folded to `~`, as a terminal shows it.
    static func displayPath(_ dir: URL, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> String {
        let path = dir.standardizedFileURL.path
        let homePath = home.standardizedFileURL.path
        if path == homePath { return "~" }
        if path.hasPrefix(homePath + "/") { return "~" + path.dropFirst(homePath.count) }
        return path
    }
}
