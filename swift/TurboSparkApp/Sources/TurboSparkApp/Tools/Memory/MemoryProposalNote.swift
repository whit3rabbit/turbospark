import Foundation

/// The transcript note for one `memory` tool save: what the model asked to
/// remember, staged as a pending claim for review.
///
/// The executor's output names no claim id (the model-visible result stays
/// one sentence). Instead the staged claim records the call's id
/// (`MemoryClaim.sourceToolCallID`), and `resolution` finds it by that id
/// alone. Never by text: a text match could dismiss the wrong claim.
public struct MemoryProposalNote: Equatable, Sendable {
    public enum Scope: Equatable, Sendable { case profile, project }

    public let callID: UUID
    public let scope: Scope
    /// One line, at most `maxTextLength` characters plus an ellipsis.
    public let text: String

    public static let maxTextLength = 120

    /// The executor's success sentences (`MemoryToolExecutor`). Matching the
    /// output, not only the arguments, keeps a refused, failed, or
    /// hook-rewritten call from drawing a claim that was never staged.
    static let projectSavePrefix = "Proposed memory '"
    static let profileSavePrefix = "Proposed a profile memory for review"

    /// Nil unless `call` is a `memory` save whose matching result reports a
    /// staged proposal. Nil when unsure.
    public init?(call: AppToolCall, result: AppToolResult?) {
        guard call.name == "memory" || call.name == "remember",
              let result, result.callID == call.id, !result.isError,
              ["save", "write", "remember"].contains((call.arguments["action"] ?? "").lowercased())
        else { return nil }
        let arguments = call.arguments
        let body = arguments["content"] ?? arguments["memory"] ?? arguments["text"] ?? ""
        let output = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
        let raw: String
        if (arguments["scope"] ?? "project").lowercased() == "profile" {
            guard output.hasPrefix(Self.profileSavePrefix) else { return nil }
            scope = .profile
            raw = body
        } else {
            guard output.hasPrefix(Self.projectSavePrefix) else { return nil }
            scope = .project
            // The same summary the executor indexes: the description, else
            // the body's first non-empty line.
            raw = arguments["description"] ?? body
        }
        let line = raw.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty } ?? ""
        guard !line.isEmpty else { return nil }
        callID = call.id
        text = line.count > Self.maxTextLength
            ? String(line.prefix(Self.maxTextLength)) + "..." : line
    }

    /// Where the claim this call staged stands now.
    public enum Resolution: Equatable, Sendable {
        /// Still awaiting review; Undo and Approve act on this claim id.
        case pending(UUID)
        /// Decided already (here or in Settings): the status to show.
        case settled(MemoryClaimStatus)
    }

    /// The state of the one claim staged by `callID`. Nil when no claim
    /// carries that id (a ledger from before the field, or a forgotten
    /// record) or when more than one does: unsure means no buttons.
    public static func resolution(callID: UUID, claims: [MemoryClaim]) -> Resolution? {
        let matches = claims.filter { $0.sourceToolCallID == callID }
        guard matches.count == 1, let claim = matches.first else { return nil }
        return claim.status == .pending ? .pending(claim.id) : .settled(claim.status)
    }

    /// What the row's buttons do to a pending claim.
    public enum Action: Sendable { case undo, approve }

    /// Notes for one assistant step, in call order.
    public static func notes(calls: [AppToolCall], results: [AppToolResult]) -> [MemoryProposalNote] {
        calls.compactMap { call in
            MemoryProposalNote(call: call, result: results.first { $0.callID == call.id })
        }
    }
}
