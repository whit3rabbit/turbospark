import Foundation

/// Edit-before-run for a pending shell call (component audit,
/// edit-before-run). Pure so every rule is testable without a model.
///
/// An edit never runs anything and never inherits the original's approval:
/// it builds a NEW call (new id) that the app evaluates from scratch, and the
/// original is recorded as superseded. Only the shell aliases qualify, since
/// their one free-text argument is the whole action; a path or URL tool has
/// other arguments the edit would have to keep consistent.
enum ToolCallCommandEdit {
    /// The executor's shell aliases (`AppToolRegistry.execute`).
    static let shellToolNames: Set<String> = ["run_command", "bash", "shell", "exec", "terminal"]

    /// The argument keys `AppToolCall.shellCommand` reads, in its order. The
    /// edit replaces the key that accessor resolved, so the executor runs the
    /// edited text and never a shadowed alias.
    static let commandKeys = ["command", "cmd", "CommandLine", "code"]

    /// Whether the card may offer Edit for `call` at all.
    static func isEditableTool(_ call: AppToolCall) -> Bool {
        shellToolNames.contains(call.name.lowercased()) && call.shellCommand != nil
    }

    /// Why Edit is unavailable on an editable tool's card, or nil when it is
    /// offered. A parked batch shares one verdict across its members, and a
    /// fused validation leg is a second command the edit cannot see, so both
    /// keep the plain Approve and Deny.
    enum Unavailable: Equatable {
        case batch
        case compound
    }

    static func unavailableReason(batchCount: Int?, hasValidationLeg: Bool) -> Unavailable? {
        if let batchCount, batchCount > 1 { return .batch }
        if hasValidationLeg { return .compound }
        return nil
    }

    /// The trimmed edit, or nil when it is empty or the same command after
    /// trimming (both behave like Cancel).
    static func normalizedEdit(_ text: String, original: String) -> String? {
        let edited = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !edited.isEmpty,
            edited != original.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }
        return edited
    }

    /// A new call carrying `text` as its command, or nil when the edit is a
    /// no-op or `original` is not an editable shell call.
    ///
    /// Copied: name, category, every other argument. Fresh: id, approval id,
    /// timestamp, status. Cleared: the risk rating (the caller re-rates the
    /// edited text), any auto-approval mark and approval record, all of which
    /// described the model's command.
    static func editedCall(
        from original: AppToolCall, text: String, id: UUID = UUID(), now: Date = Date()
    ) -> AppToolCall? {
        guard isEditableTool(original),
            let key = commandKeys.first(where: { original.arguments[$0] != nil }),
            let originalCommand = original.arguments[key],
            let edited = normalizedEdit(text, original: originalCommand)
        else { return nil }
        var arguments = original.arguments
        arguments[key] = edited
        var typed = original.typedArguments?.values
        if typed?[key] != nil { typed?[key] = .string(edited) }
        return AppToolCall(
            id: id,
            name: original.name,
            arguments: arguments,
            // The model never emitted this text; replaying the original
            // markup would show it the command it did not get.
            rawInvocation: "",
            status: .pendingApproval,
            category: original.category,
            riskAssessment: nil,
            autoApprovedBy: nil,
            createdAt: now,
            typedArguments: typed.map(TransientTypedArguments.init(values:)),
            // A native row replays every call by id, so the edit needs its
            // own id and JSON that says what actually ran.
            nativeCallID: original.nativeCallID.map { "\($0)_edited_\(id.uuidString.prefix(8).lowercased())" },
            nativeArgumentsJSON: original.nativeCallID == nil
                ? nil : nativeJSON(original.nativeArgumentsJSON, key: key, value: edited, fallback: arguments),
            editedFromCommand: originalCommand)
    }

    /// The model's JSON with only the command key replaced, types of the
    /// other keys kept. Falls back to the flat string arguments when the
    /// original JSON is missing or not an object.
    static func nativeJSON(_ json: String?, key: String, value: String, fallback: [String: String]) -> String? {
        var object: [String: Any] = fallback
        if let json, let data = json.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            object = parsed
            object[key] = value
        }
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    /// Standing grants ("Always Allow", allow for this chat) are for commands
    /// the model proposed and the policy can match again. An edited command
    /// is approved once only.
    static func allowsStandingGrant(_ call: AppToolCall) -> Bool {
        call.editedFromCommand == nil
    }

    /// The result recorded on the superseded original, so the model's call
    /// has an answer and says where the edit went.
    static func supersededResult(for original: AppToolCall, editedCommand: String) -> AppToolResult {
        AppToolResult(
            callID: original.id,
            output: "Not run: the user edited this command before approving it. "
                + "The edited command was submitted as a separate call:\n\(editedCommand)",
            isError: true,
            durationSeconds: 0.0)
    }

    /// The result of an edited call, prefixed with both commands so the
    /// model does not read the output as its own command's. Unchanged for a
    /// call that was not edited.
    static func annotate(_ result: AppToolResult, for call: AppToolCall) -> AppToolResult {
        guard let original = call.editedFromCommand else { return result }
        var annotated = result
        annotated.output = "Note: the user edited your proposed command before it was evaluated.\n"
            + "Your command: \(original)\n"
            + "Edited command (this is the one this result is for): \(call.shellCommand ?? "")\n\n"
            + result.output
        return annotated
    }
}
