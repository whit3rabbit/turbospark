import Foundation

/// When a failed tool card offers Retry, Skip and Tell the model.
///
/// The agent loop already feeds a failed call's error back to the model,
/// which usually tries again on its own. The footer is for the turn that
/// stopped on the failure instead (the model gave up, the user stopped it,
/// or the app quit mid-call). Each button only sends or drafts an ordinary
/// user reply, so a retried call is a fresh model call that goes through
/// risk classification, the approval card and the permission mode like any
/// other. Offering them while the agent is still acting would race it, so:
/// the turn is idle, the call itself failed (a denial is an answer, not a
/// failure), and nothing has happened after its batch (no user reply, no
/// further tool call). Members of one batch run together, so any failed
/// member of the last batch qualifies.
enum RunRecoveryActions {
    static func isOffered(callID: UUID, in messages: [AppChatMessage], isIdle: Bool) -> Bool {
        guard isIdle,
            let index = messages.firstIndex(where: { $0.toolCalls.contains { $0.id == callID } })
        else { return false }
        let message = messages[index]
        guard message.toolCalls.first(where: { $0.id == callID })?.status == .failed,
            message.toolResults.first(where: { $0.callID == callID })?.isError == true
        else { return false }
        return !messages[(index + 1)...].contains { $0.role == .user || !$0.toolCalls.isEmpty }
    }
}
