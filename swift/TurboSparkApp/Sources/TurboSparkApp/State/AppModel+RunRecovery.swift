import Foundation

extension AppModel {
    /// The failed-call footer's Retry. A reply, not a re-execution: the model
    /// issues a new call, which gets the same risk classification, approval
    /// card and permission mode as any other. Re-running the stored call
    /// directly would need its own approval semantics.
    ///
    /// The tool name is the model's own (`call.name`), so it can tell which
    /// call is meant; it is code, not prose, and stays untranslated.
    func retryFailedCallFromCard(_ call: AppToolCall) {
        sendReplyFromCard(AppStrings.localized("Retry the failed \(call.name) call."))
    }

    /// Skip: the model already holds the error result, so skipping is only
    /// the instruction to carry on without it.
    func skipFailedCallFromCard(_ call: AppToolCall) {
        sendReplyFromCard(AppStrings.localized("Skip the failed \(call.name) call and continue without it."))
    }

    /// Tell the model: drafts a lead-in and focuses the composer. Nothing is
    /// sent, so the user's own words explain the failure.
    func explainFailedCallFromCard(_ call: AppToolCall) {
        composeInChat(AppStrings.localized("About the failed \(call.name) call:") + " ")
    }
}
