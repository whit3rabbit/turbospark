import Foundation

/// When the context ring popover's Compact and New chat buttons (and the
/// context-full stop notice's Compact button) are refused, and why. Pure so
/// the decision is testable without a model. The buttons call the existing
/// `/compact` and `createChat` paths; this only mirrors their guards so a
/// disabled button can say why instead of silently doing nothing.
enum ContextMeterActions {
    enum Blocker: Equatable {
        case noModel
        case turnRunning
        case compacting
    }

    /// Mirrors `handleCompactCommand`'s guards: a loaded session, and no
    /// generation, submission or pending tool approval in flight.
    static func compactBlocker(
        hasSession: Bool, isCompacting: Bool, generating: Bool,
        submitting: Bool, awaitingApproval: Bool
    ) -> Blocker? {
        // Checked first: a manual compaction runs under `submitting`, and the
        // reason should name the compaction, not a generic running turn.
        if isCompacting { return .compacting }
        if generating || submitting || awaitingApproval { return .turnRunning }
        if !hasSession { return .noModel }
        return nil
    }

    /// The same rule `NewChatSuggestionBanner` applies to its New Chat button.
    static func newChatBlocker(generating: Bool) -> Blocker? {
        generating ? .turnRunning : nil
    }
}

extension AppModel {
    var contextMeterCompactBlocker: ContextMeterActions.Blocker? {
        ContextMeterActions.compactBlocker(
            hasSession: session != nil, isCompacting: isCompacting,
            generating: generating, submitting: submitting,
            awaitingApproval: pendingToolCall != nil)
    }

    /// Runs the manual `/compact` path from a button. The typed command
    /// clears the composer because the command WAS the draft; a button press
    /// is not, so whatever the user was typing is put back. The compaction
    /// task itself never touches `promptText`.
    func compactKeepingDraft() {
        let draft = promptText
        handleCompactCommand("/compact")
        // Only when it was cleared: the setter persists the chat row.
        if promptText != draft { promptText = draft }
    }
}
