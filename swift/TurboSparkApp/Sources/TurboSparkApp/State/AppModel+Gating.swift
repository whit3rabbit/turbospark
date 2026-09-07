import Foundation
import TurboSpark

/// What the app will and will not let you do right now.
///
/// **THESE ARE THE FILE'S OWN SUBJECT AND THEY ARE WHERE THE BUGS WERE.**
/// Four of them gained a term in the third state review because a sibling
/// already carried it and they did not: `canRun` had no `pendingToolCall`
/// where `canCancel` did (state#76), `canDeleteModel` did not exist at all
/// and its three call sites each spelled their own (state#73), and
/// `unloadModel` checked one of `canUnloadModel`'s three (state#85).
/// Together in one file, a missing term is visible by reading down the
/// column; scattered through 800 lines of published properties it was not.
extension AppModel {
    /// Whether generation is currently running.
    public var isRunning: Bool { generating }

    /// Whether at least one model is installed locally.
    public var isModelInstalled: Bool { !installed.isEmpty }

    /// Whether no models are installed and initial installation is required.
    public var requiresModelInstallation: Bool { installed.isEmpty }

    /// Whether a model session is currently loaded into memory.
    public var isModelAvailable: Bool { session != nil }

    /// Whether conditions allow starting a new generation run.
    ///
    /// **A PENDING TOOL CALL COUNTS** (state#76). `generating` is lowered on
    /// purpose while a call waits for a human (state#9), so Send was live for
    /// the whole time an approval card sat on screen: a second `generate()`
    /// started beside the one the card belongs to, overwrote `runTask`, and
    /// left the first turn running with nothing able to cancel it. This is
    /// `canCancel`'s third term, in the predicate that has to agree with it.
    public var canRun: Bool {
        Self.canRunTerms(
            generating: generating,
            submitting: submitting,
            opening: opening,
            hasPendingCall: pendingToolCall != nil,
            hasSession: session != nil,
            hasInput: !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !promptAttachments.isEmpty)
    }

    /// `canRun`'s terms, as a pure function of them (state#76).
    ///
    /// Split out for `swift/CLAUDE.md` Gotcha 26's reason: `session` is
    /// non-nil only with a real install open, so the interesting combinations
    /// of this predicate cannot be reached from a test at all -- and the term
    /// that was missing is one of the ones a test could not see.
    static func canRunTerms(
        generating: Bool, submitting: Bool, opening: Bool, hasPendingCall: Bool,
        hasSession: Bool, hasInput: Bool
    ) -> Bool {
        !generating && !submitting && !opening && !hasPendingCall && hasSession && hasInput
    }

    /// Whether a submitted prompt would be QUEUED rather than dropped.
    ///
    /// The message-queue contract (`AppModel+Queue.swift`): when the ONLY
    /// reason `canRun` is false is that something is already running --
    /// a turn, an awaited hook, or an approval card -- Send keeps working
    /// and parks the draft. Anything else (no session, an open in flight,
    /// an empty draft) refuses exactly as before; in particular `opening`
    /// queues nothing, because nothing drains the queue at the END of an
    /// open, and a prompt parked there would sit forever.
    public var canQueue: Bool {
        Self.canQueueTerms(
            generating: generating,
            submitting: submitting,
            hasPendingCall: pendingToolCall != nil,
            hasSession: session != nil,
            hasInput: !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !promptAttachments.isEmpty)
    }

    /// `canQueue`'s terms, pure for the same testability reason as
    /// `canRunTerms`: the busy terms are a DISJUNCTION here, which is the
    /// shape a truth-table test wants spelled out.
    static func canQueueTerms(
        generating: Bool, submitting: Bool, hasPendingCall: Bool,
        hasSession: Bool, hasInput: Bool
    ) -> Bool {
        hasSession && hasInput && (generating || submitting || hasPendingCall)
    }

    /// Whether Send/Return should do SOMETHING: run, or queue behind the
    /// running work. This is the predicate the composer's key handler and
    /// the Generate menu item are gated on.
    public var canRunOrQueue: Bool { canRun || canQueue }

    /// Whether the turn can be stopped.
    ///
    /// **THREE STATES COUNT, NOT ONE** (state#33). `generating` is the
    /// obvious one. `submitting` is the window `run()` spends awaiting a
    /// `UserPromptSubmit` hook before anything is appended: Send is already
    /// refused there, so leaving Stop refused too gave a wedged hook no exit.
    /// And a pending tool call lowers `generating` on purpose (state#9), so
    /// without `pendingToolCall` here the only ways out of an approval card
    /// were Approve, Deny, or deleting the chat -- and `cancel()`'s own
    /// `clearPendingToolCall()` was unreachable (state#34).
    public var canCancel: Bool {
        (generating || submitting || pendingToolCall != nil) && !isCancellationPending
    }

    /// Whether an active model download can be cancelled.
    public var canCancelInstall: Bool { isInstallingModel }

    /// Whether the selected model can be loaded.
    ///
    /// `!submitting` for the same reason `!generating` is here: unloading or
    /// swapping the model while `run()` is awaiting its hook appends the user
    /// message, clears the draft, and then returns silently at
    /// `executeGenerationTurn`'s `session` guard -- a prompt consumed with no
    /// reply and no error (state#33).
    public var canLoadModel: Bool {
        !generating && !submitting && !opening && session == nil && selected != nil
    }

    /// Whether the active model session can be reloaded.
    public var canReloadModel: Bool {
        !generating && !submitting && !opening && session != nil
    }

    /// Whether a model's files may be removed from disk right now (state#73).
    ///
    /// The same three flags the load/unload predicates carry, and `opening`
    /// is the one that was missing at every call site: an open runs on the
    /// binding's private queue for tens of seconds on a real install, and
    /// `deleteModel` would happily remove the directory under it.
    public var canDeleteModel: Bool {
        !generating && !submitting && !opening
    }

    /// Whether the active model session can be unloaded.
    public var canUnloadModel: Bool {
        !generating && !submitting && !opening && session != nil
    }
}
