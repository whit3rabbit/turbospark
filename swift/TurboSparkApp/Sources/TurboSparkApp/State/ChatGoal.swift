import Foundation

/// One chat's active `/goal`: a completion condition the agent keeps working
/// toward until an evaluator model judges it met, judges it impossible, or
/// the user clears it (Claude Code's built-in goal, `tengu_joyful_globe`).
///
/// The spec this type encodes lives in `swift/docs/SWIFT_GOALS.md`; the
/// seams that drive it live in `AppModel+Goal.swift`. Everything here is a
/// pure value so the state machine -- backoff, deferral runs, check-in
/// text, verdict parsing -- is assertable without a session or a model.
public struct ChatGoalState: Codable, Equatable, Sendable {
    /// The completion condition, in the user's own words. The evaluator
    /// judges the conversation against this text and nothing else.
    public var condition: String
    /// When the goal was set. Elapsed time in the banner and the status
    /// view read off this; a relaunch keeps it (CC's `--resume` behavior).
    public var setAt: Date
    /// How many times the evaluator has returned "not yet met" and the loop
    /// continued anyway.
    public var iterations: Int
    /// The evaluator's latest "not yet met" reason, or nil before the first
    /// evaluation. What the banner shows and the assembly-time reminder
    /// carries to the model.
    public var lastReason: String?
    /// When deferral began for the CURRENT run of background work, or nil
    /// while nothing defers evaluation. Cleared when the work drains.
    public var deferredSince: Date?
    /// Check-ins delivered during the current deferral run; drives the
    /// doubling backoff. Reset when deferral restarts as a new run.
    public var checkinCount: Int
    /// The last time the stop seam passed over a deferral. A background
    /// task STARTED after this instant begins a NEW deferral run, which
    /// resets the backoff (CC's `lastDeferralPassAt` new-run detection).
    public var lastDeferralPassAt: Date?
    /// Idle check-ins delivered since the user last submitted anything.
    /// The idle cap is per goal between prompts, so this survives deferral
    /// runs and resets only on a user prompt.
    public var idleCheckinCount: Int
    /// Tokens the main model spent on turns since the goal was set,
    /// accumulated from the engine's own per-turn counts. Display only.
    public var tokensSpent: Int
    /// Stall-paused: the evaluator saw enough consecutive tool-free turns
    /// that the loop stopped itself. Evaluation resumes after the user's
    /// next prompt.
    public var isPaused: Bool

    public init(
        condition: String,
        setAt: Date = Date(),
        iterations: Int = 0,
        lastReason: String? = nil,
        deferredSince: Date? = nil,
        checkinCount: Int = 0,
        lastDeferralPassAt: Date? = nil,
        idleCheckinCount: Int = 0,
        tokensSpent: Int = 0,
        isPaused: Bool = false
    ) {
        self.condition = condition
        self.setAt = setAt
        self.iterations = iterations
        self.lastReason = lastReason
        self.deferredSince = deferredSince
        self.checkinCount = checkinCount
        self.lastDeferralPassAt = lastDeferralPassAt
        self.idleCheckinCount = idleCheckinCount
        self.tokensSpent = tokensSpent
        self.isPaused = isPaused
    }

    /// What a relaunch restores: the condition and when it was set, and
    /// nothing else (CC's `--resume` rule -- turn count, timer and token
    /// baseline reset). The runtime state (deferral, stall, idle cap) is
    /// meaningless across a process boundary anyway.
    public func restoredForRelaunch() -> ChatGoalState {
        ChatGoalState(condition: condition, setAt: setAt)
    }

    /// Whether an idle check-in may still be delivered: the cap is
    /// `GoalPolicy.maxIdleCheckins` per goal between user prompts, and the
    /// delivery that reaches the cap is the one that announces the pause.
    public var canIdleCheckin: Bool {
        idleCheckinCount < GoalPolicy.maxIdleCheckins
    }
}

/// One piece of background work that defers goal evaluation, flattened out
/// of the two registries that hold it (background agents, background
/// shells) so the policy below never reads live state.
public struct GoalBackgroundTask: Equatable, Sendable {
    public enum Kind: String, Sendable {
        case agent
        case shell
    }

    public let id: String
    public let kind: Kind
    /// One-line human label: the agent's display name or the shell's
    /// description.
    public let label: String
    /// Longer detail: the agent's task description or the shell command
    /// head.
    public let detail: String
    public let startedAt: Date

    public init(
        id: String, kind: Kind, label: String, detail: String, startedAt: Date
    ) {
        self.id = id
        self.kind = kind
        self.label = label
        self.detail = detail
        self.startedAt = startedAt
    }
}

/// The one evaluator verdict, already parsed.
public enum GoalVerdict: Equatable, Sendable {
    case met(reason: String?)
    case notMet(reason: String)
    case impossible(reason: String)
}

/// Constants and pure decisions for the goal loop.
///
/// The values are Claude Code's published ones (see
/// `swift/docs/SWIFT_GOALS.md`): a 30 minute first check-in that doubles
/// per check-in capped at 4x, three idle check-ins per goal between user
/// prompts, and stall detection instead of any iteration cap.
public enum GoalPolicy {
    /// A condition is a sentence, not a document (CC's own limit).
    public static let maxConditionLength = 4000
    /// How much of the condition the banner and the check-in text quote.
    public static let conditionPreviewLength = 120
    /// Minutes until the first check-in while background work defers
    /// evaluation.
    public static let checkinBaseInterval: TimeInterval = 30 * 60
    /// The backoff ceiling, as a multiple of the base (4x: 30 min -> 1 hr
    /// -> 2 hr -> 2 hr ...).
    public static let checkinCapMultiplier = 4
    /// Idle check-ins per goal between user prompts. The third announces
    /// that idle check-ins are paused until the user prompts again.
    public static let maxIdleCheckins = 3
    /// Consecutive evaluated turns with no tool use before the loop
    /// stalls itself. CC's stall detection; there is deliberately no
    /// iteration cap on top of this.
    public static let stallThreshold = 3
    /// The evaluator's budget. It judges one transcript tail and must be
    /// quick; a timeout or a malformed answer is a TRANSIENT failure that
    /// leaves the goal active (CC: only unrecoverable errors clear it).
    public static let evaluatorTimeoutSeconds: TimeInterval = 30
    public static let evaluatorTemperature = 0.2
    public static let evaluatorMaxNewTokens = 200
    /// Conversation rows the evaluator sees, when nothing bounds the slice
    /// (the usual bound is "since the last evaluation").
    public static let evaluatorTailMessages = 20
    /// Minimum spacing between idle-timer wakeups while the chat is busy
    /// and a check-in must wait for a turn tail.
    public static let idleTimerRetryInterval: TimeInterval = 60

    /// The backoff: base doubled per check-in already delivered this run,
    /// capped at `checkinCapMultiplier` x base. The exponent caps at
    /// log2(cap): 2^2 is the 4x ceiling, so the third and every later
    /// check-in sit on the same 2 hr interval.
    public static func checkinInterval(afterCheckins count: Int) -> TimeInterval {
        let capExponent = Int(log2(Double(checkinCapMultiplier)))
        let capped = min(count, capExponent)
        return checkinBaseInterval * pow(2.0, Double(capped))
    }

    /// The result of one pass over the stop seam while background work
    /// defers evaluation.
    public struct DeferralPass: Equatable {
        public let deferredSince: Date
        public let checkinCount: Int
        public let checkinDue: Bool
        /// True when this pass detected that the previous deferral run
        /// ended and a new one began (counters were reset).
        public let isNewRun: Bool
    }

    /// Whether a check-in is due NOW, and what the goal's deferral state
    /// becomes after this pass.
    ///
    /// CC's new-run rule: background work that STARTED after the last
    /// deferral pass began a new run, but only if the last pass is also
    /// older than one base interval -- fresh work started seconds after a
    /// pass does not reset the clock the user has already been waiting on.
    public static func deferralPass(
        goal: ChatGoalState, tasks: [GoalBackgroundTask], now: Date
    ) -> DeferralPass {
        if goal.deferredSince == nil {
            return DeferralPass(
                deferredSince: now, checkinCount: 0, checkinDue: false, isNewRun: true)
        }
        let earliestStart = tasks.map(\.startedAt).min()
        let isNewRun =
            goal.lastDeferralPassAt != nil
            && earliestStart.map { $0 > goal.lastDeferralPassAt! } == true
            && now.timeIntervalSince(goal.lastDeferralPassAt!) > checkinBaseInterval
        let since = isNewRun ? now : goal.deferredSince!
        let count = isNewRun ? 0 : goal.checkinCount
        let elapsed = now.timeIntervalSince(since)
        return DeferralPass(
            deferredSince: since,
            checkinCount: count,
            checkinDue: elapsed >= checkinInterval(afterCheckins: count),
            isNewRun: isNewRun)
    }

    /// How far ahead the idle timer should be armed after a pass, given
    /// time already elapsed in the current deferral run.
    public static func idleTimerDelay(
        pass: DeferralPass, now: Date
    ) -> TimeInterval {
        let target = checkinInterval(afterCheckins: pass.checkinCount)
        let elapsed = now.timeIntervalSince(pass.deferredSince)
        return max(target - elapsed, idleTimerRetryInterval)
    }

    /// The condition quoted for a banner or a check-in, clipped to
    /// `conditionPreviewLength`.
    public static func conditionPreview(_ condition: String) -> String {
        guard condition.count > conditionPreviewLength else { return condition }
        return String(condition.prefix(conditionPreviewLength)) + "..."
    }

    /// The visible `<goal_checkin>` user turn: what the model reads when a
    /// deferred evaluation surfaces as a check-in. Two shapes, CC's two:
    /// work still running (list it, tell the model to check on it), and
    /// work no longer running (nothing defers the goal anymore; continue).
    public static func checkinMessage(
        condition: String, tasks: [GoalBackgroundTask], announcingPause: Bool
    ) -> String {
        let header =
            announcingPause
            ? "Idle check-ins are paused until you send a message; this goal keeps waiting."
            : ""
        if tasks.isEmpty {
            var lines = [
                "<goal_checkin>",
                "The goal \"\(conditionPreview(condition))\" is still active, and the "
                    + "background work that was deferring its evaluation is no longer "
                    + "running (it finished or was stopped without reporting back). "
                    + "Continue toward the goal.",
            ]
            if !header.isEmpty { lines.append(header) }
            lines.append("</goal_checkin>")
            return lines.joined(separator: "\n")
        }
        var lines = [
            "<goal_checkin>",
            "The goal \"\(conditionPreview(condition))\" is still active, and evaluation "
                + "has been deferred because background work is still running:",
        ]
        for task in tasks {
            lines.append("- \(task.id) - \(task.kind.rawValue) - \(task.label): \(task.detail)")
        }
        lines.append(
            "Check on their progress (e.g. read their output). If they are progressing, "
                + "say so briefly and keep waiting; if they are stuck or no longer needed, "
                + "fix or stop them and continue toward the goal.")
        if !header.isEmpty { lines.append(header) }
        lines.append("</goal_checkin>")
        return lines.joined(separator: "\n")
    }
}
