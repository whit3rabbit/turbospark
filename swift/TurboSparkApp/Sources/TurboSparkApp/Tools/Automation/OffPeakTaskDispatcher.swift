import Foundation

/// The keep-awake side of an off-peak run. The real port is
/// `ProcessInfo.beginActivity`; tests stub it to count begin and end calls.
public protocol OffPeakActivityPort: AnyObject {
    /// Holds off idle system sleep. Nil means the hold could not be taken;
    /// the dispatcher proceeds anyway (the task then runs only while the Mac
    /// stays awake on its own).
    func begin(reason: String) -> AnyObject?
    func end(_ token: AnyObject)
}

/// `.idleSystemSleepDisabled` keeps the Mac from sleeping on idle while an
/// off-peak task runs. It does not keep the display on and does not stop a
/// lid-close or user-requested sleep. On battery it drains power: the user
/// turned the feature on, and the pane says what it does.
public final class ProcessInfoActivityPort: OffPeakActivityPort {
    public init() {}

    public func begin(reason: String) -> AnyObject? {
        ProcessInfo.processInfo.beginActivity(
            options: [.idleSystemSleepDisabled], reason: reason) as AnyObject
    }

    public func end(_ token: AnyObject) {
        if let activity = token as? NSObjectProtocol {
            ProcessInfo.processInfo.endActivity(activity)
        }
    }
}

/// What the dispatcher needs from the app. `AppModel` implements it; a test
/// stubs it so no model or session is built.
@MainActor
public protocol OffPeakHost: AnyObject {
    func offPeakChatExists(_ chatID: UUID) -> Bool
    /// The chat is selected, idle, has a model ready, nothing queued ahead
    /// and no draft that delivery would disturb.
    func offPeakCanDispatch(chatID: UUID) -> Bool
    /// Starts the turn. True only when a generation actually began.
    func offPeakDispatch(prompt: String, chatID: UUID) -> Bool
    /// A tool approval card is parked in this chat. Nobody is there to
    /// answer it, so the run ends instead of waiting forever.
    func offPeakIsAwaitingApproval(chatID: UUID) -> Bool
}

/// Admits and dispatches off-peak tasks, one at a time, into the chat each
/// was queued from.
///
/// **THE ADMISSION GATE IS THE USER'S TOGGLE.** The feature is off by
/// default. With it on, a queued task is admitted when its chat can take
/// input right now (selected, idle, model loaded): the same delivery
/// conditions as a cron prompt, which is why concurrent chats (ZP-23) are
/// the later relaxation. Holding the no-sleep assertion is an OUTPUT of a
/// running task, never part of the gate: an assertion says nothing about
/// whether the user is at the machine.
///
/// **UNATTENDED POLICY.** The dispatcher never approves a tool call. When a
/// run parks on an approval card it ends as interrupted with outcome
/// `approval_needed` and releases the assertion; the card stays for the user.
@MainActor
public final class OffPeakTaskDispatcher {
    public let store: OffPeakTaskStore
    private weak var host: (any OffPeakHost)?
    private let activity: any OffPeakActivityPort
    private let isEnabled: () -> Bool
    private var activityToken: AnyObject?

    public init(
        store: OffPeakTaskStore = .shared,
        host: any OffPeakHost,
        activity: any OffPeakActivityPort = ProcessInfoActivityPort(),
        isEnabled: @escaping () -> Bool
    ) {
        self.store = store
        self.host = host
        self.activity = activity
        self.isEnabled = isEnabled
    }

    /// Whether the no-sleep hold is currently taken (tests and diagnostics).
    public var isHoldingActivity: Bool { activityToken != nil }

    /// One admission pass, from the app's own timer.
    public func tick(now: Date = Date()) {
        guard let host else { return }
        if let running = store.runningTask() {
            if !host.offPeakChatExists(running.chatID) {
                finish(running.id, .interrupted, "chat_missing", now)
            } else if host.offPeakIsAwaitingApproval(chatID: running.chatID) {
                finish(running.id, .interrupted, "approval_needed", now)
            }
            return
        }
        guard isEnabled() else { return }

        // A task whose chat was deleted is recorded and never dispatched.
        for task in store.list() where task.state == .queued && !host.offPeakChatExists(task.chatID) {
            store.skip(id: task.id, outcome: "chat_missing", now: now)
        }
        guard let claimed = store.claimNext(
            now: now, eligible: { host.offPeakCanDispatch(chatID: $0.chatID) })
        else { return }

        // The claim is already on disk. Take the hold, then deliver.
        activityToken = activity.begin(reason: "TurboSpark off-peak task")
        if host.offPeakDispatch(prompt: claimed.prompt, chatID: claimed.chatID) {
            return
        }
        releaseActivity()
        store.releaseClaim(id: claimed.id)
    }

    /// Called from the end of a turn. `stillBusy` covers a queued prompt
    /// that started the next turn, or an approval card that parked this one.
    public func turnDidFinish(chatID: UUID, cancelled: Bool, stillBusy: Bool, now: Date = Date()) {
        guard !stillBusy, let running = store.runningTask(), running.chatID == chatID else { return }
        if cancelled {
            finish(running.id, .interrupted, "stopped", now)
        } else {
            finish(running.id, .completed, "completed", now)
        }
    }

    /// Quit: the run is cut short, so it is recorded as interrupted now
    /// rather than left for the launch sweep, and the hold is released.
    public func shutdown(now: Date = Date()) {
        if let running = store.runningTask() {
            store.finish(id: running.id, state: .interrupted, outcome: "app_quit", now: now)
        }
        releaseActivity()
    }

    private func finish(_ id: String, _ state: OffPeakTaskState, _ outcome: String, _ now: Date) {
        store.finish(id: id, state: state, outcome: outcome, now: now)
        releaseActivity()
    }

    private func releaseActivity() {
        if let token = activityToken {
            activity.end(token)
            activityToken = nil
        }
    }
}
