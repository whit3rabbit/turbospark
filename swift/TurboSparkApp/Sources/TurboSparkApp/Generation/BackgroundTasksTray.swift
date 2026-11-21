import Foundation

/// What a background task is. Two kinds exist today and each has its own
/// owner (`AppModel.backgroundAgentRuns`, `BackgroundShellManager`); the
/// tray reads both and owns neither.
enum BackgroundTaskKind: Equatable {
    case agent
    case shell
}

/// One per-task action a tray row can offer. Each is the strip's own action
/// for that task, so the tray never introduces a second way to stop work.
enum BackgroundTaskAction: Equatable {
    /// Select the task's chat, where its strip shows the full live card.
    case open
    /// Stop a running background agent (`stopBackgroundAgent`).
    case stop
    /// Kill a running background shell, after confirmation (`killBackgroundShell`).
    case kill
    /// Clear a finished agent's card (`dismissBackgroundAgent`).
    case dismiss
}

/// A snapshot of one background task for the tray. Built from the live
/// records on each render; nothing is stored.
struct BackgroundTaskEntry: Identifiable, Equatable {
    let taskID: String
    let kind: BackgroundTaskKind
    let title: String
    /// A second line (the task or the command), nil when it would repeat the title.
    let detail: String?
    let startedAt: Date
    /// The engine's raw status word ("running", "completed", "killed", ...).
    let status: String
    let chatID: UUID?
    let canDismiss: Bool

    var id: String { (kind == .agent ? "agent:" : "shell:") + taskID }
    var isRunning: Bool { status == "running" }
}

extension BackgroundTaskEntry {
    /// The same title and fallback order the `/tasks` sheet uses.
    @MainActor
    init(agent run: SubagentRunState) {
        let title = run.displayName.isEmpty ? run.agentName : run.displayName
        let detail = run.taskDescription.isEmpty ? run.promptHead : run.taskDescription
        self.init(
            taskID: run.id, kind: .agent,
            title: title.isEmpty ? run.id : title,
            detail: detail.isEmpty ? nil : detail,
            startedAt: run.startedAt, status: run.status, chatID: run.chatID,
            // `status` flips before completion is recorded, and dismissing in
            // that window drops the task notification (see
            // `SubagentRunState.isRecordedComplete`).
            canDismiss: run.status != "running" && run.isRecordedComplete)
    }

    /// The shell strip's layout: the description as title with the command
    /// under it, or the command alone. Summaries list running shells only.
    init(shell: BackgroundShellSummary) {
        let description = shell.description ?? ""
        let described = !description.isEmpty
        self.init(
            taskID: shell.id, kind: .shell,
            title: described ? description : shell.commandHead,
            detail: described ? shell.commandHead : nil,
            startedAt: shell.startedAt, status: "running", chatID: shell.chatID,
            canDismiss: false)
    }
}

/// The merged, cross-chat view over background agents and shells behind the
/// status bar's tray chip. The per-chat strips under the transcript stay:
/// a background agent's card is the only record of its result.
enum BackgroundTasksTray {
    enum Summary: Equatable {
        case running(Int)
        /// Only finished agent cards remain (shells leave when they end).
        case allFinished
    }

    @MainActor
    static func merge(agents: [SubagentRunState], shells: [BackgroundShellSummary])
        -> [BackgroundTaskEntry]
    {
        sorted(agents.map(BackgroundTaskEntry.init(agent:)) + shells.map(BackgroundTaskEntry.init(shell:)))
    }

    /// Running first, then most recently started. Equal start times fall
    /// back to kind and a numeric id compare so rows do not swap places
    /// between renders.
    static func sorted(_ entries: [BackgroundTaskEntry]) -> [BackgroundTaskEntry] {
        entries.sorted { lhs, rhs in
            if lhs.isRunning != rhs.isRunning { return lhs.isRunning }
            if lhs.startedAt != rhs.startedAt { return lhs.startedAt > rhs.startedAt }
            if lhs.kind != rhs.kind { return lhs.kind == .agent }
            return lhs.taskID.compare(rhs.taskID, options: .numeric) == .orderedAscending
        }
    }

    /// Nil when there is nothing to show, so the chip is absent.
    static func summary(_ entries: [BackgroundTaskEntry]) -> Summary? {
        guard !entries.isEmpty else { return nil }
        let running = entries.filter(\.isRunning).count
        return running > 0 ? .running(running) : .allFinished
    }

    /// The actions a row offers. Stop, Kill and Dismiss follow the strips'
    /// visibility rule (the task's own chat, or unscoped): `killBackgroundShell`
    /// refuses another chat's shell by design, so the tray offers Open there
    /// instead of a button that would silently do nothing.
    static func actions(
        for entry: BackgroundTaskEntry, selectedChatID: UUID, isChatVisible: Bool
    ) -> [BackgroundTaskAction] {
        let inSelectedChat = entry.chatID == nil || entry.chatID == selectedChatID
        guard inSelectedChat else { return [.open] }
        var actions: [BackgroundTaskAction] = isChatVisible ? [] : [.open]
        if entry.isRunning {
            actions.append(entry.kind == .agent ? .stop : .kill)
        } else if entry.canDismiss {
            actions.append(.dismiss)
        }
        return actions
    }
}
