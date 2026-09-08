import Foundation

/// User-facing kill surfaces for background shells.
///
/// The model has had `KillShell` since background shells landed; the USER
/// had nothing, so a hung `yes` loop or dev server could only be ended by
/// quitting the app, which orphanized it anyway. The strip renders from
/// `backgroundShellSummaries`, rebuilt off the registry's change hook
/// (launch, kill, natural completion). No poll timer: the strip's elapsed
/// counter is `Text(_:style: .timer)`, which updates itself the same way
/// the subagent cards' do, and a record leaves the running set only
/// through paths that fire the hook.
extension AppModel {
    /// Wires the registry's change hook to the published summaries. Called
    /// once from `init`.
    func installBackgroundShellObserver() {
        BackgroundShellManager.shared.setChangeObserver { [weak self] in
            self?.refreshBackgroundShellSummaries()
        }
        refreshBackgroundShellSummaries()
    }

    /// Rebuilds `backgroundShellSummaries` from the running records. ALL
    /// running shells are published; the visibility rule (this chat's, or
    /// unscoped) is applied by the strip view, which already re-renders on
    /// chat switches because it observes the model.
    func refreshBackgroundShellSummaries() {
        let now = Date()
        backgroundShellSummaries = BackgroundShellManager.shared.runningRecords().map { record in
            BackgroundShellSummary(
                id: record.id,
                commandHead: Self.commandHead(of: record.command),
                description: record.displayDescription,
                startedAt: record.startedAt,
                chatID: record.chatID,
                elapsedSeconds: max(0, Int(now.timeIntervalSince(record.startedAt))))
        }
    }

    /// Kills one running shell by id, restricted to what the kill UI can
    /// show: unscoped, or scoped to the selected chat. Returns false for an
    /// unknown id, a finished shell, or another chat's -- the registry
    /// deliberately does not distinguish these (see
    /// `BackgroundShellManager.record(id:chatID:)`).
    @discardableResult
    public func killBackgroundShell(id: String) -> Bool {
        guard let record = BackgroundShellManager.shared.runningRecords().first(where: {
            $0.id == id && ($0.chatID == nil || $0.chatID == selectedChatID)
        }) else { return false }
        return BackgroundShellManager.shared.kill(record)
    }

    /// Kills every running shell, any chat. The Stop All command's shell
    /// arm.
    @discardableResult
    public func killAllBackgroundShells() -> Int {
        BackgroundShellManager.shared.killAll()
    }

    /// Ends everything a deleted chat leaves behind: its running background
    /// agents (stopped through the same insert-and-cancel shape as
    /// `stopBackgroundAgent`, WITHOUT dropping the task entry -- the
    /// completion watcher owns that removal, and `stopBackgroundAgent`
    /// reads a missing entry as "already finished") and its running
    /// background shells. A background agent whose chat is gone would
    /// otherwise run to completion for an audience of nobody -- the
    /// completion path drops its notification when the chat row is missing.
    func stopBackgroundWork(forDeletedChat chatID: UUID) {
        for (id, state) in backgroundAgentRuns
        where state.chatID == chatID && state.status == "running" {
            killedBackgroundAgentIDs.insert(id)
            backgroundAgentTasks[id]?.cancel()
        }
        for record in BackgroundShellManager.shared.runningRecords(chatID: chatID) {
            BackgroundShellManager.shared.kill(record)
        }
    }

    /// Everything background that must not outlive the process. Called from
    /// `shutdown()`. Shells are SIGKILLed tree-style with no grace period:
    /// at quit, SIGTERM's output-flush window buys nothing and the ladder's
    /// waits would stall exit by seconds per stubborn child. Agent tasks
    /// are cancelled so a completion cannot race the final persist.
    func stopAllBackgroundWorkForShutdown() {
        for id in backgroundAgentTasks.keys {
            killedBackgroundAgentIDs.insert(id)
            backgroundAgentTasks[id]?.cancel()
        }
        backgroundAgentTasks.removeAll()
        BackgroundShellManager.shared.killAllForShutdown()
    }

    /// First line of the command, whitespace-trimmed and capped, for the
    /// strip's label.
    static func commandHead(of command: String) -> String {
        let firstLine = command.split(separator: "\n", maxSplits: 1).first
            .map(String.init) ?? command
        let trimmed = firstLine.trimmingCharacters(in: .whitespaces)
        guard trimmed.count > 96 else { return trimmed }
        return String(trimmed.prefix(96)) + "..."
    }
}
