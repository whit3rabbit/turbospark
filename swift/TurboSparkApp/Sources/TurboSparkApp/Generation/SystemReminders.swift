import Foundation

/// Claude-Code-style `<system-reminder>` injection for todo and plan-mode
/// state.
///
/// Claude Code reference: `src/utils/attachments.ts` -- supplementary context
/// wrapped in reminder markup and injected into the conversation at assembly
/// time rather than persisted as transcript content. The problem it solves
/// here is verified dead state: `TodoWriteExecutor.onTodosUpdated` was
/// UI-only (the model wrote a checklist nothing ever reminded it about), and
/// `PlanModeExecutor`'s active flag was never read by prompt assembly (the
/// model could leave plan mode without anything telling it not to edit).
///
/// **EPHEMERAL BY CONSTRUCTION.** Reminders are computed fresh on every
/// `buildAppendOnlyHistory` pass and appended to the model-bound copy of the
/// LAST user message; the stored transcript row is untouched. Nothing here
/// is persisted, nothing survives a chat switch, and a reminder that fired
/// on step 3 leaves no trace on step 4 except by qualifying again.
///
/// Deliberately NOT wired into the SKILL.state history path: that prompt is
/// O(1) in step count with its own state patch format (`docs/SKILL_STATE.md`),
/// and a growing per-turn reminder would defeat the bound it exists for.
enum SystemReminders {
    /// How many assistant steps must pass after the last TodoWrite before
    /// the checklist reminder fires. 1 would nag after the very next step
    /// (which legitimately spent itself on the tool call the todo list
    /// asked for); 2 means the model has done at least one further step of
    /// its own without touching the list.
    static let todoStaleTurnThreshold = 2

    /// The reminder text for one assembled prompt, or nil when nothing
    /// applies. Pure: every input is a value, so the matrix is assertable
    /// without a session.
    static func reminder(
        todos: [TodoItem], messages: [AppChatMessage], planModeActive: Bool
    ) -> String? {
        var sections: [String] = []
        if let todo = todoSection(todos: todos, messages: messages) {
            sections.append(todo)
        }
        if planModeActive {
            sections.append(planSection)
        }
        guard !sections.isEmpty else { return nil }
        return sections
            .map { "<system-reminder>\n\($0)\n</system-reminder>" }
            .joined(separator: "\n\n")
    }

    // MARK: - Todos

    /// The checklist reminder, or nil. Fires when the chat has todos that
    /// are neither completed nor cancelled AND the model has gone
    /// `todoStaleTurnThreshold` assistant steps without a TodoWrite call.
    static func todoSection(todos: [TodoItem], messages: [AppChatMessage]) -> String? {
        let open = todos.filter { !$0.isCompleted && !$0.isCancelled }
        guard !open.isEmpty else { return nil }
        let steps = assistantTurnsSinceLastTodoWrite(in: messages)
        guard steps >= todoStaleTurnThreshold else { return nil }

        var lines = [
            "Your task checklist has not been updated in \(steps) assistant turns. Current items:"
        ]
        for (index, item) in todos.enumerated() {
            let mark = item.isCompleted ? "[x]" : (item.isCancelled ? "[-]" : "[ ]")
            let active = item.isInProgress && !item.activeForm.isEmpty
                ? " (active: \(item.activeForm))" : ""
            lines.append("\(index + 1). \(mark) \(item.content)\(active)")
        }
        lines.append(
            "Keep the list current with the todowrite tool: mark an item in_progress before "
                + "starting it and completed when done, and add items as new work is discovered.")
        return lines.joined(separator: "\n")
    }

    /// Assistant steps after the last message carrying a TodoWrite call.
    /// Zero when the model has written todos this very step or since; the
    /// full count when it never has (a chat whose todos arrived any other
    /// way is at least as stale).
    static func assistantTurnsSinceLastTodoWrite(in messages: [AppChatMessage]) -> Int {
        var steps = 0
        for message in messages.reversed() {
            if message.toolCalls.contains(where: { isTodoWriteName($0.name) }) {
                break
            }
            if message.role == .assistant {
                steps += 1
            }
        }
        return steps
    }

    /// The registry's spellings for the todo tool (`"todowrite",
    /// "todo_write"`), matched case-insensitively because `AppToolCall.name`
    /// records what the dialect emitted.
    static func isTodoWriteName(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return lowered == "todowrite" || lowered == "todo_write"
    }

    // MARK: - Plan mode

    /// While plan mode is active, every prompt restates the constraint.
    /// `exit_plan_mode` is the registry's canonical spelling.
    static var planSection: String {
        """
        Plan mode is ACTIVE for this conversation. Do not edit files, write files, apply \
        patches, or run state-changing commands: research and design only. When the plan is \
        ready, call exit_plan_mode with the finalized plan text.
        """
    }
}
