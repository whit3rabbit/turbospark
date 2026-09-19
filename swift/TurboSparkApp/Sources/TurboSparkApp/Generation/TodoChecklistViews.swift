import SwiftUI

/// Counter text shared by the per-call card header and the live checklist
/// panel, so the two surfaces always describe the same list the same way.
/// Format matches Claude Code's "(3/5)" convention: "2/4 done (1 in progress)".
public enum TodoChecklistSummary {
    public static func text(_ todos: [TodoItem]) -> String {
        let completed = todos.filter { $0.isCompleted }.count
        let inProgress = todos.filter { $0.isInProgress }.count
        let base = "\(completed)/\(todos.count) done"
        return inProgress > 0 ? "\(base) (\(inProgress) in progress)" : base
    }
}

/// Pure display rules for the live checklist panel, extracted from the view
/// so the truncation and collapse behavior is unit-testable without SwiftUI.
public enum TaskChecklistPanelModel {
    /// Number of completed rows the expanded panel shows before the rest fold
    /// into a dim overflow line. In-progress and pending rows are never folded.
    public static let settledRowLimit = 6

    /// A list where every item is completed or cancelled collapses to its
    /// summary line. An empty list never reaches the panel, but answers false
    /// so a hypothetical empty render would expand rather than vanish.
    public static func isAllSettled(_ todos: [TodoItem]) -> Bool {
        !todos.isEmpty && todos.allSatisfy { $0.isCompleted || $0.isCancelled }
    }

    /// Rows the expanded panel shows, and how many completed rows folded into
    /// the overflow line. Original order is preserved; when the cap bites, the
    /// OLDEST completed rows fold first (the recent ones stay as evidence of
    /// what just finished), and pending/in-progress/cancelled always render.
    public static func displayItems(
        _ todos: [TodoItem], settledLimit: Int = TaskChecklistPanelModel.settledRowLimit
    ) -> (shown: [TodoItem], foldedCompleted: Int) {
        let completed = todos.filter { $0.isCompleted }
        guard completed.count > settledLimit else { return (todos, 0) }
        let kept = Set(completed.suffix(settledLimit).map(\.id))
        let shown = todos.filter { !$0.isCompleted || kept.contains($0.id) }
        return (shown, completed.count - settledLimit)
    }
}

/// One checklist row, shared by the per-call tool card and the live panel so
/// the status conventions (icons, strikethrough, active form) cannot drift
/// between the two surfaces.
struct TodoItemRow: View {
    let item: TodoItem
    var compact = false

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            statusIcon.frame(width: ConversationLayout.activityIconWidth)

            VStack(alignment: .leading, spacing: 2) {
                Text(item.content)
                    .themedFont(compact ? .small : .base, weight: item.isInProgress ? .semibold : .regular)
                    .foregroundStyle(item.isCompleted ? .secondary : .primary)
                    .strikethrough(item.isCompleted || item.isCancelled, color: .secondary)

                if item.isInProgress && !item.activeForm.isEmpty && item.activeForm != item.content {
                    Text(item.activeForm)
                        .themedFont(.tiny)
                        .foregroundStyle(.appAccent)
                }
            }

            Spacer()

            if item.isInProgress && !compact {
                Text("In Progress", bundle: .module)
                    .themedFont(.tiny, weight: .medium)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.orange.opacity(0.15), in: Capsule())
                    .foregroundStyle(Color.orange)
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var statusIcon: some View {
        if item.isCompleted {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(Color.green)
                .themedFont(.small)
        } else if item.isInProgress {
            TaskProgressFlameIcon(size: 12)
        } else if item.isCancelled {
            Image(systemName: "minus.circle.fill")
                .foregroundStyle(Color.secondary)
                .themedFont(.small)
        } else {
            Image(systemName: "circle")
                .foregroundStyle(Color.secondary.opacity(0.7))
                .themedFont(.small)
        }
    }
}

/// The live checklist for the selected chat, rendered in the transcript
/// OUTSIDE the streaming row (the same rationale as BackgroundAgentsStripView):
/// it must stay visible while the turn runs AND after it ends, because the
/// box is the one surface that updates in place as the model marks tasks off.
/// The per-call tool cards stay collapsed snapshots beside it.
///
/// Repaints come for free: the view reads `model.currentTodos`, and
/// `AppModel.updateTodos(for:todos:)` mutates `@Published chats` (or bumps the
/// ghost vault row) on every TodoWrite call.
struct TaskChecklistPanelView: View {
    @ObservedObject var model: AppModel
    /// Re-expansion of a finished list is viewer-local state, deliberately not
    /// persisted: the collapsed summary is what a finished turn should show.
    @State private var showsRowsWhenSettled = false

    var body: some View {
        let todos = model.currentTodos
        if !todos.isEmpty {
            panel(todos)
        }
    }

    @ViewBuilder
    private func panel(_ todos: [TodoItem]) -> some View {
        let settled = TaskChecklistPanelModel.isAllSettled(todos)
        VStack(alignment: .leading, spacing: 8) {
            header(todos, settled: settled)
            if !settled || showsRowsWhenSettled {
                rows(todos)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            Color(nsColor: .controlBackgroundColor).opacity(0.85),
            in: RoundedRectangle(cornerRadius: 10, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private func header(_ todos: [TodoItem], settled: Bool) -> some View {
        HStack(spacing: 6) {
            Image(systemName: settled ? "checkmark.seal.fill" : "checklist")
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(settled ? Color.green : TurboSparkTheme.accentColor)
                .accessibilityHidden(true)

            Text("Tasks", bundle: .module)
                .themedFont(.small, weight: .semibold)

            Text(TodoChecklistSummary.text(todos))
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)

            Spacer()

            if settled {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        showsRowsWhenSettled.toggle()
                    }
                } label: {
                    Image(systemName: showsRowsWhenSettled ? "chevron.up" : "chevron.down")
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(
                    showsRowsWhenSettled
                        ? Text("Hide completed tasks", bundle: .module)
                        : Text("Show completed tasks", bundle: .module))
            }
        }
    }

    private func rows(_ todos: [TodoItem]) -> some View {
        let display = TaskChecklistPanelModel.displayItems(todos)
        return VStack(alignment: .leading, spacing: 4) {
            ForEach(display.shown) { item in
                TodoItemRow(item: item)
            }
            if display.foldedCompleted > 0 {
                Text(verbatim: "+\(display.foldedCompleted) completed")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .padding(.leading, 20)
            }
        }
    }
}
