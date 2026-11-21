import Combine
import SwiftUI

/// Status bar tray for every background agent and shell, in every chat.
/// The strips under the transcript only show the selected chat's tasks, so
/// work started elsewhere was invisible from any other chat or pane. This
/// reads the same two sources (`backgroundAgentRuns`,
/// `backgroundShellSummaries`) and calls the strips' own stop, kill and
/// dismiss paths; it owns no task state. Renders nothing when there are no
/// tasks, so an idle status bar is unchanged.
///
/// Reachability does not depend on this tray: the strips and the `/tasks`
/// sheet still list the selected chat's tasks, and Stop All covers all.
@MainActor
struct BackgroundTasksTrayView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var isPresented = false
    @State private var shellToKill: BackgroundTaskEntry?
    /// Bumped when any run's status or completion flag changes. The runs
    /// dictionary publishes only on insert and remove, so without this the
    /// count kept "1 running" after a run finished (the same trap
    /// `TasksStatusSheet.AgentRunRow` documents).
    @State private var statusTick = 0

    private var entries: [BackgroundTaskEntry] {
        BackgroundTasksTray.merge(
            agents: Array(model.backgroundAgentRuns.values),
            shells: model.backgroundShellSummaries)
    }

    /// Status and completion only, not `objectWillChange`: a run's streamed
    /// text changes per token, and the chip does not need those renders.
    /// `dropFirst` skips the current value each subscription replays.
    private var runChanges: AnyPublisher<Void, Never> {
        Publishers.MergeMany(model.backgroundAgentRuns.values.flatMap { run in
            [
                run.$status.removeDuplicates().dropFirst().map { _ in () }.eraseToAnyPublisher(),
                run.$isRecordedComplete.removeDuplicates().dropFirst().map { _ in () }
                    .eraseToAnyPublisher(),
            ]
        })
        .eraseToAnyPublisher()
    }

    var body: some View {
        let _ = statusTick
        let entries = entries
        if let summary = BackgroundTasksTray.summary(entries) {
            HStack(spacing: 0) {
                trayButton(summary)
                    .popover(isPresented: $isPresented, arrowEdge: .top) {
                        popoverContent(entries)
                    }
                // The divider travels with the tray (as `FanReadoutView`'s
                // does), so an idle status bar has no dangling mark.
                Rectangle()
                    .fill(.appBorder)
                    .frame(width: 0.5, height: 11)
                    .padding(.horizontal, 9)
            }
            // A narrow window squeezes the readouts, never this: it is the
            // only status bar item that leads somewhere.
            .fixedSize()
            .layoutPriority(1)
            .onReceive(runChanges) { statusTick &+= 1 }
        }
    }

    // MARK: - Chip

    private func summaryText(_ summary: BackgroundTasksTray.Summary) -> Text {
        switch summary {
        case .running(let count): return Text("\(count) running", bundle: .module)
        case .allFinished: return Text("All finished", bundle: .module)
        }
    }

    private func trayButton(_ summary: BackgroundTasksTray.Summary) -> some View {
        Button {
            isPresented.toggle()
        } label: {
            HStack(spacing: 5) {
                if case .running = summary {
                    // Honors Reduce Motion itself (`TaskProgressFlameIcon`).
                    TaskProgressFlameIcon(size: 11)
                        .accessibilityHidden(true)
                } else {
                    Image(systemName: "checkmark.circle")
                        .font(theme.ui(.tiny))
                        .accessibilityHidden(true)
                }
                summaryText(summary)
                    .monospacedDigit()
                    .foregroundStyle(.appText)
                    .lineLimit(1)
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(isPresented ? theme.accent.opacity(0.12) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .help(Text("Background Tasks", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Background Tasks", bundle: .module))
        .accessibilityValue(summaryText(summary))
        .accessibilityAddTraits(.isButton)
        .appPointerCursor()
    }

    // MARK: - Popover

    private func popoverContent(_ entries: [BackgroundTaskEntry]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label { Text("Background Tasks", bundle: .module) } icon: {
                Image(systemName: "square.stack.3d.up")
            }
            .font(theme.ui(.callout, weight: .semibold))
            .accessibilityAddTraits(.isHeader)

            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(entries) { entry in
                        row(entry)
                    }
                }
            }
            .frame(maxHeight: 360)
        }
        .padding(14)
        .frame(width: 380 * AppMetrics.textScale(theme))
        .confirmationDialog(
            Text("Kill", bundle: .module),
            isPresented: Binding(
                get: { shellToKill != nil },
                set: { if !$0 { shellToKill = nil } }
            ),
            presenting: shellToKill
        ) { target in
            Button(role: .destructive) {
                model.killBackgroundShell(id: target.taskID)
            } label: {
                Text("Kill", bundle: .module)
            }
            Button(role: .cancel) {
                shellToKill = nil
            } label: {
                Text("Cancel", bundle: .module)
            }
        } message: { _ in
            Text("Kill this background shell (its child processes are killed with it)", bundle: .module)
        }
    }

    private func row(_ entry: BackgroundTaskEntry) -> some View {
        let actions = BackgroundTasksTray.actions(
            for: entry, selectedChatID: model.selectedChatID,
            isChatVisible: model.activeSection == .chat)
        return HStack(alignment: .center, spacing: 8) {
            // Glyph plus word: the kind never rests on the icon alone.
            VStack(spacing: 2) {
                Image(systemName: entry.kind == .agent ? "person.2.wave.2" : "terminal")
                    .font(theme.ui(.small, weight: .semibold))
                    .foregroundStyle(.appAccentText)
                kindText(entry.kind)
                    .font(theme.ui(.micro, weight: .medium))
                    .foregroundStyle(.appSecondary)
            }
            .frame(minWidth: 40 * AppMetrics.textScale(theme))

            VStack(alignment: .leading, spacing: 2) {
                Text(entry.title)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.tail)
                if let detail = entry.detail {
                    Text(detail)
                        .font(theme.code(.small))
                        .foregroundStyle(theme.metadataForeground)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                HStack(spacing: 6) {
                    statusText(entry)
                    if entry.isRunning {
                        Text(entry.startedAt, style: .timer)
                            .monospacedDigit()
                    }
                    if entry.chatID != nil, entry.chatID != model.selectedChatID {
                        Text("In another chat", bundle: .module)
                    }
                }
                .themedFont(.tiny)
                .foregroundStyle(theme.metadataForeground)
            }
            Spacer(minLength: 8)
            ForEach(actions, id: \.self) { action in
                actionButton(action, entry: entry)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.appSurface)
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(theme.border, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .accessibilityElement(children: .combine)
    }

    private func kindText(_ kind: BackgroundTaskKind) -> Text {
        kind == .agent ? Text("Agent", bundle: .module) : Text("Shell", bundle: .module)
    }

    /// The subagent card's status vocabulary; a status the engine adds
    /// later shows as sent rather than vanishing.
    private func statusText(_ entry: BackgroundTaskEntry) -> Text {
        switch entry.status {
        case "running": return Text("Running", bundle: .module)
        case "completed": return Text("Completed", bundle: .module)
        case "killed": return Text("Killed", bundle: .module)
        case "cancelled": return Text("Cancelled", bundle: .module)
        case "error": return Text("Error", bundle: .module)
        case "failed": return Text("Failed", bundle: .module)
        default: return Text(verbatim: entry.status)
        }
    }

    @ViewBuilder
    private func actionButton(_ action: BackgroundTaskAction, entry: BackgroundTaskEntry) -> some View {
        switch action {
        case .open:
            Button {
                open(entry)
            } label: { Text("Open", bundle: .module) }
            .appAction(.quiet, size: .small)
            .help(Text("Show this task in its chat", bundle: .module))
        case .stop:
            Button {
                stopAgent(entry.taskID)
            } label: { Text("Stop", bundle: .module) }
            .appAction(.quiet, size: .small)
        case .kill:
            // Same confirmation as the shell strip: the kill takes the
            // shell's child processes with it.
            Button(role: .destructive) {
                shellToKill = entry
            } label: { Text("Kill", bundle: .module) }
            .appAction(.destructive, size: .small)
        case .dismiss:
            Button {
                model.dismissBackgroundAgent(entry.taskID)
            } label: { Text("Dismiss", bundle: .module) }
            .appAction(.quiet, size: .small)
        }
    }

    /// Selects the task's chat (or, for an unscoped task, brings the chat
    /// pane back), where the strip shows the full live card and output.
    private func open(_ entry: BackgroundTaskEntry) {
        isPresented = false
        // A deleted chat's tasks are stopped with it
        // (`stopBackgroundWork(forDeletedChat:)`), so a missing chat here is
        // a race the tray resolves on its next render; nothing to report.
        _ = model.selectChatFromNotification(entry.chatID ?? model.selectedChatID)
    }

    /// The agent strip's stop path, toast included.
    private func stopAgent(_ id: String) {
        Task { @MainActor in
            do {
                let message = try await model.stopBackgroundAgent(id)
                model.showToast(message, style: .info)
            } catch {
                model.showToast(error.localizedDescription, style: .error)
            }
        }
    }
}
