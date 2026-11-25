import Combine
import SwiftUI

/// The off-peak half of the Scheduled Tasks pane: the on/off switch, a
/// field to queue a prompt in the open chat, and the queue with each task's
/// state. Cron jobs stay above it, untouched: the two dispatchers share a
/// pane, not a store.
///
/// Status is glyph plus text, never color alone. Rows are re-read on a slow
/// timer because `OffPeakTaskStore` is not observable.
@MainActor
struct OffPeakTasksSectionView: View {
    @ObservedObject var model: AppModel
    @State private var revision = 0
    @State private var draft = ""
    @State private var addFailed = false
    private let refresh = Timer.publish(every: 5, on: .main, in: .common).autoconnect()

    private var tasks: [AppOffPeakTask] {
        _ = revision
        return OffPeakTaskStore.shared.list()
    }

    private var selectedChatIsTemporary: Bool {
        model.chats.first(where: { $0.id == model.selectedChatID })?.isGhost ?? false
    }

    private var canAdd: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !selectedChatIsTemporary
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            AppSectionHeader(
                title: Text("Off-Peak Tasks", bundle: .module),
                subtitle: Text("Off-peak tasks wait in a queue and run one at a time in the chat they were added from, only while this is on and that chat is open and idle. A running task keeps the Mac from sleeping on idle, which uses more battery. A task that needs your approval for a tool stops instead of waiting.", bundle: .module))
            Toggle(isOn: Binding(
                get: { model.offPeakEnabled },
                set: { model.setOffPeakEnabled($0) }
            )) {
                Text("Run off-peak tasks", bundle: .module)
            }
            .toggleStyle(.switch)
            .settingsControl("Run off-peak tasks", pane: .automation, timing: .immediate)
            addRow
            if tasks.isEmpty {
                Text("No off-peak tasks yet.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                Text("Off-peak queue: \(tasks.filter { $0.state.isActive }.count) active", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(tasks) { task in row(task) }
                    }
                }
                .frame(maxHeight: 260)
            }
        }
        .padding(ExtensionPaneLayout.contentInsets)
        .frame(maxWidth: .infinity, alignment: .leading)
        .onReceive(refresh) { _ in revision += 1 }
    }

    private var addRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                TextField(
                    text: $draft,
                    prompt: Text("What should run when the chat is idle?", bundle: .module)
                ) {
                    Text("What should run when the chat is idle?", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
                .onSubmit(add)
                Button(action: add) {
                    Text("Add to Queue", bundle: .module)
                }
                .appAction(.primary)
                .disabled(!canAdd)
                .help(selectedChatIsTemporary
                    ? Text("Temporary chats are never saved, so they cannot hold off-peak tasks.", bundle: .module)
                    : Text("Adds the prompt to the queue for the open chat", bundle: .module))
            }
            if addFailed {
                Text("Could not add the task. The queue may be full or the prompt too long.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(AppAttention.color)
            }
        }
    }

    private func add() {
        guard canAdd else { return }
        do {
            _ = try OffPeakTaskStore.shared.enqueue(
                prompt: draft, chatID: model.selectedChatID, origin: .user)
            draft = ""
            addFailed = false
        } catch {
            addFailed = true
        }
        revision += 1
    }

    private func stateGlyph(_ state: OffPeakTaskState) -> String {
        switch state {
        case .queued: return "clock"
        case .running: return "play.circle.fill"
        case .completed: return "checkmark.circle"
        case .interrupted: return "exclamationmark.triangle"
        case .cancelled: return "xmark.circle"
        case .skipped: return "minus.circle"
        }
    }

    private func stateText(_ state: OffPeakTaskState) -> Text {
        switch state {
        case .queued: return Text("Queued", bundle: .module)
        case .running: return Text("Running", bundle: .module)
        case .completed: return Text("Completed", bundle: .module)
        case .interrupted: return Text("Interrupted", bundle: .module)
        case .cancelled: return Text("Cancelled", bundle: .module)
        case .skipped: return Text("Skipped", bundle: .module)
        }
    }

    /// Why the last run ended, for the outcomes a person can act on.
    private func outcomeText(_ outcome: String?) -> Text? {
        switch outcome {
        case "approval_needed": return Text("Stopped: a tool needed your approval", bundle: .module)
        case "app_quit": return Text("Stopped when the app quit", bundle: .module)
        case "crash_recovered": return Text("Stopped unexpectedly", bundle: .module)
        case "stopped": return Text("Stopped by you", bundle: .module)
        case "chat_missing": return Text("Its chat was deleted", bundle: .module)
        default: return nil
        }
    }

    private func row(_ task: AppOffPeakTask) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: stateGlyph(task.state))
                    .themedFont(.small)
                    .foregroundStyle(task.state == .interrupted
                        ? AnyShapeStyle(AppAttention.color) : AnyShapeStyle(.appSecondary))
                    .accessibilityHidden(true)
                stateText(task.state)
                    .themedFont(.small, weight: .semibold)
                if task.origin == .model {
                    Text("Added by the assistant", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                rowActions(task)
            }
            Text(task.prompt)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .lineLimit(2)
            if let outcome = outcomeText(task.runs.first?.outcome) {
                outcome
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
        }
        .appCardSurface(padding: 10)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private func rowActions(_ task: AppOffPeakTask) -> some View {
        switch task.state {
        case .queued:
            AppIconButton("xmark.circle", label: Text("Cancel this task", bundle: .module)) {
                try? OffPeakTaskStore.shared.cancel(id: task.id)
                revision += 1
            }
            .appAction(.quiet, size: .small)
        case .interrupted:
            AppIconButton("arrow.clockwise", label: Text("Run this task again", bundle: .module)) {
                try? OffPeakTaskStore.shared.requeue(id: task.id)
                revision += 1
            }
            .appAction(.quiet, size: .small)
            removeButton(task)
        case .completed, .cancelled, .skipped:
            removeButton(task)
        case .running:
            EmptyView()
        }
    }

    /// Quiet, not destructive: removing a finished row from the history
    /// loses no work the user cannot redo.
    private func removeButton(_ task: AppOffPeakTask) -> some View {
        AppIconButton("minus.circle", label: Text("Remove this task from the list", bundle: .module)) {
            _ = OffPeakTaskStore.shared.remove(id: task.id)
            revision += 1
        }
        .appAction(.quiet, size: .small)
    }
}
