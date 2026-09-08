import SwiftUI

/// The active-goal banner in the transcript: the `/goal active` indicator,
/// reduced to this app's chrome. Reads `AppModel.activeGoals[selectedChatID]`,
/// which `AppModel+Goal.updateGoal` keeps published, so elapsed time and the
/// latest evaluator reason refresh with every state write.
///
/// Deliberately OUTSIDE the streaming row, like `TaskChecklistPanelView` and
/// the background strips: a goal must stay visible while its turns run and
/// after they end, because the banner is also the only stop button that does
/// not require typing `/goal clear`.
struct GoalBannerView: View {
    @ObservedObject var model: AppModel
    @State private var now = Date()

    var body: some View {
        if let goal = model.activeGoals[model.selectedChatID] {
            banner(goal)
        }
    }

    @ViewBuilder
    private func banner(_ goal: ChatGoalState) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: goal.isPaused ? "pause.circle" : "circle.dotted")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(TurboSparkTheme.accentColor)
                    .accessibilityHidden(true)
                Text("Goal active", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Text(AppModel.formatGoalElapsed(since: goal.setAt, now: now))
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    model.clearGoal(chatID: model.selectedChatID)
                } label: {
                    Text("Stop goal", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
                }
                .buttonStyle(.link)
                .accessibilityLabel(Text("Stop goal", bundle: .module))
            }
            Text(GoalPolicy.conditionPreview(goal.condition))
                .themedFont(.small)
                .textSelection(.enabled)
            HStack(spacing: 12) {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .accessibilityHidden(true)
                    Text("Iterations", bundle: .module)
                    Text(verbatim: "\(goal.iterations)")
                }
                .themedFont(.tiny)
                .foregroundStyle(.secondary)

                if model.isEvaluatingGoal {
                    HStack(spacing: 4) {
                        ProgressView()
                            .controlSize(.mini)
                        Text("Evaluating", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.secondary)
                    }
                }
                if goal.deferredSince != nil {
                    Label {
                        Text("Waiting for background work", bundle: .module)
                    } icon: {
                        Image(systemName: "clock")
                            .accessibilityHidden(true)
                    }
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
                }
                if goal.isPaused {
                    Label {
                        Text("Paused", bundle: .module)
                    } icon: {
                        Image(systemName: "pause")
                            .accessibilityHidden(true)
                    }
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
                }
                Spacer()
            }
            if let reason = goal.lastReason, !reason.isEmpty {
                Text(reason)
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .textSelection(.enabled)
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
        .accessibilityElement(children: .combine)
        .onAppear { now = Date() }
        .onChange(of: model.activeGoals) { now = Date() }
    }
}
