import SwiftUI

/// The Scheduled Tasks pane (qwen-code scheduled-tasks port): every cron
/// job the model created through `CronCreate`, with its schedule, the
/// prompt it fires, next run, run history, and pause/delete controls.
///
/// Jobs are created BY THE MODEL (`CronCreate` tool) or by asking for one;
/// this pane is the human surface over the same store, which is why there
/// is no create form: describing a schedule in prose to the assistant is
/// the intended path and re-implementing it here would fork the validation.
struct CronJobsSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    /// Bumped to re-read the store after a mutation; `CronScheduler` is not
    /// observable, and a refresh button is honest about that.
    @State private var revision = 0
    @State private var expandedJobID: String? = nil
    @State private var jobPendingDeletion: AppCronJob? = nil

    private var jobs: [AppCronJob] {
        _ = revision
        return CronScheduler.shared.list()
    }

    var body: some View {
        Group {
            if jobs.isEmpty {
                emptyState
            } else {
                jobList
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .alert(
            "Delete scheduled task?",
            isPresented: deletionAlertPresented,
            presenting: jobPendingDeletion
        ) { job in
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
            Button(role: .destructive) {
                _ = CronScheduler.shared.delete(id: job.id)
                revision += 1
            } label: { Text("Delete", bundle: .module) }
        } message: { job in
            Text("The schedule \"\(job.cron)\" will stop firing.", bundle: .module)
        }
    }

    private var deletionAlertPresented: Binding<Bool> {
        Binding(
            get: { jobPendingDeletion != nil },
            set: { if !$0 { jobPendingDeletion = nil } })
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "clock.badge.questionmark")
                .font(theme.ui(.title2))
                .foregroundStyle(.tertiary)
                .padding(.bottom, 2)
                .accessibilityHidden(true)
            Text("No scheduled tasks", bundle: .module)
                .themedFont(.base, weight: .medium)
            Text("Ask the assistant to schedule one, for example: \"every morning at 9, summarize my notes\" -- it will use CronCreate and the job appears here.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        .padding(24)
    }

    private var jobList: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Scheduled tasks: \(jobs.count)", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Button {
                    revision += 1
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.plain)
                .help(Text("Refresh", bundle: .module))
                .accessibilityLabel("Refresh scheduled tasks")
            }

            ForEach(jobs) { job in
                jobRow(job)
            }

            Text("Tasks fire while the app is running and submit their prompt into the chat they were created in. One-shot tasks remove themselves after firing.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.tertiary)
        }
        .padding(16)
    }

    /// Composed OUTSIDE `Text` so the call stays a catalog-key call; these
    /// are status readouts, not translatable prose yet.
    private func runSummary(of job: AppCronJob) -> String {
        if job.recentRuns.isEmpty { return String(localized: "No runs yet", bundle: .module) }
        let relative = job.recentRuns[0].date.formatted(.relative(presentation: .named))
        return String(
            localized: "Last run \(relative) (\(job.recentRuns.count) recorded)",
            bundle: .module)
    }

    private func jobRow(_ job: AppCronJob) -> some View {
        let schedule = CronSchedule(job.cron)?.describe() ?? job.cron
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: job.enabled ? "clock.fill" : "clock")
                    .themedFont(.small)
                    .foregroundStyle(job.enabled ? TurboSparkTheme.accentColor : Color.secondary)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 1) {
                    Text(schedule)
                        .themedFont(.small, weight: .semibold)
                    Text(job.cron)
                        .themedCode(.tiny)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                if job.enabled, let next = job.nextFireAt {
                    VStack(alignment: .trailing, spacing: 1) {
                        Text(next, style: .relative)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .monospacedDigit()
                        Text("next run", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                    }
                } else if !job.enabled {
                    Text("Paused", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Toggle("", isOn: Binding(
                    get: { job.enabled },
                    set: { enabled in
                        _ = CronScheduler.shared.setEnabled(id: job.id, enabled: enabled)
                        revision += 1
                    }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
                .controlSize(.small)
                .help(job.enabled ? "Pause this schedule" : "Resume this schedule")
                .accessibilityLabel("\(job.enabled ? "Pause" : "Resume") schedule \(schedule)")
                Button {
                    jobPendingDeletion = job
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(Color.red.opacity(0.75))
                }
                .buttonStyle(.plain)
                .help("Delete scheduled task")
                .accessibilityLabel("Delete scheduled task \(schedule)")
            }

            Text(job.prompt)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .lineLimit(2)

            Button {
                expandedJobID = expandedJobID == job.id ? nil : job.id
            } label: {
                Text(verbatim: runSummary(of: job))
                    .themedFont(.tiny)
                    .foregroundStyle(job.recentRuns.contains(where: { $0.status == "missed" })
                        ? Color.orange : Color.secondary)
            }
            .buttonStyle(.plain)
            .help("Show run history")
            .accessibilityLabel("Run history for \(schedule)")

            if expandedJobID == job.id {
                VStack(alignment: .leading, spacing: 3) {
                    if job.recentRuns.isEmpty {
                        Text("This schedule has not fired yet.", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                    }
                    ForEach(job.recentRuns.indices, id: \.self) { index in
                        let run = job.recentRuns[index]
                        HStack(spacing: 6) {
                            Image(systemName: run.status == "fired" ? "checkmark.circle" : "xmark.circle")
                                .themedFont(.tiny)
                                .foregroundStyle(run.status == "fired" ? Color.green : Color.orange)
                                .accessibilityHidden(true)
                            Text(run.date.formatted(date: .abbreviated, time: .shortened))
                                .themedFont(.tiny)
                            Text(run.status == "fired"
                                ? String(localized: "delivered", bundle: .module)
                                : String(localized: "chat missing", bundle: .module))
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.03))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.04))
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}
