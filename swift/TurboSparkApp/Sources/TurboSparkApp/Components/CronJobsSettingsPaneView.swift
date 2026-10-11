import SwiftUI

/// The Scheduled Tasks pane (qwen-code scheduled-tasks port): every cron
/// job the model created through `CronCreate`, with its schedule, the
/// prompt it fires, next run, run history, and pause/delete controls.
///
/// Jobs are created BY THE MODEL (`CronCreate` tool) or by asking for one;
/// this pane is the human surface over the same store, which is why there
/// is no create form: describing a schedule in prose to the assistant is
/// the intended path and re-implementing it here would fork the validation.
/// "Schedule a Task..." makes that path visible: it opens the chat with an
/// editable request the user finishes and sends (`AppModel.composeInChat`),
/// so nothing reaches the model until they do. No copy here names the tool:
/// a tool identifier means nothing to someone who has never seen one.
@MainActor
struct CronJobsSettingsPaneView: View {
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
                VStack(spacing: 0) {
                    header
                    jobList
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .alert(
            Text("Delete scheduled task?", bundle: .module),
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

    /// Seeds the chat composer with a scheduling request to complete. The
    /// draft is in the UI language: it is the user's own message, edited
    /// and sent by them, and the assistant reads any language.
    private func scheduleInChat() {
        model.composeInChat(AppStrings.localized("Schedule a task for me. When: every weekday at 9 AM. What to do: "))
    }

    private var scheduleButton: some View {
        Button(action: scheduleInChat) {
            Label { Text("Schedule a Task...", bundle: .module) } icon: { Image(systemName: "calendar.badge.plus") }
        }
        .appAction(.primary)
        .help(Text("Opens a chat with a scheduling request for you to finish and send", bundle: .module))
    }

    private var emptyState: some View {
        AppEmptyState(
            systemImage: "clock.badge.questionmark",
            title: Text("No scheduled tasks", bundle: .module),
            message: Text("A scheduled task runs a prompt for you on a timetable, such as every weekday morning. Describe what you want done and when in a chat, and the assistant sets it up. Tasks appear here so you can pause or delete them.", bundle: .module)
        ) {
            scheduleButton
        }
    }

    private var header: some View {
        ExtensionPaneHeader {
            ExtensionPaneIntro(text: Text("Tasks fire while the app is running and submit their prompt into the chat they were created in. One-shot tasks remove themselves after firing.", bundle: .module))
        } filters: {
            Text("Scheduled tasks: \(jobs.count)", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
        } actions: {
            ExtensionRefreshButton(label: Text("Refresh", bundle: .module)) { revision += 1 }
            scheduleButton
        }
    }

    private var jobList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(jobs) { job in
                    jobRow(job)
                }
            }
            .padding(ExtensionPaneLayout.contentInsets)
        }
    }

    /// Composed OUTSIDE `Text` so the call stays a catalog-key call; these
    /// are status readouts. `AppStrings` so they follow the in-app language.
    private func runSummary(of job: AppCronJob) -> String {
        if job.recentRuns.isEmpty { return AppStrings.localized("No runs yet") }
        let relative = job.recentRuns[0].date.formatted(.relative(presentation: .named))
        return AppStrings.localized("Last run \(relative). Runs recorded: \(job.recentRuns.count)")
    }

    private func jobRow(_ job: AppCronJob) -> some View {
        // A one-shot wakeup's cron field is a placeholder; only
        // `localizedHumanSchedule` knows to read `fireAt` instead.
        let schedule = job.localizedHumanSchedule
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
                // Named by the schedule it controls; on/off is its value.
                Toggle(isOn: Binding(
                    get: { job.enabled },
                    set: { enabled in
                        _ = CronScheduler.shared.setEnabled(id: job.id, enabled: enabled)
                        revision += 1
                    }
                )) {
                    Text(schedule)
                }
                .toggleStyle(.switch)
                .labelsHidden()
                .controlSize(.small)
                .help(job.enabled
                    ? Text("Pause this schedule", bundle: .module)
                    : Text("Resume this schedule", bundle: .module))
                Button {
                    jobPendingDeletion = job
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(Color.red.opacity(0.75))
                }
                .buttonStyle(.plain)
                .help(Text("Delete scheduled task", bundle: .module))
                .accessibilityLabel(Text("Delete scheduled task \(schedule)", bundle: .module))
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
            .help(Text("Show run history", bundle: .module))
            .accessibilityLabel(Text("Run history for \(schedule)", bundle: .module))

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
                            Group {
                                if run.status == "fired" {
                                    Text("delivered", bundle: .module)
                                } else {
                                    Text("chat missing", bundle: .module)
                                }
                            }
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
