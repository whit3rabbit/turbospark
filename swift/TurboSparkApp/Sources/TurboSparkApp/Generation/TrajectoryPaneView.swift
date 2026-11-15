import SwiftUI
import AppKit

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (DESIGN.md section 8).
/// The per-chat trajectory pane (ZP-03): a role-separated timeline of the
/// model calls and tool runs the chat's local log holds. Read-only. Opens
/// from the chat's context menu as a sheet.
@MainActor
struct TrajectoryPaneView: View {
    @ObservedObject var model: AppModel
    let chatID: UUID
    let onDismiss: () -> Void

    @StateObject private var state: TrajectoryPaneState
    @State private var query = ""
    @State private var confirmingDelete = false
    @Environment(\.appTheme) private var theme

    init(model: AppModel, chatID: UUID, onDismiss: @escaping () -> Void) {
        self.model = model
        self.chatID = chatID
        self.onDismiss = onDismiss
        _state = StateObject(wrappedValue: TrajectoryPaneState(chatID: chatID))
    }

    /// The loaded window, narrowed by the search box; whole-log results
    /// replace it while shown.
    private var visibleRecords: [RolloutRecord] {
        let base = state.searchResults ?? state.records
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty, state.searchResults == nil else { return base }
        return base.filter { $0.entry.matches(needle) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .frame(minWidth: 560, minHeight: 420)
        .task { await state.reload() }
        .confirmationDialog(
            Text("Delete this trajectory log?", bundle: .module),
            isPresented: $confirmingDelete
        ) {
            Button(role: .destructive) {
                Task { await state.deleteLog() }
            } label: { Text("Delete Log", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This permanently deletes the saved log for this chat. The chat itself is not changed.", bundle: .module)
        }
    }

    // MARK: - Header

    private var header: some View {
        AppAdaptiveHeader {
            Label { Text("View Trajectory", bundle: .module) } icon: { Image(systemName: "timeline.selection") }
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
        } actions: {
            searchControls
            AppIconButton("arrow.clockwise", label: Text("Refresh", bundle: .module)) {
                Task { await state.reload() }
            }
        } compactActions: {
            searchControls
        }
        .padding(16)
    }

    @ViewBuilder
    private var searchControls: some View {
        AppSearchField(text: $query, prompt: Text("Search this log", bundle: .module))
        Button {
            Task { await state.searchAll(query) }
        } label: { Text("Search entire log", bundle: .module) }
        .appAction(.secondary, size: .small)
        .disabled(query.trimmingCharacters(in: .whitespaces).isEmpty || state.isLoading)
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if state.records.isEmpty && state.searchResults == nil && !state.isLoading {
            AppEmptyState(
                systemImage: "timeline.selection",
                title: Text("No trajectory log yet", bundle: .module),
                message: model.trajectoryLoggingEnabled
                    ? Text("Model calls and tool runs appear here after they finish.", bundle: .module)
                    : Text("Trajectory logging is off. Turn it on in Settings, General. Model calls and tool runs are then recorded here as they finish.", bundle: .module)
            ) { EmptyView() }
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    notices
                    if state.hasEarlier && state.searchResults == nil {
                        Button {
                            Task { await state.loadEarlier() }
                        } label: { Text("Load earlier entries", bundle: .module) }
                        .appAction(.quiet, size: .small)
                        .disabled(state.isLoading)
                    }
                    ForEach(visibleRecords) { record in
                        TrajectoryEntryRow(entry: record.entry)
                    }
                }
                .padding(16)
            }
        }
    }

    @ViewBuilder
    private var notices: some View {
        if state.searchResults != nil {
            HStack {
                Text(
                    state.searchHitLimit
                        ? AppStrings.localized("Showing the first \(visibleRecords.count) matches.")
                        : AppStrings.localized("Matches in the whole log: \(visibleRecords.count)"))
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                Button { state.clearSearch() } label: { Text("Back to recent entries", bundle: .module) }
                    .appAction(.quiet, size: .small)
            }
        }
        if state.unreadableCount > 0 {
            Label {
                Text(AppStrings.localized("Skipped unreadable lines: \(state.unreadableCount)"))
            } icon: { Image(systemName: "exclamationmark.triangle") }
            .themedFont(.small)
            .foregroundStyle(.appWarning)
            .accessibilityElement(children: .combine)
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            Button(role: .destructive) { confirmingDelete = true } label: {
                Text("Delete Log", bundle: .module)
            }
            .appAction(.destructive, size: .small)
            .disabled(state.records.isEmpty && state.logBytes == 0)
            Button { exportRedacted() } label: { Text("Export Redacted Copy", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(state.records.isEmpty)
            Spacer()
            Button { onDismiss() } label: { Text("Done", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
        }
        .padding(16)
    }

    private func exportRedacted() {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "trajectory.jsonl"
        panel.message = AppStrings.localized("Redaction is best effort. Review the file before sharing it.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do {
                let count = try await state.export(to: url)
                model.showToast(
                    AppStrings.localized("Exported \(count) entries to \(url.lastPathComponent)"),
                    style: .success)
            } catch {
                model.showToast(
                    AppStrings.localized("Export failed: \(error.localizedDescription)"),
                    style: .error)
            }
        }
    }
}

/// One log entry: a summary line, expandable to its arguments and output.
private struct TrajectoryEntryRow: View {
    let entry: RolloutEntry
    @State private var expanded = false

    private var isTool: Bool { entry.kind == .toolExec }
    private var failed: Bool { entry.error != nil || entry.finishReason == "error" }

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            details
        } label: {
            summary
        }
        .padding(10)
        .appCardSurface(padding: 0)
        .accessibilityElement(children: .contain)
    }

    private var summary: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Image(systemName: isTool ? "wrench.and.screwdriver" : "sparkles")
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
                if isTool {
                    Text("Tool run", bundle: .module).themedFont(.small, weight: .semibold)
                } else {
                    Text("Model call", bundle: .module).themedFont(.small, weight: .semibold)
                }
                Text(verbatim: entry.name ?? entry.model ?? "")
                    .themedCode(.small)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if failed {
                    Image(systemName: "exclamationmark.octagon")
                        .foregroundStyle(.appDanger)
                        .accessibilityLabel(Text("Error", bundle: .module))
                }
            }
            Text(metrics)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Tokens, latency and finish reason as one wrapping line.
    private var metrics: String {
        var parts: [String] = [entry.ts.formatted(date: .omitted, time: .standard)]
        if let input = entry.promptTokens, let output = entry.completionTokens {
            parts.append(AppStrings.localized("Tokens: \(input) in, \(output) out"))
        }
        if let ms = entry.latencyMs { parts.append(MetricFormat.seconds(Double(ms) / 1_000)) }
        if let reason = entry.finishReason {
            parts.append(AppStrings.localized("Finish reason: \(reason)"))
        }
        return parts.joined(separator: "  ")
    }

    private var details: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let args = entry.args, !args.isEmpty {
                block(Text("Arguments", bundle: .module),
                      args.keys.sorted().map { "\($0): \(args[$0] ?? "")" }.joined(separator: "\n"))
            }
            if let result = entry.result { block(Text("Output", bundle: .module), result) }
            if let error = entry.error { block(Text("Error", bundle: .module), error) }
            if entry.truncated == true {
                Text("Clipped to the size limit", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(.top, 6)
    }

    private func block(_ title: Text, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            title.themedFont(.small, weight: .semibold).accessibilityAddTraits(.isHeader)
            Text(verbatim: body)
                .themedCode(.small)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
