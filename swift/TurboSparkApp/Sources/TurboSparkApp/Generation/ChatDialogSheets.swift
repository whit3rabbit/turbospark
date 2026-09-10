import SwiftUI

/// The `/stats` sheet: session statistics for the selected chat, computed
/// by `AppChatStats` and rendered as plain rows. Presented from
/// `OutputPaneView` while `model.showSessionStats` is true.
struct SessionStatsSheet: View {
    @Environment(\.appTheme) private var theme
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel

    var body: some View {
        let stats = AppChatStats(chat: model.selectedChat, modelAlias: model.selected?.alias)
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "chart.bar")
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Session Stats", bundle: .module)
                    .themedFont(.callout, weight: .semibold)
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Close")
                .accessibilityLabel("Close session stats")
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    statsGroup("Conversation", bundle: .module) {
                        statsRow("Title", stats.title)
                        statsRow("Created", stats.createdAt.formatted(date: .abbreviated, time: .shortened))
                        statsRow("Last activity", stats.updatedAt.formatted(date: .abbreviated, time: .shortened))
                        statsRow("Active duration", stats.durationText)
                        if let alias = stats.modelAlias {
                            statsRow("Model", alias)
                        }
                    }
                    statsGroup("Turns", bundle: .module) {
                        statsRow("Your prompts", String(stats.userTurns))
                        statsRow("Assistant replies", String(stats.assistantTurns))
                        statsRow("Tool calls", String(stats.toolCalls))
                        if stats.failedToolCalls > 0 {
                            statsRow("Failed tool calls", String(stats.failedToolCalls))
                        }
                        statsRow("Images attached", String(stats.images))
                        if stats.todoTotal > 0 {
                            statsRow(
                                "Checklist",
                                "\(stats.todoCompleted)/\(stats.todoTotal) done")
                        }
                    }
                    statsGroup("Context", bundle: .module) {
                        if stats.compactedMessages > 0 {
                            statsRow("Compacted messages", String(stats.compactedMessages))
                        }
                        statsRow(
                            "Compaction summary",
                            stats.hasSummary ? "Present" : "None")
                        if let summary = model.contextUsageSummary {
                            statsRow(
                                "Context window",
                                "\(summary.usedTokens.formatted(.number.notation(.compactName))) / "
                                    + "\(summary.windowTokens.formatted(.number.notation(.compactName))) tokens")
                        }
                    }
                    statsGroup("Token usage", bundle: .module) {
                        if let usage = stats.usage {
                            statsRow("Prompt tokens", usage.totalPromptTokens.formatted())
                            statsRow("Output tokens", usage.totalOutputTokens.formatted())
                            statsRow("Recorded turns", String(usage.turns))
                        } else {
                            Text("No recorded usage yet. Tokens are counted from the model's own per-turn totals as turns complete.", bundle: .module)
                                .themedFont(.small)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: 460, height: 480)
    }

    private func statsGroup(
        _ title: LocalizedStringKey, bundle: Bundle, @ViewBuilder rows: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title, bundle: bundle)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
            VStack(alignment: .leading, spacing: 4) {
                rows()
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.03))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private func statsRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Spacer(minLength: 12)
            Text(value)
                .themedFont(.small, weight: .medium)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
    }
}

/// The `/help` sheet: every slash command the composer offers plus the
/// fixed keyboard shortcuts. Presented from `OutputPaneView` while
/// `model.showHelpSheet` is true.
struct HelpSheetView: View {
    @Environment(\.appTheme) private var theme
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "questionmark.circle")
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Help", bundle: .module)
                    .themedFont(.callout, weight: .semibold)
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Close")
                .accessibilityLabel("Close help")
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    helpSection("Slash commands", bundle: .module) {
                        ForEach(BuiltInSlashCommand.all) { command in
                            helpRow(
                                "/\(command.name)",
                                command.summary,
                                mono: true)
                        }
                        ForEach(model.userSkills.filter { $0.isEnabled }) { skill in
                            helpRow("/\(skill.name)", skill.skillDescription, mono: true)
                        }
                    }
                    helpSection("Keyboard shortcuts", bundle: .module) {
                        ForEach(KeyboardShortcutCatalog.sections) { section in
                            Text(section.title)
                                .themedFont(.small, weight: .semibold)
                                .foregroundStyle(.appSecondary)
                                .padding(.top, 6)
                            ForEach(section.rows) { row in
                                HStack(alignment: .firstTextBaseline) {
                                    Text(row.label)
                                        .themedFont(.small)
                                    Spacer(minLength: 12)
                                    Text(row.keys)
                                        .themedCode(.small)
                                        .foregroundStyle(.appSecondary)
                                }
                            }
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: 520, height: 560)
    }

    private func helpSection(
        _ title: LocalizedStringKey, bundle: Bundle, @ViewBuilder rows: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title, bundle: bundle)
                .themedFont(.callout, weight: .semibold)
            VStack(alignment: .leading, spacing: 5) {
                rows()
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.03))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private func helpRow(_ name: String, _ detail: String, mono: Bool) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(name)
                .font(mono ? theme.code(.small) : theme.ui(.small, weight: .medium))
                .foregroundStyle(.appAccent)
            Spacer(minLength: 12)
            Text(detail)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.trailing)
        }
    }
}
