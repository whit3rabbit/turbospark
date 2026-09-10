import Charts
import SwiftUI

extension InspectorView {
    /// Cross-chat token usage, the qwen-code usage-dashboard port: recorded
    /// totals, a GitHub-style day heatmap over the last 12 weeks, a 30-day
    /// output chart, and the chats that used the most. Renders RECORDED
    /// usage only -- chats from before the ledger existed contribute
    /// nothing rather than an estimate.
    var usageDashboardSection: some View {
        UsageDashboardSectionView(model: model)
    }
}

/// Its own struct rather than an extension computed property: the section
/// reads a per-day series it formats once, and holding the derivation in a
/// body-local lets the heatmap and the chart share it.
private struct UsageDashboardSectionView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        let totals = model.usageTotals
        Section("Token Usage (recorded)") {
            if totals.turns == 0 {
                Text("No recorded usage yet. Tokens are counted from the model's own per-turn totals as turns complete.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                LabeledContent("Turns") {
                    Text(totals.turns.formatted())
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
                LabeledContent("Prompt tokens") {
                    Text(totals.promptTokens.formatted())
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
                LabeledContent("Output tokens") {
                    Text(totals.outputTokens.formatted())
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }

                heatmap(totals: totals)

                dailyChart(totals: totals)

                topChats
            }
        }
    }

    /// 12-week grid, one column per week, top row Sunday. Intensity is
    /// total tokens that day relative to the busiest day in the window.
    private func heatmap(totals: AppChatUsageTotals) -> some View {
        let series = totals.dailySeries(endingAt: Date(), days: 12 * 7)
        let maxDay = series.map { $0.day.promptTokens + $0.day.outputTokens }.max() ?? 0
        // Pad so the first cell lands on its weekday row, then chunk into
        // week columns.
        let leadingWeekday = Self.weekday(of: series.first?.key)
        let padded = Array(repeating: nil as (key: String, day: AppChatUsageDay)?, count: leadingWeekday)
            + series.map { Optional($0) }
        let weeks = stride(from: 0, to: padded.count, by: 7).map { Array(padded[$0 ..< min($0 + 7, padded.count)]) }

        return VStack(alignment: .leading, spacing: 6) {
            Text("Last 12 weeks", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
            HStack(alignment: .top, spacing: 3) {
                ForEach(weeks.indices, id: \.self) { weekIndex in
                    VStack(spacing: 3) {
                        ForEach(0..<7, id: \.self) { dayIndex in
                            if dayIndex < weeks[weekIndex].count,
                                let cell = weeks[weekIndex][dayIndex] {
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Self.heatColor(
                                        tokens: cell.day.promptTokens + cell.day.outputTokens,
                                        maxTokens: maxDay))
                                    .frame(width: 11, height: 11)
                                    .help("\(cell.key): \((cell.day.promptTokens + cell.day.outputTokens).formatted()) tokens")
                            } else {
                                Color.clear
                                    .frame(width: 11, height: 11)
                            }
                        }
                    }
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Token usage heatmap for the last 12 weeks")
        }
        .padding(.vertical, 4)
    }

    private func dailyChart(totals: AppChatUsageTotals) -> some View {
        let series = totals.dailySeries(endingAt: Date(), days: 30).compactMap { entry -> (date: Date, tokens: Int)? in
            guard let date = DateFormatter.utcDay.date(from: entry.key) else { return nil }
            return (date, entry.day.outputTokens)
        }
        return VStack(alignment: .leading, spacing: 6) {
            Text("Output tokens, last 30 days", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
            Chart(series, id: \.date) { entry in
                BarMark(
                    x: .value("Day", entry.date, unit: .day),
                    y: .value("Tokens", entry.tokens)
                )
                .foregroundStyle(.appAccent.opacity(0.75))
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: 7)) {
                    AxisValueLabel(format: .dateTime.day().month(.abbreviated))
                }
            }
            .frame(height: 90)
        }
        .padding(.vertical, 4)
    }

    private var topChats: some View {
        let rows = model.chats
            .filter { !$0.isGhost && $0.usage != nil }
            .sorted {
                ($0.usage!.totalPromptTokens + $0.usage!.totalOutputTokens)
                    > ($1.usage!.totalPromptTokens + $1.usage!.totalOutputTokens)
            }
            .prefix(5)
        return VStack(alignment: .leading, spacing: 6) {
            Text("Top chats", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
            ForEach(Array(rows)) { chat in
                Button {
                    model.selectChat(id: chat.id)
                } label: {
                    HStack {
                        Text(chat.title)
                            .themedFont(.small)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Text((chat.usage!.totalPromptTokens + chat.usage!.totalOutputTokens)
                            .formatted(.number.notation(.compactName)))
                            .themedFont(.small).monospacedDigit()
                            .foregroundStyle(.appSecondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Open \(chat.title)")
                .accessibilityLabel("Open chat \(chat.title), \(chat.usage!.totalPromptTokens + chat.usage!.totalOutputTokens) tokens recorded")
            }
        }
        .padding(.vertical, 4)
    }

    /// 0 (Sunday) through 6 for a "yyyy-MM-dd" key, so the heatmap's first
    /// column starts on the right row. Unparseable keys land on Monday.
    static func weekday(of dayKey: String?) -> Int {
        guard let dayKey, let date = DateFormatter.utcDay.date(from: dayKey) else { return 1 }
        return Calendar(identifier: .gregorian).component(.weekday, from: date) - 1
    }

    static func heatColor(tokens: Int, maxTokens: Int) -> Color {
        guard maxTokens > 0, tokens > 0 else { return Color.primary.opacity(0.05) }
        let fraction = Double(min(tokens, maxTokens)) / Double(maxTokens)
        return TurboSparkTheme.accentColor.opacity(0.2 + 0.8 * fraction)
    }
}

private extension DateFormatter {
    static let utcDay: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter
    }()
}
