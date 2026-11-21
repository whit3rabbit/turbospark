import SwiftUI

/// The "Context" group of a reply's sources popover: what conversation
/// recall brought back for the turn and whether it reached the prompt.
/// Read-only and in memory only (`RecallInspection`).
///
/// Rows show a RANK, not the score: syntext's scores are unnormalized, so a
/// bare number or a bar would suggest a scale that does not exist. The raw
/// value stays in the row's help for debugging.
struct RecallInspectionGroupView: View {
    let inspection: RecallInspection

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Context", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
                .accessibilityAddTraits(.isHeader)
            summary
            ForEach(inspection.rankedBlocks, id: \.block.id) { entry in
                RecallInspectionRowView(rank: entry.rank, block: entry.block)
            }
        }
    }

    @ViewBuilder
    private var summary: some View {
        switch inspection.summary {
        case .counts(let retrieved, let used, let trimmed):
            VStack(alignment: .leading, spacing: 2) {
                Text("Retrieved: \(retrieved), used: \(used), trimmed: \(trimmed)", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appText)
                if let retrieval = inspection.retrieval {
                    Text(
                        "Evidence: \(retrieval.evidenceTokens) of \(retrieval.budget) tokens",
                        bundle: .module)
                        .themedFont(.tiny)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
            }
            .fixedSize(horizontal: false, vertical: true)
        case .skipped(let reason):
            Text(Self.skippedText(reason))
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    static func skippedText(_ reason: RecallInspection.DropReason) -> String {
        switch reason {
        case .noRoom:
            return AppStrings.localized("Recall skipped: not enough room in the context window")
        case .nothingFound:
            return AppStrings.localized("Recall skipped: nothing relevant was found")
        case .unavailable:
            return AppStrings.localized("Recall skipped: the recall store was unavailable")
        case .overBudget:
            return AppStrings.localized("Recall skipped: the evidence would not fit")
        }
    }
}

private struct RecallInspectionRowView: View {
    let rank: Int
    let block: RecallInspection.Block

    /// syntext's passage kinds; an unknown kind shows as syntext wrote it.
    private var kindLabel: String {
        switch block.kind {
        case "turn": return AppStrings.localized("Earlier turn")
        case "prose": return AppStrings.localized("Prose")
        case "code": return AppStrings.localized("Code")
        case "definition": return AppStrings.localized("Definition")
        case "data": return AppStrings.localized("Data")
        default: return block.kind
        }
    }

    private var stateLabel: String {
        block.condensed ? AppStrings.localized("Trimmed") : AppStrings.localized("Included")
    }

    private var scoreText: String { String(format: "%.3f", block.score) }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(verbatim: "\(rank)")
                .themedFont(.small, weight: .semibold)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)
                .frame(minWidth: 16, alignment: .trailing)
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: block.source)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appText)
                    .lineLimit(2)
                    .truncationMode(.middle)
                Text(verbatim: block.breadcrumb.isEmpty ? kindLabel : "\(kindLabel) - \(block.breadcrumb)")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer(minLength: 0)
            // Glyph plus word, never color alone (StatusLabel guide).
            HStack(spacing: 4) {
                Image(systemName: block.condensed ? "scissors" : "checkmark.circle")
                    .foregroundStyle(block.condensed ? AppThemeColor.appSecondary : AppThemeColor.appSuccess)
                    .accessibilityHidden(true)
                Text(verbatim: stateLabel)
                    .foregroundStyle(.appSecondary)
            }
            .themedFont(.tiny, weight: .medium)
        }
        .padding(.vertical, 2)
        .help(Text("Relevance score \(scoreText) (unnormalized; only the order is meaningful)", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: "\(AppStrings.localized("Rank \(rank)")), \(block.source), \(kindLabel), \(stateLabel)"))
    }
}
