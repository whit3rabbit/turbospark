import SwiftUI

/// Per-change review under the rewrite: each change can be kept (the
/// rewrite's wording) or rejected (the draft's wording), and Copy sends
/// `HumanizeDiff.compose` of the choices. Display only: nothing here reaches
/// the model or changes `controller.rewrite`.
struct HumanizeChangeReview: View {
    let source: String
    let rewrite: String
    let changes: [HumanizeChange]
    @Binding var rejected: Set<Int>
    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("Changes", bundle: .module)
                    .font(theme.ui(.small, weight: .semibold))
                    .accessibilityAddTraits(.isHeader)
                Spacer(minLength: 8)
                Button { rejected = [] } label: { Text("Accept All", bundle: .module) }
                    .appAction(.quiet, size: .small)
                    .disabled(rejected.isEmpty)
                Button { rejected = Set(changes.indices) } label: { Text("Reject All", bundle: .module) }
                    .appAction(.quiet, size: .small)
                    .disabled(rejected.count == changes.count)
            }
            LazyVStack(alignment: .leading, spacing: 6) {
                ForEach(changes.indices, id: \.self) { index in
                    row(index)
                }
            }
        }
    }

    private func row(_ index: Int) -> some View {
        let accepted = !rejected.contains(index)
        return HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text("\(index + 1) of \(changes.count)", bundle: .module)
                .font(theme.ui(.tiny)).monospacedDigit().foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            excerpt(changes[index])
                .font(theme.ui(.small))
                .lineLimit(3)
                .frame(maxWidth: .infinity, alignment: .leading)
            // The state is a word and a glyph, never only a tint, so it reads
            // under Differentiate Without Color and to VoiceOver alike.
            Button {
                if accepted { rejected.insert(index) } else { rejected.remove(index) }
            } label: {
                Label {
                    accepted ? Text("Accepted", bundle: .module) : Text("Rejected", bundle: .module)
                } icon: {
                    Image(systemName: accepted ? "checkmark.circle.fill" : "arrow.uturn.backward.circle")
                }
            }
            .appAction(.quiet, size: .small)
            .fixedSize()
            .help(accepted ? Text("Reject", bundle: .module) : Text("Accept", bundle: .module))
            .accessibilityLabel(Text("Change \(index + 1) of \(changes.count)", bundle: .module))
            .accessibilityValue(accepted ? Text("Accepted", bundle: .module) : Text("Rejected", bundle: .module))
        }
    }

    /// The draft's wording struck through, then the rewrite's underlined, the
    /// same marks as the two cards. A spacing-only change is named instead.
    private func excerpt(_ change: HumanizeChange) -> Text {
        let before = (source as NSString).substring(with: change.source)
        let after = (rewrite as NSString).substring(with: change.rewrite)
        let blank: (String) -> Bool = { $0.allSatisfy(\.isWhitespace) }
        if blank(before), blank(after) {
            return Text("Spacing", bundle: .module).italic().foregroundColor(theme.secondaryText)
        }
        var text = AttributedString()
        if !blank(before) {
            var cut = AttributedString(Self.oneLine(before))
            cut.foregroundColor = theme.danger.opacity(0.85)
            cut.strikethroughStyle = Text.LineStyle(pattern: .solid, color: theme.danger)
            text.append(cut)
        }
        if !blank(before), !blank(after) { text.append(AttributedString(" \u{2192} ")) }
        if !blank(after) {
            var added = AttributedString(Self.oneLine(after))
            added.underlineStyle = Text.LineStyle(pattern: .solid, color: theme.accent)
            text.append(added)
        }
        return Text(text)
    }

    /// Line breaks inside a change would make the list row as tall as the
    /// passage; the row shows them as spaces.
    static func oneLine(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .newlines).filter { !$0.isEmpty }.joined(separator: " ")
    }
}
