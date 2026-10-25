import AppKit
import SwiftUI
import TurboSpark

/// The line marking where compaction summarized the transcript's prefix.
///
/// The rows below it are exactly the ones the prompt no longer carries; the
/// summary that replaced them expands in place. A disclosure rather than a
/// toast because a summary a user can never read is a summary they cannot
/// correct.
struct CompactionDividerView: View {
    @Environment(\.appTheme) private var theme
    let summary: String
    let summarizedRows: Int

    @State private var isExpanded: Bool = false

    var body: some View {
        DisclosureGroup(
            isExpanded: $isExpanded
        ) {
            Text(summary)
                .font(theme.code(.small))
                .foregroundStyle(theme.metadataForeground)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color.primary.opacity(0.03))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .textSelection(.enabled)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "rectangle.compress.vertical")
                    .themedFont(.tiny)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Group {
                    if summarizedRows == 1 {
                        Text("Earlier message compacted into a summary", bundle: .module)
                    } else {
                        Text("Earlier messages compacted into a summary: \(summarizedRows)", bundle: .module)
                    }
                }
                .themedFont(.small, weight: .medium)
                .foregroundStyle(theme.metadataForeground)
            }
        }
        .padding(.vertical, 2)
        .accessibilityLabel(Text("Earlier conversation compacted into a summary", bundle: .module))
        .accessibilityHint(Text("Expands to show the summary the model sees instead of those messages", bundle: .module))
    }
}
