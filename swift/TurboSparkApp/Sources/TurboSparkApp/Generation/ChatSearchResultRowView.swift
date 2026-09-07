import SwiftUI

/// One result row of the "Search Chats" dialog: title, snippet with the
/// matched tokens in the accent color, a match count, and a relative date
/// bucket. The selection highlight mirrors `ChatSidebarChatRowView`.
@MainActor
struct ChatSearchResultRowView: View {
    @Environment(\.appTheme) private var theme
    let hit: ChatSearch.Hit
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "bubble.left")
                .font(theme.ui(points: 11))
                .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(hit.title)
                    .font(theme.ui(points: 12, weight: isSelected ? .semibold : .medium))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                if let snippet = hit.snippet {
                    snippetText(snippet)
                        .font(theme.ui(points: 11))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 0)
            VStack(alignment: .trailing, spacing: 2) {
                if hit.matchCount > 0 {
                    Text(hit.matchCount == 1 ? "1 match" : "\(hit.matchCount) matches")
                        .font(theme.ui(points: 10, weight: .medium))
                        .foregroundStyle(TurboSparkTheme.accentColor)
                }
                Text(ChatSearch.dateBucket(hit.updatedAt))
                    .font(theme.ui(points: 10))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.leading, 9)
        .padding(.vertical, 7)
        .padding(.trailing, 9)
        .contentShape(.rect)
    }

    /// Folds the snippet's pre-cut segments into one `Text`, coloring the
    /// matched tokens. The engine did the searching; this only renders it.
    private func snippetText(_ snippet: ChatSearch.Snippet) -> Text {
        var text = Text("")
        for segment in snippet.segments {
            let piece = Text(segment.text)
            text = text + (segment.isMatch
                ? piece.foregroundColor(TurboSparkTheme.accentColor).fontWeight(.semibold)
                : piece)
        }
        return text
    }
}
