import SwiftUI

/// Transcript and composer share one measure so cards never drift from the text.
enum ConversationLayout {
    static let readingWidth: CGFloat = 760
    static let horizontalInset: CGFloat = 24
    static let activityHeight: CGFloat = 36
    static let activityIconWidth: CGFloat = 20
    static var cardRadius: CGFloat { AppMetrics.cardRadius }
}

private struct ConversationColumnModifier: ViewModifier {
    @Environment(\.appTheme) private var theme

    func body(content: Content) -> some View {
        content
            .frame(maxWidth: (ConversationLayout.readingWidth * AppMetrics.textScale(theme)).rounded())
            .padding(.horizontal, ConversationLayout.horizontalInset)
            .frame(maxWidth: .infinity)
    }
}

extension View {
    func conversationColumn() -> some View {
        modifier(ConversationColumnModifier())
    }
}

struct ConversationDivider: View {
    var body: some View {
        Rectangle().fill(.appBorder).frame(height: 1)
            .accessibilityHidden(true)
    }
}
