import SwiftUI

/// Transcript and composer share one measure so cards never drift from the text.
enum ConversationLayout {
    static let readingWidth: CGFloat = 760
    static let horizontalInset: CGFloat = 24
    static let activityHeight: CGFloat = 36
    static let activityIconWidth: CGFloat = 20
    static let cardRadius: CGFloat = 10
}

extension View {
    func conversationColumn() -> some View {
        frame(maxWidth: ConversationLayout.readingWidth)
            .padding(.horizontal, ConversationLayout.horizontalInset)
            .frame(maxWidth: .infinity)
    }
}

struct ConversationDivider: View {
    var body: some View {
        Rectangle().fill(.appBorder).frame(height: 1)
            .accessibilityHidden(true)
    }
}
