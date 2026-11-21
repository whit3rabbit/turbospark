import SwiftUI

/// The opt-in chip row under the newest reply (`FollowUpSuggestions`). Each
/// chip drafts its ask into the composer; nothing is sent from here.
struct FollowUpSuggestionsRow: View {
    let draft: (String) -> Void

    var body: some View {
        // One line when it fits; stacked in a narrow transcript instead of
        // clipping a label.
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 6) { chips }
            VStack(alignment: .leading, spacing: 6) { chips }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Follow-up suggestions", bundle: .module))
    }

    @ViewBuilder
    private var chips: some View {
        ForEach(FollowUpSuggestions.templates, id: \.self) { template in
            let text = AppStrings.localized(String.LocalizationValue(template))
            AppChip(Text(verbatim: text)) { draft(text) }
                .accessibilityHint(Text("Adds this to your message without sending it.", bundle: .module))
        }
    }
}
