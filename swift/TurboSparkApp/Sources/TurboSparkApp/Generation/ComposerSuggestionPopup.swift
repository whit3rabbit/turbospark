import SwiftUI

/// The inline suggestion list the composer's `/` and `@` autocomplete shows.
///
/// Sits INSIDE the composer box, above the editor, rather than floating over
/// the transcript: an overlay would have to guess its own height to avoid
/// covering the composer, and a guessed height is wrong the day the list is
/// taller or shorter than the guess.
@MainActor
struct ComposerSuggestionPopup: View {
    @ObservedObject var controller: ComposerAutocompleteController
    /// Called when the user clicks a row; keyboard acceptance is handled in
    /// the editor's key handler through the same controller.
    let onPick: () -> Void

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if let hint = controller.hint {
                        hintRow(hint)
                    }
                    ForEach(Array(controller.suggestions.enumerated()), id: \.element.id) {
                        index, suggestion in
                        row(suggestion, index: index)
                    }
                }
            }
            .onChange(of: controller.selectedIndex) { _, newIndex in
                guard controller.suggestions.indices.contains(newIndex) else { return }
                proxy.scrollTo(controller.suggestions[newIndex].id, anchor: .center)
            }
        }
        .frame(maxHeight: 260)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Color.primary.opacity(0.14), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.16), radius: 8, y: 2)
        .accessibilityLabel("Suggestions")
        .accessibilityHint("Use the up and down arrows to choose, Return to accept, Escape to dismiss.")
    }

    private func hintRow(_ text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "info.circle")
                .foregroundStyle(.secondary)
            Text(text)
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
        .themedFont(.small)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .accessibilityElement(children: .combine)
    }

    private func row(_ suggestion: ComposerSuggestion, index: Int) -> some View {
        let isSelected = index == controller.selectedIndex
        return Button {
            // Select first, THEN let the shared accept path run: accept
            // reads `selectedIndex`, and without this a click on any row
            // but the keyboard-highlighted one inserted THAT row instead.
            controller.select(index)
            onPick()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: suggestion.iconName)
                    .frame(width: 16)
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(suggestion.title)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let subtitle = suggestion.subtitle {
                        Text(subtitle)
                            .themedFont(.tiny)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
            .background(isSelected ? Color.primary.opacity(0.09) : Color.clear)
        }
        .buttonStyle(.plain)
        .id(suggestion.id)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(suggestion.title)
        .accessibilityHint(suggestion.subtitle ?? "")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}
