import SwiftUI

/// The inline suggestion list the composer's `/` and `@` autocomplete shows.
///
/// Sits INSIDE the composer box, above the editor, rather than floating over
/// the transcript: an overlay would have to guess its own height to avoid
/// covering the composer, and a guessed height is wrong the day the list is
/// taller or shorter than the guess.
@MainActor
struct ComposerSuggestionPopup: View {
    @Environment(\.appTheme) private var theme
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
        .background(.appElevated)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(theme.border, lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.16), radius: 8, y: 2)
        .accessibilityLabel(Text("Suggestions", bundle: .module))
        .accessibilityHint(Text("Use the up and down arrows to choose, Return to accept, Escape to dismiss.", bundle: .module))
    }

    private func hintRow(_ text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "info.circle")
                .foregroundStyle(.appSecondary)
            // Every hint the controller sets is fixed English and a catalog key.
            Text(LocalizedStringKey(text), bundle: .module)
                .foregroundStyle(.appSecondary)
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
                    .foregroundStyle(.appSecondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(suggestion.title)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let subtitle = Self.subtitleText(suggestion) {
                        subtitle
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
            .background(
                isSelected ? AnyShapeStyle(theme.accent.opacity(0.12)) : AnyShapeStyle(Color.clear)
            )
        }
        .buttonStyle(.plain)
        .id(suggestion.id)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(suggestion.title)
        .accessibilityHint(Self.subtitleText(suggestion) ?? Text(verbatim: ""))
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    /// The row's second line in the UI language. Command summaries, the
    /// typed-reference rows and the folder/scheme markers are fixed English
    /// from `ComposerAutocompleteEngine` and are catalog keys; a skill's
    /// description is its author's text and stays as written.
    private static func subtitleText(_ suggestion: ComposerSuggestion) -> Text? {
        guard let subtitle = suggestion.subtitle else { return nil }
        switch suggestion.kind {
        case .skill:
            return Text(subtitle)
        case .builtInCommand, .reference, .file, .folder:
            return Text(LocalizedStringKey(subtitle), bundle: .module)
        }
    }
}
