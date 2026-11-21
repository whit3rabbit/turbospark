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
        VStack(alignment: .leading, spacing: 0) {
            if !controller.slashFilterGroups.isEmpty {
                slashFilterChips
            }
            suggestionList
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

    /// The `/` menu's kind filter. Pinned above the scrolling rows so it
    /// stays reachable; the arrows keep walking the rows it narrows.
    private var slashFilterChips: some View {
        HStack(spacing: 6) {
            AppChip(Text("All", bundle: .module), isOn: controller.slashFilter == nil) {
                controller.setSlashFilter(nil)
            }
            ForEach(controller.slashFilterGroups, id: \.self) { group in
                // Every group title is fixed English and a catalog key.
                AppChip(
                    Text(LocalizedStringKey(group.title), bundle: .module),
                    isOn: controller.slashFilter == group
                ) {
                    controller.setSlashFilter(group)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 8)
        .padding(.top, 7)
        .padding(.bottom, 3)
        .accessibilityElement(children: .contain)
    }

    private var suggestionList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if let hint = controller.hint {
                        hintRow(hint)
                    }
                    if controller.showsGroupHeaders {
                        // Grouped `@` menu. `suggestions` is stored in this
                        // same order, so startIndex + offset is the row's
                        // selectedIndex and the arrows cross groups in
                        // reading order.
                        ForEach(controller.sections, id: \.group) { section in
                            VStack(alignment: .leading, spacing: 0) {
                                groupHeader(section.group)
                                ForEach(Array(section.rows.enumerated()), id: \.element.id) {
                                    offset, suggestion in
                                    row(suggestion, index: section.startIndex + offset)
                                }
                            }
                            .accessibilityElement(children: .contain)
                            .accessibilityLabel(
                                Text(LocalizedStringKey(section.group.title), bundle: .module))
                        }
                    } else {
                        ForEach(Array(controller.suggestions.enumerated()), id: \.element.id) {
                            index, suggestion in
                            row(suggestion, index: index)
                        }
                    }
                }
            }
            .onChange(of: controller.selectedIndex) { _, newIndex in
                guard controller.suggestions.indices.contains(newIndex) else { return }
                proxy.scrollTo(controller.suggestions[newIndex].id, anchor: .center)
            }
        }
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

    private func groupHeader(_ group: ComposerSuggestionGroup) -> some View {
        // Every group title is fixed English and a catalog key.
        Text(LocalizedStringKey(group.title), bundle: .module)
            .themedFont(.tiny)
            .fontWeight(.medium)
            .foregroundStyle(.appSecondary)
            .padding(.horizontal, 10)
            .padding(.top, 7)
            .padding(.bottom, 2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityAddTraits(.isHeader)
    }

    /// Glyph tint by kind. Secondary signal only: the glyph shape and the
    /// group header word carry the kind. Typed references (`@diff`,
    /// `@git:`) take the accent because they are not paths in the tree;
    /// files, folders, commands and skills stay neutral, told apart by
    /// glyph. No status color: a kind is not a state.
    private static func glyphStyle(_ kind: ComposerSuggestion.Kind) -> AnyShapeStyle {
        switch kind {
        case .reference: return AnyShapeStyle(.appAccentText)
        case .file, .folder, .builtInCommand, .skill: return AnyShapeStyle(.appSecondary)
        }
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
                    .foregroundStyle(Self.glyphStyle(suggestion.kind))
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
