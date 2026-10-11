import SwiftUI
import TurboSpark

/// Filter and control bar for ModelManagerView (search, favorites, architecture/drafter/source pickers, sort/group, tags).
@MainActor
struct ModelManagerFilterBarView: View {
    @Binding var searchText: String
    @Binding var architectureFilter: ModelManagerView.ArchitectureFilter
    @Binding var drafterFilter: ModelManagerView.DrafterFilter
    @Binding var sourceFilter: ModelManagerView.SourceFilter
    @Binding var sortOption: ModelManagerView.SortOption
    @Binding var groupOption: ModelManagerView.GroupOption
    @Binding var selectedTag: String?
    @Binding var showFavoritesOnly: Bool

    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // A wrapping row: at the largest text size or in a long
            // translation the menus move to a second line instead of
            // squeezing the search field.
            FlowLayout(spacing: 8, lineSpacing: 6) {
                AppSearchField(
                    text: $searchText,
                    prompt: Text("Filter installed models...", bundle: .module))

                Toggle(isOn: $showFavoritesOnly) {
                    Label { Text("Show Favorites Only", bundle: .module) } icon: {
                        Image(systemName: showFavoritesOnly ? "star.fill" : "star")
                    }
                    .labelStyle(.iconOnly)
                }
                .toggleStyle(.button)
                .controlSize(.small)
                .help(Text("Show Favorites Only", bundle: .module))

                ModelFilterMenu(
                    name: Text("Filter by architecture", bundle: .module),
                    value: optionText(architectureFilter.rawValue)
                ) {
                    Picker(selection: $architectureFilter) {
                        ForEach(ModelManagerView.ArchitectureFilter.allCases) { filter in
                            optionText(filter.rawValue).tag(filter)
                        }
                    } label: { Text("Architecture", bundle: .module) }
                }

                ModelFilterMenu(
                    name: Text("Filter by drafter", bundle: .module),
                    value: optionText(drafterFilter.rawValue)
                ) {
                    Picker(selection: $drafterFilter) {
                        ForEach(ModelManagerView.DrafterFilter.allCases) { filter in
                            optionText(filter.rawValue).tag(filter)
                        }
                    } label: { Text("Drafter", bundle: .module) }
                }

                ModelFilterMenu(
                    name: Text("Filter by source", bundle: .module),
                    value: optionText(sourceFilter.rawValue)
                ) {
                    Picker(selection: $sourceFilter) {
                        ForEach(ModelManagerView.SourceFilter.allCases) { filter in
                            optionText(filter.rawValue).tag(filter)
                        }
                    } label: { Text("Source", bundle: .module) }
                }

                ModelFilterMenu(
                    name: Text("Sort and group models", bundle: .module),
                    value: optionText(sortOption.rawValue)
                ) {
                    Section(header: Text("Sort By", bundle: .module)) {
                        Picker(selection: $sortOption) {
                            ForEach(ModelManagerView.SortOption.allCases) { opt in
                                optionText(opt.rawValue).tag(opt)
                            }
                        } label: { Text("Sort", bundle: .module) }
                    }
                    Section(header: Text("Group By", bundle: .module)) {
                        Picker(selection: $groupOption) {
                            ForEach(ModelManagerView.GroupOption.allCases) { grp in
                                optionText(grp.rawValue).tag(grp)
                            }
                        } label: { Text("Group", bundle: .module) }
                    }
                }
            }

            // Tag filter bar if tags exist
            let knownTags = orgStore.allKnownTags
            if !knownTags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        Text("Tags:", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)

                        AppChip(Text("All", bundle: .module), systemImage: "tag", isOn: selectedTag == nil) {
                            selectedTag = nil
                        }

                        ForEach(knownTags, id: \.self) { tag in
                            AppChip(Text(verbatim: tag), systemImage: "tag", isOn: selectedTag == tag) {
                                selectedTag = selectedTag == tag ? nil : tag
                            }
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        // Removing the last use of the selected tag hides the whole tag bar,
        // which would leave a filter active with no control to clear it.
        .onChange(of: orgStore.allKnownTags) { _, tags in
            if let tag = selectedTag, !tags.contains(tag) { selectedTag = nil }
        }
    }

    /// The filter enums keep English raw values as stable identifiers; the
    /// display text looks each one up in the catalog. Values that are names
    /// (LM Studio, Dynamic Sloth) have no key and render as written.
    private func optionText(_ rawValue: String) -> Text {
        Text(LocalizedStringKey(rawValue), bundle: .module)
    }
}
