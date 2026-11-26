import SwiftUI
import TurboSpark

/// Search and the one Filter popover of ModelManagerView (favorites, available now, architecture, drafter, source, tag, sort, group).
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
    /// Hide models whose drive is not connected and models whose files are
    /// missing (requirement 3.6).
    @Binding var availableOnly: Bool
    /// Multi-select mode for deleting several models at once.
    @Binding var isSelecting: Bool
    /// False when there is nothing to select (an empty library).
    var canSelect: Bool = true

    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    private var activeCount: Int {
        ModelManagerView.activeFilterCount(
            showFavoritesOnly: showFavoritesOnly,
            availableOnly: availableOnly,
            architectureFilter: architectureFilter,
            drafterFilter: drafterFilter,
            sourceFilter: sourceFilter,
            selectedTag: selectedTag)
    }

    var body: some View {
        // Search leads and takes the spare width; the one Filter button holds
        // everything else, so the bar stays one short row at any text size.
        HStack(spacing: 8) {
            AppSearchField(
                text: $searchText,
                prompt: Text("Filter installed models...", bundle: .module),
                maxWidth: .infinity)

            selectButton

            ModelFilterButton(activeCount: activeCount) {
                filterSection
                orderSection
                tagSection
                if activeCount > 0 { clearSection }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        // Removing the last use of the selected tag hides the tag picker,
        // which would leave a filter active with no control to clear it.
        .onChange(of: orgStore.allKnownTags) { _, tags in
            if let tag = selectedTag, !tags.contains(tag) { selectedTag = nil }
        }
    }

    /// Select and Done are the same control: the label says which it is, and
    /// the glyph differs too, so the mode never rests on color.
    private var selectButton: some View {
        Button {
            isSelecting.toggle()
        } label: {
            Label {
                if isSelecting {
                    Text("Done", bundle: .module)
                } else {
                    Text("Select", bundle: .module)
                }
            } icon: {
                Image(systemName: isSelecting ? "checkmark" : "checklist")
            }
        }
        .appAction(.secondary, size: .small)
        .disabled(!canSelect && !isSelecting)
        .help(Text("Choose several models to delete at once", bundle: .module))
    }

    // One computed property per Section (DESIGN.md section 3, Forms).

    private var filterSection: some View {
        Section {
            Toggle(isOn: $showFavoritesOnly) {
                Text("Show Favorites Only", bundle: .module)
            }
            // The state is the switch, not a color.
            Toggle(isOn: $availableOnly) {
                Text("Available now", bundle: .module)
            }
            .help(Text("Hide models whose drive is not connected or whose files are missing", bundle: .module))

            Picker(selection: $architectureFilter) {
                ForEach(ModelManagerView.ArchitectureFilter.allCases) { filter in
                    optionText(filter.rawValue).tag(filter)
                }
            } label: { Text("Architecture", bundle: .module) }

            Picker(selection: $drafterFilter) {
                ForEach(ModelManagerView.DrafterFilter.allCases) { filter in
                    optionText(filter.rawValue).tag(filter)
                }
            } label: { Text("Drafter", bundle: .module) }

            Picker(selection: $sourceFilter) {
                ForEach(ModelManagerView.SourceFilter.allCases) { filter in
                    optionText(filter.rawValue).tag(filter)
                }
            } label: { Text("Source", bundle: .module) }
        }
    }

    private var orderSection: some View {
        Section {
            Picker(selection: $sortOption) {
                ForEach(ModelManagerView.SortOption.allCases) { opt in
                    optionText(opt.rawValue).tag(opt)
                }
            } label: { Text("Sort By", bundle: .module) }

            Picker(selection: $groupOption) {
                ForEach(ModelManagerView.GroupOption.allCases) { grp in
                    optionText(grp.rawValue).tag(grp)
                }
            } label: { Text("Group By", bundle: .module) }
        }
    }

    @ViewBuilder private var tagSection: some View {
        let knownTags = orgStore.allKnownTags
        if !knownTags.isEmpty {
            Section {
                Picker(selection: $selectedTag) {
                    Text("All", bundle: .module).tag(String?.none)
                    ForEach(knownTags, id: \.self) { tag in
                        Text(verbatim: tag).tag(String?.some(tag))
                    }
                } label: { Text("Tags", bundle: .module) }
            }
        }
    }

    private var clearSection: some View {
        Section {
            Button {
                showFavoritesOnly = false
                availableOnly = false
                architectureFilter = .all
                drafterFilter = .all
                sourceFilter = .all
                selectedTag = nil
            } label: {
                Text("Clear filters", bundle: .module)
            }
            .appAction(.quiet, size: .small)
        }
    }

    /// The filter enums keep English raw values as stable identifiers; the
    /// display text looks each one up in the catalog. Values that are names
    /// (LM Studio, Dynamic Sloth) have no key and render as written.
    private func optionText(_ rawValue: String) -> Text {
        Text(LocalizedStringKey(rawValue), bundle: .module)
    }
}
