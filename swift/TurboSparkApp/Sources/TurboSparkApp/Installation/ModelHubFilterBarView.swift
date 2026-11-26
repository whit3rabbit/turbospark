import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Search, catalog view and the one Filter popover of the Model Hub.
@MainActor
struct ModelHubFilterBarView: View {
    @Binding var filter: ModelHubFilter
    let catalog: [CatalogEntry]
    let recommendations: [String: ModelRecommendation]

    var body: some View {
        // Search leads and takes the spare width. The catalog view (2 to 4
        // short options) sits beside it and everything else is in the one
        // Filter popover. When the row cannot hold all three, the view
        // switcher drops to its own line instead of squeezing the search.
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                searchField
                catalogView
                filterButton
            }
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    searchField
                    filterButton
                }
                catalogView
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .onChange(of: filter.tab) { _, tab in
            guard tab == .audio else { return }
            filter.clearFilters()
        }
    }

    private var searchField: some View {
        AppSearchField(
            text: $filter.searchText,
            // "Search", not "Filter": the same box also sends the query to
            // Hugging Face on Return, so it is the one place to look for a model.
            prompt: Text("Search", bundle: .module),
            maxWidth: .infinity)
            .help(Text("Filters by alias, name, family or notes", bundle: .module))
    }

    // "Installed only" narrows the CATALOG to entries already on this
    // Mac. It is not the Installed workspace, and the tooltip says so.
    private var catalogView: some View {
        Picker(selection: $filter.tab) {
            ForEach(ModelHubFilter.Tab.allCases) { tab in
                tabText(tab).tag(tab)
            }
        } label: { Text("Catalog view", bundle: .module) }
        .pickerStyle(.segmented)
        .controlSize(.small)
        .labelsHidden()
        .fixedSize()
        .help(Text("Catalog view", bundle: .module))
    }

    private var filterButton: some View {
        ModelFilterButton(activeCount: filter.activeFilterCount) {
            if filter.tab != .audio { dropdownSection }
            sortSection
            if filter.activeFilterCount > 0 { clearSection }
        }
    }

    private var dropdownSection: some View {
        Section {
            filterPicker(
                name: Text("Format & Package", bundle: .module),
                options: ModelHubFilter.formatOptions(for: catalog),
                selection: $filter.format)
            filterPicker(
                name: Text("Architecture", bundle: .module),
                options: ModelHubFilter.capabilityOptions(for: catalog),
                selection: $filter.capability)
            filterPicker(
                name: Text("Fits Memory", bundle: .module),
                options: ModelHubFilter.fitOptions(for: catalog, recommendations: recommendations),
                selection: $filter.fit)
        }
    }

    private var sortSection: some View {
        Section {
            Picker(selection: $filter.sort) {
                ForEach(ModelHubFilter.SortOption.allCases) { option in
                    sortOptionText(option, audio: filter.tab == .audio).tag(option)
                }
            } label: { Text("Sort By", bundle: .module) }
        }
    }

    private var clearSection: some View {
        Section {
            Button {
                filter.clearFilters()
            } label: {
                Text("Clear filters", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            .help(Text("Clear every filter", bundle: .module))
        }
    }

    /// A labeled dropdown: "All" when inactive, the choice when set. Options
    /// are catalog data: words with a catalog key (Reasoning, Tight fit) are
    /// translated, names and format labels render as written.
    private func filterPicker(
        name: Text,
        options: [String],
        selection: Binding<ModelHubFilter.Selection>
    ) -> some View {
        Picker(selection: selection) {
            Text("All", bundle: .module).tag(ModelHubFilter.Selection.none)
            if !options.isEmpty {
                Divider()
                ForEach(options, id: \.self) { option in
                    optionText(option).tag(ModelHubFilter.Selection.some(option))
                }
            }
        } label: { name }
        .disabled(options.isEmpty && selection.wrappedValue == nil)
    }

    private func optionText(_ option: String) -> Text {
        Text(LocalizedStringKey(option), bundle: .module)
    }

    private func tabText(_ tab: ModelHubFilter.Tab) -> Text {
        switch tab {
        case .featured: return Text("Featured", bundle: .module)
        // Not "Discover": that is the section's own name, so a tab with the
        // same word did not say how it differs from the others.
        case .discover: return Text("All models", bundle: .module)
        case .audio: return Text("Audio", bundle: .module)
        case .onDevice: return Text("Installed only", bundle: .module)
        }
    }

    private func sortOptionText(_ option: ModelHubFilter.SortOption, audio: Bool = false) -> Text {
        switch option {
        case .recommended: return Text("Recommended", bundle: .module)
        case .name: return Text("Name", bundle: .module)
        case .size: return audio ? Text("Download Size", bundle: .module) : Text("Installed size", bundle: .module)
        }
    }
}
