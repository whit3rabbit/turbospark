import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Filter bar and chips for narrowing and sorting models in the Model Hub.
@MainActor
struct ModelHubFilterBarView: View {
    @Binding var filter: ModelHubFilter
    let catalog: [CatalogEntry]
    let recommendations: [String: ModelRecommendation]

    var body: some View {
        // Same row as the Installed filter bar: search first, then the
        // filters in the shared menu treatment, wrapping to a second line
        // instead of scrolling sideways or truncating.
        FlowLayout(spacing: 8, lineSpacing: 6) {
            AppSearchField(
                text: $filter.searchText,
                prompt: Text("Filter", bundle: .module))
                .help(Text("Filters by alias, name, family or notes", bundle: .module))

            // "Installed only" narrows the CATALOG to entries already on this
            // Mac. It is not the Installed workspace, and the tooltip says so.
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

            if filter.tab != .audio {
                filterMenu(
                    name: Text("Format & Package", bundle: .module),
                    options: ModelHubFilter.formatOptions(for: catalog),
                    selection: $filter.format)

                filterMenu(
                    name: Text("Architecture", bundle: .module),
                    options: ModelHubFilter.capabilityOptions(for: catalog),
                    selection: $filter.capability)

                filterMenu(
                    name: Text("Fits Memory", bundle: .module),
                    options: ModelHubFilter.fitOptions(
                        for: catalog,
                        recommendations: recommendations),
                    selection: $filter.fit)
            }

            ModelFilterMenu(
                name: Text("Sort By", bundle: .module),
                value: sortOptionText(filter.sort, audio: filter.tab == .audio)
            ) {
                Picker(selection: $filter.sort) {
                    ForEach(ModelHubFilter.SortOption.allCases) { option in
                        sortOptionText(option, audio: filter.tab == .audio).tag(option)
                    }
                } label: { Text("Sort By", bundle: .module) }
                .pickerStyle(.inline)
            }

            if filter.isNarrowed {
                Button {
                    filter.clearNarrowing()
                } label: {
                    Label { Text("Clear filters", bundle: .module) } icon: { Image(systemName: "xmark") }
                }
                .appAction(.quiet, size: .small)
                .help(Text("Clear every filter", bundle: .module))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .onChange(of: filter.tab) { _, tab in
            guard tab == .audio else { return }
            filter.format = nil
            filter.capability = nil
            filter.fit = nil
        }
    }

    /// A dropdown that reads "Format & Package" when inactive and "MLX INT4"
    /// when set, so an active filter is visible without opening it. Options
    /// are catalog data: words with a catalog key (Reasoning, Tight fit) are
    /// translated, names and format labels render as written.
    private func filterMenu(
        name: Text,
        options: [String],
        selection: Binding<ModelHubFilter.Selection>
    ) -> some View {
        ModelFilterMenu(
            name: name,
            value: selection.wrappedValue.map(optionText) ?? name
        ) {
            Picker(selection: selection) {
                Text("All", bundle: .module).tag(ModelHubFilter.Selection.none)
                if !options.isEmpty {
                    Divider()
                    ForEach(options, id: \.self) { option in
                        optionText(option).tag(ModelHubFilter.Selection.some(option))
                    }
                }
            } label: { name }
            .pickerStyle(.inline)
        }
        .disabled(options.isEmpty && selection.wrappedValue == nil)
    }

    private func optionText(_ option: String) -> Text {
        Text(LocalizedStringKey(option), bundle: .module)
    }

    private func tabText(_ tab: ModelHubFilter.Tab) -> Text {
        switch tab {
        case .recommended: return Text("Recommended", bundle: .module)
        case .discover: return Text("Discover", bundle: .module)
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
