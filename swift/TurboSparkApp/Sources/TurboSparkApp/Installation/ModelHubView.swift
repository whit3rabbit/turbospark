import AppKit
import SwiftUI
import TurboSpark

/// The Models section: browse the catalog, see what fits this machine, install.
///
/// The chrome matches the Files section rather than carrying its own: no page
/// title block, no telemetry pills. Chip and RAM used to be repeated here and
/// now live in the window's status strip, which is on screen in every section
/// (`swift/CLAUDE.md` Gotcha 17).
struct ModelHubView: View {
    @ObservedObject var model: AppModel

    @State private var filter = ModelHubFilter()
    @State private var selectedAlias: String?
    @State private var recommendations: [String: ModelRecommendation] = [:]
    @State private var showingProbeSheet = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            filterBar
            Divider()
            masterDetailContent
        }
        .sheet(isPresented: $showingProbeSheet) {
            ModelProbeSheet(model: model)
        }
        .task {
            loadRecommendations()
            if selectedAlias == nil {
                selectedAlias = model.selected?.alias ?? model.catalog.first?.alias
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                Text("Models")
                    .font(.system(size: 14, weight: .semibold))
                    .accessibilityAddTraits(.isHeader)
                Text(summaryText)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            searchField

            Button {
                showingProbeSheet = true
            } label: {
                Label("Probe HF", systemImage: "magnifyingglass")
                    .font(.system(size: 11, weight: .medium))
                    .frame(height: 22)
                    .padding(.horizontal, 9)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .background(TurboSparkTheme.accentColor.opacity(0.14), in: Capsule())
            .foregroundStyle(TurboSparkTheme.accentColor)
            .help("Probe an arbitrary Hugging Face repository by header alone")
            .accessibilityLabel("Probe a Hugging Face repository")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var summaryText: String {
        let installed = model.installed
        var parts = ["\(model.catalog.count) in catalog"]
        parts.append("\(installed.count) installed")
        let onDisk = installed.reduce(UInt64(0)) { $0 + $1.installBytes }
        if onDisk > 0 {
            parts.append("\(MetricFormat.storage(onDisk)) on disk")
        }
        return parts.joined(separator: " \u{2022} ")
    }

    private var searchField: some View {
        HStack(spacing: 5) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            TextField("Filter", text: $filter.searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 11))
                .frame(width: 130)
                .accessibilityLabel("Search models")
                .accessibilityHint("Filters by alias, name, family or notes")
            if !filter.searchText.isEmpty {
                Button {
                    filter.searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tertiary)
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 8)
        .frame(height: 22)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay { Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5) }
    }

    // MARK: - Filters

    private var filterBar: some View {
        HStack(spacing: 8) {
            Picker("View", selection: $filter.tab) {
                ForEach(ModelHubFilter.Tab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .controlSize(.small)
            .frame(width: 150)
            .labelsHidden()
            .accessibilityLabel("Catalog view")

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    filterChip(
                        title: "Format",
                        options: ModelHubFilter.formatOptions(for: model.catalog),
                        selection: $filter.format)

                    filterChip(
                        title: "Capability",
                        options: ModelHubFilter.capabilityOptions(for: model.catalog),
                        selection: $filter.capability)

                    filterChip(
                        title: "Fit",
                        options: ModelHubFilter.fitOptions(
                            for: model.catalog,
                            recommendations: recommendations),
                        selection: $filter.fit)

                    sortChip

                    if filter.isNarrowed {
                        Button {
                            filter.clearNarrowing()
                        } label: {
                            Label("Clear", systemImage: "xmark")
                                .font(.system(size: 10, weight: .medium))
                                .labelStyle(.titleOnly)
                                .padding(.horizontal, 8)
                                .frame(height: 22)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Clear every filter")
                        .accessibilityLabel("Clear filters")
                    }
                }
                .padding(.trailing, 4)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    /// A dropdown that reads "Format" when inactive and "MLX INT4" when set,
    /// so an active filter is visible without opening it.
    private func filterChip(
        title: String,
        options: [String],
        selection: Binding<ModelHubFilter.Selection>
    ) -> some View {
        Menu {
            Button(ModelHubFilter.anyOption) { selection.wrappedValue = nil }
            if !options.isEmpty {
                Divider()
                ForEach(options, id: \.self) { option in
                    Button {
                        selection.wrappedValue = option
                    } label: {
                        if selection.wrappedValue == option {
                            Label(option, systemImage: "checkmark")
                        } else {
                            Text(option)
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text(selection.wrappedValue ?? title)
                    .font(.system(size: 11, weight: selection.wrappedValue == nil ? .regular : .medium))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 9)
            .frame(height: 22)
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(selection.wrappedValue == nil ? Color.secondary : TurboSparkTheme.accentColor)
        .background(
            selection.wrappedValue == nil
                ? TurboSparkTheme.surfaceColor
                : TurboSparkTheme.accentColor.opacity(0.14),
            in: Capsule())
        .overlay { Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5) }
        .disabled(options.isEmpty && selection.wrappedValue == nil)
        .help("Filter by \(title.lowercased())")
        .accessibilityLabel("\(title) filter")
        .accessibilityValue(selection.wrappedValue ?? ModelHubFilter.anyOption)
    }

    private var sortChip: some View {
        Menu {
            ForEach(ModelHubFilter.SortOption.allCases) { option in
                Button {
                    filter.sort = option
                } label: {
                    if filter.sort == option {
                        Label(option.rawValue, systemImage: "checkmark")
                    } else {
                        Text(option.rawValue)
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.up.arrow.down")
                    .font(.system(size: 8, weight: .bold))
                Text(filter.sort.rawValue)
                    .font(.system(size: 11))
                    .lineLimit(1)
            }
            .padding(.horizontal, 9)
            .frame(height: 22)
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(.secondary)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay { Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5) }
        .help("Sort the list")
        .accessibilityLabel("Sort order")
        .accessibilityValue(filter.sort.rawValue)
    }

    // MARK: - List and detail

    private var masterDetailContent: some View {
        HStack(spacing: 0) {
            masterList
                .frame(minWidth: 250, idealWidth: 300, maxWidth: 340)
                .frame(maxHeight: .infinity)

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(width: AppChromeLayout.dividerWidth)

            detailPane
                .frame(minWidth: 260, maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var masterList: some View {
        ScrollView {
            LazyVStack(spacing: 3) {
                if filteredEntries.isEmpty {
                    emptyListState
                } else {
                    ForEach(filteredEntries) { entry in
                        cardView(for: entry)
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
        }
    }

    private func cardView(for entry: CatalogEntry) -> some View {
        let isInstalled = model.installed.contains(where: { $0.alias == entry.alias }) || entry.installed
        let isActive = model.selected?.alias == entry.alias && model.session != nil
        let rec = recommendations[entry.alias]
        let isSelected = selectedAlias == entry.alias

        return ModelCardView(
            alias: entry.alias,
            name: entry.name,
            family: entry.family,
            status: entry.status,
            downloadBytes: entry.downloadBytes,
            isInstalled: isInstalled,
            isActive: isActive,
            isDownloading: model.isInstallingModel,
            downloadFraction: model.installProgressFraction,
            recommendation: rec,
            isSelected: isSelected,
            onSelect: { selectedAlias = entry.alias }
        )
    }

    private var emptyListState: some View {
        VStack(spacing: 8) {
            Image(systemName: filter.tab == .onDevice ? "internaldrive" : "tray")
                .font(.system(size: 26))
                .foregroundStyle(.quaternary)
                .accessibilityHidden(true)
            Text(emptyTitle)
                .font(.callout.weight(.medium))
            Text(emptyDetail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if filter.isNarrowed {
                Button("Clear filters") { filter.clearNarrowing() }
                    .buttonStyle(.link)
                    .font(.caption)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 48)
        .accessibilityElement(children: .contain)
    }

    private var emptyTitle: String {
        if filter.tab == .onDevice && !filter.isNarrowed { return "No models installed" }
        return "No matching models"
    }

    private var emptyDetail: String {
        if filter.tab == .onDevice && !filter.isNarrowed {
            return "Switch to Discover to browse the catalog and install one."
        }
        return "Nothing in this view matches the current filters."
    }

    @ViewBuilder
    private var detailPane: some View {
        if let selectedEntry {
            ModelDetailPaneView(
                model: model,
                entry: selectedEntry,
                installedModel: model.installed.first(where: { $0.alias == selectedEntry.alias }),
                recommendation: recommendations[selectedEntry.alias])
        } else {
            VStack(spacing: 8) {
                Image(systemName: "shippingbox")
                    .font(.system(size: 30))
                    .foregroundStyle(.quaternary)
                    .accessibilityHidden(true)
                Text("No model selected")
                    .font(.callout.weight(.medium))
                Text("Pick a row to see what it costs and whether it fits this machine.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 320)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityElement(children: .combine)
        }
    }

    /// The selected row, falling back to the first visible one so the detail
    /// pane never shows an entry the filters just hid.
    private var selectedEntry: CatalogEntry? {
        if let selectedAlias, let entry = filteredEntries.first(where: { $0.alias == selectedAlias }) {
            return entry
        }
        return filteredEntries.first
    }

    private var filteredEntries: [CatalogEntry] {
        filter.apply(
            to: model.catalog,
            installedAliases: Set(model.installed.map(\.alias)),
            recommendations: recommendations)
    }

    private func loadRecommendations() {
        guard let recommended = try? TurboSparkCatalog.recommend() else { return }
        recommendations = Dictionary(
            recommended.map { ($0.alias, $0) },
            uniquingKeysWith: { first, _ in first })
    }
}
