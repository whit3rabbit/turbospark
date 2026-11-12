import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The Models section: browse the catalog, see what fits this machine, install.
///
/// The chrome matches the Files section rather than carrying its own: no page
/// title block, no telemetry pills. Chip and RAM used to be repeated here and
/// now live in the window's status strip, which is on screen in every section
/// (`swift/CLAUDE.md` Gotcha 17).
@MainActor
struct ModelHubView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme

    @State private var filter = ModelHubFilter()
    @State private var selectedAlias: String?
    @State private var recommendations: [String: ModelRecommendation] = [:]
    @State private var isLoadingRecommendations = true
    @State private var recommendationProgress: ModelRecommendationProbeProgress?
    @State private var recommendationError: String?
    @State private var showingProbeSheet = false
    /// Live Hugging Face search. It sends nothing until asked (Return in the
    /// search box, or the row in the list that says so).
    @StateObject private var hub = HubSearchController()
    @State private var selectedHubRepo: String?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            // `onSubmit` is applied to the filter bar alone: it reacts to any
            // text field below it, and the install pane's "Save as" field
            // must not start a search when the user presses Return there.
            ModelHubFilterBarView(
                filter: $filter,
                catalog: model.catalog,
                recommendations: recommendations
            )
            .onSubmit { runHubSearch() }
            Divider()
            masterDetailContent
        }
        .onChange(of: filter.searchText) {
            // A new query makes the old answer, and a selection in it, stale.
            hub.reset()
            selectedHubRepo = nil
        }
        .onChange(of: filter.tab) {
            if !offersHubSearch { selectedHubRepo = nil }
        }
        // Native controls without their own role inherit the same live theme
        // as the explicit text roles below, including font family and size.
        .themedFont(.small)
        .sheet(isPresented: $showingProbeSheet) {
            ModelProbeSheet(model: model)
        }
        .task(id: model.fitRecommendationConfigurationID) {
            await loadRecommendations()
            if selectedAlias == nil {
                let activeAlias = model.selected?.alias
                selectedAlias = filteredEntries.first(where: { $0.alias == activeAlias })?.alias
                    ?? filteredEntries.first?.alias
            }
        }
    }

    // MARK: - Header

    /// Same structure as the Installed header (`ModelManagerHeaderView`):
    /// title block on the leading side, actions in the shared treatment, and
    /// search in the filter bar below rather than squeezed into this row.
    private var header: some View {
        AppAdaptiveHeader {
            ModelWorkspaceTitle(
                systemImage: "shippingbox.fill",
                tint: .appAccent,
                title: Text("Discover Models", bundle: .module),
                badge: Text("ONLINE CATALOG", bundle: .module),
                summary: summaryText)
        } actions: {
            Button {
                showingProbeSheet = true
            } label: {
                Label { Text("Probe HF Repo", bundle: .module) } icon: { Image(systemName: "sparkle.magnifyingglass") }
            }
            .appAction(.secondary)
            .help(Text("Probe an arbitrary Hugging Face repository by header alone", bundle: .module))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var summaryText: Text {
        let installed = model.installed
        let separator = Text(verbatim: " \u{2022} ")
        var line = Text("In catalog: \(model.catalog.count)", bundle: .module)
            + separator + Text("Installed models: \(installed.count)", bundle: .module)
        let onDisk = installed.reduce(UInt64(0)) { $0 + $1.installBytes }
        if onDisk > 0 {
            line = line + separator + Text("On disk: \(MetricFormat.storage(onDisk))", bundle: .module)
        }
        return line
    }

    // MARK: - List and detail

    @ViewBuilder private var masterDetailContent: some View {
        if filter.tab == .audio {
            AudioDiscoverCatalogView(model: model, controller: model.audioWorkspace, filter: $filter)
        } else {
            HStack(spacing: 0) {
                masterList
                    .frame(minWidth: 250, idealWidth: 300, maxWidth: 340)
                    .frame(maxHeight: .infinity)

                Rectangle()
                    .fill(.appBorder)
                    .frame(width: AppChromeLayout.dividerWidth)

                detailPane
                    .frame(minWidth: 260, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private var masterList: some View {
        ScrollView {
            LazyVStack(spacing: 3) {
                if filter.tab == .recommended,
                   !isLoadingRecommendations,
                   recommendationError == nil,
                   !filteredEntries.isEmpty {
                    recommendedHeaderCard
                }
                if filter.tab == .recommended && isLoadingRecommendations {
                    recommendationLoadingState
                } else if filter.tab == .recommended, let recommendationError {
                    recommendationFailureState(recommendationError)
                } else if filteredEntries.isEmpty {
                    emptyListState
                } else if let sections = recommendedSections {
                    // Grouped by fit, so the order is something a reader can
                    // see and not a ranking only the code knew.
                    ForEach(sections) { section in
                        RecommendedSectionHeader(verdict: section.verdict)
                        ForEach(section.entries) { entry in
                            cardView(for: entry)
                        }
                    }
                } else {
                    ForEach(filteredEntries) { entry in
                        cardView(for: entry)
                    }
                }
                if showsHubSection {
                    HubSearchSectionView(
                        controller: hub,
                        query: trimmedQuery,
                        directRepo: HubSearchController.exactRepo(from: trimmedQuery),
                        selectedRepo: $selectedHubRepo,
                        onSearch: runHubSearch,
                        onSettings: { model.openSettings(tab: .models) })
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
        }
    }

    private var recommendedHeaderCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appAccent)
                Text("Optimized for Your Mac", bundle: .module)
                    .themedFont(.tiny, weight: .bold)
                    .foregroundStyle(.appText)
                Spacer()
                if let chip = model.telemetry?.chip {
                    Text(verbatim: chip)
                        .themedFont(.micro, weight: .medium)
                        .foregroundStyle(.appSecondary)
                }
                if let ram = model.telemetry?.physicalMemoryBytes {
                    Text(verbatim: "(\(MetricFormat.memory(ram)))")
                        .themedFont(.micro)
                        .foregroundStyle(.appSecondary)
                }
            }
        }
        .padding(8)
        .background(.appAccent.opacity(0.08), in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(theme.border, lineWidth: 0.5)
        }
        .padding(.bottom, 4)
    }

    private func cardView(for entry: CatalogEntry) -> some View {
        let isInstalled = model.installed.contains(where: { $0.alias == entry.alias }) || entry.installed
        let isActive = model.selected?.alias == entry.alias && model.session != nil
        let rec = recommendations[entry.alias]
        let isSelected = selectedEntry?.alias == entry.alias

        return ModelCardView(
            alias: entry.alias,
            name: entry.name,
            family: entry.family,
            status: entry.status,
            downloadBytes: entry.downloadBytes,
            isInstalled: isInstalled,
            isActive: isActive,
            isDownloading: model.isInstallingModel && model.installingAlias == entry.alias,
            downloadFraction: model.isInstallingModel && model.installingAlias == entry.alias
                ? model.installProgressFraction : nil,
            recommendation: rec,
            isSelected: isSelected && selectedHubRepo == nil,
            onSelect: {
                selectedAlias = entry.alias
                selectedHubRepo = nil
            }
        )
    }

    // MARK: - Hugging Face search

    private var trimmedQuery: String {
        filter.searchText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Live search is offered next to the two catalog tabs, not under
    /// "Installed only" (which is about this Mac) or Audio (its own catalog).
    private var offersHubSearch: Bool {
        filter.tab == .discover || filter.tab == .recommended
    }

    private var showsHubSection: Bool {
        offersHubSearch && trimmedQuery.count >= HubSearchController.minimumQueryLength
    }

    private func runHubSearch() {
        guard showsHubSection else { return }
        let query = trimmedQuery
        selectedHubRepo = nil
        Task { await hub.search(query: query) }
    }

    private var emptyListState: some View {
        AppEmptyState(systemImage: emptyIconName, title: emptyTitle) {
            if filter.isNarrowed {
                Button { filter.clearNarrowing() } label: { Text("Clear filters", bundle: .module) }
                    .appAction(.secondary, size: .small)
            } else if filter.tab == .recommended {
                Button { filter.tab = .discover } label: { Text("All models", bundle: .module) }
                    .appAction(.secondary, size: .small)
            } else if filter.tab == .onDevice {
                Button { filter.tab = .discover } label: { Text("Discover", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
        }
        .padding(.top, 16)
    }

    private var emptyIconName: String {
        switch filter.tab {
        case .onDevice: return "internaldrive"
        case .recommended: return "sparkles"
        case .discover: return "tray"
        case .audio: return "waveform"
        }
    }

    private var emptyTitle: Text {
        if filter.tab == .onDevice && !filter.isNarrowed {
            return Text("No models installed", bundle: .module)
        } else if filter.tab == .recommended && !filter.isNarrowed {
            return Text("No recommended models", bundle: .module)
        }
        return Text("No matching models", bundle: .module)
    }

    private var recommendationLoadingState: some View {
        ModelRecommendationLoadingView(progress: recommendationProgress)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 48)
    }

    private func recommendationFailureState(_ message: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .themedFont(.hero)
                .foregroundStyle(AppAttention.color)
                .accessibilityHidden(true)
            Text(message)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
            Button {
                Task { await loadRecommendations() }
            } label: {
                Text("Refresh", bundle: .module)
            }
            .appAction(.secondary, size: .small)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 48)
    }

    @ViewBuilder
    private var detailPane: some View {
        if let repo = selectedHubRepo, showsHubSection {
            HubResultDetailView(
                model: model,
                controller: hub,
                repoId: repo,
                result: hub.row(for: repo))
        } else if let selectedEntry {
            ModelDetailPaneView(
                model: model,
                entry: selectedEntry,
                installedModel: model.installed.first(where: { $0.alias == selectedEntry.alias }),
                recommendation: recommendations[selectedEntry.alias])
        } else {
            AppEmptyState(
                systemImage: "shippingbox",
                title: Text("No model selected", bundle: .module),
                message: Text("Pick a row to see what it costs and whether it fits this machine.", bundle: .module))
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

    /// The Recommended tab's fit groups, or `nil` when the list is not
    /// grouped: another tab, or another sort. A list sorted by name or size
    /// has no meaningful group boundaries, so it stays flat.
    private var recommendedSections: [ModelHubFilter.RecommendedSection]? {
        guard filter.tab == .recommended, filter.sort == .recommended else { return nil }
        return ModelHubFilter.recommendedSections(filteredEntries, recommendations: recommendations)
    }

    private func loadRecommendations() async {
        let configurationID = model.fitRecommendationConfigurationID
        isLoadingRecommendations = true
        recommendationProgress = nil
        recommendationError = nil
        do {
            let rows = try await model.loadFitRecommendations { completed, total in
                guard configurationID == model.fitRecommendationConfigurationID else { return }
                recommendationProgress = ModelRecommendationProbeProgress(
                    completed: completed,
                    total: total)
            }
            guard !Task.isCancelled,
                  configurationID == model.fitRecommendationConfigurationID else { return }
            recommendations = Dictionary(
                rows.map { ($0.alias, $0) },
                uniquingKeysWith: { first, _ in first })
            recommendationProgress = nil
            isLoadingRecommendations = false
        } catch {
            guard !Task.isCancelled,
                  configurationID == model.fitRecommendationConfigurationID else { return }
            recommendations = [:]
            recommendationProgress = nil
            recommendationError = error.localizedDescription
            isLoadingRecommendations = false
        }
    }
}
