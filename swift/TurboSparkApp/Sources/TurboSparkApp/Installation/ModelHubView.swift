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

    @State private var filter = ModelHubFilter()
    @State private var selectedAlias: String?
    @State private var recommendations: [String: ModelRecommendation] = [:]
    @State private var isLoadingRecommendations = true
    @State private var recommendationError: String?
    @State private var showingProbeSheet = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ModelHubFilterBarView(
                filter: $filter,
                catalog: model.catalog,
                recommendations: recommendations
            )
            Divider()
            masterDetailContent
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

    private var header: some View {
        HStack(spacing: 12) {
            HStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(.appAccent.opacity(0.15))
                        .frame(width: 30, height: 30)
                    Image(systemName: "shippingbox.fill")
                        .themedFont(.callout, weight: .semibold)
                        .foregroundStyle(.appAccent)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Discover Models", bundle: .module)
                            .themedFont(.callout, weight: .semibold)
                            .accessibilityAddTraits(.isHeader)

                        Text("ONLINE CATALOG", bundle: .module)
                            .themedFont(.micro, weight: .bold)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(.appAccent.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                            .foregroundStyle(.appAccent)
                    }

                    Text(summaryText)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }

            Spacer(minLength: 8)

            searchField

            Button {
                showingProbeSheet = true
            } label: {
                Label { Text("Probe HF Repo", bundle: .module) } icon: { Image(systemName: "sparkle.magnifyingglass") }
                    .themedFont(.tiny, weight: .medium)
                    .frame(height: 22)
                    .padding(.horizontal, 9)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .background(.appAccent.opacity(0.14), in: Capsule())
            .foregroundStyle(.appAccent)
            .help("Probe an arbitrary Hugging Face repository by header alone")
            .accessibilityLabel("Probe a Hugging Face repository")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
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
                .themedFont(.tiny)
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            TextField("Filter", text: $filter.searchText)
                .textFieldStyle(.plain)
                .themedFont(.tiny)
                .frame(width: 130)
                .accessibilityLabel("Search models")
                .accessibilityHint("Filters by alias, name, family or notes")
            if !filter.searchText.isEmpty {
                Button {
                    filter.searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .themedFont(.tiny)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tertiary)
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 8)
        .frame(height: 22)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay { Capsule().stroke(.appBorder, lineWidth: 0.5) }
    }

    // MARK: - List and detail

    private var masterDetailContent: some View {
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

    private var recommendedHeaderCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appAccent)
                Text("Optimized for Your Mac", bundle: .module)
                    .themedFont(.tiny, weight: .bold)
                    .foregroundStyle(.primary)
                Spacer()
                if let chip = model.telemetry?.chip {
                    Text(chip)
                        .themedFont(.micro, weight: .medium)
                        .foregroundStyle(.appSecondary)
                }
                if let ram = model.telemetry?.physicalMemoryBytes {
                    Text(verbatim: "(\(MetricFormat.memory(ram)))")
                        .themedFont(.micro)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(8)
        .background(.appAccent.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(.appAccent.opacity(0.2), lineWidth: 0.5)
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
            isDownloading: model.isInstallingModel,
            downloadFraction: model.installProgressFraction,
            recommendation: rec,
            isSelected: isSelected,
            onSelect: { selectedAlias = entry.alias }
        )
    }

    private var emptyListState: some View {
        VStack(spacing: 8) {
            Image(systemName: emptyIconName)
                .themedFont(.hero)
                .foregroundStyle(.quaternary)
                .accessibilityHidden(true)
            emptyTitle
            if filter.isNarrowed {
                Button { filter.clearNarrowing() } label: { Text("Clear filters", bundle: .module) }
                    .buttonStyle(.link)
                    .themedFont(.small)
            } else if filter.tab == .recommended {
                Button { filter.tab = .discover } label: { Text("All models", bundle: .module) }
                    .buttonStyle(.link)
                    .themedFont(.small)
            } else if filter.tab == .onDevice {
                Button { filter.tab = .discover } label: { Text("Discover", bundle: .module) }
                    .buttonStyle(.link)
                    .themedFont(.small)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 48)
        .accessibilityElement(children: .contain)
    }

    private var emptyIconName: String {
        switch filter.tab {
        case .onDevice: return "internaldrive"
        case .recommended: return "sparkles"
        case .discover: return "tray"
        }
    }

    @ViewBuilder
    private var emptyTitle: some View {
        if filter.tab == .onDevice && !filter.isNarrowed {
            Text("No models installed", bundle: .module)
                .themedFont(.base, weight: .medium)
        } else if filter.tab == .recommended && !filter.isNarrowed {
            Text("No recommended models", bundle: .module)
                .themedFont(.base, weight: .medium)
        } else {
            Text("No matching models", bundle: .module)
                .themedFont(.base, weight: .medium)
        }
    }

    private var recommendationLoadingState: some View {
        VStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
            Text("Calculating hardware fit recommendations...", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 48)
    }

    private func recommendationFailureState(_ message: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .themedFont(.hero)
                .foregroundStyle(.orange)
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
            .buttonStyle(.link)
            .themedFont(.small)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 48)
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
                    .themedFont(.hero)
                    .foregroundStyle(.quaternary)
                    .accessibilityHidden(true)
                Text("No model selected", bundle: .module)
                    .themedFont(.base, weight: .medium)
                Text("Pick a row to see what it costs and whether it fits this machine.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
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

    private func loadRecommendations() async {
        let configurationID = model.fitRecommendationConfigurationID
        isLoadingRecommendations = true
        recommendationError = nil
        do {
            let rows = try await model.loadFitRecommendations()
            guard !Task.isCancelled,
                  configurationID == model.fitRecommendationConfigurationID else { return }
            recommendations = Dictionary(
                rows.map { ($0.alias, $0) },
                uniquingKeysWith: { first, _ in first })
            isLoadingRecommendations = false
        } catch {
            guard !Task.isCancelled,
                  configurationID == model.fitRecommendationConfigurationID else { return }
            recommendations = [:]
            recommendationError = error.localizedDescription
            isLoadingRecommendations = false
        }
    }
}
