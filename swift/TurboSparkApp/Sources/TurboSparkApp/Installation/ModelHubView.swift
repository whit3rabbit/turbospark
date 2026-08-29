import AppKit
import SwiftUI
import TurboSpark

/// The complete Unsloth Studio-style Model Hub and Library.
struct ModelHubView: View {
    @ObservedObject var model: AppModel

    enum HubTab: String, CaseIterable, Identifiable {
        case discover = "Discover"
        case onDevice = "On Device"
        var id: String { rawValue }
    }

    enum FormatFilter: String, CaseIterable, Identifiable {
        case all = "All Formats"
        case mlx = "MLX INT4"
        case gguf = "GGUF"
        var id: String { rawValue }
    }

    enum CapabilityFilter: String, CaseIterable, Identifiable {
        case all = "All Capabilities"
        case conversational = "Conversational"
        case reasoning = "Reasoning"
        case moe = "MoE"
        case dense = "Dense"
        case coding = "Coding"
        var id: String { rawValue }
    }

    enum SortOption: String, CaseIterable, Identifiable {
        case recommended = "Recommended"
        case name = "Name (A-Z)"
        case size = "Smallest Size"
        var id: String { rawValue }
    }

    @State private var selectedTab: HubTab = .discover
    @State private var searchText = ""
    @State private var formatFilter: FormatFilter = .all
    @State private var capabilityFilter: CapabilityFilter = .all
    @State private var sortOption: SortOption = .recommended
    @State private var selectedAlias: String? = nil
    @State private var recommendations: [String: ModelRecommendation] = [:]
    @State private var showingProbeSheet = false

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()
            toolbarFilterRow
            Divider()
            masterDetailContent
        }
        .frame(minWidth: 540, minHeight: 460)
        .task {
            loadRecommendations()
            if selectedAlias == nil {
                selectedAlias = model.selected?.alias ?? model.catalog.first?.alias
            }
        }
    }

    private var headerBar: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text("Model hub")
                        .font(.title2.weight(.bold))
                    Text("LOCAL")
                        .font(.caption2.weight(.heavy))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.accentColor)
                }
                Text("Discover, download, and run inference models locally.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 8)

            telemetryPills
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var telemetryPills: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                if let telemetry = model.telemetry {
                    HStack(spacing: 4) {
                        Image(systemName: "memorychip")
                        Text(telemetry.chip ?? "Apple Silicon")
                    }
                    .font(.caption2.weight(.medium))
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3.5)
                    .background(Color(nsColor: .controlBackgroundColor), in: Capsule())
                    .overlay(Capsule().stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5))
                }

                HStack(spacing: 4) {
                    Image(systemName: "internaldrive")
                    Text("\(model.installed.count) Local")
                }
                .font(.caption2.weight(.medium))
                .padding(.horizontal, 7)
                .padding(.vertical, 3.5)
                .background(Color(nsColor: .controlBackgroundColor), in: Capsule())
                .overlay(Capsule().stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5))

                if let mem = model.currentProcessMemoryBytes {
                    HStack(spacing: 4) {
                        Image(systemName: "chart.bar.fill")
                        Text("\(MetricFormat.storage(mem)) RAM")
                    }
                    .font(.caption2.weight(.medium))
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3.5)
                    .background(Color(nsColor: .controlBackgroundColor), in: Capsule())
                    .overlay(Capsule().stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5))
                }

                Button {
                    showingProbeSheet = true
                } label: {
                    Label("Probe HF…", systemImage: "magnifyingglass.circle")
                        .font(.caption2.weight(.medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Probe an arbitrary Hugging Face repository by header")
            }
        }
        .sheet(isPresented: $showingProbeSheet) {
            ModelProbeSheet(model: model)
        }
    }

    private var toolbarFilterRow: some View {
        HStack(spacing: 10) {
            Picker("Tab", selection: $selectedTab) {
                ForEach(HubTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 155)
            .accessibilityLabel("Catalog view")

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .font(.caption)
                    .accessibilityHidden(true)
                TextField("Search models…", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.callout)
                    .accessibilityLabel("Search models")
                    .accessibilityHint("Filters the model catalog by name, family, or notes")
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                            .accessibilityHidden(true)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Clear search")
                    .accessibilityHint("Empties the search field")
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5))
            .frame(minWidth: 100)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    Picker("Format", selection: $formatFilter) {
                        ForEach(FormatFilter.allCases) { f in
                            Text(f.rawValue).tag(f)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 110)
                    .accessibilityLabel("Format filter")

                    Picker("Capability", selection: $capabilityFilter) {
                        ForEach(CapabilityFilter.allCases) { c in
                            Text(c.rawValue).tag(c)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 125)
                    .accessibilityLabel("Capability filter")

                    Picker("Sort", selection: $sortOption) {
                        ForEach(SortOption.allCases) { s in
                            Text(s.rawValue).tag(s)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 120)
                    .accessibilityLabel("Sort order")
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var masterDetailContent: some View {
        HStack(spacing: 0) {
            masterList
                .frame(minWidth: 260, idealWidth: 310, maxWidth: 360)
                .frame(maxHeight: .infinity)

            Divider()

            detailPane
                .frame(minWidth: 280, maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var masterList: some View {
        ScrollView {
            LazyVStack(spacing: 6) {
                HStack {
                    Text(selectedTab == .discover ? "Curated Models" : "On Device Models")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(filteredEntries.count) available")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 12)
                .padding(.top, 10)
                .padding(.bottom, 4)

                if filteredEntries.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "tray")
                            .font(.largeTitle)
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                        Text("No matching models found.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 60)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("No matching models found")
                } else {
                    ForEach(filteredEntries) { entry in
                        let isInst = model.installed.contains(where: { $0.alias == entry.alias }) || entry.installed
                        let isAct = model.selected?.alias == entry.alias && model.session != nil
                        let isDownloading = model.isInstallingModel
                        let fraction = model.installProgressFraction
                        let rec = recommendations[entry.alias]

                        ModelCardView(
                            alias: entry.alias,
                            name: entry.name,
                            family: entry.family,
                            downloadBytes: entry.downloadBytes,
                            isInstalled: isInst,
                            isActive: isAct,
                            isDownloading: isDownloading,
                            downloadFraction: fraction,
                            recommendation: rec,
                            isSelected: selectedAlias == entry.alias,
                            onSelect: {
                                selectedAlias = entry.alias
                            }
                        )
                        .padding(.horizontal, 8)
                    }
                }
            }
            .padding(.bottom, 12)
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
    }

    @ViewBuilder
    private var detailPane: some View {
        if let selectedEntry {
            let inst = model.installed.first(where: { $0.alias == selectedEntry.alias })
            let rec = recommendations[selectedEntry.alias]
            ModelDetailPaneView(
                model: model,
                entry: selectedEntry,
                installedModel: inst,
                recommendation: rec
            )
        } else {
            VStack(spacing: 12) {
                Image(systemName: "square.grid.2x2")
                    .font(.system(size: 40))
                    .foregroundStyle(.secondary)
                Text("Select a model to view details and manage installation.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var selectedEntry: CatalogEntry? {
        if let selectedAlias, let entry = model.catalog.first(where: { $0.alias == selectedAlias }) {
            return entry
        }
        return filteredEntries.first
    }

    private var filteredEntries: [CatalogEntry] {
        var list = model.catalog

        if selectedTab == .onDevice {
            list = list.filter { entry in
                model.installed.contains(where: { $0.alias == entry.alias }) || entry.installed
            }
        }

        if !searchText.isEmpty {
            let query = searchText.lowercased()
            list = list.filter { entry in
                entry.alias.lowercased().contains(query)
                    || entry.name.lowercased().contains(query)
                    || entry.family.lowercased().contains(query)
                    || (entry.notes?.lowercased().contains(query) ?? false)
            }
        }

        switch formatFilter {
        case .all:
            break
        case .mlx:
            list = list.filter { !($0.alias.contains("gguf") || $0.name.contains("GGUF")) }
        case .gguf:
            list = list.filter { $0.alias.contains("gguf") || $0.name.contains("GGUF") }
        }

        switch capabilityFilter {
        case .all:
            break
        case .conversational:
            break
        case .reasoning:
            list = list.filter { $0.alias.contains("gptoss") || $0.alias.contains("museglimmer") || $0.alias.contains("qwen36") }
        case .moe:
            list = list.filter { $0.alias.contains("gemma") || $0.alias.contains("qwen") || $0.alias.contains("ornith") || $0.alias.contains("mixtral") || $0.alias.contains("ternary") || $0.alias.contains("gptoss") }
        case .dense:
            list = list.filter { $0.alias.contains("mistral") || $0.alias.contains("tinyllama") || $0.alias.contains("bonsai") || $0.alias.contains("museglimmer") }
        case .coding:
            list = list.filter { $0.alias.contains("gemma") || $0.alias.contains("qwen") || $0.alias.contains("mistral") }
        }

        switch sortOption {
        case .recommended:
            list.sort { lhs, rhs in
                let lhsRank = verdictRank(recommendations[lhs.alias]?.verdict)
                let rhsRank = verdictRank(recommendations[rhs.alias]?.verdict)
                if lhsRank != rhsRank { return lhsRank < rhsRank }
                return lhs.alias < rhs.alias
            }
        case .name:
            list.sort { $0.alias < $1.alias }
        case .size:
            list.sort { $0.downloadBytes < $1.downloadBytes }
        }

        return list
    }

    private func verdictRank(_ verdict: ModelRecommendation.FitVerdict?) -> Int {
        switch verdict {
        case .resident: return 0
        case .streams: return 1
        case .tight: return 2
        case .unknown, .none: return 3
        case .refused: return 4
        }
    }

    private func loadRecommendations() {
        if let recs = try? TurboSparkCatalog.recommend() {
            var map: [String: ModelRecommendation] = [:]
            for r in recs {
                map[r.alias] = r
            }
            recommendations = map
        }
    }
}
