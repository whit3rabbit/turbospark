import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The dedicated Model Manager view for organizing, inspecting, and managing downloaded models.
@MainActor
struct ModelManagerView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    @State private var searchText = ""
    @State private var architectureFilter: ArchitectureFilter = .all
    @State private var drafterFilter: DrafterFilter = .all
    @State private var sourceFilter: SourceFilter = .all
    @State private var selectedTag: String? = nil
    @State private var showFavoritesOnly = false
    @State private var sortOption: SortOption = .name
    @State private var groupOption: GroupOption = .none
    @State private var selectedModelPath: String?

    var body: some View {
        VStack(spacing: 0) {
            ModelManagerHeaderView(model: model)
            Divider()
            // The kind is chosen in the panel beside the rail
            // (`InstalledSidebarPanel`), which also says where audio models live.
            if model.installedModelKind == .text {
                textModelContent
            } else {
                ImageModelManagerView(model: model)
            }
        }
        .task {
            if selectedModelPath == nil {
                selectedModelPath = model.selected?.path ?? model.installed.first?.path
            }
        }
    }

    private var textModelContent: some View {
        VStack(spacing: 0) {
            ModelManagerFilterBarView(
                searchText: $searchText,
                architectureFilter: $architectureFilter,
                drafterFilter: $drafterFilter,
                sourceFilter: $sourceFilter,
                sortOption: $sortOption,
                groupOption: $groupOption,
                selectedTag: $selectedTag,
                showFavoritesOnly: $showFavoritesOnly
            )
            Divider()
            masterDetailContent
        }
    }

    // MARK: - Master Detail Content

    private var masterDetailContent: some View {
        HStack(spacing: 0) {
            masterList
                .frame(minWidth: 260, idealWidth: 320, maxWidth: 360)
                .frame(maxHeight: .infinity)

            Rectangle()
                .fill(.appBorder)
                .frame(width: AppChromeLayout.dividerWidth)

            detailPane
                .frame(minWidth: 280, maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: - Master List

    private var masterList: some View {
        ScrollView {
            LazyVStack(spacing: 4) {
                if filteredModels.isEmpty {
                    emptyListState
                } else if groupOption == .none {
                    ForEach(filteredModels) { m in
                        rowFor(m)
                    }
                } else {
                    ForEach(groupedModels, id: \.groupTitle) { group in
                        Section {
                            ForEach(group.models) { m in
                                rowFor(m)
                            }
                        } header: {
                            HStack {
                                Text(LocalizedStringKey(group.groupTitle), bundle: .module)
                                    .themedFont(.small, weight: .bold)
                                    .foregroundStyle(.appSecondary)
                                Spacer()
                                Text(verbatim: "\(group.models.count)")
                                    .themedFont(.tiny).monospacedDigit()
                                    .foregroundStyle(.tertiary)
                            }
                            .padding(.horizontal, 8)
                            .padding(.top, 10)
                            .padding(.bottom, 2)
                        }
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
        }
    }

    private func rowFor(_ m: InstalledModel) -> some View {
        InstalledModelRowView(
            modelItem: m,
            isSelected: selectedModelPath == m.path,
            isLoaded: model.selected?.path == m.path && model.session != nil,
            isFavorite: orgStore.isFavorite(alias: m.alias, path: m.path),
            tags: orgStore.tags(alias: m.alias, path: m.path),
            onSelect: {
                selectedModelPath = m.path
            }
        )
    }

    @ViewBuilder
    private var emptyListState: some View {
        if model.installed.isEmpty {
            AppEmptyState(
                systemImage: "internaldrive",
                title: Text("No Models Installed", bundle: .module),
                message: Text("Download models from the Discover catalog or scan an external folder.", bundle: .module)
            ) {
                Button {
                    model.activeSection = .modelHub
                } label: {
                    Label { Text("Discover Models", bundle: .module) } icon: { Image(systemName: "shippingbox") }
                }
                .appAction(.secondary, size: .small)
            }
            .padding(.top, 8)
        } else {
            AppEmptyState(
                systemImage: "line.3.horizontal.decrease.circle",
                title: Text("No Matching Models", bundle: .module),
                message: Text("Try clearing search or filter terms.", bundle: .module))
            .padding(.top, 8)
        }
    }

    // MARK: - Detail Pane

    @ViewBuilder
    private var detailPane: some View {
        if let selectedModel {
            InstalledModelDetailPaneView(model: model, installedModel: selectedModel)
        } else {
            AppEmptyState(
                systemImage: "internaldrive",
                title: Text("No Model Selected", bundle: .module),
                message: Text(
                    "Select an installed model to view specifications, runtime options, or load it into memory.",
                    bundle: .module))
        }
    }

    private var selectedModel: InstalledModel? {
        // Looks up in `model.installed` (the master list), not
        // `filteredModels`: a model hidden by the current filter is still
        // installed and its selection should persist. If the tracked path
        // matches nothing there, the model actually left the list (deleted,
        // or the store re-synced), and the fix is to show the "nothing
        // selected" empty state below -- NOT to silently substitute
        // whatever `filteredModels.first` happens to be, which used to
        // rebind the detail pane's live Load/Delete actions to a model the
        // user never chose (U6).
        guard let selectedModelPath else { return nil }
        return model.installed.first(where: { $0.path == selectedModelPath })
    }

    // MARK: - Filtering & Grouping Logic

    private var filteredModels: [InstalledModel] {
        Self.filterModels(
            installed: model.installed,
            searchText: searchText,
            showFavoritesOnly: showFavoritesOnly,
            architectureFilter: architectureFilter,
            drafterFilter: drafterFilter,
            sourceFilter: sourceFilter,
            selectedTag: selectedTag,
            sortOption: sortOption,
            orgStore: orgStore
        )
    }

    private var groupedModels: [GroupedModels] {
        Self.groupModels(
            filteredModels: filteredModels,
            groupOption: groupOption
        )
    }
}
