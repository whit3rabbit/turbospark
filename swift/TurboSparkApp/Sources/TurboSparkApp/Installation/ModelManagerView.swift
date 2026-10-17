import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The dedicated Model Manager view for organizing, inspecting, and managing downloaded models.
@MainActor
struct ModelManagerView: View {
    @ObservedObject var model: AppModel
    /// Observed directly: `AppModel` does not publish `libraryState` (a mount
    /// would redraw every screen), so a list that reads offline rows from it must
    /// hold it, or a drive that is unplugged would not redraw this view.
    @ObservedObject private var libraryState: LibraryState
    @ObservedObject private var orgStore = ModelOrganizationStore.shared
    /// How the list is grouped and whether it shows only what is available now:
    /// kept across launches (task 6.5).
    @StateObject private var preferences = ModelManagerFilterPreferences()
    /// Read so the strings built from `AppStrings` follow the language picker.
    @Environment(\.locale) private var locale

    private enum ManagerSection: String, CaseIterable, Identifiable {
        case text
        case images

        var id: String { rawValue }
    }

    @State private var managerSection: ManagerSection = .text
    @State private var searchText = ""
    @State private var architectureFilter: ArchitectureFilter = .all
    @State private var drafterFilter: DrafterFilter = .all
    @State private var sourceFilter: SourceFilter = .all
    @State private var selectedTag: String? = nil
    @State private var showFavoritesOnly = false
    @State private var sortOption: SortOption = .name
    @State private var selectedModelPath: String?

    init(model: AppModel) {
        self.model = model
        _libraryState = ObservedObject(wrappedValue: model.libraryState)
    }

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    /// Everything the list can show: the available rows and the rows of libraries
    /// that are away or whose folder is gone.
    private var displayModels: [InstalledModel] {
        model.installedForDisplay
    }

    var body: some View {
        VStack(spacing: 0) {
            ModelManagerHeaderView(model: model)
            managerSectionPicker
            Divider()
            if managerSection == .text {
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

    /// The model-type switch plus the scope line. The picker's label is
    /// hidden (it is the VoiceOver name only) and the control sizes itself:
    /// the old 240pt frame squeezed a visible label until it wrapped.
    /// Audio installs live in the Audio workspace, so the scope line says so
    /// and offers the way there instead of leaving a third category missing.
    private var managerSectionPicker: some View {
        AppAdaptiveHeader {
            Picker(selection: $managerSection) {
                Text("Text Models", bundle: .module).tag(ManagerSection.text)
                Text("Image Models", bundle: .module).tag(ManagerSection.images)
            } label: {
                Text("Installed model type", bundle: .module)
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .fixedSize()
        } actions: {
            Text("Audio models are managed in Audio.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Button {
                model.activeSection = .audio
            } label: {
                Label { Text("Open Audio", bundle: .module) } icon: { Image(systemName: "waveform") }
            }
            .appAction(.quiet, size: .small)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var textModelContent: some View {
        VStack(spacing: 0) {
            ModelManagerFilterBarView(
                searchText: $searchText,
                architectureFilter: $architectureFilter,
                drafterFilter: $drafterFilter,
                sourceFilter: $sourceFilter,
                sortOption: $sortOption,
                groupOption: $preferences.groupOption,
                selectedTag: $selectedTag,
                showFavoritesOnly: $showFavoritesOnly,
                availableOnly: $preferences.availableOnly
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
                } else if preferences.groupOption == .none {
                    ForEach(filteredModels) { m in
                        rowFor(m)
                    }
                } else {
                    ForEach(groupedModels) { group in
                        Section {
                            ForEach(group.models) { m in
                                rowFor(m)
                            }
                        } header: {
                            HStack {
                                groupTitle(group)
                                    .themedFont(.small, weight: .bold)
                                    .foregroundStyle(.appSecondary)
                                    .accessibilityAddTraits(.isHeader)
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
            rowModel: InstalledModelRowModel.make(for: m, in: libraryState, language: language),
            isSelected: selectedModelPath == m.path,
            isLoaded: model.selected?.path == m.path && model.session != nil,
            isFavorite: orgStore.isFavorite(m),
            tags: orgStore.tags(m),
            onSelect: {
                selectedModelPath = m.path
            }
        )
    }

    /// A location group's title is already the text to show (a library name from
    /// a drive, or This Mac in the language); the other modes name a catalog key.
    private func groupTitle(_ group: GroupedModels) -> Text {
        group.titleIsVerbatim
            ? Text(verbatim: group.groupTitle)
            : Text(LocalizedStringKey(group.groupTitle), bundle: .module)
    }

    @ViewBuilder
    private var emptyListState: some View {
        if displayModels.isEmpty {
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
        } else if preferences.availableOnly && hiddenOnlyByAvailableNow {
            // Every model that matches is on a drive that is away or has lost
            // its files: say so, and point at the switch in the filter bar that
            // brings them back (a second switch here would read as "turn it on").
            AppEmptyState(
                systemImage: "externaldrive.badge.xmark",
                title: Text("No Models Available Now", bundle: .module),
                message: Text("Connect a drive, or turn off Available now to see every model.", bundle: .module))
            .padding(.top, 8)
        } else {
            AppEmptyState(
                systemImage: "line.3.horizontal.decrease.circle",
                title: Text("No Matching Models", bundle: .module),
                message: Text("Try clearing search or filter terms.", bundle: .module))
            .padding(.top, 8)
        }
    }

    /// Turning "Available now" off would show something: the filter is what
    /// emptied the list, not the search or the other filters.
    private var hiddenOnlyByAvailableNow: Bool {
        !Self.filterModels(
            installed: displayModels,
            searchText: searchText,
            showFavoritesOnly: showFavoritesOnly,
            availableOnly: false,
            architectureFilter: architectureFilter,
            drafterFilter: drafterFilter,
            sourceFilter: sourceFilter,
            selectedTag: selectedTag,
            sortOption: sortOption,
            orgStore: orgStore
        ).isEmpty
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
        // Looks up in the master list (`displayModels`, which includes the
        // offline rows), not `filteredModels`: a model hidden by the current
        // filter, "Available now" included, is still listed and its selection
        // should persist (`selectedRow`).
        Self.selectedRow(path: selectedModelPath, in: displayModels)
    }

    // MARK: - Filtering & Grouping Logic

    private var filteredModels: [InstalledModel] {
        Self.filterModels(
            installed: displayModels,
            searchText: searchText,
            showFavoritesOnly: showFavoritesOnly,
            availableOnly: preferences.availableOnly,
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
            groupOption: preferences.groupOption,
            libraries: libraryState.libraries,
            language: language
        )
    }
}
