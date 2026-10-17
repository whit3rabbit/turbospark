import Foundation
import TurboSpark

extension ModelManagerView {
    enum ArchitectureFilter: String, CaseIterable, Identifiable {
        case all = "All"
        case moeOnly = "MoE Only"
        case denseOnly = "Dense Only"

        var id: String { rawValue }
    }

    enum DrafterFilter: String, CaseIterable, Identifiable {
        case all = "All Drafters"
        case dynamicSloth = "Dynamic Sloth"
        case mtp = "MTP Head"

        var id: String { rawValue }
    }

    enum SourceFilter: String, CaseIterable, Identifiable {
        case all = "All Sources"
        case turboSpark = "TurboSpark"
        case lmStudio = "LM Studio"
        case custom = "Custom"

        var id: String { rawValue }
    }

    enum SortOption: String, CaseIterable, Identifiable {
        case name = "Name (A-Z)"
        case sizeDescending = "Size (Largest)"
        case sizeAscending = "Size (Smallest)"
        case dateDescending = "Recently Added"
        case family = "Family"

        var id: String { rawValue }
    }

    /// How the list is grouped. The raw values are catalog keys the sort menu
    /// renders; `location` (requirement 3.5) is the design's `GroupMode.location`.
    enum GroupOption: String, CaseIterable, Identifiable {
        case none = "None"
        case source = "By Source"
        case architecture = "By Architecture"
        case family = "By Family"
        case location = "By Location"

        var id: String { rawValue }
    }

    struct GroupedModels: Identifiable {
        /// Distinct per group even when two libraries share a name.
        let id: String
        let groupTitle: String
        let models: [InstalledModel]
        /// True when the title is already the text to show (a library name from
        /// a drive, or This Mac in the language). False when it is a catalog
        /// key the header looks up.
        let titleIsVerbatim: Bool

        init(id: String? = nil, groupTitle: String, models: [InstalledModel], titleIsVerbatim: Bool = false) {
            self.id = id ?? groupTitle
            self.groupTitle = groupTitle
            self.models = models
            self.titleIsVerbatim = titleIsVerbatim
        }
    }

    /// The row the detail pane shows for the selected path. It looks in the list
    /// the filters read FROM, not the one they produce: a model hidden by a
    /// filter is still listed and its selection should persist. If nothing there
    /// has the path, the model left the list (deleted, forgotten, the store
    /// re-synced), and the pane shows "nothing selected" instead of silently
    /// rebinding its Load and Delete to a model the user never chose (U6).
    static func selectedRow(path: String?, in rows: [InstalledModel]) -> InstalledModel? {
        guard let path else { return nil }
        return rows.first { $0.path == path }
    }

    static func filterModels(
        installed: [InstalledModel],
        searchText: String,
        showFavoritesOnly: Bool,
        availableOnly: Bool = false,
        architectureFilter: ArchitectureFilter,
        drafterFilter: DrafterFilter,
        sourceFilter: SourceFilter,
        selectedTag: String?,
        sortOption: SortOption,
        orgStore: ModelOrganizationStore
    ) -> [InstalledModel] {
        var list = installed

        // 1. Text search
        let trimmedQuery = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !trimmedQuery.isEmpty {
            list = list.filter { m in
                let notes = orgStore.notes(m).lowercased()
                let tags = orgStore.tags(m).joined(separator: " ").lowercased()
                let matchAlias = m.alias.lowercased().contains(trimmedQuery)
                let matchFamily = m.family.lowercased().contains(trimmedQuery)
                let matchPath = m.path.lowercased().contains(trimmedQuery)
                let matchNotes = notes.contains(trimmedQuery)
                let matchTags = tags.contains(trimmedQuery)
                return matchAlias || matchFamily || matchPath || matchNotes || matchTags
            }
        }

        // 2. Favorites only
        if showFavoritesOnly {
            list = list.filter { orgStore.isFavorite($0) }
        }

        // 2b. Available now (requirement 3.6): hide models whose drive is not
        // connected and models whose folder is gone. Decided from the row's own
        // availability and nothing else.
        if availableOnly {
            list = list.filter { $0.availability == .available }
        }

        // 3. Architecture filter
        switch architectureFilter {
        case .all: break
        case .moeOnly:
            list = list.filter { ModelFeatureDescriptor.resolve(installedModel: $0).routingType == .moe }
        case .denseOnly:
            list = list.filter { ModelFeatureDescriptor.resolve(installedModel: $0).routingType == .dense }
        }

        // 4. Drafter filter
        switch drafterFilter {
        case .all: break
        case .dynamicSloth:
            list = list.filter { ModelFeatureDescriptor.resolve(installedModel: $0).speculativeDrafter == .dynamicSloth }
        case .mtp:
            list = list.filter { ModelFeatureDescriptor.resolve(installedModel: $0).speculativeDrafter == .mtp }
        }

        // 5. Source filter
        switch sourceFilter {
        case .all: break
        case .turboSpark:
            list = list.filter { ModelFeatureDescriptor.resolve(installedModel: $0).storageSource == .turboSpark }
        case .lmStudio:
            list = list.filter { ModelFeatureDescriptor.resolve(installedModel: $0).storageSource == .lmStudio }
        case .custom:
            list = list.filter { ModelFeatureDescriptor.resolve(installedModel: $0).storageSource == .custom }
        }

        // 6. Tag filter
        if let selectedTag {
            list = list.filter { orgStore.tags($0).contains(selectedTag) }
        }

        // 7. Sort
        switch sortOption {
        case .name:
            list.sort { $0.alias.localizedStandardCompare($1.alias) == .orderedAscending }
        case .sizeDescending:
            list.sort { $0.installBytes > $1.installBytes }
        case .sizeAscending:
            list.sort { $0.installBytes < $1.installBytes }
        case .dateDescending:
            list.sort { $0.installedOn > $1.installedOn }
        case .family:
            list.sort { $0.family.localizedStandardCompare($1.family) == .orderedAscending }
        }

        return list
    }

    /// - Parameters:
    ///   - libraries: the libraries as `LibraryState` lists them, This Mac
    ///     first. Only `.location` reads them.
    static func groupModels(
        filteredModels: [InstalledModel],
        groupOption: GroupOption,
        libraries: [ModelLibrary] = [],
        language: AppLanguage = AppStrings.currentLanguage
    ) -> [GroupedModels] {
        switch groupOption {
        case .location:
            return locationGroups(filteredModels, libraries: libraries, language: language)
        case .none:
            return [GroupedModels(groupTitle: "All Models", models: filteredModels)]
        case .source:
            let groups = Dictionary(grouping: filteredModels) { m in
                ModelFeatureDescriptor.resolve(installedModel: m).storageSource.rawValue
            }
            return groups.keys.sorted().map { GroupedModels(groupTitle: $0, models: groups[$0] ?? []) }
        case .architecture:
            let groups = Dictionary(grouping: filteredModels) { m in
                ModelFeatureDescriptor.resolve(installedModel: m).routingType.rawValue
            }
            return groups.keys.sorted().map { GroupedModels(groupTitle: $0, models: groups[$0] ?? []) }
        case .family:
            let groups = Dictionary(grouping: filteredModels) { m in
                m.family.isEmpty ? "Other" : m.family.capitalized
            }
            return groups.keys.sorted().map { GroupedModels(groupTitle: $0, models: groups[$0] ?? []) }
        }
    }

    /// Groups the list under where each model is (requirement 3.5).
    ///
    /// **THE ORDER IS WHAT THE USER NEEDS FIRST:** This Mac, then the libraries
    /// that are connected, then the ones that are away, then custom folders
    /// (`pull --out` installs and every folder only a scan found: LM Studio and
    /// the user's extra scan folders), then models of a library nothing knows.
    /// Libraries keep the order `LibraryState` lists them in. A group with no
    /// rows is not drawn, and rows keep the order the sort gave them.
    ///
    /// **A GROUP IS A LIBRARY, NOT A NAME.** Two drives can both be called
    /// "Backup", so a group's id is the library's, and its title is the library's
    /// name through `UntrustedText.display` (a marker can name its library
    /// anything). This Mac's title is localized from the row's kind, never from
    /// the name, which a marker can also use. A row of a library the list does
    /// not know any more is grouped under the name the row itself carries.
    private static func locationGroups(
        _ models: [InstalledModel], libraries: [ModelLibrary], language: AppLanguage
    ) -> [GroupedModels] {
        struct Bucket {
            let id: String
            let title: String
            let rank: Int
            let order: Int
            var models: [InstalledModel]
        }
        var buckets: [String: Bucket] = [:]
        var firstSeen: [String: Int] = [:]
        let listed = Dictionary(libraries.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        var listPosition: [String: Int] = [:]
        for (index, library) in libraries.enumerated() where listPosition[library.id] == nil {
            listPosition[library.id] = index
        }

        for model in models {
            let id: String
            let title: String
            let rank: Int
            var order = 0
            if ModelIdentityKey.isPlaceholder(model) {
                (id, title, rank) = ("custom", AppStrings.localized("Custom Folder", language: language), 3)
            } else {
                switch model.locationKind {
                case .thisMac:
                    (id, title, rank) = ("internal", AppStrings.localized("This Mac", language: language), 0)
                case .library:
                    let libraryId = model.library ?? ""
                    let library = listed[libraryId]
                    id = "library:\(libraryId)"
                    rank = library?.available == true ? 1 : 2
                    order = listPosition[libraryId] ?? Int.max
                    let name = (library?.name).flatMap { $0.isEmpty ? nil : $0 }
                        ?? model.libraryName.flatMap { $0.isEmpty ? nil : $0 }
                    title = name.map { UntrustedText.display($0) }
                        ?? AppStrings.localized("Unknown library", language: language)
                case .custom:
                    (id, title, rank) = ("custom", AppStrings.localized("Custom Folder", language: language), 3)
                case .unknown:
                    (id, title, rank) = ("unknown", AppStrings.localized("Unknown library", language: language), 4)
                }
            }
            if buckets[id] == nil {
                firstSeen[id] = firstSeen.count
                buckets[id] = Bucket(id: id, title: title, rank: rank, order: order, models: [])
            }
            buckets[id]?.models.append(model)
        }

        return buckets.values
            .sorted { lhs, rhs in
                (lhs.rank, lhs.order, firstSeen[lhs.id] ?? 0) < (rhs.rank, rhs.order, firstSeen[rhs.id] ?? 0)
            }
            .map { GroupedModels(id: $0.id, groupTitle: $0.title, models: $0.models, titleIsVerbatim: true) }
    }
}
