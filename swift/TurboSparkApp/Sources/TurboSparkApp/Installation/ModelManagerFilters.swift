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

    enum GroupOption: String, CaseIterable, Identifiable {
        case none = "None"
        case source = "By Source"
        case architecture = "By Architecture"
        case family = "By Family"

        var id: String { rawValue }
    }

    struct GroupedModels {
        let groupTitle: String
        let models: [InstalledModel]
    }

    static func filterModels(
        installed: [InstalledModel],
        searchText: String,
        showFavoritesOnly: Bool,
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
                let notes = orgStore.notes(alias: m.alias, path: m.path).lowercased()
                let tags = orgStore.tags(alias: m.alias, path: m.path).joined(separator: " ").lowercased()
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
            list = list.filter { orgStore.isFavorite(alias: $0.alias, path: $0.path) }
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
            list = list.filter { orgStore.tags(alias: $0.alias, path: $0.path).contains(selectedTag) }
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

    static func groupModels(
        filteredModels: [InstalledModel],
        groupOption: GroupOption
    ) -> [GroupedModels] {
        switch groupOption {
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
}
