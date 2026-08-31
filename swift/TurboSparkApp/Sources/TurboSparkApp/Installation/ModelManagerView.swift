import AppKit
import SwiftUI
import TurboSpark

/// The dedicated Model Manager view for organizing, inspecting, and managing downloaded models.
struct ModelManagerView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var orgStore = ModelOrganizationStore.shared

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
            header
            Divider()
            filterAndControlBar
            Divider()
            masterDetailContent
        }
        .task {
            if selectedModelPath == nil {
                selectedModelPath = model.selected?.path ?? model.installed.first?.path
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            HStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.teal.opacity(0.15))
                        .frame(width: 30, height: 30)
                    Image(systemName: "internaldrive.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Color.teal)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Installed Models")
                            .font(.system(size: 14, weight: .semibold))
                            .accessibilityAddTraits(.isHeader)

                        Text("LOCAL STORAGE")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(Color.teal.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                            .foregroundStyle(Color.teal)
                    }

                    Text(summaryText)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
            }

            Spacer(minLength: 8)

            Button {
                model.refreshModels()
                model.showToast("Refreshed model libraries", style: .info)
            } label: {
                Label("Rescan", systemImage: "arrow.clockwise")
                    .font(.system(size: 11, weight: .medium))
                    .frame(height: 22)
                    .padding(.horizontal, 8)
            }
            .buttonStyle(.plain)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: Capsule())
            .help("Rescan storage directories for models")

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label("Add Folder…", systemImage: "folder.badge.plus")
                    .font(.system(size: 11, weight: .medium))
                    .frame(height: 22)
                    .padding(.horizontal, 8)
            }
            .buttonStyle(.plain)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: Capsule())
            .help("Add an external folder containing .gturbo bundles or .gguf files")

            Button {
                model.activeSection = .modelHub
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "shippingbox.fill")
                        .font(.system(size: 10))
                    Text("Discover Hub →")
                        .font(.system(size: 11, weight: .medium))
                }
                .frame(height: 22)
                .padding(.horizontal, 10)
            }
            .buttonStyle(.plain)
            .background(TurboSparkTheme.accentColor.opacity(0.14), in: Capsule())
            .foregroundStyle(TurboSparkTheme.accentColor)
            .help("Browse the curated catalog and download models")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var summaryText: String {
        let count = model.installed.count
        let totalBytes = model.installed.reduce(UInt64(0)) { $0 + $1.installBytes }
        var parts = ["\(count) installed"]
        if totalBytes > 0 {
            parts.append("\(MetricFormat.storage(totalBytes)) on disk")
        }
        if let sel = model.selected, model.session != nil {
            parts.append("Active: \(sel.alias)")
        }
        return parts.joined(separator: " \u{2022} ")
    }

    // MARK: - Filter Bar

    private var filterAndControlBar: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                searchField

                Button {
                    showFavoritesOnly.toggle()
                } label: {
                    Image(systemName: showFavoritesOnly ? "star.fill" : "star")
                        .font(.system(size: 11))
                        .foregroundStyle(showFavoritesOnly ? Color.yellow : Color.secondary)
                        .padding(.horizontal, 7)
                        .frame(height: 22)
                        .background(
                            showFavoritesOnly ? Color.yellow.opacity(0.18) : Color(nsColor: .quaternaryLabelColor).opacity(0.3),
                            in: RoundedRectangle(cornerRadius: 6)
                        )
                }
                .buttonStyle(.plain)
                .help("Show Favorites Only")

                Menu {
                    Picker("Architecture", selection: $architectureFilter) {
                        ForEach(ArchitectureFilter.allCases) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(architectureFilter.rawValue)
                            .font(.system(size: 11))
                        Image(systemName: "chevron.down")
                            .font(.system(size: 8))
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                Menu {
                    Picker("Drafter", selection: $drafterFilter) {
                        ForEach(DrafterFilter.allCases) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(drafterFilter.rawValue)
                            .font(.system(size: 11))
                        Image(systemName: "chevron.down")
                            .font(.system(size: 8))
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                Menu {
                    Picker("Source", selection: $sourceFilter) {
                        ForEach(SourceFilter.allCases) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(sourceFilter.rawValue)
                            .font(.system(size: 11))
                        Image(systemName: "chevron.down")
                            .font(.system(size: 8))
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                Spacer(minLength: 4)

                Menu {
                    Section("Sort By") {
                        Picker("Sort", selection: $sortOption) {
                            ForEach(SortOption.allCases) { opt in
                                Text(opt.rawValue).tag(opt)
                            }
                        }
                    }
                    Section("Group By") {
                        Picker("Group", selection: $groupOption) {
                            ForEach(GroupOption.allCases) { grp in
                                Text(grp.rawValue).tag(grp)
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.up.arrow.down")
                            .font(.system(size: 10))
                        Text(sortOption.rawValue)
                            .font(.system(size: 11))
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }

            // Tag filter bar if tags exist
            let knownTags = orgStore.allKnownTags
            if !knownTags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        Text("Tags:")
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)

                        Button {
                            selectedTag = nil
                        } label: {
                            Text("All")
                                .font(.system(size: 10, weight: selectedTag == nil ? .bold : .regular))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(selectedTag == nil ? Color.accentColor.opacity(0.2) : Color.clear, in: Capsule())
                                .foregroundStyle(selectedTag == nil ? Color.accentColor : Color.secondary)
                        }
                        .buttonStyle(.plain)

                        ForEach(knownTags, id: \.self) { tag in
                            Button {
                                if selectedTag == tag {
                                    selectedTag = nil
                                } else {
                                    selectedTag = tag
                                }
                            } label: {
                                Text(tag)
                                    .font(.system(size: 10, weight: selectedTag == tag ? .bold : .regular))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(selectedTag == tag ? Color.accentColor.opacity(0.2) : Color.clear, in: Capsule())
                                    .foregroundStyle(selectedTag == tag ? Color.accentColor : Color.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var searchField: some View {
        HStack(spacing: 5) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
            TextField("Filter installed models...", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 11))
                .frame(minWidth: 120, idealWidth: 160)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 8)
        .frame(height: 22)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay { Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5) }
    }

    // MARK: - Master Detail Content

    private var masterDetailContent: some View {
        HStack(spacing: 0) {
            masterList
                .frame(minWidth: 260, idealWidth: 320, maxWidth: 360)
                .frame(maxHeight: .infinity)

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
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
                        modelRow(m)
                    }
                } else {
                    ForEach(groupedModels, id: \.groupTitle) { group in
                        Section {
                            ForEach(group.models) { m in
                                modelRow(m)
                            }
                        } header: {
                            HStack {
                                Text(group.groupTitle)
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text("\(group.models.count)")
                                    .font(.caption2.monospacedDigit())
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

    private func modelRow(_ m: InstalledModel) -> some View {
        // Keyed on path, not alias (U6): a scanned LM Studio/Custom row
        // can share an alias with another installed row, and selection must
        // not silently resolve to the wrong one.
        let isSelected = selectedModelPath == m.path
        let isLoaded = model.selected?.path == m.path && model.session != nil
        let isFav = orgStore.isFavorite(alias: m.alias, path: m.path)
        let desc = ModelFeatureDescriptor.resolve(installedModel: m)
        let visuals = ModelFamilyVisuals.resolve(alias: m.alias, family: m.family, name: m.alias)
        let tags = orgStore.tags(alias: m.alias, path: m.path)

        return Button {
            selectedModelPath = m.path
        } label: {
            HStack(alignment: .center, spacing: 10) {
                ModelLogoView(visuals: visuals, size: 36, cornerRadius: 8)

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 4) {
                        Text(m.alias)
                            .font(.system(.body, design: .rounded).weight(.semibold))
                            .lineLimit(1)

                        if isFav {
                            Image(systemName: "star.fill")
                                .font(.system(size: 9))
                                .foregroundStyle(Color.yellow)
                        }

                        Spacer(minLength: 2)

                        if isLoaded {
                            HStack(spacing: 3) {
                                Circle().fill(Color.green).frame(width: 5, height: 5)
                                Text("Loaded")
                                    .font(.caption2.weight(.bold))
                                    .foregroundStyle(Color.green)
                            }
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(Color.green.opacity(0.12), in: Capsule())
                        }
                    }

                    // Feature & Source Badges
                    HStack(spacing: 4) {
                        Text(desc.storageSource.shortLabel)
                            .font(.system(size: 8, weight: .semibold))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1.5)
                            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.35), in: RoundedRectangle(cornerRadius: 3))
                            .foregroundStyle(.secondary)

                        if desc.routingType == .moe {
                            ModelFeatureBadgeView.moe(details: "MoE", style: .compact)
                        } else {
                            ModelFeatureBadgeView.dense(details: "Dense", style: .compact)
                        }

                        ModelFeatureBadgeView.quant(desc.quantFormat, style: .compact)

                        if desc.speculativeDrafter == .dynamicSloth {
                            ModelFeatureBadgeView.dynamicSloth(style: .compact)
                        } else if desc.speculativeDrafter == .mtp {
                            ModelFeatureBadgeView.mtp(style: .compact)
                        }

                        Spacer(minLength: 2)

                        HStack(spacing: 2) {
                            Image(systemName: "internaldrive")
                                .font(.system(size: 8))
                                .foregroundStyle(.tertiary)
                            Text(MetricFormat.storage(m.installBytes))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }

                    if !tags.isEmpty {
                        HStack(spacing: 3) {
                            ForEach(tags.prefix(3), id: \.self) { tag in
                                Text(tag)
                                    .font(.system(size: 8, weight: .medium))
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 1)
                                    .background(Color.teal.opacity(0.12), in: RoundedRectangle(cornerRadius: 3))
                                    .foregroundStyle(Color.teal)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .background(
            isSelected
                ? Color.teal.opacity(0.14)
                : Color.clear,
            in: RoundedRectangle(cornerRadius: 8)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    isSelected ? Color.teal.opacity(0.4) : Color.clear,
                    lineWidth: 1
                )
        )
    }

    private var emptyListState: some View {
        VStack(spacing: 12) {
            Image(systemName: "internaldrive")
                .font(.system(size: 32))
                .foregroundStyle(.quaternary)
            Text(model.installed.isEmpty ? "No Models Installed" : "No Matching Models")
                .font(.callout.weight(.medium))
            Text(model.installed.isEmpty
                ? "Download models from the Discover catalog or scan an external folder."
                : "Try clearing search or filter terms.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 240)

            if model.installed.isEmpty {
                Button {
                    model.activeSection = .modelHub
                } label: {
                    Label("Discover Models in Hub", systemImage: "shippingbox.fill")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.plain)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(.white)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.top, 40)
    }

    // MARK: - Detail Pane

    @ViewBuilder
    private var detailPane: some View {
        if let selectedModel {
            InstalledModelDetailPaneView(model: model, installedModel: selectedModel)
        } else {
            VStack(spacing: 8) {
                Image(systemName: "internaldrive")
                    .font(.system(size: 30))
                    .foregroundStyle(.quaternary)
                Text("No Model Selected")
                    .font(.callout.weight(.medium))
                Text("Select an installed model to view specifications, runtime options, or load it into memory.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 300)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
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
        var list = model.installed

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

    private struct GroupedModels {
        let groupTitle: String
        let models: [InstalledModel]
    }

    private var groupedModels: [GroupedModels] {
        let list = filteredModels
        switch groupOption {
        case .none:
            return [GroupedModels(groupTitle: "All Models", models: list)]
        case .source:
            let groups = Dictionary(grouping: list) { m in
                ModelFeatureDescriptor.resolve(installedModel: m).storageSource.rawValue
            }
            return groups.keys.sorted().map { GroupedModels(groupTitle: $0, models: groups[$0] ?? []) }
        case .architecture:
            let groups = Dictionary(grouping: list) { m in
                ModelFeatureDescriptor.resolve(installedModel: m).routingType.rawValue
            }
            return groups.keys.sorted().map { GroupedModels(groupTitle: $0, models: groups[$0] ?? []) }
        case .family:
            let groups = Dictionary(grouping: list) { m in
                m.family.isEmpty ? "Other" : m.family.capitalized
            }
            return groups.keys.sorted().map { GroupedModels(groupTitle: $0, models: groups[$0] ?? []) }
        }
    }
}
