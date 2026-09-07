import SwiftUI
import TurboSpark

/// Filter and control bar for ModelManagerView (search, favorites, architecture/drafter/source pickers, sort/group, tags).
struct ModelManagerFilterBarView: View {
    @Binding var searchText: String
    @Binding var architectureFilter: ModelManagerView.ArchitectureFilter
    @Binding var drafterFilter: ModelManagerView.DrafterFilter
    @Binding var sourceFilter: ModelManagerView.SourceFilter
    @Binding var sortOption: ModelManagerView.SortOption
    @Binding var groupOption: ModelManagerView.GroupOption
    @Binding var selectedTag: String?
    @Binding var showFavoritesOnly: Bool

    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                searchField

                Button {
                    showFavoritesOnly.toggle()
                } label: {
                    Image(systemName: showFavoritesOnly ? "star.fill" : "star")
                        .themedFont(points: 11)
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
                .accessibilityLabel("Filter favorites")
                .accessibilityValue(showFavoritesOnly ? "Favorites only" : "All models")
                .accessibilityAddTraits(showFavoritesOnly ? [.isButton, .isSelected] : .isButton)

                Menu {
                    Picker("Architecture", selection: $architectureFilter) {
                        ForEach(ModelManagerView.ArchitectureFilter.allCases) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(architectureFilter.rawValue)
                            .themedFont(points: 11)
                        Image(systemName: "chevron.down")
                            .themedFont(points: 8)
                            .accessibilityHidden(true)
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Filter by architecture")
                .accessibilityLabel("Filter by architecture")
                .accessibilityValue(architectureFilter.rawValue)

                Menu {
                    Picker("Drafter", selection: $drafterFilter) {
                        ForEach(ModelManagerView.DrafterFilter.allCases) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(drafterFilter.rawValue)
                            .themedFont(points: 11)
                        Image(systemName: "chevron.down")
                            .themedFont(points: 8)
                            .accessibilityHidden(true)
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Filter by drafter")
                .accessibilityLabel("Filter by drafter")
                .accessibilityValue(drafterFilter.rawValue)

                Menu {
                    Picker("Source", selection: $sourceFilter) {
                        ForEach(ModelManagerView.SourceFilter.allCases) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(sourceFilter.rawValue)
                            .themedFont(points: 11)
                        Image(systemName: "chevron.down")
                            .themedFont(points: 8)
                            .accessibilityHidden(true)
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Filter by source")
                .accessibilityLabel("Filter by source")
                .accessibilityValue(sourceFilter.rawValue)

                Spacer(minLength: 4)

                Menu {
                    Section("Sort By") {
                        Picker("Sort", selection: $sortOption) {
                            ForEach(ModelManagerView.SortOption.allCases) { opt in
                                Text(opt.rawValue).tag(opt)
                            }
                        }
                    }
                    Section("Group By") {
                        Picker("Group", selection: $groupOption) {
                            ForEach(ModelManagerView.GroupOption.allCases) { grp in
                                Text(grp.rawValue).tag(grp)
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.up.arrow.down")
                            .themedFont(points: 10)
                            .accessibilityHidden(true)
                        Text(sortOption.rawValue)
                            .themedFont(points: 11)
                    }
                    .padding(.horizontal, 8)
                    .frame(height: 22)
                    .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Sort and group models")
                .accessibilityLabel("Sort and group models")
                .accessibilityValue("\(sortOption.rawValue), \(groupOption.rawValue)")
            }

            // Tag filter bar if tags exist
            let knownTags = orgStore.allKnownTags
            if !knownTags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        Text("Tags:", bundle: .module)
                            .themedFont(points: 10)
                            .foregroundStyle(.tertiary)

                        Button {
                            selectedTag = nil
                        } label: {
                            Text("All", bundle: .module)
                                .themedFont(points: 10, weight: selectedTag == nil ? .bold : .regular)
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
                                    .themedFont(points: 10, weight: selectedTag == tag ? .bold : .regular)
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
                .themedFont(points: 10)
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            TextField("Filter installed models...", text: $searchText)
                .textFieldStyle(.plain)
                .themedFont(points: 11)
                .frame(minWidth: 120, idealWidth: 160)
                .accessibilityLabel("Filter installed models")
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .themedFont(points: 10)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tertiary)
                .help("Clear search")
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 8)
        .frame(height: 22)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay { Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5) }
    }
}
