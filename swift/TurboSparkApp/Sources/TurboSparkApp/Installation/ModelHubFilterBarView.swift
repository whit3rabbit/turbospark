import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Filter bar and chips for narrowing and sorting models in the Model Hub.
@MainActor
struct ModelHubFilterBarView: View {
    @Binding var filter: ModelHubFilter
    let catalog: [CatalogEntry]
    let recommendations: [String: ModelRecommendation]

    var body: some View {
        HStack(spacing: 8) {
            Picker(selection: $filter.tab) {
                ForEach(ModelHubFilter.Tab.allCases) { tab in
                    tabText(tab).tag(tab)
                }
            } label: { Text("View", bundle: .module) }
            .pickerStyle(.segmented)
            .controlSize(.small)
            .themedFont(.tiny)
            .frame(width: 260)
            .labelsHidden()
            .accessibilityLabel("Catalog view")

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    filterChip(
                        title: "Format & Package",
                        options: ModelHubFilter.formatOptions(for: catalog),
                        selection: $filter.format)

                    filterChip(
                        title: "Architecture",
                        options: ModelHubFilter.capabilityOptions(for: catalog),
                        selection: $filter.capability)

                    filterChip(
                        title: "Fits Memory",
                        options: ModelHubFilter.fitOptions(
                            for: catalog,
                            recommendations: recommendations),
                        selection: $filter.fit)

                    sortChip

                    if filter.isNarrowed {
                        Button {
                            filter.clearNarrowing()
                        } label: {
                            Label { Text("Clear", bundle: .module) } icon: { Image(systemName: "xmark") }
                                .themedFont(.tiny, weight: .medium)
                                .labelStyle(.titleOnly)
                                .padding(.horizontal, 8)
                                .frame(height: 22)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.appSecondary)
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
        title: LocalizedStringKey,
        options: [String],
        selection: Binding<ModelHubFilter.Selection>
    ) -> some View {
        Menu {
            Button { selection.wrappedValue = nil } label: {
                Text("All", bundle: .module)
            }
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
                if let selected = selection.wrappedValue {
                    Text(selected)
                        .themedFont(.tiny, weight: .medium)
                        .lineLimit(1)
                } else {
                    Text(title, bundle: .module)
                        .themedFont(.tiny)
                        .lineLimit(1)
                }
                Image(systemName: "chevron.down")
                    .themedFont(.micro, weight: .bold)
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
        .overlay { Capsule().stroke(.appBorder, lineWidth: 0.5) }
        .disabled(options.isEmpty && selection.wrappedValue == nil)
        .help(Text(title, bundle: .module))
        .accessibilityLabel(Text(title, bundle: .module))
        .accessibilityValue(
            selection.wrappedValue.map { Text(verbatim: $0) }
                ?? Text("All", bundle: .module))
    }

    private var sortChip: some View {
        Menu {
            ForEach(ModelHubFilter.SortOption.allCases) { option in
                Button {
                    filter.sort = option
                } label: {
                    if filter.sort == option {
                        Label { sortOptionText(option) } icon: { Image(systemName: "checkmark") }
                    } else {
                        sortOptionText(option)
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.up.arrow.down")
                    .themedFont(.micro, weight: .bold)
                sortOptionText(filter.sort)
                    .themedFont(.tiny)
                    .lineLimit(1)
            }
            .padding(.horizontal, 9)
            .frame(height: 22)
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(.appSecondary)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay { Capsule().stroke(.appBorder, lineWidth: 0.5) }
        .help(Text("Sort and group models", bundle: .module))
        .accessibilityLabel(Text("Sort By", bundle: .module))
        .accessibilityValue(sortOptionText(filter.sort))
    }

    private func tabText(_ tab: ModelHubFilter.Tab) -> Text {
        switch tab {
        case .recommended: return Text("Recommended", bundle: .module)
        case .discover: return Text("Discover", bundle: .module)
        case .onDevice: return Text("Installed", bundle: .module)
        }
    }

    private func sortOptionText(_ option: ModelHubFilter.SortOption) -> Text {
        switch option {
        case .recommended: return Text("Recommended", bundle: .module)
        case .name: return Text("Name", bundle: .module)
        case .size: return Text("Installed size", bundle: .module)
        }
    }
}
