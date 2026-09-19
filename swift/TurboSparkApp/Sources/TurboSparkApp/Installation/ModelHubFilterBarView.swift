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
                    Text(tab.rawValue).tag(tab)
                }
            } label: { Text("View", bundle: .module) }
            .pickerStyle(.segmented)
            .controlSize(.small)
            .frame(width: 260)
            .labelsHidden()
            .accessibilityLabel("Catalog view")

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    filterChip(
                        title: "Format",
                        options: ModelHubFilter.formatOptions(for: catalog),
                        selection: $filter.format)

                    filterChip(
                        title: "Capability",
                        options: ModelHubFilter.capabilityOptions(for: catalog),
                        selection: $filter.capability)

                    filterChip(
                        title: "Fit",
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
        title: String,
        options: [String],
        selection: Binding<ModelHubFilter.Selection>
    ) -> some View {
        Menu {
            Button(ModelHubFilter.anyOption) { selection.wrappedValue = nil }
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
                Text(selection.wrappedValue ?? title)
                    .themedFont(.tiny, weight: selection.wrappedValue == nil ? .regular : .medium)
                    .lineLimit(1)
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
        .help(title.lowercased() == "architecture" ? Text("Filter by architecture", bundle: .module) : Text("Filter by source", bundle: .module))
        .accessibilityLabel(title.lowercased() == "architecture" ? Text("Filter by architecture", bundle: .module) : Text("Filter by source", bundle: .module))
        .accessibilityValue(selection.wrappedValue ?? ModelHubFilter.anyOption)
    }

    private var sortChip: some View {
        Menu {
            ForEach(ModelHubFilter.SortOption.allCases) { option in
                Button {
                    filter.sort = option
                } label: {
                    if filter.sort == option {
                        Label(option.rawValue, systemImage: "checkmark")
                    } else {
                        Text(option.rawValue)
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "arrow.up.arrow.down")
                    .themedFont(.micro, weight: .bold)
                Text(filter.sort.rawValue)
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
        .accessibilityValue(filter.sort.rawValue)
    }
}
