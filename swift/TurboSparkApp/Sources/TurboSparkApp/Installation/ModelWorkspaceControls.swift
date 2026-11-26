import SwiftUI
import TurboSpark

/// The two halves of the Models area. Installed is `.modelManager` and
/// Discover is `.modelHub` in `AppNavigationSection`; this is the pure mapping
/// between them so the switcher and its test share one definition.
enum ModelWorkspaceMode: Hashable {
    case installed
    case discover

    init(section: AppModel.AppNavigationSection) {
        self = section == .modelHub ? .discover : .installed
    }

    var section: AppModel.AppNavigationSection {
        self == .discover ? .modelHub : .modelManager
    }
}

/// "Installed | Discover" at the leading edge of every Models header, so a
/// user can move between what they have and what they can get without going
/// back to the sidebar. The sidebar rows stay: this is a second way in, and it
/// reads the same `activeSection` they do.
@MainActor
struct ModelWorkspaceSwitcher: View {
    @ObservedObject var model: AppModel

    var body: some View {
        AppPillPicker(
            selection: Binding(
                get: { ModelWorkspaceMode(section: model.activeSection) },
                set: { model.activeSection = $0.section }),
            items: [
                AppPillItem(
                    value: ModelWorkspaceMode.installed,
                    label: Text("Installed", bundle: .module),
                    systemImage: "internaldrive"),
                AppPillItem(
                    value: ModelWorkspaceMode.discover,
                    label: Text("Discover", bundle: .module),
                    systemImage: "shippingbox"),
            ],
            label: Text("Models", bundle: .module))
    }
}

/// The one "Filter" button of the Installed and Discover filter bars. It
/// replaces a row of dropdowns: the advanced filters live in a popover, and
/// the button carries how many of them are narrowing the list, as a number
/// and in its accessibility value, never color alone.
@MainActor
struct ModelFilterButton<Content: View>: View {
    /// Filters that differ from their default. Sort and grouping are not
    /// filters, so they never count.
    let activeCount: Int
    @ViewBuilder let content: () -> Content

    @State private var isShown = false
    @Environment(\.appTheme) private var theme

    var body: some View {
        Button {
            isShown.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: activeCount > 0
                    ? "line.3.horizontal.decrease.circle.fill"
                    : "line.3.horizontal.decrease.circle")
                    .accessibilityHidden(true)
                Text("Filter", bundle: .module)
                if activeCount > 0 {
                    Text(verbatim: "\(activeCount)")
                        .themedFont(.tiny, weight: .bold).monospacedDigit()
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(.appAccent.opacity(0.18), in: Capsule())
                        .foregroundStyle(.appAccentText)
                        .accessibilityHidden(true)
                }
            }
        }
        .appAction(.secondary, size: .small)
        .help(Text("Filter", bundle: .module))
        .accessibilityLabel(Text("Filter", bundle: .module))
        .accessibilityValue(
            activeCount > 0 ? Text("\(activeCount) active", bundle: .module) : Text("None", bundle: .module))
        .popover(isPresented: $isShown, arrowEdge: .bottom) {
            let scale = AppMetrics.textScale(theme)
            Form { content() }
                .formStyle(.grouped)
                .scrollContentBackground(.hidden)
                .background(.appPage)
                .frame(minWidth: 300 * scale, idealWidth: 340 * scale, maxWidth: 440 * scale)
                .frame(minHeight: 120, idealHeight: 380 * scale, maxHeight: 560 * scale)
                .themedFont(.small)
        }
    }
}

extension ModelManagerView {
    /// How many narrowing filters of the Installed list are set. Sort and
    /// grouping describe the order, not the contents, so they are not counted.
    static func activeFilterCount(
        showFavoritesOnly: Bool,
        availableOnly: Bool,
        architectureFilter: ArchitectureFilter,
        drafterFilter: DrafterFilter,
        sourceFilter: SourceFilter,
        selectedTag: String?
    ) -> Int {
        var count = 0
        if showFavoritesOnly { count += 1 }
        if availableOnly { count += 1 }
        if architectureFilter != .all { count += 1 }
        if drafterFilter != .all { count += 1 }
        if sourceFilter != .all { count += 1 }
        if selectedTag != nil { count += 1 }
        return count
    }
}
