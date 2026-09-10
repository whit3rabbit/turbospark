import SwiftUI

public enum SettingsApplicationTiming: String, Sendable {
    case immediate = "Applies immediately"
    case nextTurn = "Applies to the next turn"
    case modelReload = "Applies after model reload"
    case relaunch = "Requires relaunch"
    case action = "Applies when you use this action"
}

struct SettingsControlDescriptor: Identifiable {
    let title: String
    let pane: AppSettingsView.SettingsTab
    let timing: SettingsApplicationTiming
    var id: String { pane.rawValue + ":" + title }
    var navigationID: String {
        // A menu item is not mounted until its parent menu opens.
        if pane == .appearance {
            if ["Copy theme JSON", "Export JSON file", "Import JSON file", "Import from clipboard"].contains(title) {
                return pane.rawValue + ":Import / Export"
            }
            if ["Delete saved theme", "Rename"].contains(title) { return pane.rawValue + ":Theme gallery" }
        }
        return id
    }
}

private struct SettingsTargetKey: EnvironmentKey { static let defaultValue: String? = nil }
extension EnvironmentValues {
    var settingsTarget: String? {
        get { self[SettingsTargetKey.self] }
        set { self[SettingsTargetKey.self] = newValue }
    }
}

private struct SettingsControlModifier: ViewModifier {
    @Environment(\.settingsTarget) private var target
    @Environment(\.appTheme) private var theme
    let descriptor: SettingsControlDescriptor
    func body(content: Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            content
            if descriptor.timing == .modelReload || descriptor.timing == .relaunch {
                Text(LocalizedStringKey(descriptor.timing.rawValue), bundle: .module)
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
            }
        }
            .id(descriptor.id)
            .overlay {
                if target == descriptor.id {
                    RoundedRectangle(cornerRadius: 4).stroke(theme.accent, lineWidth: 2)
                        .padding(-4).allowsHitTesting(false)
                }
            }
            .accessibilityHint(Text(LocalizedStringKey(descriptor.timing.rawValue), bundle: .module))
    }
}

public extension View {
    /// The catalogue is generated from these declarations next to the rendered control.
    func settingsControl(_ title: String, pane: AppSettingsView.SettingsTab,
                         timing: SettingsApplicationTiming = .immediate) -> some View {
        modifier(SettingsControlModifier(descriptor: .init(title: title, pane: pane, timing: timing)))
    }
}
