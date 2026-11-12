import SwiftUI

/// Settings pane for SOUL.md and agent personality directives.
struct SoulSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        Form {
            Text("Global personality and communication guidance for the agent.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)

            SoulSettingsSection(model: model)
            PersonalitySettingsSection(model: model)
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
    }
}
