import SwiftUI

/// Settings pane for SOUL.md and agent personality directives.
struct SoulSettingsPaneView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Form {
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
