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
        .padding(16)
    }
}
