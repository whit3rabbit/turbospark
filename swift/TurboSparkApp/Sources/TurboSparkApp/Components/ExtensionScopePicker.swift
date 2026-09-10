import SwiftUI

/// Scope is a captured project identifier, never an alias for the selected chat.
struct ExtensionScopePicker: View {
    @ObservedObject var model: AppModel
    @Binding var projectID: UUID?

    var body: some View {
        HStack {
            Text(model.currentProfile.name).themedFont(.small, weight: .medium)
            Picker(selection: $projectID) {
                Text("User", bundle: .module).tag(nil as UUID?)
                ForEach(model.projects.filter { $0.rootDirectoryURL != nil }) { project in
                    Text("Project: \(project.name)", bundle: .module).tag(Optional(project.id))
                }
            } label: { Text("Scope", bundle: .module) }
            .frame(maxWidth: 320)
            Spacer()
        }
        .padding(12)
    }
}

struct ExtensionOverridePicker: View {
    @Binding var enabled: Bool?
    var body: some View {
        Picker(selection: $enabled) {
            Text("Use user default", bundle: .module).tag(nil as Bool?)
            Text("Enabled", bundle: .module).tag(Optional(true))
            Text("Disabled", bundle: .module).tag(Optional(false))
        } label: { Text("Project override", bundle: .module) }
        .frame(maxWidth: 260)
    }
}

struct McpManagementView: View {
    @ObservedObject var model: AppModel
    @State var projectID: UUID?
    var body: some View {
        VStack(spacing: 0) {
            ExtensionScopePicker(model: model, projectID: $projectID)
            if let projectID {
                ProjectMcpSettingsSheet(model: model, projectID: projectID, onDismiss: {})
                    .id(projectID)
            } else {
                McpSettingsPaneView(model: model)
            }
        }
    }
}
