import SwiftUI

/// Scope is a captured project identifier, never an alias for the selected chat.
///
/// Inline and sized to its content so it sits first in an extension pane's
/// header row, beside the search field, instead of taking a row of its own.
/// The profile name it used to print beside the menu moved to the tooltip:
/// "User" means the current profile's user scope.
struct ExtensionScopePicker: View {
    @ObservedObject var model: AppModel
    @Binding var projectID: UUID?

    var body: some View {
        Picker(selection: $projectID) {
            Text("User", bundle: .module).tag(nil as UUID?)
            ForEach(model.projects.filter { $0.rootDirectoryURL != nil }) { project in
                Text("Project: \(project.name)", bundle: .module).tag(Optional(project.id))
            }
        } label: { Text("Scope", bundle: .module) }
        .pickerStyle(.menu)
        .fixedSize()
        .help(Text("User applies to every chat in the \(model.currentProfile.name) profile. A project scope applies to that project only.", bundle: .module))
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

/// The MCP Servers pane: the global manager at User scope, the project
/// manager at a project scope. The scope picker stays in the header either
/// way so the user can always switch back.
struct McpManagementView: View {
    @ObservedObject var model: AppModel
    @State var projectID: UUID?
    var body: some View {
        if let projectID {
            VStack(spacing: 0) {
                ExtensionPaneHeader {
                    ExtensionScopePicker(model: model, projectID: $projectID)
                }
                // Embedded in Settings, which already titles the pane and has no
                // sheet to dismiss: hide the sheet's own title and Close.
                ProjectMcpSettingsSheet(
                    model: model, projectID: projectID, showsChrome: false, onDismiss: {})
                    .id(projectID)
            }
        } else {
            McpSettingsPaneView(model: model, scopeProjectID: $projectID)
        }
    }
}
