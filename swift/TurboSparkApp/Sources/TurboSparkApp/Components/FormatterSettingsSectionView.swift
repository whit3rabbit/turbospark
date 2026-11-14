import SwiftUI

/// OC-11: the Formatters section of the General pane. One master switch (off
/// by default) and one row per formatter. Nothing runs from here: a formatter
/// runs only after the agent writes a file the row covers. Commands live in the
/// user's own defaults, so no project file can add or enable a row.
@MainActor
struct FormatterSettingsSectionView: View {
    @Environment(\.appTheme) private var theme
    @State private var enabled = FormatterSettings.isEnabled()
    @State private var rows = FormatterSettings.rows()
    @State private var newName = ""
    @State private var newExtensions = ""
    @State private var newCommand = ""
    @State private var newArguments = ""

    var body: some View {
        Section(header: Text("Formatters", bundle: .module)) {
            Toggle(isOn: Binding(get: { enabled }, set: setEnabled)) {
                Text("Format files after the agent edits them", bundle: .module)
            }
            .settingsControl("Format files after the agent edits them", pane: .general, timing: .nextTurn)
            Text(
                "After the agent writes or edits a file, runs a formatter you installed yourself on that one file, and tells the agent when the file changed. Nothing is downloaded or installed. A formatter can run code from a project's own config files, so turn this on only for projects you trust.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
            if enabled {
                ForEach(rows) { row in formatterRow(row) }
                addRow
            }
        }
    }

    private func setEnabled(_ value: Bool) {
        enabled = value
        FormatterSettings.setEnabled(value)
    }

    private func save() { FormatterSettings.save(rows) }

    private func resolvedPath(_ row: FormatterRow) -> String? {
        // A cheap, non-executing check using the PATH captured so far. It never
        // runs the login shell from a settings screen.
        try? FormatterLaunchPolicy.resolve(
            row: row, rootURL: URL(fileURLWithPath: "/nonexistent-root"),
            searchPath: LoginShellPath.cached, filePath: "{file}").executable
    }

    private func formatterRow(_ row: FormatterRow) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle(isOn: Binding(
                get: { rows.first { $0.id == row.id }?.enabled ?? false },
                set: { value in
                    if let index = rows.firstIndex(where: { $0.id == row.id }) {
                        rows[index].enabled = value
                        save()
                    }
                })
            ) {
                Text(verbatim: row.name)
            }
            if let path = resolvedPath(row) {
                SettingsPathText(path)
            } else {
                Text("Not found on this Mac", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            Text(verbatim: row.extensions.joined(separator: ", "))
                .themedCode(.small)
                .foregroundStyle(.appSecondary)
            if !row.isBuiltin {
                Button { remove(row) } label: { Text("Remove", bundle: .module) }
                    .appAction(.quiet, size: .small)
            }
        }
    }

    private var addRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextField(text: $newName, prompt: Text("Name", bundle: .module)) { Text("Name", bundle: .module) }
            TextField(text: $newExtensions, prompt: Text("File extensions (for example c, h)", bundle: .module)) {
                Text("File extensions (for example c, h)", bundle: .module)
            }
            TextField(text: $newCommand, prompt: Text("Command (full path or a name on your PATH)", bundle: .module)) {
                Text("Command (full path or a name on your PATH)", bundle: .module)
            }
            TextField(text: $newArguments, prompt: Text("Arguments (use {file} for the file)", bundle: .module)) {
                Text("Arguments (use {file} for the file)", bundle: .module)
            }
            Button { add() } label: { Text("Add formatter", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(!canAdd)
                .settingsControl("Add formatter", pane: .general, timing: .action)
        }
    }

    private var canAdd: Bool {
        !newName.trimmingCharacters(in: .whitespaces).isEmpty
            && !LSPSettings.parseExtensions(newExtensions).isEmpty
            && !newCommand.trimmingCharacters(in: .whitespaces).isEmpty
            && rows.count < FormatterSettings.maximumRows
    }

    private func add() {
        let row = FormatterRow(
            id: "custom-" + String(UUID().uuidString.lowercased().prefix(8)),
            name: newName.trimmingCharacters(in: .whitespaces),
            extensions: LSPSettings.parseExtensions(newExtensions),
            command: newCommand.trimmingCharacters(in: .whitespaces),
            args: FormatterSettings.parseArguments(newArguments), enabled: true, isBuiltin: false)
        rows.append(row)
        save()
        newName = ""; newExtensions = ""; newCommand = ""; newArguments = ""
    }

    private func remove(_ row: FormatterRow) {
        rows.removeAll { $0.id == row.id }
        save()
    }
}
