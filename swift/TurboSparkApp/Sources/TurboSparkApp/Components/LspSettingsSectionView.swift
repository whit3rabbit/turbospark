import SwiftUI

/// OC-10: the Language servers section of the General pane. One master switch
/// (off by default) and one row per server. Nothing starts from here except
/// the Test button, which the user presses, and which runs in an empty
/// temporary folder. Server commands live in the user's own defaults: no
/// project file can add or enable a row.
@MainActor
struct LspSettingsSectionView: View {
    @Environment(\.appTheme) private var theme
    @State private var enabled = LSPSettings.isEnabled()
    @State private var servers = LSPSettings.servers()
    @State private var testing: Set<String> = []
    @State private var testResults: [String: Result<String, Error>] = [:]
    @State private var newName = ""
    @State private var newExtensions = ""
    @State private var newCommand = ""
    /// What the user is typing in the per-row fields. The stored row keeps
    /// only what parsed (`LSPSettings.parseArguments`, `LSPEnvironmentPolicy`),
    /// so the draft is what lets a refused name stay visible next to its note.
    @State private var argumentDrafts: [String: String] = [:]
    @State private var environmentDrafts: [String: String] = [:]

    var body: some View {
        Section(header: Text("Language servers", bundle: .module)) {
            Toggle(isOn: Binding(get: { enabled }, set: { setEnabled($0) })) {
                Text("Use language servers", bundle: .module)
            }
            .settingsControl("Use language servers", pane: .general, timing: .nextTurn)
            Text(
                "Lets the agent check the files it edits and look up symbols through language servers you installed yourself. Nothing is downloaded or installed, and a server starts only after you turn on its row and the agent uses it. Each server runs with a minimal environment inside the open project and stops when the project closes. Some servers read a project's build files when they start, so enable them only for projects you trust.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
            if enabled {
                ForEach(servers) { server in serverRow(server) }
                addRow
            }
        }
    }

    private func setEnabled(_ value: Bool) {
        enabled = value
        LSPSettings.setEnabled(value)
        // Turning the feature off stops every server now, not at the next use.
        if !value { Task { await LanguageServerRegistry.shared.stopAll() } }
    }

    private func save() {
        LSPSettings.save(servers)
        // A changed row is restarted by the manager on next use (it compares
        // the whole row), so nothing else to do here.
    }

    private func serverRow(_ server: LSPServerConfig) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle(isOn: Binding(
                get: { servers.first { $0.id == server.id }?.enabled ?? false },
                set: { value in
                    if let index = servers.firstIndex(where: { $0.id == server.id }) {
                        servers[index].enabled = value
                        save()
                    }
                })
            ) {
                Text(verbatim: server.name)
            }
            if server.isDiscovered {
                if let path = LSPDiscovery.sourcekitLSPPath() {
                    SettingsPathText(path)
                } else {
                    Text("Not found on this Mac", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                }
            } else {
                SettingsPathText(server.command)
                Text(verbatim: server.extensions.joined(separator: ", "))
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)
            }
            if !server.isDiscovered { launchOptions(server) }
            HStack {
                Button { runTest(server) } label: { Text("Test", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(testing.contains(server.id))
                if !server.isDiscovered {
                    Button { remove(server) } label: { Text("Remove", bundle: .module) }
                        .appAction(.quiet, size: .small)
                }
                testResultView(server.id)
            }
        }
    }

    /// Arguments and environment passthrough for a user-added row. Both live in
    /// this app's own defaults; nothing here is read from a project. The
    /// launch environment stays the minimal allowlist plus the NAMES typed here.
    private func launchOptions(_ server: LSPServerConfig) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 6) {
                Text("Arguments, one per line. They are passed to the server as typed, never through a shell.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                TextEditor(text: Binding(
                    get: { argumentDrafts[server.id] ?? server.args.joined(separator: "\n") },
                    set: { text in
                        argumentDrafts[server.id] = text
                        update(server.id) { $0.args = LSPSettings.parseArguments(text) }
                    }))
                    .themedCode(.small)
                    .frame(minHeight: 44, maxHeight: 96)
                    .accessibilityLabel(Text("Arguments", bundle: .module))
                Text("Environment variable names to pass through, separated by commas or spaces. Only these named variables are copied from this app's own environment. Names that look like secrets or that can inject code are refused.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                TextField(text: Binding(
                    get: { environmentDrafts[server.id] ?? server.envPassthrough.joined(separator: ", ") },
                    set: { text in
                        environmentDrafts[server.id] = text
                        update(server.id) { $0.envPassthrough = LSPEnvironmentPolicy.parse(text).accepted }
                    }),
                    prompt: Text("Names, for example LANG_SERVER_HOME", bundle: .module)
                ) { Text("Environment variable names", bundle: .module) }
                let refused = LSPEnvironmentPolicy.parse(
                    environmentDrafts[server.id] ?? server.envPassthrough.joined(separator: ", ")).refused
                if !refused.isEmpty {
                    Label {
                        HStack(spacing: 4) {
                            Text("Not passed:", bundle: .module)
                            Text(verbatim: refused.joined(separator: ", ")).lineLimit(2)
                        }
                    } icon: { Image(systemName: "exclamationmark.triangle") }
                        .font(theme.ui(.small))
                        .foregroundStyle(.appWarning)
                }
            }
            .padding(.top, 4)
        } label: {
            Text("Arguments and environment", bundle: .module)
        }
    }

    private func update(_ id: String, _ change: (inout LSPServerConfig) -> Void) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        change(&servers[index])
        save()
    }

    @ViewBuilder
    private func testResultView(_ id: String) -> some View {
        switch testResults[id] {
        case .success(let name):
            Label {
                HStack(spacing: 4) {
                    Text("Test passed", bundle: .module)
                    Text(verbatim: name)
                }
            } icon: { Image(systemName: "checkmark.circle") }
                .font(theme.ui(.small))
                .foregroundStyle(.appSuccess)
        case .failure(let error):
            Label {
                HStack(spacing: 4) {
                    Text("Test failed", bundle: .module)
                    Text(verbatim: error.localizedDescription).lineLimit(2)
                }
            } icon: { Image(systemName: "exclamationmark.triangle") }
                .font(theme.ui(.small))
                .foregroundStyle(.appWarning)
        case nil:
            EmptyView()
        }
    }

    private func runTest(_ server: LSPServerConfig) {
        testing.insert(server.id)
        testResults[server.id] = nil
        Task {
            let outcome = await LSPServerProbe.test(server)
            testResults[server.id] = outcome
            testing.remove(server.id)
        }
    }

    private func remove(_ server: LSPServerConfig) {
        servers.removeAll { $0.id == server.id }
        testResults[server.id] = nil
        save()
        Task { await LanguageServerRegistry.shared.stopAll() }
    }

    private var canAdd: Bool {
        !newName.trimmingCharacters(in: .whitespaces).isEmpty
            && !LSPSettings.parseExtensions(newExtensions).isEmpty
            && !newCommand.trimmingCharacters(in: .whitespaces).isEmpty
            && servers.count < LSPSettings.maximumServers
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
            Button { add() } label: { Text("Add server", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(!canAdd)
                .settingsControl("Add server", pane: .general, timing: .action)
        }
    }

    private func add() {
        let extensions = LSPSettings.parseExtensions(newExtensions)
        let row = LSPServerConfig(
            id: "custom-" + String(UUID().uuidString.lowercased().prefix(8)), name: newName.trimmingCharacters(in: .whitespaces),
            extensions: extensions, languageID: extensions.first ?? "plaintext",
            command: newCommand.trimmingCharacters(in: .whitespaces), args: [], enabled: true, isDiscovered: false)
        servers.append(row)
        save()
        newName = ""; newExtensions = ""; newCommand = ""
    }
}
