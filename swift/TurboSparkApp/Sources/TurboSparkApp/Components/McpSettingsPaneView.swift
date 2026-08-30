import SwiftUI

/// Codex-style MCP server management pane for app settings and preferences.
public struct McpSettingsPaneView: View {
    @ObservedObject var model: AppModel

    @State private var searchText = ""
    @State private var showingEditorSheet = false
    @State private var editingServer: McpServerConfig?
    @State private var expandedServerIDs: Set<UUID> = []
    @State private var testingServerID: UUID?
    @State private var testResultToast: (id: UUID, message: String, isError: Bool)?

    public init(model: AppModel) {
        self.model = model
    }

    private var filteredServers: [McpServerConfig] {
        if searchText.trimmingCharacters(in: .whitespaces).isEmpty {
            return model.globalMcpServers
        }
        let query = searchText.lowercased()
        return model.globalMcpServers.filter {
            $0.name.lowercased().contains(query) ||
            ($0.serverDescription?.lowercased().contains(query) ?? false) ||
            $0.commandSummary.lowercased().contains(query)
        }
    }

    private var activeCount: Int {
        model.globalMcpServers.filter { $0.isEnabled }.count
    }

    private var totalToolsCount: Int {
        model.globalMcpServers.reduce(0) { $0 + $1.discoveredTools.count }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            headerBar
            filterAndPillsBar
            Divider()

            if filteredServers.isEmpty {
                emptyState
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Servers")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        ForEach(filteredServers) { server in
                            serverCard(server)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .padding(16)
        .sheet(isPresented: $showingEditorSheet) {
            McpServerEditorSheet(
                existingConfig: editingServer,
                workingDirectory: nil,
                onSave: { updatedConfig in
                    if editingServer != nil {
                        model.updateGlobalMcpServer(updatedConfig)
                    } else {
                        model.addGlobalMcpServer(updatedConfig)
                    }
                    showingEditorSheet = false
                    editingServer = nil
                },
                onDismiss: {
                    showingEditorSheet = false
                    editingServer = nil
                }
            )
        }
    }

    private var headerBar: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Plugins & MCPs")
                    .font(.title2.weight(.bold))
                HStack(spacing: 4) {
                    Text("Manage Model Context Protocol servers, external tools, and integrations.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if let url = URL(string: "https://modelcontextprotocol.io") {
                        Link("Documentation", destination: url)
                            .font(.caption)
                            .help("Open MCP official documentation: https://modelcontextprotocol.io")
                            .accessibilityLabel("MCP Documentation")
                            .accessibilityHint("Opens Model Context Protocol documentation in web browser")
                            .accessibilityAddTraits(.isLink)
                    }
                }
            }
            Spacer()
            Button {
                editingServer = nil
                showingEditorSheet = true
            } label: {
                Label("Add Server", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .help("Add a new global MCP server")
            .accessibilityLabel("Add MCP Server")
            .accessibilityHint("Opens editor sheet to configure a new MCP server")
        }
    }

    private var filterAndPillsBar: some View {
        HStack(spacing: 12) {
            HStack(spacing: 6) {
                summaryPill(label: "MCPs", value: "\(model.globalMcpServers.count)", isSelected: true)
                summaryPill(label: "Active", value: "\(activeCount)", isSelected: false)
                summaryPill(label: "Tools", value: "\(totalToolsCount)", isSelected: false)
            }

            Spacer()

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .font(.caption)
                TextField("Search MCP servers", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.caption)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2), lineWidth: 0.5))
            .frame(width: 200)
        }
    }

    private func summaryPill(label: String, value: String, isSelected: Bool) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption.weight(.medium))
                .foregroundStyle(isSelected ? .primary : .secondary)
            Text(value)
                .font(.caption.monospacedDigit().weight(.bold))
                .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.secondary.opacity(0.08))
        .clipShape(Capsule())
    }

    private func serverCard(_ server: McpServerConfig) -> some View {
        let isExpanded = expandedServerIDs.contains(server.id)
        let isTesting = testingServerID == server.id

        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: server.isEnabled ? "server.rack" : "server.rack")
                    .font(.title3)
                    .foregroundStyle(server.isEnabled ? TurboSparkTheme.accentColor : Color.secondary)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(server.name)
                            .font(.headline)
                            .foregroundStyle(.primary)

                        if !server.discoveredTools.isEmpty {
                            Text("\(server.discoveredTools.count) tools")
                                .font(.caption2.weight(.medium))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.12))
                                .clipShape(Capsule())
                        }
                    }

                    Text(server.commandSummary)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer()

                // Actions
                HStack(spacing: 8) {
                    Button {
                        Task { await testServer(server) }
                    } label: {
                        if isTesting {
                            ProgressView()
                                .scaleEffect(0.5)
                                .frame(width: 16, height: 16)
                        } else {
                            Image(systemName: "bolt.fill")
                                .font(.caption)
                        }
                    }
                    .buttonStyle(.borderless)
                    .help("Test connection and discover tools")

                    Button {
                        editingServer = server
                        showingEditorSheet = true
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("Edit server settings")

                    Toggle("", isOn: Binding(
                        get: { server.isEnabled },
                        set: { model.toggleGlobalMcpServer(id: server.id, isEnabled: $0) }
                    ))
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)

                    Menu {
                        Button("Edit Server", systemImage: "pencil") {
                            editingServer = server
                            showingEditorSheet = true
                        }
                        Button("Re-query Tools", systemImage: "arrow.clockwise") {
                            Task { await testServer(server) }
                        }
                        Divider()
                        Button("Delete Server", systemImage: "trash", role: .destructive) {
                            model.deleteGlobalMcpServer(id: server.id)
                        }
                    } label: {
                        Image(systemName: "ellipsis")
                            .font(.caption)
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .frame(width: 18)
                    .help("Server actions")
                }
            }

            if let toast = testResultToast, toast.id == server.id {
                Text(toast.message)
                    .font(.caption)
                    .foregroundStyle(toast.isError ? .red : .green)
                    .padding(6)
                    .background(toast.isError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            if !server.discoveredTools.isEmpty {
                Button {
                    if isExpanded {
                        expandedServerIDs.remove(server.id)
                    } else {
                        expandedServerIDs.insert(server.id)
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.caption2)
                        Text(isExpanded ? "Hide Discovered Tools" : "Show Discovered Tools (\(server.discoveredTools.count))")
                            .font(.caption2.weight(.medium))
                    }
                    .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)

                if isExpanded {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(server.discoveredTools) { tool in
                            HStack(alignment: .top, spacing: 6) {
                                Image(systemName: "wrench.and.screwdriver")
                                    .font(.caption2)
                                    .foregroundStyle(TurboSparkTheme.accentColor)
                                    .padding(.top, 2)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(tool.name)
                                        .font(.caption.monospaced().weight(.semibold))
                                    Text(tool.description)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 2)
                        }
                    }
                    .padding(8)
                    .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.secondary.opacity(0.15), lineWidth: 0.5))
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "server.rack")
                .font(.system(size: 36))
                .foregroundStyle(.secondary.opacity(0.5))
            Text("No MCP Servers Configured")
                .font(.headline)
            Text("Add external MCP servers (like memory, GitHub, filesystem, database, or browser tools) to extend assistant capabilities.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
            Button {
                editingServer = nil
                showingEditorSheet = true
            } label: {
                Label("Add MCP Server", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }

    private func testServer(_ server: McpServerConfig) async {
        testingServerID = server.id
        testResultToast = nil
        let result = await model.testMcpServer(server)
        switch result {
        case .success(let tools):
            var updated = server
            updated.discoveredTools = tools
            model.updateGlobalMcpServer(updated)
            testResultToast = (id: server.id, message: "Connected: Discovered \(tools.count) tools.", isError: false)
        case .failure(let error):
            testResultToast = (id: server.id, message: "Error: \(error.localizedDescription)", isError: true)
        }
        testingServerID = nil
    }
}
