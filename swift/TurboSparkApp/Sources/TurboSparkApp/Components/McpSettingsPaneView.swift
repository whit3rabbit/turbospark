import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Codex-style MCP server management pane for app settings and preferences.
@MainActor
public struct McpSettingsPaneView: View {
    @ObservedObject var model: AppModel
    /// Supplied by `McpManagementView` so the scope picker sits in this
    /// pane's header row; nil where the pane is shown without a scope choice.
    private let scopeProjectID: Binding<UUID?>?

    @State private var searchText = ""
    @State private var showingEditorSheet = false
    @State private var showingImportSheet = false
    @State private var showingAgentImportSheet = false
    @State private var editingServer: McpServerConfig?
    @State private var expandedServerIDs: Set<UUID> = []
    @State private var testingServerID: UUID?
    @State private var testResultToast: (id: UUID, message: String, isError: Bool)?
    /// Backed by the same `UserDefaults` key `CodemodeSettings.isEnabled`
    /// reads. A `Binding(get:set:)` over `UserDefaults` is invisible to
    /// SwiftUI, so the switch never redrew after a tap; `@AppStorage` is
    /// observed and invalidates this view.
    @AppStorage(CodemodeSettings.storageKey) private var codemodeEnabled = false

    public init(model: AppModel) {
        self.model = model
        self.scopeProjectID = nil
    }

    init(model: AppModel, scopeProjectID: Binding<UUID?>) {
        self.model = model
        self.scopeProjectID = scopeProjectID
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
        VStack(spacing: 0) {
            header

            VStack(alignment: .leading, spacing: 14) {
                summaryBar
                codemodeFeatureToggle
                Divider()

                if model.globalMcpServers.isEmpty {
                    emptyState
                } else if filteredServers.isEmpty {
                    ExtensionNoMatchesState(query: searchText)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Servers", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                                .foregroundStyle(.appSecondary)
                                .accessibilityAddTraits(.isHeader)

                            ForEach(filteredServers) { server in
                                serverCard(server)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .padding(ExtensionPaneLayout.contentInsets)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .sheet(isPresented: $showingEditorSheet) {
            McpServerEditorSheet(
                existingConfig: editingServer,
                workingDirectory: nil,
                existingNames: model.globalMcpServers
                    .filter { $0.id != editingServer?.id }
                    .map(\.name),
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
        .sheet(isPresented: $showingImportSheet) {
            McpImportSheet(model: model) { showingImportSheet = false }
        }
        .sheet(isPresented: $showingAgentImportSheet) {
            AgentContentImportSheet(
                model: model, initialCategory: .mcpServers, projectID: nil)
        }
    }

    // MARK: - Header

    /// `[scope][search] ... [Import][Add Server]`, the layout every extension
    /// pane shares. Both import sources live in the one Import menu: four
    /// bordered buttons beside the title and description truncated at the
    /// Settings window's minimum width.
    private var header: some View {
        ExtensionPaneHeader {
            ExtensionPaneIntro(
                text: Text("Manage Model Context Protocol servers, external tools, and integrations.", bundle: .module),
                documentation: URL(string: "https://modelcontextprotocol.io"),
                documentationHint: Text("Opens the Model Context Protocol documentation in your browser.", bundle: .module))
                .settingsControl("MCP Servers", pane: .mcp, timing: .nextTurn)
        } filters: {
            if let scopeProjectID {
                ExtensionScopePicker(model: model, projectID: scopeProjectID)
            }
            AppSearchField(text: $searchText, prompt: Text("Search MCP servers", bundle: .module))
        } actions: {
            ExtensionImportMenu { importMenuItems }
            addServerButton
        }
    }

    @ViewBuilder
    private var importMenuItems: some View {
        Button {
            showingAgentImportSheet = true
        } label: {
            Label { Text("From Other Agents...", bundle: .module) } icon: { Image(systemName: "person.2") }
        }
        .help(Text("Import servers from Claude, Codex, Cursor, or Gemini global configs; they are added disabled", bundle: .module))

        Button {
            showingImportSheet = true
        } label: {
            Label { Text("Browse Marketplace", bundle: .module) } icon: { Image(systemName: "square.grid.2x2") }
        }
        .help(Text("Install a server from a published catalog", bundle: .module))
    }

    private var addServerButton: some View {
        Button {
            editingServer = nil
            showingEditorSheet = true
        } label: {
            Label { Text("Add Server", bundle: .module) } icon: { Image(systemName: "plus") }
        }
        .appAction(.primary)
        .help(Text("Add a new global MCP server", bundle: .module))
    }

    // MARK: - Content

    /// Opt-in codemode scripting. Catalog advertisement and execution both
    /// read `CodemodeSettings.isEnabled`, which is this same `UserDefaults`
    /// key, so they agree with the switch; changes land on the next turn like
    /// the rest of this pane.
    private var codemodeFeatureToggle: some View {
        VStack(alignment: .leading, spacing: 2) {
            Toggle(isOn: $codemodeEnabled) {
                Text("Codemode", bundle: .module)
                    .themedFont(.callout, weight: .medium)
            }
            .toggleStyle(.switch)
            .help(Text("Runs model-written JavaScript that can batch deferred MCP tool calls; every inner call still passes its permission and hook gates", bundle: .module))
            .accessibilityLabel(Text("Codemode", bundle: .module))
            .accessibilityHint(Text("Batch MCP tool calls inside one sandboxed script.", bundle: .module))
            Text("Batch MCP tool calls inside one sandboxed script.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    /// Counts, not filters: these used to tint the first one as "selected",
    /// which read as a filter that could not be changed.
    private var summaryBar: some View {
        HStack(spacing: 6) {
            summaryPill(label: Text("Servers", bundle: .module), value: model.globalMcpServers.count)
            summaryPill(label: Text("Active", bundle: .module), value: activeCount)
            summaryPill(label: Text("Tools", bundle: .module), value: totalToolsCount)
        }
    }

    private func summaryPill(label: Text, value: Int) -> some View {
        HStack(spacing: 4) {
            label
                .themedFont(.small, weight: .medium)
                .foregroundStyle(.appSecondary)
            Text(verbatim: "\(value)")
                .themedFont(.small, weight: .bold).monospacedDigit()
                .foregroundStyle(.appText)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.appElevated)
        .clipShape(Capsule())
        .accessibilityElement(children: .combine)
    }

    private func serverCard(_ server: McpServerConfig) -> some View {
        McpServerCard(
            model: model,
            server: server,
            isExpanded: Binding(
                get: { expandedServerIDs.contains(server.id) },
                set: { expanded in
                    if expanded {
                        expandedServerIDs.insert(server.id)
                    } else {
                        expandedServerIDs.remove(server.id)
                    }
                }),
            isTesting: testingServerID == server.id,
            result: testResult(for: server),
            onTest: { Task { await testServer(server) } },
            onEdit: {
                editingServer = server
                showingEditorSheet = true
            })
    }

    private func testResult(for server: McpServerConfig) -> (message: String, isError: Bool)? {
        guard let toast = testResultToast, toast.id == server.id else { return nil }
        return (message: toast.message, isError: toast.isError)
    }

    /// Empty-state actions are secondary: the header already carries the one
    /// primary action (Add Server), so this offers the routes the Import menu
    /// hides.
    private var emptyState: some View {
        AppEmptyState(
            systemImage: "server.rack",
            title: Text("No MCP Servers Configured", bundle: .module),
            message: Text("Add external MCP servers (like memory, GitHub, filesystem, database, or browser tools) to extend assistant capabilities.", bundle: .module)
        ) {
            Button {
                showingAgentImportSheet = true
            } label: {
                Text("From Other Agents...", bundle: .module)
            }
            .appAction(.secondary)
            Button {
                showingImportSheet = true
            } label: {
                Text("Browse Marketplace", bundle: .module)
            }
            .appAction(.secondary)
        }
    }

    private func testServer(_ server: McpServerConfig) async {
        testingServerID = server.id
        testResultToast = nil
        let result = await model.testMcpServer(server)
        switch result {
        case .success(let tools):
            model.setGlobalMcpDiscoveredTools(id: server.id, tools: tools, transport: server.transport)
            testResultToast = (
                id: server.id,
                message: AppStrings.localized("Connected. Tools discovered: \(tools.count)."),
                isError: false)
        case .failure(let error):
            testResultToast = (
                id: server.id,
                message: AppStrings.localized("Error: \(error.localizedDescription)"),
                isError: true)
        }
        testingServerID = nil
    }
}
