import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Project-scoped MCP server manager and auto-detection importer.
@MainActor
public struct ProjectMcpSettingsSheet: View {
    @ObservedObject var model: AppModel
    let projectID: UUID
    let onDismiss: () -> Void

    @State private var detectedFiles: [DetectedProjectMcpFile] = []
    @State private var isDetecting: Bool = false
    @State private var showingEditorSheet: Bool = false
    @State private var editingServer: McpServerConfig?
    @State private var expandedServerIDs: Set<UUID> = []
    @State private var testingServerID: UUID?
    @State private var testResultToast: (id: UUID, message: String, isError: Bool)?

    public init(model: AppModel, projectID: UUID, onDismiss: @escaping () -> Void) {
        self.model = model
        self.projectID = projectID
        self.onDismiss = onDismiss
    }

    private var project: AppProject? {
        model.projects.first { $0.id == projectID }
    }

    private var projectServers: [McpServerConfig] {
        project?.mcpServers ?? []
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    ProjectMcpDetectionSectionView(
                        project: project,
                        detectedFiles: detectedFiles,
                        isDetecting: isDetecting,
                        projectServers: projectServers,
                        onScan: runDetection,
                        onImportAll: importAllServers,
                        onImportSingle: { server in
                            model.addProjectMcpServer(projectID: projectID, config: server)
                        }
                    )
                    Divider()
                    serversSection
                }
                .padding(20)
            }
            Divider()
            footer
        }
        .frame(width: 600, height: 680)
        .onAppear(perform: runDetection)
        .sheet(isPresented: $showingEditorSheet) {
            McpServerEditorSheet(
                existingConfig: editingServer,
                workingDirectory: project?.rootDirectoryURL,
                // GLOBAL names count here too. `executeMcpCall` resolves over
                // `globalServers + projectServers` and a global wins the
                // collision by design (state#61), so a project server sharing a
                // global name is not merely a duplicate: it can never be
                // dialled at all.
                existingNames: reservedServerNames,
                onSave: { updatedConfig in
                    if editingServer != nil {
                        model.updateProjectMcpServer(projectID: projectID, config: updatedConfig)
                    } else {
                        model.addProjectMcpServer(projectID: projectID, config: updatedConfig)
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

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Image(systemName: "folder.badge.gearshape")
                        .foregroundStyle(TurboSparkTheme.accentColor)
                        .help("Project MCP Tool Configuration")
                    Text("\(project?.name ?? "Project") – MCP External Tools")
                        .font(.headline)
                }
                Text("Manage project-specific MCP servers and import configs from codebase.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { onDismiss() }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var serversSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Configured Project Servers (\(projectServers.count))")
                        .font(.subheadline.weight(.semibold))
                    Text("Active servers are available to the assistant during turns in this project.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    editingServer = nil
                    showingEditorSheet = true
                } label: {
                    Label("Add Server", systemImage: "plus")
                }
                .buttonStyle(.bordered)
            }

            if projectServers.isEmpty {
                VStack(spacing: 8) {
                    Text("No project MCP servers configured.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(20)
                .frame(maxWidth: .infinity)
                .background(Color.secondary.opacity(0.04))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                VStack(spacing: 8) {
                    ForEach(projectServers) { server in
                        ProjectMcpServerRowView(
                            server: server,
                            isExpanded: expandedServerIDs.contains(server.id),
                            isTesting: testingServerID == server.id,
                            testResultToast: testResultToast,
                            onToggleExpand: {
                                if expandedServerIDs.contains(server.id) {
                                    expandedServerIDs.remove(server.id)
                                } else {
                                    expandedServerIDs.insert(server.id)
                                }
                            },
                            onToggleEnabled: { isEnabled in
                                model.toggleProjectMcpServer(projectID: projectID, serverID: server.id, isEnabled: isEnabled)
                            },
                            onTest: {
                                Task { await testServer(server) }
                            },
                            onEdit: {
                                editingServer = server
                                showingEditorSheet = true
                            },
                            onDelete: {
                                model.deleteProjectMcpServer(projectID: projectID, serverID: server.id)
                            }
                        )
                    }
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            Spacer()
            Button("Done") { onDismiss() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    /// Every name a project server would collide with: the other project
    /// servers, minus the one being edited, plus every global server.
    private var reservedServerNames: [String] {
        let siblings = (project?.mcpServers ?? [])
            .filter { $0.id != editingServer?.id }
            .map(\.name)
        return siblings + model.globalMcpServers.map(\.name)
    }

    private func runDetection() {
        guard let rootURL = project?.rootDirectoryURL else {
            detectedFiles = []
            return
        }
        isDetecting = true
        let detected = ProjectMcpDetector.detectInProject(rootURL: rootURL)
        detectedFiles = detected
        isDetecting = false
    }

    private func importAllServers(from file: DetectedProjectMcpFile) {
        for server in file.servers {
            if !projectServers.contains(where: { $0.name == server.name }) {
                model.addProjectMcpServer(projectID: projectID, config: server)
            }
        }
    }

    private func testServer(_ server: McpServerConfig) async {
        testingServerID = server.id
        testResultToast = nil
        let result = await model.testMcpServer(server, workingDirectory: project?.rootDirectoryURL)
        switch result {
        case .success(let tools):
            var updated = server
            updated.discoveredTools = tools
            model.updateProjectMcpServer(projectID: projectID, config: updated)
            testResultToast = (id: server.id, message: "Connected: Discovered \(tools.count) tools.", isError: false)
        case .failure(let error):
            testResultToast = (id: server.id, message: "Error: \(error.localizedDescription)", isError: true)
        }
        testingServerID = nil
    }
}
