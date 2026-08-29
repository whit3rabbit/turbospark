import SwiftUI

/// Project-scoped MCP server manager and auto-detection importer.
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
                    detectionSection
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

    private var detectionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Codebase Import & Detection")
                        .font(.subheadline.weight(.semibold))
                    Text("Scans project root for .mcp.json, opencode.json, .cursor, .vscode, and .agents configs.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    runDetection()
                } label: {
                    if isDetecting {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 16, height: 16)
                    } else {
                        Label("Scan Folder", systemImage: "arrow.clockwise")
                    }
                }
                .buttonStyle(.bordered)
                .help("Scan codebase root directory for MCP configuration files")
                .disabled(project?.rootDirectoryURL == nil || isDetecting)
            }

            if let root = project?.rootDirectoryURL {
                if detectedFiles.isEmpty {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle")
                            .foregroundStyle(.secondary)
                        Text("No external MCP config files detected in \(root.lastPathComponent).")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.secondary.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(detectedFiles) { detected in
                            detectedFileCard(detected)
                        }
                    }
                }
            } else {
                Text("Assign a Codebase Root Directory in Project Settings to enable automatic MCP file detection.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(8)
                    .background(Color.secondary.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private func detectedFileCard(_ file: DetectedProjectMcpFile) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "doc.text.fill")
                    .foregroundStyle(TurboSparkTheme.accentColor)
                VStack(alignment: .leading, spacing: 1) {
                    Text(file.formatLabel)
                        .font(.caption.weight(.semibold))
                    Text(file.relativePath)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Import All (\(file.servers.count))") {
                    importAllServers(from: file)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 4) {
                ForEach(file.servers) { server in
                    HStack {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(server.name)
                                .font(.caption.monospaced().weight(.semibold))
                            Text(server.commandSummary)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Spacer()
                        let alreadyImported = projectServers.contains { $0.name == server.name }
                        if alreadyImported {
                            Text("Imported")
                                .font(.caption2.weight(.medium))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.12))
                                .clipShape(Capsule())
                        } else {
                            Button("Import") {
                                model.addProjectMcpServer(projectID: projectID, config: server)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
            .padding(8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.8))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(TurboSparkTheme.accentColor.opacity(0.3), lineWidth: 1))
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
                        serverRow(server)
                    }
                }
            }
        }
    }

    private func serverRow(_ server: McpServerConfig) -> some View {
        let isExpanded = expandedServerIDs.contains(server.id)
        let isTesting = testingServerID == server.id

        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "server.rack")
                    .font(.headline)
                    .foregroundStyle(server.isEnabled ? TurboSparkTheme.accentColor : Color.secondary)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(server.name)
                            .font(.headline)
                        if let src = server.sourcePath {
                            Text(URL(fileURLWithPath: src).lastPathComponent)
                                .font(.caption2)
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
                    .help("Test connection")

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
                        set: { model.toggleProjectMcpServer(projectID: projectID, serverID: server.id, isEnabled: $0) }
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
                            model.deleteProjectMcpServer(projectID: projectID, serverID: server.id)
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
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.15), lineWidth: 0.5))
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
