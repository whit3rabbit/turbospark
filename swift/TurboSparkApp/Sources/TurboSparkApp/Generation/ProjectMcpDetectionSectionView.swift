import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Section in ProjectMcpSettingsSheet for scanning and importing MCP configuration files from the project codebase.
@MainActor
struct ProjectMcpDetectionSectionView: View {
    let project: AppProject?
    let detectedFiles: [DetectedProjectMcpFile]
    let isDetecting: Bool
    let projectServers: [McpServerConfig]
    let onScan: () -> Void
    let onImportAll: (DetectedProjectMcpFile) -> Void
    let onImportSingle: (McpServerConfig) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Codebase Import & Detection", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                    Text("Scans project root for .mcp.json, opencode.json, .cursor, .vscode, and .agents configs.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Button {
                    onScan()
                } label: {
                    if isDetecting {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 16, height: 16)
                    } else {
                        Label { Text("Scan Folder", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
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
                            .foregroundStyle(.appSecondary)
                        Text("No external MCP config files detected in \(root.lastPathComponent).", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
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
                Text("Assign a Codebase Root Directory in Project Settings to enable automatic MCP file detection.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
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
                    .foregroundStyle(.appAccent)
                VStack(alignment: .leading, spacing: 1) {
                    Text(file.formatLabel)
                        .themedFont(.small, weight: .semibold)
                    Text(file.relativePath)
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Button("Import All (\(file.servers.count))") {
                    onImportAll(file)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 4) {
                ForEach(file.servers) { server in
                    HStack {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(server.name)
                                .themedCode(.small, weight: .semibold)
                            Text(server.commandSummary)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                                .lineLimit(1)
                        }
                        Spacer()
                        let alreadyImported = projectServers.contains { $0.name == server.name }
                        if alreadyImported {
                            Text("Imported", bundle: .module)
                                .themedFont(.tiny, weight: .medium)
                                .foregroundStyle(.appSecondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.12))
                                .clipShape(Capsule())
                        } else {
                            Button {
                                onImportSingle(server)
                            } label: { Text("Import", bundle: .module) }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
            .padding(8)
            .background(.appSurface.opacity(0.8))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(10)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.appAccent.opacity(0.3), lineWidth: 1))
    }
}
