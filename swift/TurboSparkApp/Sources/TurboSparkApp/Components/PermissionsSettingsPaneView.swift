import AppKit
import SwiftUI

/// macOS File and Privacy Permissions management pane for app settings.
public struct PermissionsSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var permissionsManager = SystemPermissionsManager.shared

    @State private var hoveredFolder: String?
    @State private var isRefreshing = false

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                headerBar
                systemFoldersSection
                customFoldersSection
                systemPrivacyLinksSection
                securityModelCallout
            }
            .padding(16)
        }
    }

    // MARK: - Header
    private var headerBar: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Files & Privacy Permissions")
                    .font(.title2.weight(.bold))
                Text("Inspect and manage macOS filesystem access, protected user folders, and tool execution permissions.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                isRefreshing = true
                permissionsManager.refreshAllStatuses()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    isRefreshing = false
                }
            } label: {
                Label("Refresh Status", systemImage: isRefreshing ? "arrow.triangle.2.circlepath.circle.fill" : "arrow.triangle.2.circlepath")
            }
            .buttonStyle(.bordered)
            .controlSize(.regular)
            .help("Re-check accessibility for all folders")
        }
    }

    // MARK: - Standard Protected Folders Section
    private var systemFoldersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("macOS Protected Folders")
                    .font(.headline)
                Spacer()
                Text("Governed by macOS TCC & Sandbox")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 8) {
                ForEach(SystemFolderType.allCases) { folder in
                    systemFolderRow(folder)
                }
            }
        }
    }

    private func systemFolderRow(_ folder: SystemFolderType) -> some View {
        let status = permissionsManager.folderStatuses[folder] ?? .unknown

        return HStack(spacing: 12) {
            Image(systemName: folder.systemImage)
                .font(.title3)
                .foregroundStyle(status.isGranted ? TurboSparkTheme.accentColor : .secondary)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text(folder.rawValue)
                        .font(.subheadline.weight(.semibold))
                    Text(folder.pathDisplay)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }

                Text(folder.description)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            // Status Badge
            HStack(spacing: 4) {
                Image(systemName: status.systemIcon)
                    .font(.caption)
                Text(status.rawValue)
                    .font(.caption.weight(.medium))
            }
            .foregroundStyle(status.statusColor)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(status.statusColor.opacity(0.12))
            .clipShape(Capsule())

            // Actions
            HStack(spacing: 6) {
                if !status.isGranted {
                    Button {
                        permissionsManager.requestFolderAccess(for: folder)
                    } label: {
                        Text("Grant Access")
                            .font(.caption)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .help("Prompt macOS to grant file access to \(folder.rawValue)")
                }

                if let url = folder.defaultURL {
                    Button {
                        permissionsManager.revealInFinder(url: url)
                    } label: {
                        Image(systemName: "folder")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("Reveal \(folder.rawValue) in Finder")
                }
            }
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.15), lineWidth: 0.5)
        )
    }

    // MARK: - Custom Granted Folders Section
    private var customFoldersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Authorized Workspace Folders")
                        .font(.headline)
                    Text("Explicitly granted project and workspace directories for file tools and code editing.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    permissionsManager.addCustomFolder()
                } label: {
                    Label("Add Folder...", systemImage: "plus")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Authorize an additional workspace directory")
            }

            if permissionsManager.customFolders.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 6) {
                        Image(systemName: "folder.badge.plus")
                            .font(.title2)
                            .foregroundStyle(.secondary.opacity(0.6))
                        Text("No custom folders authorized yet.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("Add directories where your projects live to streamline autonomous tool access.")
                            .font(.caption2)
                            .foregroundStyle(.secondary.opacity(0.8))
                    }
                    .padding(.vertical, 14)
                    Spacer()
                }
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.15), style: StrokeStyle(lineWidth: 1, dash: [4]))
                )
            } else {
                VStack(spacing: 6) {
                    ForEach(permissionsManager.customFolders) { customFolder in
                        HStack(spacing: 10) {
                            Image(systemName: "folder.fill")
                                .foregroundStyle(TurboSparkTheme.accentColor)
                                .font(.subheadline)

                            VStack(alignment: .leading, spacing: 1) {
                                Text(customFolder.name)
                                    .font(.subheadline.weight(.medium))
                                Text(customFolder.path)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }

                            Spacer()

                            Button {
                                permissionsManager.revealInFinder(url: URL(fileURLWithPath: customFolder.path))
                            } label: {
                                Image(systemName: "folder")
                                    .font(.caption)
                            }
                            .buttonStyle(.borderless)
                            .help("Reveal in Finder")

                            Button {
                                permissionsManager.removeCustomFolder(id: customFolder.id)
                            } label: {
                                Image(systemName: "trash")
                                    .font(.caption)
                                    .foregroundStyle(.red.opacity(0.8))
                            }
                            .buttonStyle(.borderless)
                            .help("Remove authorization")
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.secondary.opacity(0.12), lineWidth: 0.5)
                        )
                    }
                }
            }
        }
    }

    // MARK: - macOS System Privacy Links
    private var systemPrivacyLinksSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("macOS System Settings Links")
                .font(.headline)

            HStack(spacing: 10) {
                systemLinkCard(
                    title: "Files and Folders",
                    subtitle: "Manage per-folder permissions in macOS Privacy & Security.",
                    icon: "folder.badge.gearshape",
                    pane: .filesAndFolders
                )

                systemLinkCard(
                    title: "Full Disk Access",
                    subtitle: "Grant broad filesystem access for developer workflows.",
                    icon: "externaldrive.fill.badge.checkmark",
                    pane: .fullDiskAccess
                )

                systemLinkCard(
                    title: "Accessibility",
                    subtitle: "System control for global shortcuts and automation.",
                    icon: "figure.roll",
                    pane: .accessibility
                )
            }
        }
    }

    private func systemLinkCard(title: String, subtitle: String, icon: String, pane: MacPrivacyPane) -> some View {
        Button {
            permissionsManager.openSystemPrivacySettings(pane)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: icon)
                        .font(.headline)
                        .foregroundStyle(TurboSparkTheme.accentColor)
                    Spacer()
                    Image(systemName: "arrow.up.forward.app")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)

                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.15), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .appPointerCursor()
    }

    // MARK: - Security Model Callout
    private var securityModelCallout: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "shield.lefthalf.filled")
                .font(.title2)
                .foregroundStyle(TurboSparkTheme.accentColor)

            VStack(alignment: .leading, spacing: 4) {
                Text("TurboSpark Safety & Sandbox Guarantee")
                    .font(.subheadline.weight(.semibold))
                Text("Model tools (`FileRead`, `FileWrite`, `FileEdit`, `Terminal`) operate under strict workspace containment checks (`resolveSecurePath`). High-risk operations (e.g., editing files outside project root, deleting directories, or executing destructive terminal commands) require manual user confirmation unless overridden in Project Settings.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(TurboSparkTheme.accentColor.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(TurboSparkTheme.accentColor.opacity(0.25), lineWidth: 0.5)
        )
    }
}
