import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// macOS File and Privacy Permissions management pane for app settings.
@MainActor
public struct PermissionsSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var permissionsManager = SystemPermissionsManager.shared

    @State private var hoveredFolder: String?
    @State private var isRefreshing = false
    /// Mirrors `CommandGate.vetoEnabled`, which is a static and so cannot
    /// drive a SwiftUI binding on its own. `persistSettings` reads the static.
    @State private var commandVetoEnabled = CommandGate.vetoEnabled

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
                commandGateSection
                AgentModeHintsSectionView(model: model)
                securityModelCallout
            }
            .padding(16)
        }
    }

    // MARK: - Header
    private var headerBar: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Files & Privacy Permissions", bundle: .module)
                    .settingsControl("Files & Privacy Permissions", pane: .permissions, timing: .immediate)
                    .themedFont(.title2, weight: .bold)
                Text("Inspect and manage macOS filesystem access, protected user folders, and tool execution permissions.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button {
                isRefreshing = true
                permissionsManager.refreshAllStatuses()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    isRefreshing = false
                }
            } label: {
                Label { Text("Refresh Status", bundle: .module) } icon: { Image(systemName: isRefreshing ? "arrow.triangle.2.circlepath.circle.fill" : "arrow.triangle.2.circlepath") }
            }
            .buttonStyle(.bordered)
            .controlSize(.regular)
            .help("Re-check accessibility for all folders")
        }
        // A grant made in System Settings was invisible until the manual
        // Refresh, so the pane kept reporting "restricted" for a folder the
        // user had just allowed -- and returning to the app is exactly when
        // they have.
        .onReceive(
            NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
        ) { _ in
            permissionsManager.refreshAllStatusesInBackground()
        }
    }

    // MARK: - Standard Protected Folders Section
    private var systemFoldersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("macOS Protected Folders", bundle: .module)
                    .settingsControl("macOS Protected Folders", pane: .permissions, timing: .immediate)
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Text("Governed by macOS TCC & Sandbox", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
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
                .themedFont(.title3)
                .foregroundStyle(status.isGranted ? TurboSparkTheme.accentColor : .secondary)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text(folder.rawValue)
                        .themedFont(.small, weight: .semibold)
                    Text(folder.pathDisplay)
                        .themedCode(.small)
                        .foregroundStyle(.appSecondary)
                }

                Text(folder.description)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
            }

            Spacer()

            // Status Badge
            HStack(spacing: 4) {
                Image(systemName: status.systemIcon)
                    .themedFont(.small)
                Text(status.rawValue)
                    .themedFont(.small, weight: .medium)
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
                        Text("Grant Access", bundle: .module)
                    .settingsControl("Grant Access", pane: .permissions, timing: .immediate)
                            .themedFont(.small)
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
                            .themedFont(.small)
                    }
                    .buttonStyle(.borderless)
                    .help("Reveal \(folder.rawValue) in Finder")
                }
            }
        }
        .padding(10)
        .background(.appSurface)
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
                    Text("Authorized Workspace Folders", bundle: .module)
                    .settingsControl("Authorized Workspace Folders", pane: .permissions, timing: .immediate)
                        .themedFont(.base, weight: .semibold)
                    Text("Folders you have granted macOS read access to. This does not widen what file tools may reach: those stay inside the selected project's root.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Button {
                    permissionsManager.addCustomFolder()
                } label: {
                    Label {
                        Text("Add Folder…", bundle: .module)
                    .settingsControl("Add Folder…", pane: .permissions, timing: .immediate)
                    } icon: {
                        Image(systemName: "plus")
                    }
                        .themedFont(.small)
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
                            .themedFont(.title2)
                            .foregroundStyle(.secondary.opacity(0.6))
                        Text("No custom folders authorized yet.", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                        Text("Add a folder here to clear the macOS access prompt for it up front.", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.secondary.opacity(0.8))
                    }
                    .padding(.vertical, 14)
                    Spacer()
                }
                .background(.appSurface.opacity(0.5))
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
                                .foregroundStyle(.appAccent)
                                .themedFont(.small)

                            VStack(alignment: .leading, spacing: 1) {
                                Text(customFolder.name)
                                    .themedFont(.small, weight: .medium)
                                Text(customFolder.path)
                                    .themedCode(.small)
                                    .foregroundStyle(.appSecondary)
                                    .lineLimit(1)
                            }

                            Spacer()

                            Button {
                                permissionsManager.revealInFinder(url: URL(fileURLWithPath: customFolder.path))
                            } label: {
                                Image(systemName: "folder")
                                    .themedFont(.small)
                            }
                            .buttonStyle(.borderless)
                            .help(Text("Reveal in Finder", bundle: .module))

                            Button {
                                permissionsManager.removeCustomFolder(id: customFolder.id)
                            } label: {
                                Image(systemName: "trash")
                                    .themedFont(.small)
                                    .foregroundStyle(.red.opacity(0.8))
                            }
                            .buttonStyle(.borderless)
                            .help("Remove authorization")
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(.appSurface)
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
            Text("macOS System Settings Links", bundle: .module)
                    .settingsControl("macOS System Settings Links", pane: .permissions, timing: .immediate)
                .themedFont(.base, weight: .semibold)

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
                        .themedFont(.base, weight: .semibold)
                        .foregroundStyle(.appAccent)
                    Spacer()
                    Image(systemName: "arrow.up.forward.app")
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }

                Text(title)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appText)

                Text(subtitle)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.15), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .appPointerCursor()
        .help("Open macOS \(title) Settings in System Settings")
        .accessibilityLabel("macOS System Settings: \(title)")
        .accessibilityHint("Opens \(title) privacy settings pane in macOS System Settings")
        .accessibilityAddTraits(.isLink)
    }

    // MARK: - Command Gate
    /// The one `MacAppSettings` key that was consumed and unreachable: the
    /// classifier's veto had been on the JSON file and in `CommandGate` since
    /// it was measured, and only a hand edit could turn it on.
    private var commandGateSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label { Text("Command Classifier Veto", bundle: .module) } icon: { Image(systemName: "terminal.fill") }
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Toggle("", isOn: $commandVetoEnabled)
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .onChange(of: commandVetoEnabled) { _, newValue in
                        CommandGate.vetoEnabled = newValue
                        model.persistSettingsDebounced()
                    }
            }
            Text("Lets the local command classifier send an ALLOWLISTED shell command to the approval sheet when it scores it as hazardous. Off by default on purpose: held out by generator the classifier scores 0.70 against 0.997 in distribution, so no threshold stops it prompting on commands the Auto tier promises to run silently. Its reasons are shown either way on commands that were already going to ask.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .background(.appPage)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(.appBorder.opacity(0.4), lineWidth: 1))
    }

    // MARK: - Security Model Callout
    private var securityModelCallout: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "shield.lefthalf.filled")
                .themedFont(.title2)
                .foregroundStyle(.appAccent)

            VStack(alignment: .leading, spacing: 4) {
                Text("TurboSpark Safety & Sandbox Guarantee", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Text("Model tools (`FileRead`, `FileWrite`, `FileEdit`, `Terminal`) operate under strict workspace containment checks (`resolveSecurePath`). High-risk operations (e.g., editing files outside project root, deleting directories, or executing destructive terminal commands) require manual user confirmation unless overridden in Project Settings.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(.appAccent.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(.appAccent.opacity(0.25), lineWidth: 0.5)
        )
    }
}
