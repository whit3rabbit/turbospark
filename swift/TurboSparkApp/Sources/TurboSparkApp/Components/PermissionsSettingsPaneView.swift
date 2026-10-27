import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// macOS File and Privacy Permissions management pane for app settings.
@MainActor
public struct PermissionsSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var permissionsManager = SystemPermissionsManager.shared

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
                PermissionsSystemLinksSection(permissionsManager: permissionsManager)
                CommandGateSettingsSection(model: model)
                AgentModeHintsSectionView(model: model)
                PermissionsSecurityCallout()
            }
            .padding(SettingsLayout.contentInsets)
        }
    }

    // MARK: - Header
    /// The Settings shell already titles this pane, so the header is only
    /// the pane's purpose and its one action. The search anchor that sat on
    /// the old in-pane title moves to the purpose line so a search for it
    /// still lands at the top of the pane.
    ///
    /// A plain row rather than `AppAdaptiveHeader`: the purpose line wraps
    /// and the only action is one icon, so nothing here can truncate, and
    /// the adaptive header would measure the sentence unwrapped and always
    /// drop the icon onto a second row.
    private var headerBar: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("Inspect and manage macOS filesystem access, protected user folders, and tool execution permissions.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .settingsControl("Files & Privacy Permissions", pane: .permissions, timing: .immediate)
            AppIconButton(
                isRefreshing ? "arrow.triangle.2.circlepath.circle.fill" : "arrow.triangle.2.circlepath",
                label: Text("Refresh Status", bundle: .module)
            ) {
                isRefreshing = true
                permissionsManager.refreshAllStatuses()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    isRefreshing = false
                }
            }
            .appAction(.secondary, size: .small)
            .accessibilityHint(Text("Re-checks access for every folder listed here", bundle: .module))
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
            AppSectionHeader(
                title: Text("macOS Protected Folders", bundle: .module),
                subtitle: Text("Governed by macOS TCC & Sandbox", bundle: .module)
            )
            .settingsControl("macOS Protected Folders", pane: .permissions, timing: .immediate)

            VStack(spacing: 8) {
                ForEach(SystemFolderType.allCases) { folder in
                    systemFolderRow(folder)
                }
            }
        }
    }

    private func systemFolderRow(_ folder: SystemFolderType) -> some View {
        let status = permissionsManager.folderStatuses[folder] ?? .unknown
        let name = Self.folderName(folder)

        return HStack(spacing: 12) {
            Image(systemName: folder.systemImage)
                .themedFont(.title3)
                .foregroundStyle(status.isGranted ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
                .frame(minWidth: 24)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text(name)
                        .themedFont(.small, weight: .semibold)
                    Text(verbatim: folder.pathDisplay)
                        .themedCode(.small)
                        .foregroundStyle(.appSecondary)
                }

                Self.folderPurpose(folder)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()

            // Status badge: the explicit word, not only the color, so the
            // state reads the same to VoiceOver and to a color-blind reader.
            HStack(spacing: 4) {
                Image(systemName: status.systemIcon)
                    .themedFont(.small)
                    .accessibilityHidden(true)
                Self.statusLabel(status)
                    .themedFont(.small, weight: .medium)
            }
            .foregroundStyle(status.statusColor)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(status.statusColor.opacity(0.12))
            .clipShape(Capsule())

            HStack(spacing: 6) {
                if !status.isGranted {
                    Button {
                        permissionsManager.requestFolderAccess(for: folder)
                    } label: {
                        Text("Grant Access", bundle: .module)
                            .settingsControl("Grant Access", pane: .permissions, timing: .immediate)
                    }
                    .appAction(.secondary, size: .small)
                    .help(Text("Prompt macOS to grant file access to \(name)", bundle: .module))
                    .accessibilityLabel(Text("Grant access to \(name)", bundle: .module))
                }

                if let url = folder.defaultURL {
                    AppIconButton("folder", label: Text("Reveal \(name) in Finder", bundle: .module)) {
                        permissionsManager.revealInFinder(url: url)
                    }
                    .appAction(.quiet, size: .small)
                }
            }
        }
        .padding(10)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5)
        )
    }

    // MARK: - Custom Granted Folders Section
    private var customFoldersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            // A plain row for the same reason as `headerBar`: the subtitle
            // wraps, and the button keeps its full label beside it.
            HStack(alignment: .top, spacing: 12) {
                AppSectionHeader(
                    title: Text("Authorized Workspace Folders", bundle: .module),
                    subtitle: Text("Folders you have granted macOS read access to. This does not widen what file tools may reach: those stay inside the selected project's root.", bundle: .module)
                )
                .settingsControl("Authorized Workspace Folders", pane: .permissions, timing: .immediate)
                Button {
                    permissionsManager.addCustomFolder()
                } label: {
                    Label {
                        Text("Add Folder\u{2026}", bundle: .module)
                            .settingsControl("Add Folder\u{2026}", pane: .permissions, timing: .immediate)
                    } icon: {
                        Image(systemName: "plus")
                    }
                }
                .appAction(.secondary, size: .small)
                .fixedSize()
                .help(Text("Choose a folder to grant macOS access to.", bundle: .module))
                .accessibilityHint(Text("Choose a folder to grant macOS access to.", bundle: .module))
            }

            if permissionsManager.customFolders.isEmpty {
                customFoldersEmptyState
            } else {
                VStack(spacing: 6) {
                    ForEach(permissionsManager.customFolders) { customFolder in
                        customFolderRow(customFolder)
                    }
                }
            }
        }
    }

    private var customFoldersEmptyState: some View {
        AppEmptyState(
            systemImage: "folder.badge.plus",
            title: Text("No custom folders authorized yet.", bundle: .module),
            message: Text("Add a folder here to clear the macOS access prompt for it up front.", bundle: .module)
        )
        .padding(.vertical, 8)
    }

    private func customFolderRow(_ customFolder: GrantedCustomFolder) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "folder.fill")
                .foregroundStyle(.appAccent)
                .themedFont(.small)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(verbatim: customFolder.name)
                    .themedFont(.small, weight: .medium)
                SettingsPathText(customFolder.path)
                    .foregroundStyle(.appSecondary)
            }

            Spacer()

            AppIconButton("folder", label: Text("Reveal \(customFolder.name) in Finder", bundle: .module)) {
                permissionsManager.revealInFinder(url: URL(fileURLWithPath: customFolder.path))
            }
            .appAction(.quiet, size: .small)

            AppIconButton(
                "minus.circle",
                label: Text("Remove authorization for \(customFolder.name)", bundle: .module)
            ) {
                permissionsManager.removeCustomFolder(id: customFolder.id)
            }
            .appAction(.quiet, size: .small)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5)
        )
    }

    // MARK: - Localized folder and status names
    // `SystemFolderType` and `FolderAccessStatus` raw values are identifiers
    // shared with persistence and tests, so the display names are resolved
    // here, one literal per case, where the catalog scan can see each key.

    /// A `String` rather than a `Text` because it is also interpolated into
    /// the per-folder help and VoiceOver labels.
    static func folderName(_ folder: SystemFolderType) -> String {
        switch folder {
        case .documents: return AppStrings.localized("Documents")
        case .downloads: return AppStrings.localized("Downloads")
        case .desktop: return AppStrings.localized("Desktop")
        case .home: return AppStrings.localized("Home Directory")
        }
    }

    static func folderPurpose(_ folder: SystemFolderType) -> Text {
        switch folder {
        case .documents:
            return Text("Used for accessing attached documents, notes, and user project files.", bundle: .module)
        case .downloads:
            return Text("Used for downloading model checkpoints, datasets, and reading imported files.", bundle: .module)
        case .desktop:
            return Text("Used for accessing desktop attachments and user workspace folders.", bundle: .module)
        case .home:
            return Text("Root user folder for config files (.turbospark, .mcp, scripts).", bundle: .module)
        }
    }

    static func statusLabel(_ status: FolderAccessStatus) -> Text {
        switch status {
        case .granted: return Text("Granted", bundle: .module)
        case .restricted: return Text("Requires Permission", bundle: .module)
        case .notFound: return Text("Not Found", bundle: .module)
        case .unknown: return Text("Unknown", bundle: .module)
        }
    }
}
