import AppKit
import SwiftUI

// Split out of `PermissionsSettingsPaneView` to keep that file under the
// 400-line guideline. Isolated explicitly for `swift/CLAUDE.md` Gotcha 45's
// reason: only `body` is isolated by the protocol on the macOS 14 SDK.

/// Deep links into the three macOS Privacy & Security panes this app's
/// permissions live in.
@MainActor
struct PermissionsSystemLinksSection: View {
    @ObservedObject var permissionsManager: SystemPermissionsManager

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            AppSectionHeader(title: Text("macOS System Settings Links", bundle: .module))
                .settingsControl("macOS System Settings Links", pane: .permissions, timing: .immediate)

            HStack(spacing: 10) {
                systemLinkCard(
                    title: Text("Files and Folders", bundle: .module),
                    subtitle: Text("Manage per-folder permissions in macOS Privacy & Security.", bundle: .module),
                    icon: "folder.badge.gearshape",
                    pane: .filesAndFolders
                )
                systemLinkCard(
                    title: Text("Full Disk Access", bundle: .module),
                    subtitle: Text("Grant broad filesystem access for developer workflows.", bundle: .module),
                    icon: "externaldrive.fill.badge.checkmark",
                    pane: .fullDiskAccess
                )
                systemLinkCard(
                    title: Text("Accessibility", bundle: .module),
                    subtitle: Text("System control for global shortcuts and automation.", bundle: .module),
                    icon: "figure.roll",
                    pane: .accessibility
                )
            }
        }
    }

    /// The card's own title and subtitle are its VoiceOver name; the hint and
    /// tooltip only say where the click goes.
    private func systemLinkCard(title: Text, subtitle: Text, icon: String, pane: MacPrivacyPane) -> some View {
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
                .accessibilityHidden(true)

                title
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appText)

                subtitle
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .appPointerCursor()
        .help(Text("Opens this pane in macOS System Settings", bundle: .module))
        .accessibilityHint(Text("Opens this pane in macOS System Settings", bundle: .module))
        .accessibilityAddTraits(.isLink)
    }
}

/// The one `MacAppSettings` key that was consumed and unreachable: the
/// classifier's veto had been on the JSON file and in `CommandGate` since
/// it was measured, and only a hand edit could turn it on.
@MainActor
struct CommandGateSettingsSection: View {
    @ObservedObject var model: AppModel
    /// Mirrors `CommandGate.vetoEnabled`, which is a static and so cannot
    /// drive a SwiftUI binding on its own. `persistSettings` reads the static.
    @State private var commandVetoEnabled = CommandGate.vetoEnabled

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label { Text("Command Classifier Veto", bundle: .module) } icon: { Image(systemName: "terminal.fill") }
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Toggle(isOn: $commandVetoEnabled) {
                    Text("Command Classifier Veto", bundle: .module)
                }
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
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder.opacity(0.4), lineWidth: 1)
        )
    }
}

/// What the file tools are actually held to, stated beside the grants so a
/// macOS grant is not read as widening the tools' reach.
struct PermissionsSecurityCallout: View {
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "shield.lefthalf.filled")
                .themedFont(.title2)
                .foregroundStyle(.appAccent)
                .accessibilityHidden(true)

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
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appAccent.opacity(0.25), lineWidth: 0.5)
        )
    }
}
