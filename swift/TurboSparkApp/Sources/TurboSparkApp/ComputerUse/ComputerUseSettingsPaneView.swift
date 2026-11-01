import AppKit
import ComputerUseKit
import SwiftUI

/// Settings > Computer Use. A grouped Form, one computed property per
/// `Section` (the macOS 14 type-checker reason on
/// `EngineSettingsPaneView.body`). The shell already shows the title.
///
/// Turning the feature on launches the helper, which only reads grant
/// status. The macOS prompts come from the Allow buttons alone.
@MainActor
struct ComputerUseSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @State private var isRequesting = false

    var body: some View {
        Form {
            featureSection
            if model.computerUseSettings.enabled {
                accessSection
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .task { await model.refreshComputerUseHelperStatus() }
        // The user comes back from System Settings after flipping a switch.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await model.refreshComputerUseHelperStatus() }
        }
    }

    private var featureSection: some View {
        Section(header: Text("Computer Use", bundle: .module)) {
            Toggle(
                isOn: Binding(
                    get: { model.computerUseSettings.enabled },
                    set: { model.setComputerUseEnabled($0) }
                )
            ) {
                Text("Enable computer use", bundle: .module)
            }
            Text("Lets a local model with vision see and control the Mac apps you allow. It asks before using each app, and the menu bar shows a Stop item while it runs.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if model.computerUseSettings.enabled, model.session != nil, !(model.session?.info.vision.active ?? false) {
                Label {
                    Text("The loaded model cannot see images, so it is not offered computer tools.", bundle: .module)
                } icon: {
                    Image(systemName: "eye.slash").accessibilityHidden(true)
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
        }
    }

    @ViewBuilder private var accessSection: some View {
        Section(header: Text("Permissions", bundle: .module)) {
            if HelperComputerUsePort.installedHelperURL() == nil {
                note(Text("This build has no computer use helper. Use the packaged TurboSpark app.", bundle: .module))
            } else if let status = model.computerUseHelperStatus {
                ComputerUsePermissionRow(
                    title: Text("Accessibility", bundle: .module),
                    detail: Text("Reads the controls in allowed apps and sends clicks and keystrokes.", bundle: .module),
                    symbol: "accessibility",
                    granted: status.accessibilityGranted,
                    isBusy: isRequesting
                ) { request(.accessibility) }
                ComputerUsePermissionRow(
                    title: Text("Screen Recording", bundle: .module),
                    detail: Text("Captures the windows of allowed apps so the model can see them.", bundle: .module),
                    symbol: "rectangle.dashed.badge.record",
                    granted: status.screenRecordingGranted,
                    isBusy: isRequesting
                ) { request(.screenRecording) }
                note(Text("macOS lists these under TurboSpark Computer Use, a helper inside TurboSpark.", bundle: .module))
            } else {
                note(Text("The computer use helper is not running.", bundle: .module))
            }
            Button {
                Task { await model.refreshComputerUseHelperStatus() }
            } label: {
                Text("Refresh", bundle: .module)
            }
            .appAction(.quiet, size: .small)
        }
    }

    private func note(_ text: Text) -> some View {
        text
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// The only route to the macOS prompts: the user's click.
    private func request(_ kind: ComputerUsePermissionKind) {
        isRequesting = true
        Task {
            await model.requestComputerUsePermission(kind)
            isRequesting = false
        }
    }
}

/// One grant: status as glyph plus text, and Allow while it is missing.
/// macOS cannot tell "never asked" from "declined" for these grants, so the
/// helper prompts the first time and opens System Settings after that.
@MainActor
struct ComputerUsePermissionRow: View {
    let title: Text
    let detail: Text
    let symbol: String
    let granted: Bool
    let isBusy: Bool
    let action: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: symbol)
                    .themedFont(.large)
                    .foregroundStyle(.appAccent)
                    .frame(minWidth: 24, minHeight: 24)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    title.themedFont(.base, weight: .semibold)
                    detail
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Label {
                        granted ? Text("Allowed", bundle: .module) : Text("Not allowed yet", bundle: .module)
                    } icon: {
                        Image(systemName: granted ? "checkmark.circle.fill" : "circle.dashed")
                            .accessibilityHidden(true)
                    }
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(granted ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
                }
            }
            .accessibilityElement(children: .combine)
            Spacer(minLength: 8)
            if !granted {
                Button(action: action) { Text("Allow", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(isBusy)
                    .accessibilityLabel(Text("Allow", bundle: .module) + Text(verbatim: ", ") + title)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
