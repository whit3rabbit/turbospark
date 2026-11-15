import AppKit
import SwiftUI

/// OC-02: the Notifications section of the General pane. One toggle per
/// event, an optional system sound for it, and one status row for macOS
/// authorization. Authorization starts only from the Allow button here.
@MainActor
struct NotificationsSettingsSection: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var center: AppNotificationCenter

    var body: some View {
        Section(header: Text("Notifications", bundle: .module)) {
            Text(
                "Get a banner when a chat needs you while TurboSpark is in the background. Banners show the chat title and a short line, never the conversation. Nothing is shown while you are looking at the app.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            eventRow(.turnFinished)
                .settingsControl("Turn finished", pane: .general, timing: .immediate)
            eventRow(.needsPermission)
                .settingsControl("Needs approval", pane: .general, timing: .immediate)
            eventRow(.needsAnswer)
                .settingsControl("Has a question", pane: .general, timing: .immediate)
            eventRow(.error)
                .settingsControl("Error", pane: .general, timing: .immediate)

            agentPushRow
                .settingsControl("Allow agents to send notifications", pane: .general, timing: .immediate)

            if model.attentionPreferences.wantsBanners { authorizationRow }
        }
        .settingsControl("Notifications", pane: .general, timing: .immediate)
        // Reading the status never prompts.
        .task { await center.refreshAuthorization() }
    }

    /// Off by default: the agent's PushNotification tool does nothing until
    /// this is on. It never asks macOS for permission; Allow below does.
    private var agentPushRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle(
                isOn: Binding(
                    get: { model.attentionPreferences.allowAgentPush },
                    set: { model.setAgentPushAllowed($0) })
            ) {
                Text("Allow agents to send notifications", bundle: .module)
            }
            Text(
                "Lets the agent post a short banner of its own, such as when a long task is done. Off by default. The text is written by the model, so it can be wrong.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func title(_ event: AttentionEvent) -> Text {
        Text(LocalizedStringKey(event.settingsTitle), bundle: .module)
    }

    private func eventRow(_ event: AttentionEvent) -> some View {
        let enabled = model.attentionPreferences[event].enabled
        return VStack(alignment: .leading, spacing: 6) {
            Toggle(
                isOn: Binding(
                    get: { enabled },
                    set: { model.setAttentionEnabled(event, $0) })
            ) {
                title(event)
            }
            Picker(
                selection: Binding(
                    get: { model.attentionPreferences[event].soundChoice },
                    set: { model.setAttentionSound(event, $0) })
            ) {
                ForEach(AttentionSound.allCases) { sound in
                    soundLabel(sound).tag(sound)
                }
            } label: {
                Text("Sound", bundle: .module)
            }
            .pickerStyle(.menu)
            .disabled(!enabled)
            .accessibilityLabel(Text(verbatim: soundLabelText(event)))
        }
    }

    /// "<event> sound", so four Sound pickers do not read the same.
    private func soundLabelText(_ event: AttentionEvent) -> String {
        let name = AppStrings.localized(String.LocalizationValue(event.settingsTitle))
        return AppStrings.localized("\(name) sound")
    }

    /// The five system sounds are proper names and stay as macOS spells them.
    private func soundLabel(_ sound: AttentionSound) -> Text {
        sound == .none ? Text("No sound", bundle: .module) : Text(verbatim: sound.rawValue)
    }

    /// Status is glyph plus text, with one next step: Allow while macOS can
    /// still prompt, Open System Settings after a refusal.
    @ViewBuilder private var authorizationRow: some View {
        switch center.authorization {
        case .notDetermined:
            VStack(alignment: .leading, spacing: 6) {
                Text(
                    "TurboSpark needs your permission to show banners. macOS asks once, and you can change it later in System Settings.",
                    bundle: .module
                )
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
                Button {
                    Task { await center.requestAuthorization() }
                } label: {
                    Text("Allow Notifications", bundle: .module)
                }
                .appAction(.primary, size: .small)
                .settingsControl("Allow Notifications", pane: .general, timing: .action)
            }
        case .denied:
            VStack(alignment: .leading, spacing: 6) {
                Label {
                    Text(
                        "Notifications are off for TurboSpark in System Settings. These events stay on here, but no banner is shown.",
                        bundle: .module)
                } icon: {
                    Image(systemName: "bell.slash")
                }
                .font(theme.ui(.small))
                .foregroundStyle(.appWarning)
                Button {
                    Self.openSystemNotificationSettings()
                } label: {
                    Text("Open System Settings", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .settingsControl("Open System Settings", pane: .general, timing: .action)
            }
        case .unavailable:
            Label {
                Text(
                    "Banners work only in the installed TurboSpark app, not in a development build.",
                    bundle: .module)
            } icon: {
                Image(systemName: "info.circle")
            }
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
        case .authorized:
            EmptyView()
        }
    }

    private static func openSystemNotificationSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.Notifications-Settings.extension") {
            NSWorkspace.shared.open(url)
        }
    }
}
