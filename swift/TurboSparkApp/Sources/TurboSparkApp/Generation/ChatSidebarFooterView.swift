import AppKit
import SwiftUI

/// Footer component for the chat sidebar showing user profile, device connection, and settings.
struct ChatSidebarFooterView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    let chatCount: Int
    @Binding var languageRawValue: String

    @ScaledMetric private var actionButtonSize: CGFloat = 24
    @ScaledMetric private var avatarSize: CGFloat = 26
    @ScaledMetric private var sidebarFooterMinHeight: CGFloat = 46

    private var profileDisplayName: String {
        let active = UserProfileStore.active
        if active.id != UserProfileStore.defaultProfileID && !active.name.isEmpty {
            return active.name
        }
        let fullName = NSFullUserName()
        if !fullName.isEmpty {
            return fullName
        }
        return active.name.isEmpty ? "User" : active.name
    }

    private var profileInitial: String {
        let name = profileDisplayName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = name.first else { return "U" }
        return String(first).uppercased()
    }

    var body: some View {
        HStack(spacing: 8) {
            // User Avatar & Name Menu
            Menu {
                Section("Active Profile") {
                    Label(profileDisplayName, systemImage: "person.circle.fill")
                    if UserProfileStore.isDefault {
                        Text("Default Machine Profile", bundle: .module)
                    } else {
                        Text("Profile ID: \(UserProfileStore.active.id)", bundle: .module)
                    }
                }

                Divider()

                Section("Appearance") {
                    Picker("Theme", selection: $appearanceManager.appearance) {
                        ForEach(AppAppearance.allCases) { option in
                            Label(option.label, systemImage: option.systemImage)
                                .tag(option)
                        }
                    }

                    Picker("Text Size", selection: $appearanceManager.textSize) {
                        ForEach(AppTextSize.allCases) { size in
                            Text(size.label)
                                .tag(size)
                        }
                    }

                    Picker("Language", selection: $languageRawValue) {
                        ForEach(AppLanguage.allCases) { language in
                            Text(language.label)
                                .tag(language.rawValue)
                        }
                    }
                }

                Divider()

                Text("\(chatCount) local chats stored", bundle: .module)
            } label: {
                HStack(spacing: 7) {
                    avatarView

                    Text(profileDisplayName)
                        .font(theme.ui(.small, weight: .medium))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    badgePill
                }
                .contentShape(.rect)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .help("User Profile: \(profileDisplayName)")
            .accessibilityLabel("User Profile \(profileDisplayName)")

            Spacer(minLength: 4)

            // Device / Companion Icon
            deviceButton

            // Settings Cog
            settingsButton
        }
        .padding(.horizontal, 10)
        .frame(minHeight: sidebarFooterMinHeight)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 0.5)
        }
    }

    private var avatarView: some View {
        ZStack {
            Circle()
                .fill(Color(nsColor: .controlAccentColor).opacity(0.85))
                .frame(width: avatarSize, height: avatarSize)

            Text(profileInitial)
                .font(theme.ui(.tiny, weight: .bold))
                .foregroundStyle(.white)
        }
        .accessibilityHidden(true)
    }

    private var badgePill: some View {
        Text("P", bundle: .module)
            .font(theme.code(.small, weight: .bold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(
                Color.primary.opacity(0.08),
                in: RoundedRectangle(cornerRadius: 4, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
            )
            .accessibilityLabel("Profile badge")
    }

    private var deviceButton: some View {
        Button {
            // Companion / local server indicator
        } label: {
            Image(systemName: "iphone")
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Local Engine & Device Status")
        .accessibilityLabel("Local Engine and Device Status")
    }

    private var settingsButton: some View {
        SettingsLink {
            Image(systemName: "gearshape")
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Settings (Cmd+,)")
        .accessibilityLabel("Application Settings")
    }
}
