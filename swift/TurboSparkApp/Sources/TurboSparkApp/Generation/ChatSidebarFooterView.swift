import AppKit
import SwiftUI

/// Footer component for the chat sidebar showing user profile, device connection, and settings.
struct ChatSidebarFooterView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @ObservedObject var model: AppModel
    let chatCount: Int
    @Binding var languageRawValue: String

    @ScaledMetric private var actionButtonSize: CGFloat = 24
    @ScaledMetric private var avatarSize: CGFloat = 26
    @ScaledMetric private var sidebarFooterMinHeight: CGFloat = 46
    @State private var confirmingEndGhost = false

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
                        .foregroundStyle(.appText)
                        .lineLimit(1)

                    badgePill
                }
                .contentShape(.rect)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .help("User Profile: \(profileDisplayName)")
            .accessibilityLabel("User Profile \(profileDisplayName)")

            // Beside the profile menu rather than the top bar: ghost mode is
            // a property of WHOSE chat this is (temporary, tied to nobody's
            // history), which reads as an extension of the profile row
            // rather than of window chrome.
            ghostButton

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
                .fill(.appBorder)
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
            .foregroundStyle(.appSecondary)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(
                Color.primary.opacity(0.08),
                in: RoundedRectangle(cornerRadius: 4, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .stroke(.appBorder, lineWidth: 0.5)
            )
            .accessibilityLabel("Profile badge")
    }

    /// Ghost Mode: start a temporary chat, or end the one being viewed.
    private var ghostButton: some View {
        Button {
            if model.isInGhostChat {
                if model.ghostChatHasContent {
                    confirmingEndGhost = true
                } else {
                    model.endGhostChat()
                }
            } else {
                model.enterGhostChat()
            }
        } label: {
            GhostGlyph(size: 14, color: model.isInGhostChat ? theme.accent : Color.secondary)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // `enterGhostChat()`/`endGhostChat()` both no-op under exactly this
        // guard, matching the File menu's "New Temporary Chat".
        .disabled(model.generating || model.submitting || model.pendingToolCall != nil)
        .help(model.isInGhostChat ? "End temporary chat" : "New temporary chat")
        .accessibilityLabel(model.isInGhostChat ? "End temporary chat" : "New temporary chat")
        .accessibilityHint(
            model.isInGhostChat
                ? "Discards the temporary conversation. It is never saved."
                : "Starts a temporary chat that lives only in memory and is never saved.")
        .accessibilityAddTraits(model.isInGhostChat ? [.isButton, .isSelected] : .isButton)
        .confirmationDialog(
            "End temporary chat?", isPresented: $confirmingEndGhost,
            titleVisibility: .visible
        ) {
            Button("Discard Temporary Chat", role: .destructive) {
                model.endGhostChat()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "This conversation exists only in memory and cannot be recovered "
                    + "once discarded."
            )
        }
    }

    private var deviceButton: some View {
        Button {
            // Companion / local server indicator
        } label: {
            Image(systemName: "iphone")
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(.appSecondary)
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
                .foregroundStyle(.appSecondary)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Settings (Cmd+,)")
        .accessibilityLabel("Application Settings")
    }
}
