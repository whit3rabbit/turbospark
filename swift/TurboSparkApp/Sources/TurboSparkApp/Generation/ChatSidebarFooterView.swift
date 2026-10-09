import AppKit
import SwiftUI

/// Footer component for the chat sidebar showing user profile, device connection, and settings.
struct ChatSidebarFooterView: View {
    @Environment(\.appTheme) private var theme
    @Environment(\.locale) private var locale
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @ObservedObject var model: AppModel
    let chatCount: Int
    @Binding var languageRawValue: String

    private var actionButtonSize: CGFloat { theme.sidebarActionSize }
    private var avatarSize: CGFloat { 26 * theme.sidebarScale }
    private var sidebarFooterMinHeight: CGFloat { 46 * theme.sidebarScale }
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
        return active.name.isEmpty ? String(localized: "User", bundle: .module, locale: locale) : active.name
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
                Section(header: Text("Active Profile", bundle: .module)) {
                    Label(profileDisplayName, systemImage: "person.circle.fill")
                    if UserProfileStore.isDefault {
                        Text("Default Machine Profile", bundle: .module)
                    } else {
                        Text("Profile ID: \(UserProfileStore.active.id)", bundle: .module)
                    }
                }

                Divider()

                Section(header: Text("Appearance", bundle: .module)) {
                    Picker(selection: $appearanceManager.appearance) {
                        ForEach(AppAppearance.allCases) { option in
                            Label { Text(LocalizedStringKey(option.label), bundle: .module) } icon: { Image(systemName: option.systemImage) }
                                .tag(option)
                        }
                    } label: { Text("Theme", bundle: .module) }

                    Picker(selection: $appearanceManager.textSize) {
                        ForEach(AppTextSize.allCases) { size in
                            Text(LocalizedStringKey(size.label), bundle: .module)
                                .tag(size)
                        }
                    } label: { Text("Text Size", bundle: .module) }

                    Picker(selection: $languageRawValue) {
                        ForEach(AppLanguage.allCases) { language in
                            Group {
                                if language == .system {
                                    Text("System", bundle: .module)
                                } else {
                                    Text(verbatim: language.label)
                                }
                            }
                            .tag(language.rawValue)
                        }
                    } label: { Text("Language", bundle: .module) }
                }

                Divider()

                Text("\(chatCount) local chats", bundle: .module)
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
            .help(Text("User profile and appearance settings", bundle: .module))
            .accessibilityLabel(Text("User profile and appearance", bundle: .module))

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
                .fill(theme.selection)
                .frame(width: avatarSize, height: avatarSize)

            Text(profileInitial)
                .font(theme.ui(.tiny, weight: .bold))
                .foregroundStyle(.appText)
        }
        .accessibilityHidden(true)
    }

    private var badgePill: some View {
        Text(verbatim: "P")
            .font(theme.code(.small, weight: .bold))
            .foregroundStyle(.appSecondary)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(
                theme.foreground.opacity(0.08),
                in: RoundedRectangle(cornerRadius: 4, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .stroke(.appBorder, lineWidth: 0.5)
            )
            .accessibilityLabel(Text("Active Profile", bundle: .module))
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
            GhostGlyph(size: 14 * theme.sidebarScale, color: model.isInGhostChat ? theme.accent : theme.secondaryText)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // `enterGhostChat()`/`endGhostChat()` both no-op under exactly this
        // guard, matching the File menu's "New Temporary Chat".
        .disabled(model.generating || model.submitting || model.pendingToolCall != nil)
        .help(Text(LocalizedStringKey(model.isInGhostChat ? "Discard Temporary Chat" : "New Temporary Chat"), bundle: .module))
        .accessibilityLabel(Text(LocalizedStringKey(model.isInGhostChat ? "Discard Temporary Chat" : "New Temporary Chat"), bundle: .module))
        .accessibilityAddTraits(model.isInGhostChat ? [.isButton, .isSelected] : .isButton)
        .confirmationDialog(
            Text("Discard Temporary Chat", bundle: .module), isPresented: $confirmingEndGhost,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                model.endGhostChat()
            } label: { Text("Discard Temporary Chat", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This conversation exists only in memory and cannot be recovered once discarded.", bundle: .module)
        }
    }

    private var deviceButton: some View {
        Button {
            model.activeSection = .server
        } label: {
            Image(systemName: "iphone")
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(.appSecondary)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(Text("API", bundle: .module))
        .accessibilityLabel(Text("API", bundle: .module))
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
        .help(Text("Settings", bundle: .module) + Text(verbatim: " (Cmd+,)"))
        .accessibilityLabel(Text("Settings", bundle: .module))
    }
}
