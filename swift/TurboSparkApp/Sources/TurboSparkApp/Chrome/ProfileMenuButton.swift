import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The profile and appearance menu, as an avatar at the foot of the rail.
///
/// It used to lead `ChatSidebarFooterView`. It moved here because the rail is
/// the only chrome present in every section and state: a footer on the panel
/// would vanish with the panel and in every rail-only section. The menu
/// content is unchanged, including the shared Mode, Text Size and Language
/// option views, so this place and the Appearance menu cannot drift apart.
@MainActor
struct ProfileMenuButton: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @ObservedObject var model: AppModel
    let itemSize: CGFloat

    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    private var avatarSize: CGFloat { min(30, 26 * theme.sidebarScale) }

    /// "local chats stored" is a machine-wide figure, so it counts every
    /// persisted chat regardless of the selected project or archive filter.
    private var chatCount: Int { model.chats.filter { !$0.isGhost }.count }

    private var profileDisplayName: String {
        let active = UserProfileStore.active
        if active.id != UserProfileStore.defaultProfileID && !active.name.isEmpty {
            return active.name
        }
        let fullName = NSFullUserName()
        if !fullName.isEmpty {
            return fullName
        }
        return active.name.isEmpty ? AppStrings.localized("User") : active.name
    }

    private var profileInitial: String {
        let name = profileDisplayName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = name.first else { return "U" }
        return String(first).uppercased()
    }

    var body: some View {
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

            // The same option rows and the same "Mode" name as the
            // Appearance menu bar menu, so the two places this is
            // reachable cannot drift apart.
            Section(header: Text("Appearance", bundle: .module)) {
                Picker(selection: $appearanceManager.appearance) {
                    AppearanceModeOptions()
                } label: { Text("Mode", bundle: .module) }

                Picker(selection: $appearanceManager.textSize) {
                    TextSizeOptions()
                } label: { Text("Text Size", bundle: .module) }

                Picker(selection: $languageRawValue) {
                    LanguageOptions()
                } label: { Text("Language", bundle: .module) }
            }

            Divider()

            Text("\(chatCount) local chats", bundle: .module)
        } label: {
            ZStack {
                Circle()
                    .fill(theme.selection)
                    .frame(width: avatarSize, height: avatarSize)

                Text(profileInitial)
                    .font(theme.ui(.tiny, weight: .bold))
                    .foregroundStyle(.appText)
            }
            .frame(width: itemSize, height: itemSize)
            .contentShape(.rect)
        }
        // `.button` + `.plain`, not `.borderlessButton`: the latter FLATTENS a
        // custom label on the macOS 14 SDK to its first Text, which drew a
        // bare initial with no circle (`ModelLoaderControl` has the history).
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(Text("User profile and appearance settings", bundle: .module))
        .accessibilityLabel(Text("User profile and appearance", bundle: .module))
    }
}
