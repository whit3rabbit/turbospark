import AppKit
import SwiftUI

/// Footer component for the chat sidebar showing chat count and quick appearance picker.
struct ChatSidebarFooterView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    let chatCount: Int
    @Binding var languageRawValue: String

    @ScaledMetric private var actionButtonSize: CGFloat = 26
    @ScaledMetric private var sidebarFooterMinHeight: CGFloat = 42

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: "internaldrive")
                .accessibilityHidden(true)
            Text("\(chatCount) local chats", bundle: .module)
            Spacer()
            appearanceMenu
        }
        .font(theme.ui(points: 11))
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .frame(minHeight: sidebarFooterMinHeight)
        .help("Local chat history stored on device")
    }

    private var appearanceMenu: some View {
        let appearance = appearanceManager.appearance
        return Menu {
            Picker("Appearance", selection: $appearanceManager.appearance) {
                ForEach(AppAppearance.allCases) { option in
                    Label(option.label, systemImage: option.systemImage)
                        .tag(option)
                }
            }

            Divider()

            Picker("Text Size", selection: $appearanceManager.textSize) {
                ForEach(AppTextSize.allCases) { size in
                    Text(size.label)
                        .tag(size)
                }
            }

            Divider()

            Picker("Language", selection: $languageRawValue) {
                ForEach(AppLanguage.allCases) { language in
                    Text(language.label)
                        .tag(language.rawValue)
                }
            }
        } label: {
            Label("Display & Language Settings", systemImage: appearance.systemImage)
                .labelStyle(.iconOnly)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Circle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Appearance & Language Settings")
        .accessibilityLabel("Display and Language Settings")
        .accessibilityHint("Change theme, text size, and language")
    }
}
