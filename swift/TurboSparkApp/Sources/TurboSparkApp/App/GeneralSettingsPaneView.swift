import SwiftUI
import TurboSpark

/// Settings pane for language, localization, keyboard layout, and readability.
struct GeneralSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @ObservedObject var model: AppModel

    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    var body: some View {
        Form {
            Section("Language & Localization") {
                Picker("Language", selection: $languageRawValue) {
                    ForEach(AppLanguage.allCases) { language in
                        Text(language.label).tag(language.rawValue)
                    }
                }
                .pickerStyle(.menu)

                if let keyboard = LanguageDetector.currentKeyboardLayoutName() {
                    HStack {
                        Text("Active Keyboard Layout:")
                            .font(theme.ui(.small))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(keyboard)
                            .font(theme.code(.small))
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Text Size & Readability") {
                Picker("Text Readability", selection: $appearanceManager.textSize) {
                    ForEach(AppTextSize.allCases) { size in
                        Text(size.label).tag(size)
                    }
                }
                .pickerStyle(.segmented)
            }

            // Its own computed property: a Section inline in a larger Form
            // expression is the shape that blew the macOS 14 SDK type-checker.
            temporaryChatSection
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private var temporaryChatSection: some View {
        Section("Temporary Chats") {
            Toggle("Always start in Ghost Mode", isOn: Binding(
                get: { model.alwaysStartInGhostMode },
                set: { newValue in
                    model.alwaysStartInGhostMode = newValue
                    model.persistSettingsDebounced()
                }))
            Text(
                "Every launch opens a temporary chat that exists only in memory. "
                    + "It is never saved, even when tied to a profile, and is "
                    + "encrypted while the app runs. Ghost chats cannot be recovered."
            )
            .font(theme.ui(.small))
            .foregroundStyle(.secondary)
        }
    }
}
