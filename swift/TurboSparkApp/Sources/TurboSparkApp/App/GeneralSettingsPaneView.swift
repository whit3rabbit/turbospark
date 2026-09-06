import SwiftUI
import TurboSpark

/// Settings pane for language, localization, keyboard layout, and readability.
struct GeneralSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared

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
        }
        .formStyle(.grouped)
        .padding(16)
    }
}
