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
                        Text("Active Keyboard Layout:", bundle: .module)
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
            menuBarSection

            temporaryChatSection

            compactionSection

            hfAuthSection
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private var menuBarSection: some View {
        Section("Menu Bar & Server") {
            Toggle("Show TurboSpark in menu bar", isOn: Binding(
                get: { model.showMenuBarItem },
                set: { newValue in
                    model.showMenuBarItem = newValue
                    model.persistSettingsDebounced()
                }))
            Toggle("Automatically start server on app launch", isOn: Binding(
                get: { model.serverAutoStartOnLaunch },
                set: { newValue in
                    model.serverAutoStartOnLaunch = newValue
                    model.persistSettingsDebounced()
                }))
            Toggle("Keep running in background when window is closed", isOn: Binding(
                get: { model.keepServerRunningInBackground },
                set: { newValue in
                    model.keepServerRunningInBackground = newValue
                    model.persistSettingsDebounced()
                }))
            Text(
                "When enabled, closing the main window leaves the server and menu bar active. "
                    + "Use Quit TurboSpark from the menu bar or Command-Q to exit."
            )
            .font(theme.ui(.small))
            .foregroundStyle(.secondary)
        }
    }

    private var hfAuthSection: some View {
        Section("Hugging Face Authentication") {
            HfAuthTokenCardView(model: model)
        }
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

    private var compactionSection: some View {
        Section("Context Compaction") {
            Toggle("Summarize older turns automatically", isOn: Binding(
                get: { model.autoCompactEnabled },
                set: { newValue in
                    model.autoCompactEnabled = newValue
                    model.persistSettingsDebounced()
                }))
            Text(
                "As the conversation approaches the context window, older messages "
                    + "are summarized by the model and the newer turns kept verbatim. "
                    + "The full transcript stays on screen; only the prompt changes. "
                    + "You can also run /compact at any time."
            )
            .font(theme.ui(.small))
            .foregroundStyle(.secondary)

            Picker("Keep recent messages verbatim", selection: Binding(
                get: { model.compactionKeepRecentTurns },
                set: { newValue in
                    model.compactionKeepRecentTurns = AppChatCompaction.clampKeepRecent(newValue)
                    model.persistSettingsDebounced()
                })) {
                ForEach(AppChatCompaction.keepRecentRange, id: \.self) { count in
                    Text("\(count)", bundle: .module).tag(count)
                }
            }
            .pickerStyle(.menu)
        }
    }
}
