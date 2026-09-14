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
            Section(header: Text("Language & Localization", bundle: .module)) {
                Picker(selection: $languageRawValue) {
                    ForEach(AppLanguage.allCases) { language in
                        Text(language.label).tag(language.rawValue)
                    }
                } label: { Text("Language", bundle: .module) }
            .settingsControl("Language", pane: .general, timing: .immediate)
                .pickerStyle(.menu)

                if let keyboard = LanguageDetector.currentKeyboardLayoutName() {
                    HStack {
                        Text("Active Keyboard Layout:", bundle: .module)
                            .font(theme.ui(.small))
                            .foregroundStyle(.appSecondary)
                        Spacer()
                        Text(keyboard)
                            .font(theme.code(.small))
                            .foregroundStyle(.appSecondary)
                    }
                }
            }
            .settingsControl("Language & Localization", pane: .general, timing: .immediate)

            Section(header: Text("Text Size & Readability", bundle: .module)) {
                Picker(selection: $appearanceManager.textSize) {
                    ForEach(AppTextSize.allCases) { size in
                        Text(size.label).tag(size)
                    }
                } label: { Text("Text Readability", bundle: .module) }
            .settingsControl("Text Readability", pane: .general, timing: .immediate)
                .pickerStyle(.segmented)
            }

            // Its own computed property: a Section inline in a larger Form
            // expression is the shape that blew the macOS 14 SDK type-checker.
            menuBarSection

            temporaryChatSection

            compactionSection

            agentEfficiencySection

            codeSearchSection

            hfAuthSection
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private var codeSearchSection: some View {
        Section {
            Toggle(isOn: Binding(
                get: { model.syntextIndexingEnabled },
                set: { model.setSyntextIndexingEnabled($0) }
            )) {
                Text("Enable Syntext project indexing", bundle: .module)
            }
            .settingsControl("Enable Syntext project indexing", pane: .general, timing: .immediate)
            Text(
                "Allows opted-in projects to use fast indexed code search. Only the active project's index stays loaded in memory.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
        } header: {
            Text("Code Search & Project Indexing", bundle: .module)
        }
    }

    private var menuBarSection: some View {
        Section(header: Text("Menu Bar & Server", bundle: .module)) {
            Toggle(isOn: Binding(
                get: { model.showMenuBarItem },
                set: { newValue in
                    model.showMenuBarItem = newValue
                    model.persistSettingsDebounced()
                })) {
                Text("Show TurboSpark in menu bar", bundle: .module)
            }
            .settingsControl("Show TurboSpark in menu bar", pane: .general, timing: .immediate)
            Toggle(isOn: Binding(
                get: { model.serverAutoStartOnLaunch },
                set: { newValue in
                    model.serverAutoStartOnLaunch = newValue
                    model.persistSettingsDebounced()
                })) {
                Text("Automatically start server on app launch", bundle: .module)
            }
            .settingsControl("Automatically start server on app launch", pane: .general, timing: .relaunch)
            Toggle(isOn: Binding(
                get: { model.keepServerRunningInBackground },
                set: { newValue in
                    model.keepServerRunningInBackground = newValue
                    model.persistSettingsDebounced()
                })) {
                Text("Keep running in background when window is closed", bundle: .module)
            }
            .settingsControl("Keep running in background when window is closed", pane: .general, timing: .immediate)
            Text("When enabled, closing the main window leaves the server and menu bar active. Use Quit TurboSpark from the menu bar or Command-Q to exit.", bundle: .module)
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
        }
    }

    private var hfAuthSection: some View {
        Section(header: Text("Hugging Face Authentication", bundle: .module)) {
            HfAuthTokenCardView(model: model)
        }
            .settingsControl("Hugging Face Authentication", pane: .general, timing: .immediate)
    }

    private var temporaryChatSection: some View {
        Section(header: Text("Temporary Chats", bundle: .module)) {
            Toggle(isOn: Binding(
                get: { model.alwaysStartInGhostMode },
                set: { newValue in
                    model.alwaysStartInGhostMode = newValue
                    model.persistSettingsDebounced()
                })) {
                Text("Always start in Ghost Mode", bundle: .module)
            }
            .settingsControl("Always start in Ghost Mode", pane: .general, timing: .relaunch)
            Text("Every launch opens a temporary chat that exists only in memory. It is never saved, even when tied to a profile, and is encrypted while the app runs. Ghost chats cannot be recovered.", bundle: .module)
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
        }
    }

    private var compactionSection: some View {
        Section(header: Text("Context Compaction", bundle: .module)) {
            Toggle(isOn: Binding(
                get: { model.autoCompactEnabled },
                set: { newValue in
                    model.autoCompactEnabled = newValue
                    model.persistSettingsDebounced()
                })) {
                Text("Summarize older turns automatically", bundle: .module)
            }
            .settingsControl("Summarize older turns automatically", pane: .general, timing: .nextTurn)
            Text("As the conversation approaches the context window, older messages are summarized by the model and the newer turns kept verbatim. The full transcript stays on screen; only the prompt changes. You can also run /compact at any time.", bundle: .module)
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            Picker(selection: Binding(
                get: { model.compactionKeepRecentTurns },
                set: { newValue in
                    model.compactionKeepRecentTurns = AppChatCompaction.clampKeepRecent(newValue)
                    model.persistSettingsDebounced()
                })) {
                ForEach(AppChatCompaction.keepRecentRange, id: \.self) { count in
                    Text(verbatim: "\(count)").tag(count)
                }
            } label: { Text("Keep recent messages verbatim", bundle: .module) }
            .settingsControl("Keep recent messages verbatim", pane: .general, timing: .nextTurn)
            .pickerStyle(.menu)
        }
            .settingsControl("Context Compaction", pane: .general, timing: .immediate)
    }

    private var agentEfficiencySection: some View {
        Section(header: Text("Agent Efficiency", bundle: .module)) {
            Toggle(isOn: efficiencyBinding(\.actionFusionEnabled)) {
                Text("Fuse edits with validation", bundle: .module)
            }
            Toggle(isOn: efficiencyBinding(\.observationPackEnabled)) {
                Text("Archive and recall large tool output", bundle: .module)
            }
            Toggle(isOn: efficiencyBinding(\.evidenceReducerEnabled)) {
                Text("Verify local diagnostic reductions", bundle: .module)
            }
            Toggle(isOn: efficiencyBinding(\.todoBoundaryCompactionEnabled)) {
                Text("Compact at completed todo boundaries", bundle: .module)
            }
            Text("Validation remains permission-gated. Archived normal-chat output stays local until its chat is deleted. Ghost output stays encrypted in memory.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
        .settingsControl("Agent Efficiency", pane: .general, timing: .nextTurn)
    }

    private func efficiencyBinding(_ keyPath: ReferenceWritableKeyPath<AppModel, Bool>) -> Binding<Bool> {
        Binding(
            get: { model[keyPath: keyPath] },
            set: { value in
                model[keyPath: keyPath] = value
                model.persistSettingsDebounced()
            })
    }
}
