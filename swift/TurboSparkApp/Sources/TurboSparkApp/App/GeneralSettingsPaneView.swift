import SwiftUI
import TurboSpark

/// Settings pane for language, localization, keyboard layout, and readability.
@MainActor
struct GeneralSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @ObservedObject private var updateController = SparkleUpdateController.shared
    @ObservedObject var model: AppModel

    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    @State private var showsImportWizard = false

    var body: some View {
        Form {
            // Where a change lands is the one thing this pane mixes: most
            // rows apply at once, the chat rows from the next turn, and the
            // relaunch rows say so under their own label.
            Text("Display and app options apply immediately. Chat and agent options apply from the next turn; rows that need a relaunch say so.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)

            languageSection

            textSizeSection

            // Its own computed property: a Section inline in a larger Form
            // expression is the shape that blew the macOS 14 SDK type-checker.
            menuBarSection

            temporaryChatSection

            compactionSection

            chatRuntimeSection

            agentEfficiencySection

            codeSearchSection

            ContextEngineSettingsSection(model: model)

            integrationsSection

            softwareUpdatesSection

            // Hugging Face sign-in lives in Models & Storage only: two token
            // fields for one Keychain item read as two separate settings.
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .sheet(isPresented: $showsImportWizard) {
            AgentContentImportSheet(model: model)
        }
    }

    /// The option rows come from `LanguageOptions`, the same definition the
    /// menus use, so the native-script names and the "System" entry match
    /// everywhere the setting is reachable.
    private var languageSection: some View {
        Section(header: Text("Language & Localization", bundle: .module)) {
            Picker(selection: $languageRawValue) {
                LanguageOptions()
            } label: { Text("Language", bundle: .module) }
            .settingsControl("Language", pane: .general, timing: .immediate)
            .pickerStyle(.menu)

            if let keyboard = LanguageDetector.currentKeyboardLayoutName() {
                LabeledContent {
                    Text(keyboard)
                        .font(theme.code(.small))
                        .foregroundStyle(.appSecondary)
                } label: {
                    Text("Active Keyboard Layout:", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                }
            }
        }
        .settingsControl("Language & Localization", pane: .general, timing: .immediate)
    }

    /// "Text Size", the name the View menu and the profile menu use for the
    /// same setting, with the shared option rows.
    private var textSizeSection: some View {
        Section(header: Text("Text Size & Readability", bundle: .module)) {
            Picker(selection: $appearanceManager.textSize) {
                TextSizeOptions()
            } label: { Text("Text Size", bundle: .module) }
            .settingsControl("Text Size", pane: .general, timing: .immediate)
            .pickerStyle(.segmented)
        }
    }

    // Its own computed property like its siblings: a Section inline in a
    // larger Form expression is the shape that blew the macOS 14 SDK
    // type-checker.
    private var softwareUpdatesSection: some View {
        Section(header: Text("Software Updates", bundle: .module)) {
            Toggle(isOn: $updateController.automaticallyChecksForUpdates) {
                Text("Automatically check for updates", bundle: .module)
            }
            .disabled(!SparkleUpdateController.isBundledApp)
            .settingsControl("Automatically check for updates", pane: .general, timing: .immediate)

            Button {
                updateController.checkForUpdates()
            } label: {
                Text("Check for Updates\u{2026}", bundle: .module)
            }
            .disabled(!updateController.canCheckForUpdates)
            .settingsControl("Check for Updates\u{2026}", pane: .general, timing: .immediate)

            HStack {
                Text("Current Version", bundle: .module)
                Spacer()
                Text(SparkleUpdateController.displayVersion)
                    .font(theme.code(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
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

    /// Cross-agent content: live discovery is OFF by default, and the
    /// Import wizard is the explicit path in. Own section so the toggle and
    /// the button read as one decision.
    private var integrationsSection: some View {
        Section(header: Text("Other Agent Tools", bundle: .module)) {
            Toggle(isOn: Binding(
                get: { model.autoLoadExternalAgentContent },
                set: { model.setAutoLoadExternalAgentContent($0) }
            )) {
                Text("Auto-load skills, agents, plugins, and hooks from other agent tools", bundle: .module)
            }
            .settingsControl("Auto-load skills, agents, plugins, and hooks from other agent tools", pane: .general, timing: .immediate)
            Text("When off, other agents' home folders are never read automatically. Use Import to copy only the skills, agents, and servers you want; project folders inside an open repository are unaffected.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            Button {
                showsImportWizard = true
            } label: {
                Text("Import from Other Agents\u{2026}", bundle: .module)
            }
            .settingsControl("Import from Other Agents\u{2026}", pane: .general, timing: .action)
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

    private var chatRuntimeSection: some View {
        let controls = ChatRuntimeSettingsBindings(model: model)
        return Section(header: Text("Chat Runtime", bundle: .module)) {
            Toggle(isOn: controls.microcompactEnabled) {
                Text("Compact older tool results", bundle: .module)
            }
            .settingsControl("Compact older tool results", pane: .general, timing: .nextTurn)
            Text(
                "Older tool-result bodies are shortened only in model requests. The saved transcript remains unchanged.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            LabeledContent {
                HStack(spacing: 8) {
                    TextField(
                        value: controls.microcompactMinimumSavingsTokens,
                        format: .number.grouping(.never),
                        prompt: Text(verbatim: "64")
                    ) {
                        Text("Minimum token savings (64-4,096 tokens)", bundle: .module)
                    }
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 80, idealWidth: 100, maxWidth: 140)
                    Text("tokens", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                        .fixedSize()
                }
            } label: {
                Text("Minimum token savings (64-4,096 tokens)", bundle: .module)
                    .settingsControl("Minimum token savings (64-4,096 tokens)", pane: .general, timing: .nextTurn)
            }

            Toggle(isOn: controls.autoContinuationEnabled) {
                Text("Continue responses after output limits", bundle: .module)
            }
            .settingsControl("Continue responses after output limits", pane: .general, timing: .nextTurn)
            Text(
                "TurboSpark makes up to three follow-up requests when a response stops at the output limit.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            Toggle(isOn: controls.nativeToolCallingEnabled) {
                Text("Use the model's native tool calling", bundle: .module)
            }
            .settingsControl("Use the model's native tool calling", pane: .general, timing: .nextTurn)
            Text(
                "When the loaded model supports it, tools are rendered by its own chat template and calls are read by the engine, instead of being described in the system prompt. Turn this off to keep the prompt-described format.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            Toggle(isOn: controls.nativeToolCallingIgnoresSpeed) {
                Text("Use it even when the first turn is slow", bundle: .module)
            }
            .disabled(!controls.nativeToolCallingEnabled.wrappedValue)
            .settingsControl("Use it even when the first turn is slow", pane: .general, timing: .nextTurn)
            Text(
                "Skips the time limit on processing tool definitions. On a slow model the first turn can take minutes while they are processed; later turns reuse them. The context-window limit still applies.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            Toggle(isOn: controls.instructionPinningEnabled) {
                Text("Include pinned instructions in model requests", bundle: .module)
            }
            .settingsControl("Include pinned instructions in model requests", pane: .general, timing: .nextTurn)
            Text(
                "Pinned user messages are added in transcript order, up to the token ceiling. Set the ceiling to zero to disable injection.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)

            LabeledContent {
                HStack(spacing: 8) {
                    TextField(
                        value: controls.instructionPinTokenCeiling,
                        format: .number.grouping(.never),
                        prompt: Text(verbatim: "0")
                    ) {
                        Text("Maximum pinned instruction tokens (0-2,048 tokens)", bundle: .module)
                    }
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 80, idealWidth: 100, maxWidth: 140)
                    Text("tokens", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                        .fixedSize()
                }
            } label: {
                Text("Maximum pinned instruction tokens (0-2,048 tokens)", bundle: .module)
                    .settingsControl("Maximum pinned instruction tokens (0-2,048 tokens)", pane: .general, timing: .nextTurn)
            }
        }
        .settingsControl("Chat Runtime", pane: .general, timing: .nextTurn)
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
