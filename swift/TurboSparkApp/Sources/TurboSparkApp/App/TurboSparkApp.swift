import AppKit
import SwiftUI

private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Before the first view builds: the font pickers narrow their options
        // to what `NSFontManager` reports, so a bundled family that is not yet
        // registered would be filtered out of its own menu.
        AppFontRegistrar.registerBundledFonts()
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    /// The authoritative quit hook. `RootView`'s own
    /// `willTerminateNotification` observer fired only while the view tree
    /// existed, which -- with the policy above -- is not guaranteed at quit.
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            AppShutdownCoordinator.shared.onTerminate?()
        }
    }
}

@main
struct TurboSparkApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @StateObject private var model = AppModel()
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

    init() {
        AppFontRegistrar.registerBundledFonts()
    }

    var body: some Scene {
        let currentLanguage = AppLanguage.resolve(languageRawValue)
        Window("TurboSpark", id: "main") {
            RootView(model: model)
                .preferredColorScheme(appearanceManager.appearance.preferredColorScheme)
                .dynamicTypeSize(appearanceManager.textSize.dynamicTypeSize)
                .environment(\.locale, currentLanguage.locale)
                .environment(\.layoutDirection, currentLanguage.layoutDirection)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 760)
        .windowResizability(.contentMinSize)
        .commands {
            CommandMenu("View") {

                Button("Chat") {
                    model.activeSection = .chat
                }
                .keyboardShortcut("1", modifiers: .command)

                Button("Files") {
                    model.activeSection = .files
                }
                .keyboardShortcut("2", modifiers: .command)

                Button("Installed Models") {
                    model.activeSection = .modelManager
                }
                .keyboardShortcut("3", modifiers: .command)

                Button("Discover Models") {
                    model.activeSection = .modelHub
                }
                .keyboardShortcut("4", modifiers: .command)

                Divider()

                Button("Toggle Chat Sidebar") {
                    NotificationCenter.default.post(name: .toggleChatSidebar, object: nil)
                }
                .keyboardShortcut("s", modifiers: [.command, .control])

                Button("Toggle Inspector") {
                    NotificationCenter.default.post(name: .toggleInspector, object: nil)
                }
                .keyboardShortcut("i", modifiers: [.command, .shift])

                Divider()

                Picker("Status Bar Benchmarks", selection: $appearanceManager.statusBarViewMode) {
                    ForEach(StatusBarViewMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.systemImage).tag(mode)
                    }
                }

                Divider()

                Button("Make Text Bigger") {
                    appearanceManager.makeTextBigger()
                }
                .keyboardShortcut("+", modifiers: .command)

                Button("Make Text Smaller") {
                    appearanceManager.makeTextSmaller()
                }
                .keyboardShortcut("-", modifiers: .command)

                Button("Default Text Size") {
                    appearanceManager.resetTextSize()
                }
                .keyboardShortcut("0", modifiers: .command)
            }


            CommandMenu("Chat") {
                Button("New Chat") { model.createChat() }
                    .keyboardShortcut("n", modifiers: .command)
                    .disabled(model.isRunning)

                Button("New Temporary Chat") { model.enterGhostChat() }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
                    .disabled(model.isRunning)

                Button("Previous Chat") { model.selectPreviousChat() }
                    .keyboardShortcut("[", modifiers: .command)
                    .disabled(model.isRunning || model.orderedChats.isEmpty)

                Button("Next Chat") { model.selectNextChat() }
                    .keyboardShortcut("]", modifiers: .command)
                    .disabled(model.isRunning || model.orderedChats.isEmpty)

                Divider()

                Button("Focus Prompt") {
                    NotificationCenter.default.post(name: .focusPrompt, object: nil)
                }
                .keyboardShortcut("l", modifiers: .command)
                .accessibilityHint("Moves keyboard focus to the prompt editor")

                Button("Clear Chat History") { model.clearOutput() }
                    .keyboardShortcut("k", modifiers: [.command, .shift])
                    .disabled(model.isRunning || !model.hasOutputTranscript)
            }

            CommandMenu("Generation") {
                Button("Generate Response") { model.run() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!model.canRun)

                Button("Cancel Generation") { model.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.canCancel)

                Button("Cancel Model Installation") { model.cancelInstall() }
                    .disabled(!model.canCancelInstall)
            }

            CommandMenu("Model") {
                Button("Choose Model Folder...") {
                    ModelLocationPicker.choose(for: model)
                }
                .disabled(model.isRunning || model.isInstallingModel)

                Button("Load Model", action: model.loadModel)
                    .disabled(!model.canLoadModel)

                Button("Reload Model", action: model.reloadModel)
                    .disabled(!model.canReloadModel)

                Button("Unload Model", action: model.unloadModel)
                    .disabled(!model.canUnloadModel)
            }

            CommandMenu("Profile") {
                // Switching is a save-and-relaunch, so every entry here ends
                // the process once `switchToProfile` has flushed and saved.
                ForEach([UserProfileStore.defaultProfile] + model.profiles) { profile in
                    Button(profile.id == model.currentProfile.id
                            ? "\(profile.name) (current)"
                            : "Switch to \(profile.name)") {
                        model.switchToProfile(profile)
                    }
                    .disabled(profile.id == model.currentProfile.id || !model.canSwitchProfile)
                }

                Divider()

                Button("New Profile...") {
                    model.openSettings(tab: .profiles)
                }
            }

            CommandMenu("Appearance") {
                Picker("Appearance", selection: $appearanceManager.appearance) {
                    ForEach(AppAppearance.allCases) { appearance in
                        Label(appearance.label, systemImage: appearance.systemImage)
                            .tag(appearance)
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
            }
        }

        Settings {
            // A SEPARATE SCENE, so nothing `RootView` injects reaches it. The
            // theme has to be injected again here or every view in this window
            // reads `ResolvedAppTheme.fallback` -- the light defaults -- while
            // the window itself renders dark.
            AppSettingsView(model: model)
                .appThemed()
                .preferredColorScheme(appearanceManager.appearance.preferredColorScheme)
                .dynamicTypeSize(appearanceManager.textSize.dynamicTypeSize)
                .environment(\.locale, currentLanguage.locale)
                .environment(\.layoutDirection, currentLanguage.layoutDirection)
        }
    }
}


extension Notification.Name {
    /// Posted by the Cmd+L command; PromptComposerView listens to it and
    /// moves first-responder focus to its TextEditor.
    static let focusPrompt = Notification.Name("TurboSpark.focusPrompt")

    /// Posted to toggle the visibility of the conversation sidebar.
    static let toggleChatSidebar = Notification.Name("TurboSpark.toggleChatSidebar")

    /// Posted to toggle the visibility of the right-hand inspector.
    static let toggleInspector = Notification.Name("TurboSpark.toggleInspector")

    /// Posted to request opening the settings window on a specific tab.
    public static let openSettingsTab = Notification.Name("TurboSpark.openSettingsTab")
}

