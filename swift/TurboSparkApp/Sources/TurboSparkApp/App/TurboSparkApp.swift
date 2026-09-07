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
        let settings = MacAppSettingsFileStore.load()
        if settings.keepServerRunningInBackground || settings.showMenuBarItem {
            return false
        }
        return true
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            for window in sender.windows {
                if window.canBecomeMain {
                    window.makeKeyAndOrderFront(self)
                    return true
                }
            }
        }
        return true
    }

    /// The authoritative quit hook. `RootView`'s own
    /// `willTerminateNotification` observer fired only while the view tree
    /// existed, which -- with the policy above -- is not guaranteed at quit.
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            // Before the coordinator: the fan restore must not depend on the
            // view tree having set `onTerminate`, and a hold left pinned
            // outlives the process (FanController's header).
            FanController.shared.restoreOnQuitIfNeeded()
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
                .environment(\.locale, currentLanguage.locale)
                .environment(\.layoutDirection, currentLanguage.layoutDirection)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 760)
        .windowResizability(.contentMinSize)
        .commands {
            // Menu-bar chrome takes its text from `Text(_:bundle:)` labels
            // rather than bare `LocalizedStringKey`s: the string catalog lives
            // in `Bundle.module`, and the key-only initializers
            // (`Button("Chat")`, `CommandMenu("View")`) resolve against
            // `Bundle.main`, which carries no strings in a SwiftPM build --
            // they would render English under every language.
            CommandMenu(Text("View", bundle: .module)) {

                Button {
                    model.activeSection = .chat
                } label: {
                    Text("Chat", bundle: .module)
                }
                .keyboardShortcut("1", modifiers: .command)

                Button {
                    model.activeSection = .files
                } label: {
                    Text("Files", bundle: .module)
                }
                .keyboardShortcut("2", modifiers: .command)

                Button {
                    model.activeSection = .modelManager
                } label: {
                    Text("Installed Models", bundle: .module)
                }
                .keyboardShortcut("3", modifiers: .command)

                Button {
                    model.activeSection = .modelHub
                } label: {
                    Text("Discover Models", bundle: .module)
                }
                .keyboardShortcut("4", modifiers: .command)

                // The rail's tooltip had advertised this since the section
                // was appended (`AppSection.shortcutKey`), with nothing bound.
                Button {
                    model.activeSection = .server
                } label: {
                    Text("Server", bundle: .module)
                }
                .keyboardShortcut("5", modifiers: .command)

                Divider()

                Button {
                    NotificationCenter.default.post(name: .toggleChatSidebar, object: nil)
                } label: {
                    Text("Toggle Chat Sidebar", bundle: .module)
                }
                .keyboardShortcut("s", modifiers: [.command, .control])

                Button {
                    NotificationCenter.default.post(name: .toggleInspector, object: nil)
                } label: {
                    Text("Toggle Inspector", bundle: .module)
                }
                .keyboardShortcut("i", modifiers: [.command, .shift])

                // The Keyboard Shortcuts pane's own entry point, matching
                // unsloth studio's Cmd+/ for its shortcuts tab.
                Button {
                    model.openSettings(tab: .shortcuts)
                } label: {
                    Text("Keyboard Shortcuts...", bundle: .module)
                }
                .keyboardShortcut("/", modifiers: .command)

                Divider()

                Picker(selection: $appearanceManager.statusBarViewMode) {
                    ForEach(StatusBarViewMode.allCases) { mode in
                        Label(mode.label, systemImage: mode.systemImage).tag(mode)
                    }
                } label: {
                    Text("Status Bar Benchmarks", bundle: .module)
                }

                Divider()

                Button {
                    appearanceManager.makeTextBigger()
                } label: {
                    Text("Make Text Bigger", bundle: .module)
                }
                .keyboardShortcut("+", modifiers: .command)

                Button {
                    appearanceManager.makeTextSmaller()
                } label: {
                    Text("Make Text Smaller", bundle: .module)
                }
                .keyboardShortcut("-", modifiers: .command)

                Button {
                    appearanceManager.resetTextSize()
                } label: {
                    Text("Default Text Size", bundle: .module)
                }
                .keyboardShortcut("0", modifiers: .command)
            }


            CommandMenu(Text("Chat", bundle: .module)) {
                Button {
                    model.createChat()
                } label: {
                    Text("New Chat", bundle: .module)
                }
                .keyboardShortcut("n", modifiers: .command)
                .disabled(model.isRunning)

                Button {
                    model.enterGhostChat()
                } label: {
                    Text("New Temporary Chat", bundle: .module)
                }
                .keyboardShortcut("n", modifiers: [.command, .shift])
                .disabled(model.isRunning)

                Button {
                    model.selectPreviousChat()
                } label: {
                    Text("Previous Chat", bundle: .module)
                }
                .keyboardShortcut("[", modifiers: .command)
                .disabled(model.isRunning || model.orderedChats.isEmpty)

                Button {
                    model.selectNextChat()
                } label: {
                    Text("Next Chat", bundle: .module)
                }
                .keyboardShortcut("]", modifiers: .command)
                .disabled(model.isRunning || model.orderedChats.isEmpty)

                Divider()

                Button {
                    NotificationCenter.default.post(name: .showChatSearch, object: nil)
                } label: {
                    Text("Search Chats...", bundle: .module)
                }
                .keyboardShortcut("k", modifiers: .command)
                .accessibilityHint(Text("Searches previous chats by keyword", bundle: .module))

                Button {
                    NotificationCenter.default.post(name: .focusPrompt, object: nil)
                } label: {
                    Text("Focus Prompt", bundle: .module)
                }
                .keyboardShortcut("l", modifiers: .command)
                .accessibilityHint(Text("Moves keyboard focus to the prompt editor", bundle: .module))

                Button {
                    model.clearOutput()
                } label: {
                    Text("Clear Chat History", bundle: .module)
                }
                .keyboardShortcut("k", modifiers: [.command, .shift])
                .disabled(model.isRunning || !model.hasOutputTranscript)
            }

            CommandMenu(Text("Generation", bundle: .module)) {
                // CanRunOrQueue, not canRun: mid-turn the item QUEUES the
                // draft, so the label says which one it will do.
                Button {
                    model.run()
                } label: {
                    model.canRun
                        ? Text("Generate Response", bundle: .module)
                        : Text("Queue Message", bundle: .module)
                }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!model.canRunOrQueue)

                Button {
                    model.cancel()
                } label: {
                    Text("Cancel Generation", bundle: .module)
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!model.canCancel)

                Button {
                    model.cancelInstall()
                } label: {
                    Text("Cancel Model Installation", bundle: .module)
                }
                .disabled(!model.canCancelInstall)
            }

            CommandMenu(Text("Model", bundle: .module)) {
                Button {
                    ModelLocationPicker.choose(for: model)
                } label: {
                    Text("Choose Model Folder...", bundle: .module)
                }
                .disabled(model.isRunning || model.isInstallingModel)

                Button {
                    model.loadModel()
                } label: {
                    Text("Load Model", bundle: .module)
                }
                .disabled(!model.canLoadModel)

                Button {
                    model.reloadModel()
                } label: {
                    Text("Reload Model", bundle: .module)
                }
                .disabled(!model.canReloadModel)

                Button {
                    model.unloadModel()
                } label: {
                    Text("Unload Model", bundle: .module)
                }
                .disabled(!model.canUnloadModel)
            }

            CommandMenu(Text("Profile", bundle: .module)) {
                // Switching is a save-and-relaunch, so every entry here ends
                // the process once `switchToProfile` has flushed and saved.
                ForEach([UserProfileStore.defaultProfile] + model.profiles) { profile in
                    Button {
                        model.switchToProfile(profile)
                    } label: {
                        profile.id == model.currentProfile.id
                            ? Text("\(profile.name) (current)", bundle: .module)
                            : Text("Switch to \(profile.name)", bundle: .module)
                    }
                    .disabled(profile.id == model.currentProfile.id || !model.canSwitchProfile)
                }

                Divider()

                Button {
                    model.openSettings(tab: .profiles)
                } label: {
                    Text("New Profile...", bundle: .module)
                }
            }

            CommandMenu(Text("Appearance", bundle: .module)) {
                Picker(selection: $appearanceManager.appearance) {
                    ForEach(AppAppearance.allCases) { appearance in
                        Label(appearance.label, systemImage: appearance.systemImage)
                            .tag(appearance)
                    }
                } label: {
                    Text("Appearance", bundle: .module)
                }

                Divider()

                Picker(selection: $appearanceManager.textSize) {
                    ForEach(AppTextSize.allCases) { size in
                        Text(size.label)
                            .tag(size)
                    }
                } label: {
                    Text("Text Size", bundle: .module)
                }

                Divider()

                Picker(selection: $languageRawValue) {
                    ForEach(AppLanguage.allCases) { language in
                        Text(language.label)
                            .tag(language.rawValue)
                    }
                } label: {
                    Text("Language", bundle: .module)
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
                .environment(\.locale, currentLanguage.locale)
                .environment(\.layoutDirection, currentLanguage.layoutDirection)
        }

        MenuBarExtra(
            "TurboSpark",
            systemImage: model.server != nil ? "bolt.fill" : "bolt",
            // Guard writes to prevent an infinite re-render loop: MenuBarExtra
            // on macOS writes back its insertion state on each scene graph evaluation.
            // Directly passing $model.showMenuBarItem fires objectWillChange on
            // unchanged values and spins the main thread at 100% CPU.
            isInserted: Binding(
                get: { model.showMenuBarItem },
                set: { newValue in
                    if model.showMenuBarItem != newValue {
                        model.showMenuBarItem = newValue
                    }
                }
            )
        ) {
            // A THIRD scene: the same reason the Settings scene above injects
            // its own theme applies here, and the menu bar's own text and
            // icons were reading `ResolvedAppTheme.fallback` for the life of
            // the feature (swift/docs/SWIFT_SETTINGS_AUDIT.md item 7).
            ServerMenuBarView(model: model)
                .appThemed()
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

    /// Posted by the Cmd+K command; RootView listens to it and toggles the
    /// "Search Chats" overlay.
    static let showChatSearch = Notification.Name("TurboSpark.showChatSearch")
}

