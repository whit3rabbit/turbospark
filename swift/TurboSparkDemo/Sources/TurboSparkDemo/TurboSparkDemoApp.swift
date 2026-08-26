import AppKit
import SwiftUI

/// Main SwiftUI application entry point for the TurboSpark demo chat app.
@main
struct TurboSparkDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("TurboSpark") {
            ContentView()
        }
        .windowResizability(.contentMinSize)
    }
}

/// Promotes the process so a SwiftPM executable behaves like an app.
///
/// A `swift run` binary has no bundle, so AppKit starts it as an ACCESSORY
/// process: the window appears behind everything, cannot take focus, and
/// there is no menu bar. Without this the demo looks broken in a way that
/// has nothing to do with the binding it exists to test. A shipping app
/// carries an Info.plist and needs none of it.
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Sets regular activation policy and brings the window to front on launch.
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Terminates the application when its main window is closed.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
