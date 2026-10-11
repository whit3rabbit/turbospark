import AppKit
import SwiftUI

/// Makes the Settings window resizable while still opening at a useful size.
///
/// The Settings scene ignores `idealWidth`/`idealHeight` and opens at the
/// content's MINIMUM (see `AppSettingsView.defaultWindowSize`), which is why
/// the window used to carry a fixed frame -- and a fixed frame is a window
/// the user cannot widen to recover a truncated control (UI audit
/// 2026-10-11). Here the SwiftUI content keeps only a minimum, and the
/// AppKit window is given the default size ONCE, the first time it ever
/// opens. After that the frame autosave restores whatever size the user
/// chose.
struct SettingsWindowFrame: NSViewRepresentable {
    var minimumSize: CGSize
    var defaultSize: CGSize

    /// The autosave name; AppKit stores the frame under
    /// "NSWindow Frame <name>" in the standard defaults.
    static let autosaveName = "TurboSpark.SettingsWindow"

    func makeNSView(context: Context) -> FrameView {
        FrameView(minimumSize: minimumSize, defaultSize: defaultSize)
    }

    func updateNSView(_ view: FrameView, context: Context) {
        view.minimumSize = minimumSize
        view.defaultSize = defaultSize
    }

    final class FrameView: NSView {
        var minimumSize: CGSize
        var defaultSize: CGSize
        private weak var configuredWindow: NSWindow?

        init(minimumSize: CGSize, defaultSize: CGSize) {
            self.minimumSize = minimumSize
            self.defaultSize = defaultSize
            super.init(frame: .zero)
        }

        required init?(coder: NSCoder) { nil }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window, window !== configuredWindow else { return }
            configuredWindow = window
            makeResizable(window)
            // The Settings scene re-applies its own fixed style mask after
            // this view is attached, dropping `.resizable` again (measured
            // 2026-10-11: zoom button disabled, AXSize not settable). Apply it
            // once more after SwiftUI finishes its own window setup.
            DispatchQueue.main.async { [weak self, weak window] in
                guard let self, let window else { return }
                self.makeResizable(window)
            }
            // `setFrameAutosaveName` restores a saved frame when one exists
            // and returns false only when the name is already taken.
            let hadSavedFrame = UserDefaults.standard.string(
                forKey: "NSWindow Frame \(SettingsWindowFrame.autosaveName)") != nil
            window.setFrameAutosaveName(SettingsWindowFrame.autosaveName)
            guard !hadSavedFrame else { return }
            let size = MainWindowMinimumSize.clamped(
                defaultSize, toScreen: window.screen?.visibleFrame.size)
            window.setContentSize(size)
            window.center()
        }

        /// Re-checked on every layout pass, because a scene update can reset
        /// the style mask long after the window first appeared.
        override func layout() {
            super.layout()
            if let window, !window.styleMask.contains(.resizable) {
                makeResizable(window)
            }
        }

        private func makeResizable(_ window: NSWindow) {
            window.styleMask.insert(.resizable)
            window.contentMinSize = MainWindowMinimumSize.clamped(
                minimumSize, toScreen: window.screen?.visibleFrame.size)
            window.contentMaxSize = CGSize(
                width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        }
    }
}
