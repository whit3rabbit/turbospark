import AppKit
import CoreGraphics

/// The size a workspace window opens at: the whole visible screen.
///
/// The main window and the Settings window share this rule, so Settings
/// opens about the same size as the app instead of a fraction of it.
///
/// This asks the screen for its VISIBLE frame -- visible rather than full,
/// so the menu bar and Dock are subtracted and the window does not open
/// underneath either one.
///
/// `NSScreen.main` is nil in some launch contexts (no attached display,
/// certain headless runs), and the fallback is the standard default size
/// rather than a computed guess.
enum AppWindowDefaultSize {
    /// Fallback when there is no screen to measure.
    static let fallback = CGSize(width: 1280, height: 760)

    static func size() -> CGSize {
        size(forVisibleFrame: NSScreen.main?.visibleFrame)
    }

    /// Separate from `NSScreen` so the rule is testable without a display.
    static func size(forVisibleFrame visible: CGRect?) -> CGSize {
        guard let visible else { return fallback }
        return CGSize(width: visible.width, height: visible.height)
    }
}
