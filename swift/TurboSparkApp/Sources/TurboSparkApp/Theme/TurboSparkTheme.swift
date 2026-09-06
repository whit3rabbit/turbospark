import AppKit
import SwiftUI

/// Chrome colors, resolved for a mode.
///
/// **Every accessor here used to branch on `NSApp.effectiveAppearance`**, which
/// is an APPLICATION-level property and is not moved by SwiftUI's
/// `.preferredColorScheme`. Setting the app to Light on a dark system therefore
/// drew light chrome out of `darkConfig`: dark accent, dark background, dark
/// contrast. `AppearanceSettingsPaneView` had always derived the same question
/// correctly from `manager.appearance` plus `\.colorScheme`, so the two
/// disagreed and only the settings pane was right.
///
/// The `isDark`-taking methods are the real ones. The static properties are
/// kept as wrappers so the ~47 files already calling them did not all have to
/// move at once, and they resolve through `AppearanceManager.appearance`
/// rather than through `NSApp`, which fixes the forced-Light and forced-Dark
/// cases. They cannot see the SYSTEM scheme, so the `.system` case still falls
/// back to reading the effective appearance: prefer `\.appTheme` in any view
/// that can, and reach for these only from a non-view context.
@MainActor
public enum TurboSparkTheme {
    /// The mode to draw in, for a caller with no `\.colorScheme` in scope.
    private static var resolvedIsDark: Bool {
        switch AppearanceManager.shared.appearance {
        case .light: return false
        case .dark: return true
        case .system: return NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        }
    }

    // MARK: - Mode-explicit accessors

    public static func accentColor(isDark: Bool) -> Color {
        AppearanceManager.shared.activeAccentColor(isDark: isDark)
    }

    public static func sidebarBackgroundColor(isDark: Bool) -> Color {
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        if config.translucentSidebar {
            return Color(nsColor: .windowBackgroundColor).opacity(0.82)
        }
        return AppearanceManager.shared.activeBackgroundColor(isDark: isDark)
    }

    /// Background of the leftmost icon rail: slightly differentiated from the sidebar.
    public static func railBackgroundColor(isDark: Bool) -> Color {
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        if config.translucentSidebar {
            return Color(nsColor: .windowBackgroundColor).opacity(0.7)
        }
        // **`#available` DOES NOT HELP WITH A SYMBOL THE SDK DOES NOT HAVE.**
        // This was `Color.mix(with:by:)` behind a `macOS 15.0` guard, which
        // is correct for RUNTIME availability and does nothing for
        // compilation: the macOS 14 SDK has no such member, so the guarded
        // call is `value of type 'Color' has no member 'mix'` on any
        // toolchain older than the one it was written on. CI builds the app
        // bundle on `macos-14` and every developer here is on a far newer
        // Xcode, so it failed for nobody who could see it.
        //
        // `NSColor.blended(withFraction:of:)` is AppKit's own and predates
        // all of this. It needs both colours in one space or it returns nil,
        // and `NSColor.black` is generic gray, hence the two conversions and
        // the fallback that was already the `else` branch.
        let base = NSColor(AppearanceManager.shared.activeBackgroundColor(isDark: isDark))
        if let sRGB = base.usingColorSpace(.sRGB),
            let black = NSColor.black.usingColorSpace(.sRGB),
            let darkened = sRGB.blended(withFraction: 0.06, of: black)
        {
            return Color(nsColor: darkened)
        }
        return Color(nsColor: .underPageBackgroundColor)
    }

    /// Background of the flat top bar and bottom status strip.
    public static func barBackgroundColor(isDark: Bool) -> Color {
        AppearanceManager.shared.activeBackgroundColor(isDark: isDark)
    }

    /// Hairline used between every chrome band.
    public static func hairlineColor(isDark: Bool) -> Color {
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        let opacity = 0.25 + (config.contrast / 100.0) * 0.55
        return Color(nsColor: .separatorColor).opacity(opacity)
    }

    // MARK: - Ambient accessors

    /// Adaptive brand accent color resolved dynamically from active appearance settings.
    public static var accentColor: Color { accentColor(isDark: resolvedIsDark) }

    public static var accentNSColor: NSColor { NSColor(accentColor) }

    public static var sidebarBackgroundColor: Color { sidebarBackgroundColor(isDark: resolvedIsDark) }

    public static var railBackgroundColor: Color { railBackgroundColor(isDark: resolvedIsDark) }

    public static var barBackgroundColor: Color { barBackgroundColor(isDark: resolvedIsDark) }

    /// Fill for cards and chips: the raised surface in the app.
    public static var surfaceColor: Color {
        Color(nsColor: .controlBackgroundColor)
    }

    /// Background color specifically for the prompt composer textbox so it
    /// cleanly contrasts against the window background in both light and dark modes.
    nonisolated public static func composerBackgroundColor(isDark: Bool) -> Color {
        if isDark {
            return Color(nsColor: NSColor(white: 0.20, alpha: 1.0))
        } else {
            return Color(nsColor: NSColor(white: 0.94, alpha: 1.0))
        }
    }

    public static var composerBackgroundColor: Color {
        composerBackgroundColor(isDark: resolvedIsDark)
    }

    public static var hairlineColor: Color { hairlineColor(isDark: resolvedIsDark) }

    /// Returns high-contrast compliant text styling for metadata and captions (WCAG 2.1 AA >= 4.5:1).
    public static func metadataForeground(contrast: ColorSchemeContrast = .standard) -> Color {
        let config = AppearanceManager.shared.activeConfig(isDark: resolvedIsDark)
        if contrast == .increased || config.contrast > 70 {
            return Color.primary.opacity(0.9)
        }
        return Color.secondary
    }

    /// Returns high-contrast compliant secondary border stroke opacity.
    public static func borderStrokeOpacity(contrast: ColorSchemeContrast = .standard) -> Double {
        let config = AppearanceManager.shared.activeConfig(isDark: resolvedIsDark)
        let base = (config.contrast / 100.0) * 0.85
        return contrast == .increased ? max(0.85, base) : max(0.3, base)
    }
}
