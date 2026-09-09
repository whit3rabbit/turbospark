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

    /// The theme's own background, unmodified. Every "elevated" accessor
    /// below is a step OFF this value rather than off a system material, so
    /// a card, a bar or a hairline always reads as belonging to the same
    /// palette instead of falling back to whatever `.controlBackgroundColor`
    /// happens to render as -- which does not move when the theme's
    /// background hex changes, and is why a themed dark-navy install still
    /// drew system-neutral-gray boxes with no visible separation between
    /// them (2026-09-08).
    public static func pageBackgroundColor(isDark: Bool) -> Color {
        AppearanceManager.shared.activeBackgroundColor(isDark: isDark)
    }

    /// Blends `base` toward white (dark mode: lighter, "raised") or black
    /// (light mode: darker, "raised") by `fraction`. `NSColor.blended` is
    /// AppKit's own and predates `Color.mix(with:by:)`, which the macOS 14
    /// SDK this app still targets does not have at all -- `#available` gates
    /// execution, not symbol resolution, so a guarded call to it fails to
    /// compile on that SDK rather than merely failing at runtime.
    nonisolated private static func elevate(_ base: Color, isDark: Bool, by fraction: CGFloat) -> Color {
        guard let baseSRGB = NSColor(base).usingColorSpace(.sRGB) else { return base }
        let mixColor = isDark ? NSColor.white : NSColor.black
        guard let mixSRGB = mixColor.usingColorSpace(.sRGB),
            let blended = baseSRGB.blended(withFraction: fraction, of: mixSRGB)
        else {
            return base
        }
        return Color(nsColor: blended)
    }

    public static func sidebarBackgroundColor(isDark: Bool) -> Color {
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        let elevated = elevate(pageBackgroundColor(isDark: isDark), isDark: isDark, by: isDark ? 0.05 : 0.035)
        return config.translucentSidebar ? elevated.opacity(0.94) : elevated
    }

    /// Background of the leftmost icon rail: slightly differentiated from the sidebar.
    public static func railBackgroundColor(isDark: Bool) -> Color {
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        let elevated = elevate(pageBackgroundColor(isDark: isDark), isDark: isDark, by: isDark ? 0.03 : 0.02)
        return config.translucentSidebar ? elevated.opacity(0.9) : elevated
    }

    /// Background of the flat top bar and bottom status strip: one step
    /// lighter (dark mode) than the page, so a band reads as a distinct
    /// strip rather than as an unbroken continuation of the canvas below it.
    public static func barBackgroundColor(isDark: Bool) -> Color {
        elevate(pageBackgroundColor(isDark: isDark), isDark: isDark, by: isDark ? 0.055 : 0.03)
    }

    /// Hairline used between every chrome band: a further step lighter than
    /// a bar, so a border reads as a lighter line of the SAME palette rather
    /// than as a neutral system-gray seam laid over a colored theme.
    public static func hairlineColor(isDark: Bool) -> Color {
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        let elevated = elevate(pageBackgroundColor(isDark: isDark), isDark: isDark, by: isDark ? 0.30 : 0.18)
        let opacity = 0.4 + (config.contrast / 100.0) * 0.5
        return elevated.opacity(opacity)
    }

    // MARK: - Ambient accessors

    /// Adaptive brand accent color resolved dynamically from active appearance settings.
    public static var accentColor: Color { accentColor(isDark: resolvedIsDark) }

    public static var accentNSColor: NSColor { NSColor(accentColor) }

    public static var pageBackgroundColor: Color { pageBackgroundColor(isDark: resolvedIsDark) }

    public static var sidebarBackgroundColor: Color { sidebarBackgroundColor(isDark: resolvedIsDark) }

    public static var railBackgroundColor: Color { railBackgroundColor(isDark: resolvedIsDark) }

    public static var barBackgroundColor: Color { barBackgroundColor(isDark: resolvedIsDark) }

    /// Fill for cards and chips: the raised surface in the app. The most
    /// elevated step short of a hairline, so a card is visibly lighter than
    /// both the page behind it and the bar above it.
    public static var surfaceColor: Color {
        let isDark = resolvedIsDark
        return elevate(pageBackgroundColor(isDark: isDark), isDark: isDark, by: isDark ? 0.11 : 0.06)
    }

    /// Background color specifically for the prompt composer textbox,
    /// derived from `background` rather than reached from `AppearanceManager`
    /// directly: this stays `nonisolated` so `ResolvedAppTheme.composerBackground`
    /// (a plain, non-actor-isolated computed property) can call it without an
    /// actor hop, which means it cannot touch `AppearanceManager.shared` itself.
    nonisolated public static func composerBackgroundColor(isDark: Bool, background: Color) -> Color {
        elevate(background, isDark: isDark, by: isDark ? 0.09 : 0.05)
    }

    public static var composerBackgroundColor: Color {
        composerBackgroundColor(isDark: resolvedIsDark, background: pageBackgroundColor(isDark: resolvedIsDark))
    }

    public static var hairlineColor: Color { hairlineColor(isDark: resolvedIsDark) }
}
