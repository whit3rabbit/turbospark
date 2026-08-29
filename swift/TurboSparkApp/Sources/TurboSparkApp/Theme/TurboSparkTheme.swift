import AppKit
import SwiftUI

@MainActor
public enum TurboSparkTheme {
    /// Adaptive brand accent color resolved dynamically from active appearance settings.
    public static var accentColor: Color {
        let isDark = NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        return AppearanceManager.shared.activeAccentColor(isDark: isDark)
    }

    public static var accentNSColor: NSColor {
        NSColor(accentColor)
    }

    public static var sidebarBackgroundColor: Color {
        let isDark = NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        if config.translucentSidebar {
            return Color(nsColor: .windowBackgroundColor).opacity(0.82)
        } else {
            return AppearanceManager.shared.activeBackgroundColor(isDark: isDark)
        }
    }

    /// Background of the leftmost icon rail: slightly differentiated from the sidebar.
    public static var railBackgroundColor: Color {
        let isDark = NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        if config.translucentSidebar {
            return Color(nsColor: .windowBackgroundColor).opacity(0.7)
        } else {
            if #available(macOS 15.0, *) {
                return AppearanceManager.shared.activeBackgroundColor(isDark: isDark).mix(with: .black, by: 0.06)
            } else {
                return Color(nsColor: .underPageBackgroundColor)
            }
        }
    }

    /// Background of the flat top bar and bottom status strip.
    public static var barBackgroundColor: Color {
        let isDark = NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        return AppearanceManager.shared.activeBackgroundColor(isDark: isDark)
    }

    /// Fill for cards, chips and the composer: the raised surface in the app.
    public static var surfaceColor: Color {
        Color(nsColor: .controlBackgroundColor)
    }

    /// Hairline used between every chrome band.
    public static var hairlineColor: Color {
        let isDark = NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        let opacity = 0.25 + (config.contrast / 100.0) * 0.55
        return Color(nsColor: .separatorColor).opacity(opacity)
    }

    /// Returns high-contrast compliant text styling for metadata and captions (WCAG 2.1 AA >= 4.5:1).
    public static func metadataForeground(contrast: ColorSchemeContrast = .standard) -> Color {
        let isDark = NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        if contrast == .increased || config.contrast > 70 {
            return Color.primary.opacity(0.9)
        } else {
            return Color.secondary
        }
    }

    /// Returns high-contrast compliant secondary border stroke opacity.
    public static func borderStrokeOpacity(contrast: ColorSchemeContrast = .standard) -> Double {
        let isDark = NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark")
        let config = AppearanceManager.shared.activeConfig(isDark: isDark)
        let base = (config.contrast / 100.0) * 0.85
        return contrast == .increased ? max(0.85, base) : max(0.3, base)
    }
}
