import AppKit
import SwiftUI

public enum TurboSparkTheme {
    /// Adaptive brand accent color: vibrant emerald green in dark mode (7.2:1 contrast)
    /// and deep forest green in light mode (5.1:1 contrast) for WCAG 2.1 AA compliance,
    /// with enhanced high-contrast variants for accessibilityHighContrast appearances (9.5:1+).
    public static let accentNSColor = NSColor(name: nil) { appearance in
        let match = appearance.bestMatch(from: [
            .accessibilityHighContrastDarkAqua,
            .accessibilityHighContrastAqua,
            .darkAqua,
            .aqua
        ])
        let isDark = match == .darkAqua || match == .accessibilityHighContrastDarkAqua
        let isHighContrast = match == .accessibilityHighContrastDarkAqua || match == .accessibilityHighContrastAqua

        if isHighContrast {
            if isDark {
                return NSColor(
                    srgbRed: 130.0 / 255.0,
                    green: 225.0 / 255.0,
                    blue: 140.0 / 255.0,
                    alpha: 1.0)
            } else {
                return NSColor(
                    srgbRed: 20.0 / 255.0,
                    green: 95.0 / 255.0,
                    blue: 35.0 / 255.0,
                    alpha: 1.0)
            }
        } else if isDark {
            return NSColor(
                srgbRed: 106.0 / 255.0,
                green: 186.0 / 255.0,
                blue: 113.0 / 255.0,
                alpha: 1.0)
        } else {
            return NSColor(
                srgbRed: 35.0 / 255.0,
                green: 125.0 / 255.0,
                blue: 50.0 / 255.0,
                alpha: 1.0)
        }
    }

    public static let accentColor = Color(nsColor: accentNSColor)

    public static var sidebarBackgroundColor: Color {
        if #available(macOS 15.0, *) {
            return Color(nsColor: .windowBackgroundColor).mix(with: accentColor, by: 0.025)
        } else {
            return Color(nsColor: .windowBackgroundColor)
        }
    }

    /// Returns high-contrast compliant text styling for metadata and captions (WCAG 2.1 AA >= 4.5:1).
    public static func metadataForeground(contrast: ColorSchemeContrast = .standard) -> Color {
        contrast == .increased ? Color.primary.opacity(0.85) : Color.secondary
    }

    /// Returns high-contrast compliant secondary border stroke opacity.
    public static func borderStrokeOpacity(contrast: ColorSchemeContrast = .standard) -> Double {
        contrast == .increased ? 0.85 : 0.45
    }
}

