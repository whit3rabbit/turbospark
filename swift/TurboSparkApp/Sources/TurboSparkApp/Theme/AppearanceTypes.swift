import AppKit
import SwiftUI

/// Preference for how git / artifact diffs represent additions and deletions.
public enum DiffMarkerPreference: String, CaseIterable, Identifiable, Codable, Sendable {
    case color = "color"
    case plusMinus = "plusMinus"

    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .color: return "Color"
        case .plusMinus: return "+/-"
        }
    }
}

/// Accessibility preference for motion and animation reduction.
public enum ReduceMotionPreference: String, CaseIterable, Identifiable, Codable, Sendable {
    case system = "system"
    case on = "on"
    case off = "off"

    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .system: return "System"
        case .on: return "On"
        case .off: return "Off"
        }
    }
}

/// Selectable application dock icon styling presets.
public enum AppDockIcon: String, CaseIterable, Identifiable, Codable, Sendable {
    case emeraldSpark = "emeraldSpark"
    case codexDark = "codexDark"
    case terminalPro = "terminalPro"
    case minimalist = "minimalist"

    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .emeraldSpark: return "TurboSpark"
        case .codexDark: return "Codex"
        case .terminalPro: return "Terminal"
        case .minimalist: return "Minimal"
        }
    }
}

/// Visual theme configuration for a single appearance mode (light or dark).
public struct ThemeModeConfig: Codable, Equatable, Sendable {
    public var preset: String
    public var accentName: String
    public var accentHex: String
    public var backgroundHex: String
    public var foregroundHex: String
    public var uiFontFamily: String
    public var uiFontWeight: String
    public var codeFontFamily: String
    public var codeFontWeight: String
    public var translucentSidebar: Bool
    public var contrast: Double

    public init(
        preset: String = "Codex",
        accentName: String = "Black",
        accentHex: String = "#000000",
        backgroundHex: String = "#FFFFFF",
        foregroundHex: String = "#1A1C1F",
        uiFontFamily: String = "System default",
        uiFontWeight: String = "Regular",
        codeFontFamily: String = "System default",
        codeFontWeight: String = "Regular",
        translucentSidebar: Bool = true,
        contrast: Double = 45.0
    ) {
        self.preset = preset
        self.accentName = accentName
        self.accentHex = accentHex
        self.backgroundHex = backgroundHex
        self.foregroundHex = foregroundHex
        self.uiFontFamily = uiFontFamily
        self.uiFontWeight = uiFontWeight
        self.codeFontFamily = codeFontFamily
        self.codeFontWeight = codeFontWeight
        self.translucentSidebar = translucentSidebar
        self.contrast = contrast
    }

    public static let defaultLight = ThemeModeConfig(
        preset: "Codex",
        accentName: "Black",
        accentHex: "#111827",
        backgroundHex: "#FFFFFF",
        foregroundHex: "#1A1C1F",
        uiFontFamily: "System default",
        uiFontWeight: "Regular",
        codeFontFamily: "System default",
        codeFontWeight: "Regular",
        translucentSidebar: true,
        contrast: 45.0
    )

    public static let defaultDark = ThemeModeConfig(
        preset: "Codex",
        accentName: "White",
        accentHex: "#F3F4F6",
        backgroundHex: "#181818",
        foregroundHex: "#FFFFFF",
        uiFontFamily: "System default",
        uiFontWeight: "Regular",
        codeFontFamily: "System default",
        codeFontWeight: "Regular",
        translucentSidebar: true,
        contrast: 60.0
    )
}

/// Curated theme preset bundling matching light and dark mode configurations.
public struct ThemePreset: Identifiable, Sendable {
    public let id: String
    public let name: String
    public let light: ThemeModeConfig
    public let dark: ThemeModeConfig

    public static let presets: [ThemePreset] = [
        ThemePreset(
            id: "codex",
            name: "Codex",
            light: .defaultLight,
            dark: .defaultDark
        ),
        ThemePreset(
            id: "turbospark",
            name: "TurboSpark Emerald",
            light: ThemeModeConfig(
                preset: "TurboSpark Emerald",
                accentName: "Emerald",
                accentHex: "#237D32",
                backgroundHex: "#FFFFFF",
                foregroundHex: "#1A1C1F",
                uiFontFamily: "System default",
                uiFontWeight: "Regular",
                codeFontFamily: "System default",
                codeFontWeight: "Regular",
                translucentSidebar: true,
                contrast: 50.0
            ),
            dark: ThemeModeConfig(
                preset: "TurboSpark Emerald",
                accentName: "Emerald",
                accentHex: "#6ABA71",
                backgroundHex: "#151816",
                foregroundHex: "#F0FDF4",
                uiFontFamily: "System default",
                uiFontWeight: "Regular",
                codeFontFamily: "System default",
                codeFontWeight: "Regular",
                translucentSidebar: true,
                contrast: 65.0
            )
        ),
        ThemePreset(
            id: "midnight",
            name: "Midnight Blue",
            light: ThemeModeConfig(
                preset: "Midnight Blue",
                accentName: "Blue",
                accentHex: "#2563EB",
                backgroundHex: "#F8FAFC",
                foregroundHex: "#0F172A",
                uiFontFamily: "System default",
                uiFontWeight: "Regular",
                codeFontFamily: "System default",
                codeFontWeight: "Regular",
                translucentSidebar: true,
                contrast: 50.0
            ),
            dark: ThemeModeConfig(
                preset: "Midnight Blue",
                accentName: "Blue",
                accentHex: "#38BDF8",
                backgroundHex: "#0B0F19",
                foregroundHex: "#F1F5F9",
                uiFontFamily: "System default",
                uiFontWeight: "Regular",
                codeFontFamily: "System default",
                codeFontWeight: "Regular",
                translucentSidebar: true,
                contrast: 70.0
            )
        ),
        ThemePreset(
            id: "charcoal",
            name: "Charcoal Mono",
            light: ThemeModeConfig(
                preset: "Charcoal Mono",
                accentName: "Black",
                accentHex: "#18181B",
                backgroundHex: "#FAFAFA",
                foregroundHex: "#18181B",
                uiFontFamily: "System default",
                uiFontWeight: "Medium",
                codeFontFamily: "SF Mono",
                codeFontWeight: "Regular",
                translucentSidebar: false,
                contrast: 55.0
            ),
            dark: ThemeModeConfig(
                preset: "Charcoal Mono",
                accentName: "White",
                accentHex: "#E4E4E7",
                backgroundHex: "#09090B",
                foregroundHex: "#FAFAFA",
                uiFontFamily: "System default",
                uiFontWeight: "Medium",
                codeFontFamily: "SF Mono",
                codeFontWeight: "Regular",
                translucentSidebar: false,
                contrast: 75.0
            )
        ),
        ThemePreset(
            id: "highContrast",
            name: "High Contrast",
            light: ThemeModeConfig(
                preset: "High Contrast",
                accentName: "Black",
                accentHex: "#000000",
                backgroundHex: "#FFFFFF",
                foregroundHex: "#000000",
                uiFontFamily: "System default",
                uiFontWeight: "Semibold",
                codeFontFamily: "System default",
                codeFontWeight: "Medium",
                translucentSidebar: false,
                contrast: 95.0
            ),
            dark: ThemeModeConfig(
                preset: "High Contrast",
                accentName: "White",
                accentHex: "#FFFFFF",
                backgroundHex: "#000000",
                foregroundHex: "#FFFFFF",
                uiFontFamily: "System default",
                uiFontWeight: "Semibold",
                codeFontFamily: "System default",
                codeFontWeight: "Medium",
                translucentSidebar: false,
                contrast: 95.0
            )
        )
    ]
}

// MARK: - Color Hex Helpers

public extension Color {
    /// Initializes a SwiftUI Color from a hexadecimal color string (e.g. `#RRGGBB` or `#RRGGBBAA`).
    init?(hex: String) {
        var hexSanitized = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        hexSanitized = hexSanitized.replacingOccurrences(of: "#", with: "")

        var rgb: UInt64 = 0
        guard Scanner(string: hexSanitized).scanHexInt64(&rgb) else { return nil }

        let length = hexSanitized.count
        let r, g, b, a: Double
        if length == 6 {
            r = Double((rgb & 0xFF0000) >> 16) / 255.0
            g = Double((rgb & 0x00FF00) >> 8) / 255.0
            b = Double(rgb & 0x0000FF) / 255.0
            a = 1.0
        } else if length == 8 {
            r = Double((rgb & 0xFF000000) >> 24) / 255.0
            g = Double((rgb & 0x00FF0000) >> 16) / 255.0
            b = Double((rgb & 0x0000FF00) >> 8) / 255.0
            a = Double(rgb & 0x000000FF) / 255.0
        } else {
            return nil
        }

        self.init(.sRGB, red: r, green: g, blue: b, opacity: a)
    }

    /// Formats this color as a standard hexadecimal string.
    func toHex(includeAlpha: Bool = false) -> String {
        guard let components = NSColor(self).usingColorSpace(.sRGB) else {
            return "#000000"
        }
        let r = Float(components.redComponent)
        let g = Float(components.greenComponent)
        let b = Float(components.blueComponent)
        let a = Float(components.alphaComponent)

        if includeAlpha {
            return String(format: "#%02lX%02lX%02lX%02lX",
                          lroundf(r * 255),
                          lroundf(g * 255),
                          lroundf(b * 255),
                          lroundf(a * 255))
        } else {
            return String(format: "#%02lX%02lX%02lX",
                          lroundf(r * 255),
                          lroundf(g * 255),
                          lroundf(b * 255))
        }
    }
}

public extension Font.Weight {
    /// Maps human-readable font weight string to SwiftUI `Font.Weight`.
    static func fromName(_ name: String) -> Font.Weight {
        switch name.lowercased() {
        case "ultralight": return .ultraLight
        case "thin": return .thin
        case "light": return .light
        case "regular": return .regular
        case "medium": return .medium
        case "semibold": return .semibold
        case "bold": return .bold
        case "heavy": return .heavy
        case "black": return .black
        default: return .regular
        }
    }
}
