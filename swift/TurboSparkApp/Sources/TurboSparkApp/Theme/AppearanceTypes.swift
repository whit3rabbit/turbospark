import AppKit
import SwiftUI

/// Display mode for live benchmarks in the bottom status bar (numbers or live graphs).
public enum StatusBarViewMode: String, CaseIterable, Identifiable, Codable, Sendable {
    case text = "text"
    case graphs = "graphs"

    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .text: return "Numbers"
        case .graphs: return "Live Graphs"
        }
    }

    public var systemImage: String {
        switch self {
        case .text: return "number"
        case .graphs: return "chart.xyaxis.line"
        }
    }
}

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

    /// Decodes a config pasted as JSON, or nil when the text is not one.
    ///
    /// Pure and separate from the view so the failure case can be tested:
    /// the view's own fallback used to be "apply a preset", which no test
    /// could see and no user could tell from a successful import.
    public static func fromClipboardJSON(_ string: String) -> ThemeModeConfig? {
        guard let data = string.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(ThemeModeConfig.self, from: data)
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

/// One entry in the Accent picker: a display name and the hex it selects.
public struct AccentOption: Identifiable, Equatable, Sendable {
    public let name: String
    public let hex: String
    public var id: String { hex }

    public init(name: String, hex: String) {
        self.name = name
        self.hex = hex
    }

    /// The six curated accents for a mode.
    public static func curated(isDark: Bool) -> [AccentOption] {
        [
            AccentOption(name: "Emerald", hex: isDark ? "#6ABA71" : "#237D32"),
            AccentOption(name: "Blue", hex: isDark ? "#38BDF8" : "#2563EB"),
            AccentOption(name: "Purple", hex: isDark ? "#C084FC" : "#7C3AED"),
            AccentOption(name: "Orange", hex: isDark ? "#FB923C" : "#EA580C"),
            AccentOption(name: "Black", hex: "#000000"),
            AccentOption(name: "White", hex: "#FFFFFF")
        ]
    }

    /// The curated accents, plus the selected one when it is not among them.
    ///
    /// **A `Picker` whose selection matches no tag renders BLANK**, and the
    /// curated six did not cover the shipped presets: Codex is `#111827` /
    /// `#F3F4F6` and Charcoal Mono is `#18181B` / `#E4E4E7`, none of which is
    /// an option, so 4 of the 10 preset-and-mode combinations drew an empty
    /// Accent control -- including Codex, which is the DEFAULT. It reads as a
    /// broken control rather than as an unlisted color.
    ///
    /// Widening the curated list would fix those four and break again at the
    /// next preset or at any custom color from the wheel, so the selected
    /// value is always carried instead.
    public static func options(isDark: Bool, selectedHex: String, selectedName: String) -> [AccentOption] {
        let curated = curated(isDark: isDark)
        let normalized = selectedHex.uppercased()
        guard !curated.contains(where: { $0.hex.uppercased() == normalized }) else {
            return curated
        }
        let label = selectedName.isEmpty || selectedName == "Custom" ? "Custom" : selectedName
        return curated + [AccentOption(name: label, hex: selectedHex)]
    }

    /// The name to record for a hex, or `Custom` when it is not a curated one.
    public static func name(forHex hex: String, isDark: Bool) -> String {
        let normalized = hex.uppercased()
        return curated(isDark: isDark)
            .first(where: { $0.hex.uppercased() == normalized })?.name ?? "Custom"
    }
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

    /// This color's components in sRGB space, or nil if it cannot be
    /// represented there. Shared by `toHex` and `isLight` so the two can
    /// only ever disagree on the math applied to the components, never on
    /// how the color is read.
    private var sRGBComponents: (r: CGFloat, g: CGFloat, b: CGFloat, a: CGFloat)? {
        guard let components = NSColor(self).usingColorSpace(.sRGB) else {
            return nil
        }
        return (components.redComponent, components.greenComponent, components.blueComponent, components.alphaComponent)
    }

    /// Formats this color as a standard hexadecimal string.
    func toHex(includeAlpha: Bool = false) -> String {
        guard let c = sRGBComponents else {
            return "#000000"
        }
        if includeAlpha {
            return String(format: "#%02lX%02lX%02lX%02lX",
                          lroundf(Float(c.r) * 255),
                          lroundf(Float(c.g) * 255),
                          lroundf(Float(c.b) * 255),
                          lroundf(Float(c.a) * 255))
        } else {
            return String(format: "#%02lX%02lX%02lX",
                          lroundf(Float(c.r) * 255),
                          lroundf(Float(c.g) * 255),
                          lroundf(Float(c.b) * 255))
        }
    }

    /// Whether this color has high perceived luminance in sRGB space.
    var isLight: Bool {
        guard let c = sRGBComponents else {
            return false
        }
        let luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
        return luminance > 0.55
    }

    /// High-contrast foreground color suitable for rendering text and icons on top of this color.
    var contrastForeground: Color {
        isLight ? Color(red: 0.08, green: 0.08, blue: 0.09) : Color.white
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
