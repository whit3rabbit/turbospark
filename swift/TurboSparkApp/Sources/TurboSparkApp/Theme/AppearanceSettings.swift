import AppKit
import Combine
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

/// Global observable manager for application appearance, fonts, colors, and dock icon.
@MainActor
public final class AppearanceManager: ObservableObject {
    public static let shared = AppearanceManager()

    private let defaults = UserDefaults.standard

    // Keys
    private static let keyAppearance = "TurboSpark.appearance"
    private static let keyLightConfig = "TurboSpark.theme.lightConfig"
    private static let keyDarkConfig = "TurboSpark.theme.darkConfig"
    private static let keyUsePointerCursors = "TurboSpark.prefs.usePointerCursors"
    private static let keyDockIcon = "TurboSpark.prefs.dockIcon"
    private static let keyReduceMotion = "TurboSpark.prefs.reduceMotion"
    private static let keyUiFontSize = "TurboSpark.prefs.uiFontSize"
    private static let keyCodeFontSize = "TurboSpark.prefs.codeFontSize"
    private static let keyDiffMarkers = "TurboSpark.prefs.diffMarkers"
    private static let keyFontSmoothing = "TurboSpark.prefs.fontSmoothing"

    /// Resolved app appearance mode (system, light, dark).
    @Published public var appearance: AppAppearance {
        didSet { defaults.set(appearance.rawValue, forKey: Self.keyAppearance) }
    }

    /// Theme styling configuration used when light mode is active.
    @Published public var lightConfig: ThemeModeConfig {
        didSet { saveLightConfig() }
    }

    /// Theme styling configuration used when dark mode is active.
    @Published public var darkConfig: ThemeModeConfig {
        didSet { saveDarkConfig() }
    }

    /// Whether interactive buttons and clickable controls display a pointer cursor on hover.
    @Published public var usePointerCursors: Bool {
        didSet { defaults.set(usePointerCursors, forKey: Self.keyUsePointerCursors) }
    }

    /// Selected application dock icon variant.
    @Published public var dockIcon: AppDockIcon {
        didSet {
            defaults.set(dockIcon.rawValue, forKey: Self.keyDockIcon)
            updateDockIcon()
        }
    }

    /// Motion reduction preference for transitions and animations.
    @Published public var reduceMotion: ReduceMotionPreference {
        didSet { defaults.set(reduceMotion.rawValue, forKey: Self.keyReduceMotion) }
    }

    /// Base font point size for primary user interface text.
    @Published public var uiFontSize: Double {
        didSet { defaults.set(uiFontSize, forKey: Self.keyUiFontSize) }
    }

    /// Base font point size for code blocks, terminal, and monospace views.
    @Published public var codeFontSize: Double {
        didSet { defaults.set(codeFontSize, forKey: Self.keyCodeFontSize) }
    }

    /// Presentation style for inline and side-by-side diff markers.
    @Published public var diffMarkers: DiffMarkerPreference {
        didSet { defaults.set(diffMarkers.rawValue, forKey: Self.keyDiffMarkers) }
    }

    /// Whether subpixel font smoothing is enabled on macOS.
    @Published public var fontSmoothing: Bool {
        didSet {
            defaults.set(fontSmoothing, forKey: Self.keyFontSmoothing)
            applyFontSmoothing()
        }
    }

    public init() {
        let appStr = defaults.string(forKey: Self.keyAppearance) ?? AppAppearance.system.rawValue
        self.appearance = AppAppearance.resolve(appStr)

        if let data = defaults.data(forKey: Self.keyLightConfig),
           let config = try? JSONDecoder().decode(ThemeModeConfig.self, from: data) {
            self.lightConfig = config
        } else {
            self.lightConfig = .defaultLight
        }

        if let data = defaults.data(forKey: Self.keyDarkConfig),
           let config = try? JSONDecoder().decode(ThemeModeConfig.self, from: data) {
            self.darkConfig = config
        } else {
            self.darkConfig = .defaultDark
        }

        self.usePointerCursors = defaults.object(forKey: Self.keyUsePointerCursors) as? Bool ?? false

        let dockStr = defaults.string(forKey: Self.keyDockIcon) ?? AppDockIcon.emeraldSpark.rawValue
        self.dockIcon = AppDockIcon(rawValue: dockStr) ?? .emeraldSpark

        let motionStr = defaults.string(forKey: Self.keyReduceMotion) ?? ReduceMotionPreference.system.rawValue
        self.reduceMotion = ReduceMotionPreference(rawValue: motionStr) ?? .system

        self.uiFontSize = defaults.object(forKey: Self.keyUiFontSize) as? Double ?? 14.0
        self.codeFontSize = defaults.object(forKey: Self.keyCodeFontSize) as? Double ?? 12.0

        let diffStr = defaults.string(forKey: Self.keyDiffMarkers) ?? DiffMarkerPreference.color.rawValue
        self.diffMarkers = DiffMarkerPreference(rawValue: diffStr) ?? .color

        self.fontSmoothing = defaults.object(forKey: Self.keyFontSmoothing) as? Bool ?? true

        updateDockIcon()
        applyFontSmoothing()
    }

    /// Applies a preset configuration across light, dark, or both color modes.
    public func applyPreset(_ preset: ThemePreset, forMode isDark: Bool? = nil) {
        if let isDark = isDark {
            if isDark {
                self.darkConfig = preset.dark
            } else {
                self.lightConfig = preset.light
            }
        } else {
            self.lightConfig = preset.light
            self.darkConfig = preset.dark
        }
    }

    private func saveLightConfig() {
        if let data = try? JSONEncoder().encode(lightConfig) {
            defaults.set(data, forKey: Self.keyLightConfig)
        }
    }

    private func saveDarkConfig() {
        if let data = try? JSONEncoder().encode(darkConfig) {
            defaults.set(data, forKey: Self.keyDarkConfig)
        }
    }

    /// Retrieves active theme config for the specified brightness mode.
    public func activeConfig(isDark: Bool) -> ThemeModeConfig {
        isDark ? darkConfig : lightConfig
    }

    /// Resolves active accent color for the specified brightness mode.
    public func activeAccentColor(isDark: Bool) -> Color {
        let hex = activeConfig(isDark: isDark).accentHex
        return Color(hex: hex) ?? (isDark ? Color.white : Color.black)
    }

    /// Resolves active window background color for the specified brightness mode.
    public func activeBackgroundColor(isDark: Bool) -> Color {
        let hex = activeConfig(isDark: isDark).backgroundHex
        return Color(hex: hex) ?? (isDark ? Color(nsColor: .windowBackgroundColor) : Color.white)
    }

    /// Resolves active foreground text color for the specified brightness mode.
    public func activeForegroundColor(isDark: Bool) -> Color {
        let hex = activeConfig(isDark: isDark).foregroundHex
        return Color(hex: hex) ?? (isDark ? Color.white : Color.primary)
    }

    /// Constructs a code font with current user preferences and optional size/weight override.
    public func codeFont(size: CGFloat? = nil, weight: Font.Weight? = nil) -> Font {
        let targetSize = size ?? CGFloat(codeFontSize)
        let config = activeConfig(isDark: NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark"))
        let resolvedWeight = weight ?? Font.Weight.fromName(config.codeFontWeight)
        
        switch config.codeFontFamily {
        case "SF Mono":
            return .system(size: targetSize, weight: resolvedWeight, design: .monospaced)
        case "Menlo":
            return .custom("Menlo", size: targetSize).weight(resolvedWeight)
        case "Courier", "Courier New":
            return .custom("Courier", size: targetSize).weight(resolvedWeight)
        case "JetBrains Mono":
            return .custom("JetBrainsMono-Regular", size: targetSize).weight(resolvedWeight)
        case "Fira Code":
            return .custom("FiraCode-Regular", size: targetSize).weight(resolvedWeight)
        default:
            return .system(size: targetSize, weight: resolvedWeight, design: .monospaced)
        }
    }

    /// Constructs a UI font with current user preferences and optional size/weight override.
    public func uiFont(size: CGFloat? = nil, weight: Font.Weight? = nil) -> Font {
        let targetSize = size ?? CGFloat(uiFontSize)
        let config = activeConfig(isDark: NSApp.effectiveAppearance.name.rawValue.lowercased().contains("dark"))
        let resolvedWeight = weight ?? Font.Weight.fromName(config.uiFontWeight)

        switch config.uiFontFamily {
        case "SF Pro", "System default":
            return .system(size: targetSize, weight: resolvedWeight, design: .default)
        case "Inter":
            return .custom("Inter", size: targetSize).weight(resolvedWeight)
        case "Helvetica Neue", "Helvetica":
            return .custom("Helvetica Neue", size: targetSize).weight(resolvedWeight)
        case "Avenir":
            return .custom("Avenir", size: targetSize).weight(resolvedWeight)
        default:
            return .system(size: targetSize, weight: resolvedWeight)
        }
    }

    /// Determines whether animation/motion should be suppressed based on user or system settings.
    public func shouldReduceMotion(systemReduceMotion: Bool) -> Bool {
        switch reduceMotion {
        case .system: return systemReduceMotion
        case .on: return true
        case .off: return false
        }
    }

    private func applyFontSmoothing() {
        defaults.set(fontSmoothing ? 2 : 0, forKey: "AppleFontSmoothing")
    }

    /// Renders and updates the macOS application dock icon based on current preset choice.
    public func updateDockIcon() {
        let iconSize = NSSize(width: 128, height: 128)
        let image = NSImage(size: iconSize)
        image.lockFocus()

        let bounds = NSRect(origin: .zero, size: iconSize)
        let bgPath = NSBezierPath(roundedRect: bounds.insetBy(dx: 8, dy: 8), xRadius: 26, yRadius: 26)

        switch dockIcon {
        case .emeraldSpark:
            let bgGradient = NSGradient(
                starting: NSColor(srgbRed: 0.08, green: 0.16, blue: 0.10, alpha: 1.0),
                ending: NSColor(srgbRed: 0.02, green: 0.06, blue: 0.03, alpha: 1.0)
            )
            bgGradient?.draw(in: bgPath, angle: -45)
            
            // Outer subtle border
            NSColor(srgbRed: 0.41, green: 0.73, blue: 0.44, alpha: 0.6).setStroke()
            bgPath.lineWidth = 2.5
            bgPath.stroke()

            // Sparkle symbol
            if let spark = NSImage(systemSymbolName: "sparkles", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 48, weight: .semibold)
                if let tinted = spark.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 32, y: 32, width: 64, height: 64)
                    NSColor(srgbRed: 0.41, green: 0.88, blue: 0.44, alpha: 1.0).set()
                    tinted.draw(in: rect)
                }
            }

        case .codexDark:
            let bgGradient = NSGradient(
                starting: NSColor(srgbRed: 0.16, green: 0.16, blue: 0.18, alpha: 1.0),
                ending: NSColor(srgbRed: 0.08, green: 0.08, blue: 0.09, alpha: 1.0)
            )
            bgGradient?.draw(in: bgPath, angle: -45)
            NSColor(white: 0.35, alpha: 0.5).setStroke()
            bgPath.lineWidth = 2.0
            bgPath.stroke()

            if let symbol = NSImage(systemSymbolName: "chevron.left.forwardslash.chevron.right", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 42, weight: .bold)
                if let tinted = symbol.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 28, y: 34, width: 72, height: 60)
                    NSColor.white.set()
                    tinted.draw(in: rect)
                }
            }

        case .terminalPro:
            let bgGradient = NSGradient(
                starting: NSColor(srgbRed: 0.05, green: 0.07, blue: 0.12, alpha: 1.0),
                ending: NSColor(srgbRed: 0.01, green: 0.02, blue: 0.04, alpha: 1.0)
            )
            bgGradient?.draw(in: bgPath, angle: -45)
            NSColor(srgbRed: 0.22, green: 0.53, blue: 0.95, alpha: 0.6).setStroke()
            bgPath.lineWidth = 2.0
            bgPath.stroke()

            if let symbol = NSImage(systemSymbolName: "terminal.fill", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 46, weight: .medium)
                if let tinted = symbol.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 30, y: 30, width: 68, height: 68)
                    NSColor(srgbRed: 0.22, green: 0.65, blue: 1.0, alpha: 1.0).set()
                    tinted.draw(in: rect)
                }
            }

        case .minimalist:
            NSColor(white: 0.1, alpha: 1.0).setFill()
            bgPath.fill()
            NSColor(white: 0.3, alpha: 0.6).setStroke()
            bgPath.lineWidth = 2.0
            bgPath.stroke()

            if let symbol = NSImage(systemSymbolName: "bolt.fill", accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 46, weight: .bold)
                if let tinted = symbol.withSymbolConfiguration(config) {
                    let rect = NSRect(x: 32, y: 32, width: 64, height: 64)
                    NSColor.white.set()
                    tinted.draw(in: rect)
                }
            }
        }

        image.unlockFocus()
        NSApplication.shared.applicationIconImage = image
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
