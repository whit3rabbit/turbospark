import AppKit
import Combine
import SwiftUI

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
        AppDockIconRenderer.applyDockIcon(dockIcon)
    }
}
