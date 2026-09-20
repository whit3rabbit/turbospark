import AppKit
import Combine
import SwiftUI

/// Global observable manager for application appearance, fonts, colors, and dock icon.
///
/// Persisted through `AppearanceFileStore` (`appearance.json` under the
/// shared store root) rather than UserDefaults -- see that type's header for
/// why. Every `didSet` funnels into `persist()`; the legacy defaults keys are
/// read once at first load and removed after the first successful save.
@MainActor
public final class AppearanceManager: ObservableObject {
    public static let shared = AppearanceManager()

    /// Set when `load()` reported the archive came from the legacy
    /// UserDefaults keys; cleared after the first successful JSON save,
    /// which is what makes removing them safe.
    private var needsLegacyCleanup = false
    private var isReloadingProfile = false

    /// Resolved app appearance mode (system, light, dark).
    @Published public var appearance: AppAppearance {
        didSet { persist() }
    }

    /// User interface text scaling level (Default, Large, Extra Large).
    @Published public var textSize: AppTextSize {
        didSet { persist() }
    }

    /// Theme styling configuration used when light mode is active.
    @Published public var lightConfig: ThemeModeConfig {
        didSet { persist() }
    }

    /// Theme styling configuration used when dark mode is active.
    @Published public var darkConfig: ThemeModeConfig {
        didSet { persist() }
    }

    /// Presentation mode for bottom toolbar benchmarks (numbers or live graphs).
    @Published public var statusBarViewMode: StatusBarViewMode {
        didSet { persist() }
    }

    /// Whether interactive buttons and clickable controls display a pointer cursor on hover.
    @Published public var usePointerCursors: Bool {
        didSet { persist() }
    }

    /// Selected application dock icon variant.
    @Published public var dockIcon: AppDockIcon {
        didSet {
            persist()
            updateDockIcon()
        }
    }

    /// Motion reduction preference for transitions and animations.
    @Published public var reduceMotion: ReduceMotionPreference {
        didSet { persist() }
    }

    /// Base font point size for primary user interface text.
    @Published public var uiFontSize: Double {
        didSet { persist() }
    }

    /// Base font point size for code blocks, terminal, and monospace views.
    @Published public var codeFontSize: Double {
        didSet { persist() }
    }

    /// Presentation style for inline and side-by-side diff markers.
    @Published public var diffMarkers: DiffMarkerPreference {
        didSet { persist() }
    }

    @Published public var savedThemes: [SavedTheme] { didSet { persist() } }
    @Published public var selectedThemeID: String? { didSet { persist() } }

    public init() {
        let (archive, migrated) = AppearanceFileStore.load()
        needsLegacyCleanup = migrated

        self.savedThemes = archive.savedThemes
        self.selectedThemeID = archive.selectedThemeID
        self.appearance = AppAppearance.resolve(archive.appearance)
        self.textSize = AppTextSize.resolve(archive.textSize)
        self.lightConfig = archive.lightConfig
        self.darkConfig = archive.darkConfig
        self.statusBarViewMode = StatusBarViewMode(rawValue: archive.statusBarViewMode) ?? .text
        self.usePointerCursors = archive.usePointerCursors
        self.dockIcon = AppDockIcon(rawValue: archive.dockIcon) ?? .emeraldSpark
        self.reduceMotion = ReduceMotionPreference(rawValue: archive.reduceMotion) ?? .system
        self.uiFontSize = archive.uiFontSize
        self.codeFontSize = archive.codeFontSize
        self.diffMarkers = DiffMarkerPreference(rawValue: archive.diffMarkers) ?? .color

        AppFontRegistrar.registerBundledFonts()
        updateDockIcon()
    }

    /// Replaces the bootstrap appearance after a protected profile unlocks.
    /// Property observers are suppressed while the archive is applied so a
    /// partial set of fields is never written back to the vault.
    public func reloadFromProfile() {
        let (archive, migrated) = AppearanceFileStore.load()
        isReloadingProfile = true
        defer { isReloadingProfile = false }
        needsLegacyCleanup = migrated
        savedThemes = archive.savedThemes
        selectedThemeID = archive.selectedThemeID
        appearance = AppAppearance.resolve(archive.appearance)
        textSize = AppTextSize.resolve(archive.textSize)
        lightConfig = archive.lightConfig
        darkConfig = archive.darkConfig
        statusBarViewMode = StatusBarViewMode(rawValue: archive.statusBarViewMode) ?? .text
        usePointerCursors = archive.usePointerCursors
        dockIcon = AppDockIcon(rawValue: archive.dockIcon) ?? .emeraldSpark
        reduceMotion = ReduceMotionPreference(rawValue: archive.reduceMotion) ?? .system
        uiFontSize = archive.uiFontSize
        codeFontSize = archive.codeFontSize
        diffMarkers = DiffMarkerPreference(rawValue: archive.diffMarkers) ?? .color
        updateDockIcon()
    }

    // The four font accessors read `lightConfig` alone. Fonts are GLOBAL:
    // `setUIFont` / `setCodeFont` and the setters below are the only writers
    // and each writes both configs, so the two cannot disagree and either is
    // the answer. This used to branch on `NSApp.effectiveAppearance`, which
    // `.preferredColorScheme` does not move (swift/CLAUDE.md Gotcha 36), and
    // was right only because the invariant held.

    /// Global UI font family preference, kept in sync across both light and dark configs.
    public var uiFontFamily: String {
        get { lightConfig.uiFontFamily }
        set {
            lightConfig.uiFontFamily = newValue
            darkConfig.uiFontFamily = newValue
        }
    }

    /// Global UI font weight preference.
    public var uiFontWeight: String {
        get { lightConfig.uiFontWeight }
        set {
            lightConfig.uiFontWeight = newValue
            darkConfig.uiFontWeight = newValue
        }
    }

    /// Global code font family preference, kept in sync across both light and dark configs.
    public var codeFontFamily: String {
        get { lightConfig.codeFontFamily }
        set {
            lightConfig.codeFontFamily = newValue
            darkConfig.codeFontFamily = newValue
        }
    }

    /// Global code font weight preference.
    public var codeFontWeight: String {
        get { lightConfig.codeFontWeight }
        set {
            lightConfig.codeFontWeight = newValue
            darkConfig.codeFontWeight = newValue
        }
    }

    public func setUIFont(family: String? = nil, weight: String? = nil) {
        if let family {
            lightConfig.uiFontFamily = family
            darkConfig.uiFontFamily = family
        }
        if let weight {
            lightConfig.uiFontWeight = weight
            darkConfig.uiFontWeight = weight
        }
    }

    public func setCodeFont(family: String? = nil, weight: String? = nil) {
        if let family {
            lightConfig.codeFontFamily = family
            darkConfig.codeFontFamily = family
        }
        if let weight {
            lightConfig.codeFontWeight = weight
            darkConfig.codeFontWeight = weight
        }
    }

    /// Writes every published field into the JSON store, and completes the
    /// one-way migration from UserDefaults on the first successful save.
    private func persist() {
        guard !isReloadingProfile else { return }
        let archive = AppearanceArchive(
            appearance: appearance.rawValue,
            textSize: textSize.rawValue,
            lightConfig: lightConfig,
            darkConfig: darkConfig,
            statusBarViewMode: statusBarViewMode.rawValue,
            usePointerCursors: usePointerCursors,
            dockIcon: dockIcon.rawValue,
            reduceMotion: reduceMotion.rawValue,
            uiFontSize: uiFontSize,
            codeFontSize: codeFontSize,
            diffMarkers: diffMarkers.rawValue,
            savedThemes: savedThemes, selectedThemeID: currentThemeID)
        guard AppearanceFileStore.save(archive) else { return }
        if needsLegacyCleanup {
            AppearanceFileStore.removeLegacyKeys()
            needsLegacyCleanup = false
        }
    }

    /// Applies a preset configuration across light, dark, or both color modes.
    public func applyPreset(_ preset: ThemePreset, forMode isDark: Bool? = nil) {
        let currentUIFamily = isDark == true ? darkConfig.uiFontFamily : lightConfig.uiFontFamily
        let currentUIWeight = isDark == true ? darkConfig.uiFontWeight : lightConfig.uiFontWeight
        let currentCodeFamily = isDark == true ? darkConfig.codeFontFamily : lightConfig.codeFontFamily
        let currentCodeWeight = isDark == true ? darkConfig.codeFontWeight : lightConfig.codeFontWeight

        if let isDark = isDark {
            if isDark {
                var dark = preset.dark
                dark.uiFontFamily = currentUIFamily
                dark.uiFontWeight = currentUIWeight
                dark.codeFontFamily = currentCodeFamily
                dark.codeFontWeight = currentCodeWeight
                self.darkConfig = dark
            } else {
                var light = preset.light
                light.uiFontFamily = currentUIFamily
                light.uiFontWeight = currentUIWeight
                light.codeFontFamily = currentCodeFamily
                light.codeFontWeight = currentCodeWeight
                self.lightConfig = light
            }
        } else {
            var light = preset.light
            var dark = preset.dark
            light.uiFontFamily = currentUIFamily
            light.uiFontWeight = currentUIWeight
            light.codeFontFamily = currentCodeFamily
            light.codeFontWeight = currentCodeWeight
            dark.uiFontFamily = currentUIFamily
            dark.uiFontWeight = currentUIWeight
            dark.codeFontFamily = currentCodeFamily
            dark.codeFontWeight = currentCodeWeight
            self.lightConfig = light
            self.darkConfig = dark
        }
        selectedThemeID = isDark == nil ? preset.id : nil
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

    /// Steps the interface text size larger, up to Extra Large.
    public func makeTextBigger() {
        switch textSize {
        case .standard: textSize = .large
        case .large: textSize = .extraLarge
        case .extraLarge: break
        }
    }

    /// Steps the interface text size smaller, down to Default.
    public func makeTextSmaller() {
        switch textSize {
        case .extraLarge: textSize = .large
        case .large: textSize = .standard
        case .standard: break
        }
    }

    /// Resets the interface text size to Default (16px UI, 12px code).
    public func resetTextSize() {
        textSize = .standard
        uiFontSize = 16.0
        codeFontSize = 12.0
    }

    /// Restores every appearance field to the factory archive: system mode,
    /// Spark Blue in both color modes, standard text size, the default fonts
    /// and sizes, and the shipped preferences. `AppearanceArchive()` IS the
    /// factory defaults (its memberwise parameters carry them), and each
    /// assignment fires its `didSet`, so one call persists the whole reset
    /// and re-renders the dock icon.
    public func resetToDefaults() {
        let defaults = AppearanceArchive()
        selectedThemeID = "sparkblue"
        appearance = AppAppearance.resolve(defaults.appearance)
        textSize = AppTextSize.resolve(defaults.textSize)
        lightConfig = defaults.lightConfig
        darkConfig = defaults.darkConfig
        statusBarViewMode = StatusBarViewMode(rawValue: defaults.statusBarViewMode) ?? .text
        usePointerCursors = defaults.usePointerCursors
        dockIcon = AppDockIcon(rawValue: defaults.dockIcon) ?? .emeraldSpark
        reduceMotion = ReduceMotionPreference(rawValue: defaults.reduceMotion) ?? .system
        uiFontSize = defaults.uiFontSize
        codeFontSize = defaults.codeFontSize
        diffMarkers = DiffMarkerPreference(rawValue: defaults.diffMarkers) ?? .color
    }

    /// Builds the code-font descriptor for a mode.
    ///
    /// `isDark` is a PARAMETER. This used to read
    /// `NSApp.effectiveAppearance`, which is application-level and is not
    /// moved by SwiftUI's `.preferredColorScheme`, so an app forced to Light
    /// on a dark system resolved its fonts and colors out of `darkConfig`.
    /// The family-to-face mapping that used to live here as a `switch` is now
    /// `AppFontDescriptor.font`, which resolves any catalog family rather than
    /// the five it happened to name -- and by FAMILY rather than by a
    /// hardcoded Regular PostScript name, so a chosen weight reaches the real
    /// face instead of being synthesized off Regular.
    public func codeFontDescriptor(isDark: Bool, size: CGFloat? = nil, weight: Font.Weight? = nil) -> AppFontDescriptor {
        let config = activeConfig(isDark: isDark)
        let resolvedSize = size ?? textSize.scaled(CGFloat(codeFontSize))
        return AppFontDescriptor(
            family: config.codeFontFamily,
            weight: weight ?? Font.Weight.fromName(config.codeFontWeight),
            size: resolvedSize,
            isCode: true)
    }

    /// Builds the UI-font descriptor for a mode. See `codeFontDescriptor`.
    public func uiFontDescriptor(isDark: Bool, size: CGFloat? = nil, weight: Font.Weight? = nil) -> AppFontDescriptor {
        let config = activeConfig(isDark: isDark)
        let resolvedSize = size ?? textSize.scaled(CGFloat(uiFontSize))
        return AppFontDescriptor(
            family: config.uiFontFamily,
            weight: weight ?? Font.Weight.fromName(config.uiFontWeight),
            size: resolvedSize,
            isCode: false)
    }

    /// Constructs a code font with current user preferences and optional size/weight override.
    public func codeFont(isDark: Bool, size: CGFloat? = nil, weight: Font.Weight? = nil) -> Font {
        codeFontDescriptor(isDark: isDark, size: size, weight: weight).font
    }

    /// Constructs a UI font with current user preferences and optional size/weight override.
    public func uiFont(isDark: Bool, size: CGFloat? = nil, weight: Font.Weight? = nil) -> Font {
        uiFontDescriptor(isDark: isDark, size: size, weight: weight).font
    }

    /// Determines whether animation/motion should be suppressed based on user or system settings.
    public func shouldReduceMotion(systemReduceMotion: Bool) -> Bool {
        switch reduceMotion {
        case .system: return systemReduceMotion
        case .on: return true
        case .off: return false
        }
    }

    /// Renders and updates the macOS application dock icon based on current preset choice.
    public func updateDockIcon() {
        AppDockIconRenderer.applyDockIcon(dockIcon)
    }
}
