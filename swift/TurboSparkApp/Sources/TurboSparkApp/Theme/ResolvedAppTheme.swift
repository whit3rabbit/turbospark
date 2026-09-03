import AppKit
import SwiftUI

/// Every appearance value a view needs, resolved once for the current mode.
///
/// This exists because the settings were resolved in two incompatible ways.
/// `TurboSparkTheme`'s static accessors branched on `NSApp.effectiveAppearance`
/// while `AppearanceSettingsPaneView` derived the same question from
/// `manager.appearance` plus `\.colorScheme`, and only the second is correct
/// under `.preferredColorScheme` (see `AppAppearance.isDark(systemColorScheme:)`).
///
/// Being `Equatable` and living in the environment is what makes it reactive:
/// a static computed var re-reads its inputs but cannot tell SwiftUI that
/// anything changed, so a view that does not itself observe `AppearanceManager`
/// keeps drawing the old theme until something else invalidates it.
public struct ResolvedAppTheme: Equatable, Sendable {
    public var isDark: Bool
    public var accent: Color
    public var background: Color
    public var foreground: Color
    /// 0 to 100, from the mode's config. Drives hairlines, metadata text and
    /// border strokes.
    public var contrast: Double
    public var uiFontDescriptor: AppFontDescriptor
    public var codeFontDescriptor: AppFontDescriptor
    public var textSize: AppTextSize

    public init(
        isDark: Bool,
        accent: Color,
        background: Color,
        foreground: Color,
        contrast: Double,
        uiFontDescriptor: AppFontDescriptor,
        codeFontDescriptor: AppFontDescriptor,
        textSize: AppTextSize = .standard
    ) {
        self.isDark = isDark
        self.accent = accent
        self.background = background
        self.foreground = foreground
        self.contrast = contrast
        self.uiFontDescriptor = uiFontDescriptor
        self.codeFontDescriptor = codeFontDescriptor
        self.textSize = textSize
    }

    public var uiFont: Font { uiFontDescriptor.font }
    public var codeFont: Font { codeFontDescriptor.font }

    /// The code font at a named step, for a call site that wants a caption or
    /// a heading without leaving the user's chosen family and size.
    ///
    /// Named steps rather than raw ratios so a call site says what it means,
    /// and so every code surface moves together when the base size changes.
    public func code(_ step: AppFontStep = .base, weight: Font.Weight? = nil) -> Font {
        codeFontDescriptor.scaled(by: step.factor, weight: weight)
    }

    /// The UI font at a named step. See `code(_:weight:)`.
    public func ui(_ step: AppFontStep = .base, weight: Font.Weight? = nil) -> Font {
        uiFontDescriptor.scaled(by: step.factor, weight: weight)
    }

    /// The UI font at an explicit point size, scaled by how far the user has
    /// moved the base off `AppFontCatalog.defaultUISize`.
    ///
    /// At the default this returns exactly the size the call site asks for, so
    /// converting a `.system(size: 11, ...)` site changes the FAMILY and
    /// nothing else. `systemDesign` is honoured only while the family is the
    /// system face, which keeps a deliberately serif or monospaced site
    /// looking as designed until a real family is chosen.
    public func ui(points: CGFloat, weight: Font.Weight = .regular, systemDesign: Font.Design? = nil) -> Font {
        let scale = uiFontDescriptor.size / AppFontCatalog.defaultUISize
        return uiFontDescriptor.resolved(
            size: (points * scale).rounded(), weight: weight, systemDesign: systemDesign)
    }

    /// The code font at an explicit point size. See `ui(points:weight:)`.
    public func code(points: CGFloat, weight: Font.Weight = .regular) -> Font {
        let scale = codeFontDescriptor.size / AppFontCatalog.defaultCodeSize
        return codeFontDescriptor.resolved(
            size: (points * scale).rounded(), weight: weight, systemDesign: nil)
    }

    /// Secondary text that still clears WCAG AA at the configured contrast.
    public var metadataForeground: Color {
        contrast > 70 ? Color.primary.opacity(0.9) : Color.secondary
    }

    /// Stroke opacity for card and chip borders at the configured contrast.
    public var borderStrokeOpacity: Double {
        max(0.3, (contrast / 100.0) * 0.85)
    }
}

public extension ResolvedAppTheme {
    /// Builds the theme for a mode.
    ///
    /// `colorScheme` is a PARAMETER rather than something read from `NSApp`,
    /// which is the whole point: it comes from `@Environment(\.colorScheme)`,
    /// so it follows `.preferredColorScheme` and a test can drive both modes
    /// without touching the running application's appearance.
    @MainActor
    static func resolve(
        manager: AppearanceManager,
        colorScheme: ColorScheme,
        installedFamilies: Set<String>
    ) -> ResolvedAppTheme {
        let isDark = manager.appearance.isDark(systemColorScheme: colorScheme)
        let config = manager.activeConfig(isDark: isDark)

        let uiFamily = AppFontCatalog.resolveFamily(
            config.uiFontFamily,
            offered: AppFontCatalog.offeredUIFamilies,
            installed: installedFamilies)
        let codeFamily = AppFontCatalog.resolveFamily(
            config.codeFontFamily,
            offered: AppFontCatalog.offeredCodeFamilies,
            installed: installedFamilies)

        let uiSize = manager.textSize.scaled(CGFloat(manager.uiFontSize))
        let codeSize = manager.textSize.scaled(CGFloat(manager.codeFontSize))

        return ResolvedAppTheme(
            isDark: isDark,
            accent: manager.activeAccentColor(isDark: isDark),
            background: manager.activeBackgroundColor(isDark: isDark),
            foreground: manager.activeForegroundColor(isDark: isDark),
            contrast: config.contrast,
            uiFontDescriptor: AppFontDescriptor(
                family: uiFamily,
                weight: .fromName(config.uiFontWeight),
                size: uiSize,
                isCode: false),
            codeFontDescriptor: AppFontDescriptor(
                family: codeFamily,
                weight: .fromName(config.codeFontWeight),
                size: codeSize,
                isCode: true),
            textSize: manager.textSize)
    }

    /// Convenience for the running app, which always wants the live font list.
    @MainActor
    static func resolve(manager: AppearanceManager, colorScheme: ColorScheme) -> ResolvedAppTheme {
        resolve(
            manager: manager,
            colorScheme: colorScheme,
            installedFamilies: AppFontCatalog.installedFamilies())
    }

    /// The value a view sees before `RootView` injects one. Deliberately the
    /// shipped light defaults rather than an empty or zeroed theme, so a
    /// preview or a detached view renders the app's real look.
    ///
    /// Not actor-isolated, because an `EnvironmentKey`'s `defaultValue` is not:
    /// nothing here reads `AppearanceManager`, only its `Codable` defaults.
    static let fallback = ResolvedAppTheme(
        isDark: false,
        accent: Color(hex: ThemeModeConfig.defaultLight.accentHex) ?? .black,
        background: Color(hex: ThemeModeConfig.defaultLight.backgroundHex) ?? .white,
        foreground: Color(hex: ThemeModeConfig.defaultLight.foregroundHex) ?? .primary,
        contrast: ThemeModeConfig.defaultLight.contrast,
        uiFontDescriptor: AppFontDescriptor(
            family: AppFontCatalog.systemDefault, weight: .regular, size: 14, isCode: false),
        codeFontDescriptor: AppFontDescriptor(
            family: AppFontCatalog.systemDefault, weight: .regular, size: 12, isCode: true))
}

private struct AppThemeKey: EnvironmentKey {
    static let defaultValue = ResolvedAppTheme.fallback
}

/// Resolves the appearance settings and injects them, with the accent as tint.
///
/// This exists as a modifier rather than as code in `RootView` because THIS APP
/// HAS TWO SCENES. `Settings { ... }` is a separate window, so nothing the
/// main window's root injects reaches it, and every view there fell back to the
/// light defaults while the window itself rendered dark. Two scenes spelling
/// the resolution separately is how they drift, which is the defect this whole
/// type was introduced to remove.
public struct AppThemeInjector: ViewModifier {
    @ObservedObject private var manager = AppearanceManager.shared
    @Environment(\.colorScheme) private var colorScheme

    public init() {}

    public func body(content: Content) -> some View {
        let theme = ResolvedAppTheme.resolve(manager: manager, colorScheme: colorScheme)
        return content
            .environment(\.appTheme, theme)
            .tint(theme.accent)
    }
}

public extension View {
    /// Resolves and injects `\.appTheme` for a scene root. Every scene needs
    /// its own call; see `AppThemeInjector`.
    func appThemed() -> some View {
        modifier(AppThemeInjector())
    }
}

public extension EnvironmentValues {
    /// The resolved appearance for the current mode. Injected once by
    /// `RootView`; read it rather than calling `AppearanceManager` from a body.
    var appTheme: ResolvedAppTheme {
        get { self[AppThemeKey.self] }
        set { self[AppThemeKey.self] = newValue }
    }
}
