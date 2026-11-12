import SwiftUI

/// The success, warning and danger hues, per mode and under Increase Contrast.
///
/// System `.green`, `.orange` and `.red` ignore the theme and miss WCAG AA on
/// the app's own surfaces (light-mode system red is 3.6:1 on white). These are
/// fixed per mode instead of derived from the accent, so a custom theme keeps a
/// status that reads as a status. Each clears 4.5:1 on every surface (page,
/// surface, elevated, composer) of every shipped preset, 7:1 under Increase
/// Contrast, and 4.5:1 on its own 14% wash. `AppStatusColorTests` pins all
/// three, so a new preset or a changed surface that breaks one fails there. Status is still never color alone: pair
/// each use with a glyph or a word (DESIGN.md section 4).
struct AppStatusPalette: Equatable, Sendable {
    let success: String
    let warning: String
    let danger: String

    static func resolve(isDark: Bool, isHighContrast: Bool) -> AppStatusPalette {
        switch (isDark, isHighContrast) {
        case (true, false): return AppStatusPalette(success: "#3DDBA8", warning: "#FFB340", danger: "#FFAAA5")
        case (false, false): return AppStatusPalette(success: "#00694A", warning: "#805300", danger: "#B50012")
        case (true, true): return AppStatusPalette(success: "#5FF0C0", warning: "#FFC966", danger: "#FFC0BC")
        case (false, true): return AppStatusPalette(success: "#004D3A", warning: "#5C3B00", danger: "#8F000C")
        }
    }
}

/// Where the accent is used as TEXT or a glyph in light mode, and the floors
/// a focus edge has to clear.
enum AppAccentText {
    /// Spark Blue's light accent (`ThemeModeConfig.defaultLight`) is 4.3:1 on
    /// the composer fill, under AA. This is its text companion.
    static let sparkBlueLightAccent = "#1D6FE0"
    static let sparkBlueLightText = "#1A64CC"

    static let textFloor = 4.5
    static let focusFloor = 3.0

    /// Dark themes keep the accent. In light mode Spark Blue gets its named
    /// companion and any other accent is darkened toward black just far
    /// enough to clear `textFloor` on `surface`; one that already clears it
    /// is returned unchanged.
    static func textHex(accent: String, surface: String, isDark: Bool) -> String {
        guard !isDark else { return accent }
        if accent.uppercased() == sparkBlueLightAccent { return sparkBlueLightText }
        var candidate = accent
        var step = 0.0
        while AppContrast.ratio(candidate, surface) < textFloor, step < 1 {
            step += 0.02
            candidate = AppContrast.blend(accent, toward: "#000000", step)
        }
        return candidate
    }

    /// The accent at full strength when it clears `focusFloor` on every
    /// surface it touches, otherwise the foreground. 55% accent could not
    /// reach 3:1 on the composer (2.7:1 dark, 2.1:1 light).
    static func focusEdgeHex(accent: String, foreground: String, against surfaces: [String]) -> String {
        surfaces.allSatisfy({ AppContrast.ratio(accent, $0) >= focusFloor }) ? accent : foreground
    }
}

extension ResolvedAppTheme {
    private var statusPalette: AppStatusPalette {
        .resolve(isDark: isDark, isHighContrast: isHighContrast)
    }

    public var success: Color { Color(hex: statusPalette.success) ?? .primary }
    public var warning: Color { Color(hex: statusPalette.warning) ?? .primary }
    public var danger: Color { Color(hex: statusPalette.danger) ?? .primary }

    /// 14% washes behind a status glyph and word. The tint is a background
    /// only: the text on it is the full-strength status color.
    public var successTint: Color { success.opacity(Self.statusTintOpacity) }
    public var warningTint: Color { warning.opacity(Self.statusTintOpacity) }
    public var dangerTint: Color { danger.opacity(Self.statusTintOpacity) }

    static let statusTintOpacity = 0.14

    // The hex strings the contrast helpers need, read back from the colors
    // the theme already resolved.
    var backgroundHex: String { AppContrast.hex(of: background) }
    var foregroundHex: String { AppContrast.hex(of: foreground) }
    var accentHex: String { AppContrast.hex(of: accent) }

    /// The composer fill as a hex, read from the color the composer draws
    /// (`TurboSparkTheme.composerBackgroundColor`).
    var composerBackgroundHex: String { AppContrast.hex(of: composerBackground) }

    /// The accent for text and glyphs: `AppAccentText.textHex`, measured on
    /// the composer fill, the lightest-contrast surface text sits on.
    public var accentText: Color {
        Color(hex: AppAccentText.textHex(accent: accentHex, surface: composerBackgroundHex, isDark: isDark)) ?? accent
    }

    /// The composer's focused edge outside Increase Contrast.
    var focusEdge: Color {
        Color(hex: AppAccentText.focusEdgeHex(
            accent: accentHex, foreground: foregroundHex,
            against: [backgroundHex, composerBackgroundHex])) ?? accent
    }
}

private struct AppLinkTintModifier: ViewModifier {
    @Environment(\.appTheme) private var theme
    func body(content: Content) -> some View { content.tint(theme.accentText) }
}

extension View {
    /// Tints a `Link` or `.link` button with the accent as TEXT. Scoped to
    /// the link rather than the scene's tint, which also fills prominent
    /// buttons and keeps the plain accent.
    func appLinkTint() -> some View { modifier(AppLinkTintModifier()) }
}
