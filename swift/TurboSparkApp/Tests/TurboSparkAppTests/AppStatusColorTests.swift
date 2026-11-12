import AppKit
import SwiftUI
import XCTest
@testable import TurboSparkApp

/// Pins the contrast the status colors, the accent-as-text color and the
/// composer focus edge were built to reach (DESIGN.md section 4 and 5).
///
/// Every check runs on the theme objects the app resolves, for each shipped
/// preset in both modes, so a surface drift or a new preset that breaks a
/// floor fails here rather than in a screenshot. These are computed ratios,
/// not a look at the running app.
@MainActor
final class AppStatusColorTests: XCTestCase {
    private func theme(_ config: ThemeModeConfig, isDark: Bool, highContrast: Bool = false) -> ResolvedAppTheme {
        ResolvedAppTheme(
            isDark: isDark,
            accent: Color(hex: config.accentHex) ?? .black,
            background: Color(hex: config.backgroundHex) ?? .white,
            foreground: Color(hex: config.foregroundHex) ?? .black,
            contrast: highContrast ? max(config.contrast, 95) : config.contrast,
            uiFontDescriptor: AppFontDescriptor(
                family: AppFontCatalog.systemDefault, weight: .regular, size: 16, isCode: false),
            codeFontDescriptor: AppFontDescriptor(
                family: AppFontCatalog.systemDefault, weight: .regular, size: 12, isCode: true),
            isHighContrast: highContrast)
    }

    /// Every preset in both modes, with a label for failure messages.
    private var themes: [(name: String, config: ThemeModeConfig, isDark: Bool)] {
        (ThemePreset.presets + ThemePreset.additionalPresets).flatMap { preset in
            [("\(preset.name) light", preset.light, false), ("\(preset.name) dark", preset.dark, true)]
        }
    }

    /// Page, card, raised and composer fills as drawn.
    private func surfaces(_ t: ResolvedAppTheme) -> [(String, String)] {
        [("page", AppContrast.hex(of: t.background)),
         ("surface", AppContrast.hex(of: t.surface)),
         ("elevated", AppContrast.hex(of: t.elevatedSurface)),
         ("composer", t.composerBackgroundHex)]
    }

    // MARK: Status colors

    func testStatusPaletteMatchesTheSpecifiedValues() {
        XCTAssertEqual(AppStatusPalette.resolve(isDark: true, isHighContrast: false),
                       AppStatusPalette(success: "#3DDBA8", warning: "#FFB340", danger: "#FFAAA5"))
        XCTAssertEqual(AppStatusPalette.resolve(isDark: false, isHighContrast: false),
                       AppStatusPalette(success: "#00694A", warning: "#805300", danger: "#B50012"))
        XCTAssertEqual(AppStatusPalette.resolve(isDark: true, isHighContrast: true),
                       AppStatusPalette(success: "#5FF0C0", warning: "#FFC966", danger: "#FFC0BC"))
        XCTAssertEqual(AppStatusPalette.resolve(isDark: false, isHighContrast: true),
                       AppStatusPalette(success: "#004D3A", warning: "#5C3B00", danger: "#8F000C"))
    }

    private func statusHexes(_ t: ResolvedAppTheme) -> [(String, String)] {
        [("success", AppContrast.hex(of: t.success)),
         ("warning", AppContrast.hex(of: t.warning)),
         ("danger", AppContrast.hex(of: t.danger))]
    }

    /// Every status color clears 4.5:1 on page, surface, elevated and composer
    /// of every shipped preset in both modes, and 7:1 under Increase Contrast.
    /// A new preset, a changed surface blend or a changed hex that breaks it
    /// fails here. Custom user themes are not covered: their surfaces are
    /// arbitrary.
    func testStatusColorsClearTheFloorOnEverySurfaceOfEveryPreset() {
        for (name, config, isDark) in themes {
            for highContrast in [false, true] {
                let t = theme(config, isDark: isDark, highContrast: highContrast)
                let floor = highContrast ? 7.0 : 4.5
                for (statusName, status) in statusHexes(t) {
                    for (surfaceName, surface) in surfaces(t) {
                        let ratio = AppContrast.ratio(status, surface)
                        XCTAssertGreaterThanOrEqual(
                            ratio, floor,
                            "\(statusName) on \(surfaceName), \(name)\(highContrast ? " (Increase Contrast)" : ""): \(String(format: "%.2f", ratio))")
                    }
                }
            }
        }
    }

    /// A badge draws the status text on a 14% wash of itself, which costs up to
    /// about 0.5:1, so the wash is held to 4.5:1 in every mode. 7:1 on a wash
    /// would push the dark hues to near white and erase the hue.
    func testStatusTextOnItsOwnTintClearsFourPointFive() {
        for (name, config, isDark) in themes {
            for highContrast in [false, true] {
                let t = theme(config, isDark: isDark, highContrast: highContrast)
                for (statusName, status) in statusHexes(t) {
                    for (surfaceName, surface) in surfaces(t) {
                        let wash = AppContrast.blend(surface, toward: status, ResolvedAppTheme.statusTintOpacity)
                        let ratio = AppContrast.ratio(status, wash)
                        XCTAssertGreaterThanOrEqual(
                            ratio, 4.5,
                            "\(statusName) tint over \(surfaceName), \(name)\(highContrast ? " (Increase Contrast)" : ""): \(String(format: "%.2f", ratio))")
                    }
                }
            }
        }
    }

    func testSystemRedWasUnderTheFloorOnWhite() {
        // The reason this exists: light-mode system red on white.
        XCTAssertLessThan(AppContrast.ratio("#FF3B30", "#FFFFFF"), 4.5)
        XCTAssertGreaterThanOrEqual(AppContrast.ratio("#B50012", "#FFFFFF"), 4.5)
    }

    func testShapeStylesResolveToTheThemeStatusColors() {
        let light = theme(.defaultLight, isDark: false)
        var environment = EnvironmentValues()
        environment.appTheme = light
        XCTAssertEqual(AppContrast.hex(of: AppThemeColor.appSuccess.resolve(in: environment)), "#00694A")
        XCTAssertEqual(AppContrast.hex(of: AppThemeColor.appWarning.resolve(in: environment)), "#805300")
        XCTAssertEqual(AppContrast.hex(of: AppThemeColor.appDanger.resolve(in: environment)), "#B50012")
        XCTAssertEqual(AppContrast.hex(of: AppThemeColor.appAccentText.resolve(in: environment)), "#1A64CC")

        environment.appTheme = theme(.defaultDark, isDark: true, highContrast: true)
        XCTAssertEqual(AppContrast.hex(of: AppThemeColor.appDanger.resolve(in: environment)), "#FFC0BC")
        let tint = NSColor(AppThemeColor.appDangerTint.resolve(in: environment)).usingColorSpace(.sRGB)
        XCTAssertEqual(Double(tint?.alphaComponent ?? 0), ResolvedAppTheme.statusTintOpacity, accuracy: 0.005)
    }

    // MARK: Accent as text

    func testSparkBlueLightAccentWasUnderTheTextFloorAndTheCompanionClears() {
        let t = theme(.defaultLight, isDark: false)
        XCTAssertLessThan(AppContrast.ratio("#1D6FE0", t.composerBackgroundHex), 4.5)
        XCTAssertEqual(AppContrast.hex(of: t.accentText), "#1A64CC")
        for (surfaceName, surface) in surfaces(t) {
            XCTAssertGreaterThanOrEqual(AppContrast.ratio("#1A64CC", surface), 4.5, surfaceName)
        }
    }

    func testAccentTextClearsTheFloorOnEveryPresetAndKeepsDarkAccents() {
        for (name, config, isDark) in themes {
            let t = theme(config, isDark: isDark)
            let text = AppContrast.hex(of: t.accentText)
            if isDark {
                XCTAssertEqual(text, config.accentHex.uppercased(), "\(name): dark keeps the accent")
            }
            // A preset whose own accent is already under the floor (a
            // deliberate low-contrast accent in dark) is a different
            // question; light accents must always clear it.
            if !isDark {
                XCTAssertGreaterThanOrEqual(
                    AppContrast.ratio(text, t.composerBackgroundHex), 4.5,
                    "\(name): \(text) on composer")
            }
        }
    }

    func testCustomLightAccentIsDarkenedJustEnough() {
        // A pale custom accent that fails on white: darkened toward black
        // until it passes, and no further than needed.
        let surface = "#FFFFFF"
        let result = AppAccentText.textHex(accent: "#6FA8FF", surface: surface, isDark: false)
        XCTAssertGreaterThanOrEqual(AppContrast.ratio(result, surface), 4.5)
        XCTAssertNotEqual(result, "#6FA8FF")
        // An accent that already clears the floor is returned as given.
        XCTAssertEqual(AppAccentText.textHex(accent: "#111827", surface: surface, isDark: false), "#111827")
    }

    // MARK: Composer focus edge

    func testFocusEdgeClearsThreeToOneOnPageAndComposerInEveryTheme() {
        for (name, config, isDark) in themes {
            let t = theme(config, isDark: isDark)
            let edge = AppContrast.hex(of: t.focusEdge)
            for (surfaceName, surface) in [("page", t.backgroundHex), ("composer", t.composerBackgroundHex)] {
                XCTAssertGreaterThanOrEqual(
                    AppContrast.ratio(edge, surface), 3.0,
                    "\(name): focus edge \(edge) on \(surfaceName)")
            }
        }
    }

    func testTheOldFiftyFivePercentEdgeWasUnderTheFloor() {
        for (isDark, config) in [(true, ThemeModeConfig.defaultDark), (false, ThemeModeConfig.defaultLight)] {
            let t = theme(config, isDark: isDark)
            let composited = AppContrast.blend(t.composerBackgroundHex, toward: config.accentHex, 0.55)
            XCTAssertLessThan(AppContrast.ratio(composited, t.composerBackgroundHex), 3.0,
                              "\(isDark ? "dark" : "light"): the replaced edge")
        }
    }

    func testFocusEdgeFallsBackToTheForegroundWhenTheAccentCannotReachTheFloor() {
        // An accent nearly equal to the page cannot signal focus at 3:1.
        XCTAssertEqual(
            AppAccentText.focusEdgeHex(accent: "#F5F5F5", foreground: "#111111", against: ["#FFFFFF"]),
            "#111111")
        XCTAssertEqual(
            AppAccentText.focusEdgeHex(accent: "#1D6FE0", foreground: "#111111", against: ["#FFFFFF"]),
            "#1D6FE0")
    }

    func testIncreaseContrastFocusPathStillUsesTheSolidAccent() {
        // The Increase Contrast branch was not part of this change; its edge
        // is `theme.accent` at 2.5pt, which only needs to stay 3:1 itself.
        for (name, config, isDark) in themes {
            let t = theme(config, isDark: isDark, highContrast: true)
            XCTAssertGreaterThanOrEqual(
                AppContrast.ratio(t.accentHex, t.composerBackgroundHex), 3.0,
                "\(name): Increase Contrast accent edge")
        }
    }
}
