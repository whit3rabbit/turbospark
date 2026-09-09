import AppKit
import SwiftUI

/// The named roles of the app's canonical type scale, and the only way to
/// size themed text.
///
/// This is the ONE change point for font sizing. Every themed call site names
/// a role (`theme.ui(.small)`, `.themedFont(.callout)`); the rendered size
/// comes from this table, so retuning a role moves every surface that plays
/// it, and adding surface #2,000 with a mismatched size is a compile error
/// rather than a drift. The explicit-point-size spellings
/// (`theme.ui(points:)` and friends) were deleted 2026-09-08: they had
/// accumulated 437 call sites across 28 distinct values, with the SAME role
/// (secondary row text, empty-state symbols) written at five or six
/// neighbouring sizes across files. See `swift/docs/SWIFT_SETTINGS_AUDIT.md`
/// section 1.
///
/// The rendered sizes below are at `AppFontCatalog.defaultUISize` (16pt);
/// every one scales proportionally when the user moves the UI or code base
/// size or the Text Size setting, through the same descriptor math for every
/// role. Factors are exact size/16 ratios so the rendered value at the
/// default base is the integer the table promises after `scaled(by:)`'s
/// rounding.
///
/// | role | renders (16pt base) | plays |
/// |---|---|---|
/// | `.micro`   |  9 | overline chips, micro badges, chart ticks |
/// | `.tiny`    | 11 | fine print, timestamps, dense metadata |
/// | `.small`   | 12 | secondary rows, sidebar text |
/// | `.callout` | 14 | control text, card and sheet titles |
/// | `.base`    | 16 | body text |
/// | `.large`   | 17 | slightly emphasized body |
/// | `.title3`  | 18 | section headings |
/// | `.title2`  | 21 | pane headings, small empty-state symbols |
/// | `.title`   | 27 | window headings |
/// | `.hero`    | 30 | medium empty-state symbols |
/// | `.display` | 36 | hero empty-state symbols, display numerals |
public enum AppFontStep: CaseIterable, Sendable {
    case micro
    case tiny
    case small
    case callout
    case base
    case large
    case title3
    case title2
    case title
    case hero
    case display

    /// The multiplier applied to the user's base font size. Exactly
    /// `renderedPointsAtDefaultBase / 16`.
    public var factor: CGFloat {
        switch self {
        case .micro: 0.5625
        case .tiny: 0.6875
        case .small: 0.75
        case .callout: 0.875
        case .base: 1.0
        case .large: 1.0625
        case .title3: 1.125
        case .title2: 1.3125
        case .title: 1.6875
        case .hero: 1.875
        case .display: 2.25
        }
    }

    /// What this role renders at `AppFontCatalog.defaultUISize`. For tests
    /// and previews; call sites read the factor through the theme instead.
    public var renderedPointsAtDefaultBase: CGFloat {
        AppFontCatalog.defaultUISize * factor
    }
}

/// A font request resolved out of `ThemeModeConfig`: family, weight and size.
///
/// This is kept separate from a built `Font` because MarkdownUI cannot take
/// one. Its `FontFamily` / `FontSize` styles want the family name and the
/// point size as separate values, so the transcript needs the parts even
/// though every other surface only needs `font`.
public struct AppFontDescriptor: Equatable, Hashable, Sendable {
    /// Family name as the picker and `ThemeModeConfig` spell it, which is a
    /// catalog name ("JetBrains Mono") rather than a PostScript face name.
    public var family: String
    public var weight: Font.Weight
    public var size: CGFloat
    /// Whether this descriptor styles code. Only consulted when `family` names
    /// the system font, where it picks the monospaced design over the default.
    public var isCode: Bool

    public init(family: String, weight: Font.Weight, size: CGFloat, isCode: Bool) {
        self.family = family
        self.weight = weight
        self.size = size
        self.isCode = isCode
    }

    /// The name to hand `Font.custom`, or nil when this names the system font.
    ///
    /// `System default`, `SF Pro` and `SF Mono` are not files on disk: they are
    /// the system face, reached through `Font.system(design:)`. Sending them to
    /// `Font.custom` would look up a family that does not exist and fall back
    /// silently, which is the failure this whole type exists to remove.
    public var customFamilyName: String? {
        AppFontCatalog.systemFamilies.contains(family) ? nil : family
    }

    /// The SwiftUI font this descriptor asks for.
    public var font: Font {
        resolved(size: size, weight: weight, systemDesign: nil)
    }

    /// The font at an explicit point size and weight.
    ///
    /// `systemDesign` overrides the design ONLY when this descriptor names the
    /// system face, which is how a deliberately serif or monospaced site keeps
    /// its look at the default setting and still follows a family the user
    /// actually chose.
    public func resolved(size: CGFloat, weight: Font.Weight, systemDesign: Font.Design?) -> Font {
        guard let name = customFamilyName else {
            let design = systemDesign ?? (isCode ? .monospaced : .default)
            return .system(size: size, weight: weight, design: design)
        }
        return .custom(name, size: size).weight(weight)
    }

    /// The same font at a size relative to this one, for a surface that wants
    /// a caption or a heading without leaving the chosen family.
    ///
    /// `systemDesign` overrides the design only while this descriptor names
    /// the system face (see `resolved(size:weight:systemDesign:)`), so a
    /// deliberately monospaced or serif site keeps its look at the default
    /// family setting.
    public func scaled(
        by factor: CGFloat,
        weight override: Font.Weight? = nil,
        systemDesign: Font.Design? = nil
    ) -> Font {
        var copy = self
        copy.size = (size * factor).rounded()
        if let override { copy.weight = override }
        return copy.resolved(
            size: copy.size, weight: copy.weight, systemDesign: systemDesign)
    }
}

/// The font families the Appearance pane offers, and which of them can render.
///
/// The offered lists used to be two array literals inside
/// `ThemeConfigCardView.fontPickerRow`, which is how three families that are
/// not installed anywhere came to be on the menu: `Font.custom` falls back to
/// the system face without erroring, so picking one did nothing and looked
/// like the setting being ignored. `swift/CLAUDE.md` Gotcha 22 is the same
/// rule on the model hub's filters -- build the options from what is present.
public enum AppFontCatalog {
    public static let systemDefault = "System default"

    /// The base sizes the app was designed at, and what `AppearanceManager`
    /// defaults to. An explicit point size at a call site is expressed
    /// RELATIVE to these, so the shipped design renders unchanged at the
    /// defaults and scales as a whole when the user moves the base.
    public static let defaultUISize: CGFloat = 16
    public static let defaultCodeSize: CGFloat = 12

    /// Families that name the system face and therefore need nothing installed.
    public static let systemFamilies: Set<String> = ["System default", "SF Pro", "SF Mono"]

    /// Families bundled with the app, registered at launch by `AppFontRegistrar`.
    public static let bundledFamilies: Set<String> = ["Inter", "JetBrains Mono", "Fira Code"]

    /// Every UI family the pane may offer, in menu order.
    public static let offeredUIFamilies = [
        "System default", "SF Pro", "Inter", "Helvetica Neue", "Avenir", "Georgia", "Palatino", "Charter"
    ]

    /// Every code family the pane may offer, in menu order.
    public static let offeredCodeFamilies = [
        "System default", "SF Mono", "Menlo", "JetBrains Mono", "Fira Code", "Courier New"
    ]

    /// Narrows an offered list to the families that will actually render.
    ///
    /// Pure, with the installed set passed in rather than read from
    /// `NSFontManager`, so a test can pin the behaviour without depending on
    /// which fonts the machine running it happens to have.
    public static func available(from offered: [String], installed: Set<String>) -> [String] {
        offered.filter { systemFamilies.contains($0) || installed.contains($0) }
    }

    public static func availableUIFamilies(installed: Set<String>) -> [String] {
        available(from: offeredUIFamilies, installed: installed)
    }

    public static func availableCodeFamilies(installed: Set<String>) -> [String] {
        available(from: offeredCodeFamilies, installed: installed)
    }

    /// Resolves a persisted family name to one that can render.
    ///
    /// A config naming a family that is no longer installed falls back to the
    /// system face BY NAME, so the pane shows what is really being drawn. The
    /// alternative -- handing the missing name to `Font.custom` anyway -- draws
    /// the same glyphs while the picker still claims the missing family.
    public static func resolveFamily(
        _ name: String,
        offered: [String],
        installed: Set<String>
    ) -> String {
        available(from: offered, installed: installed).contains(name) ? name : systemDefault
    }

    /// The families macOS reports as installed, including any this process
    /// registered.
    ///
    /// MEMOIZED, because `ResolvedAppTheme.resolve` calls this from
    /// `AppThemeInjector`'s body, which `RootView` re-evaluates once per
    /// streamed token (swift/CLAUDE.md Gotcha 16), and enumerating every
    /// installed family that often made the Appearance pane and decode both
    /// feel sluggish. The cost of the cache: a font installed while the app
    /// runs appears in the pickers after a relaunch.
    @MainActor
    public static func installedFamilies() -> Set<String> {
        if let cachedInstalledFamilies { return cachedInstalledFamilies }
        AppFontRegistrar.registerBundledFonts()
        let families = Set(NSFontManager.shared.availableFontFamilies)
        cachedInstalledFamilies = families
        return families
    }

    @MainActor
    private static var cachedInstalledFamilies: Set<String>?
}
