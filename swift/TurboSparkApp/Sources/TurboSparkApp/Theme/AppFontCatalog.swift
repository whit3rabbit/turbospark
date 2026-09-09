import AppKit
import SwiftUI

/// Named size steps for themed text, standing in for SwiftUI's semantic sizes.
///
/// The factors are the approximate ratios of `.caption2`, `.caption` and
/// `.callout` to `.body` on macOS, so a site that used to say `.caption` reads
/// the same after conversion. They are relative on purpose: the point of the
/// step is that every themed surface moves together when the base size does.
///
/// `.tiny`/`.small` were recalibrated 2026-09-08: at the 16pt default base
/// they used to render at about 13.3pt/14.7pt, while the OTHER font-sizing
/// convention in this app (`theme.ui(points: N)`, used for chat/sidebar rows)
/// renders equivalent small/secondary text at about 11pt/12.5pt -- two
/// systems for the same semantic role, never reconciled, so a step-based
/// control placed next to a points-based one (a popover next to a sidebar
/// row) visibly mismatched. The new factors target those same point values;
/// `.base` and up are untouched, since body/heading text wasn't part of the
/// mismatch.
public enum AppFontStep: Sendable {
    case tiny
    case small
    case base
    case large
    case title3
    case title2
    case title
    case hero

    public var factor: CGFloat {
        switch self {
        case .tiny: 0.69
        case .small: 0.78
        case .base: 1.0
        case .large: 1.08
        case .title3: 1.15
        case .title2: 1.31
        case .title: 1.70
        case .hero: 2.00
        }
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
    public func scaled(by factor: CGFloat, weight override: Font.Weight? = nil) -> Font {
        var copy = self
        copy.size = (size * factor).rounded()
        if let override { copy.weight = override }
        return copy.font
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
