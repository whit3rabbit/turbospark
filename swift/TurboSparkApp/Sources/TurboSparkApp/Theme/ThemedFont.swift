import SwiftUI

/// Applies a themed font without requiring the call site to declare
/// `@Environment(\.appTheme)` itself.
///
/// `swift/docs/SWIFT_SETTINGS_AUDIT.md` section 1 traced why the font picker moved
/// 232 call sites and left 809 untouched: every one of those reads
/// `\.appTheme` explicitly through `ResolvedAppTheme.ui(_:weight:)` /
/// `.code(_:weight:)`, which is a correct but easy modifier to skip under a
/// plain `.font(.caption)`. This modifier reads the environment itself, so a
/// converted call site is exactly as short as the hardcoded one it replaces.
///
/// Sizing is BY ROLE only (`AppFontStep`): the explicit-point-size overloads
/// this once carried were deleted when the two coexisting sizing conventions
/// were reconciled into the one canonical scale, because a numeric size at a
/// call site is exactly how the drift came back. A genuinely new size needs a
/// new role in `AppFontStep`, where it is one edit and one review, not a
/// literal scattered across files.
private struct ThemedFontModifier: ViewModifier {
    @Environment(\.appTheme) private var theme

    enum Size {
        case step(AppFontStep)
        /// A layout-derived point value; see `themedFont(fitting:)`.
        case fitting(CGFloat)
    }

    let size: Size
    let weight: Font.Weight?
    let isCode: Bool
    let systemDesign: Font.Design?

    func body(content: Content) -> some View {
        content.font(resolvedFont)
    }

    private var resolvedFont: Font {
        switch size {
        case .step(let step):
            return isCode
                ? theme.code(step, weight: weight, systemDesign: systemDesign)
                : theme.ui(step, weight: weight, systemDesign: systemDesign)
        case .fitting(let points):
            return theme.ui(fitting: points, weight: weight ?? .regular, systemDesign: systemDesign)
        }
    }
}

public extension View {
    /// The UI font at a named step, e.g. `.themedFont(.small)` in place of
    /// `.font(.caption)`. See `AppFontStep` for the canonical scale.
    ///
    /// `systemDesign` reaches the font only while the family is the system
    /// face, for a deliberately serif or monospaced site.
    func themedFont(
        _ step: AppFontStep = .base,
        weight: Font.Weight? = nil,
        systemDesign: Font.Design? = nil
    ) -> some View {
        modifier(
            ThemedFontModifier(
                size: .step(step), weight: weight, isCode: false, systemDesign: systemDesign))
    }

    /// The UI font at a size derived from LAYOUT rather than from the type
    /// scale: a letterform proportional to the logo frame it sits in, a
    /// symbol sized to its icon frame, a label relative to a gauge's size.
    /// The value still scales with the user's base size, so the glyph tracks
    /// the setting like every themed surface.
    ///
    /// This is the ONLY spelling that takes a number, and it is for
    /// layout-derived values only: `fitting: size * 0.45`, never a constant.
    /// A constant size is a role the type scale is missing, and belongs as a
    /// new case in `AppFontStep` where it is one reviewed edit instead of a
    /// scattered literal -- that rule is what retired the old
    /// numeric `points:` API, which had drifted to 28 distinct values across
    /// 437 call sites playing the same roles.
    func themedFont(
        fitting points: CGFloat,
        weight: Font.Weight? = nil,
        systemDesign: Font.Design? = nil
    ) -> some View {
        modifier(
            ThemedFontModifier(
                size: .fitting(points), weight: weight, isCode: false,
                systemDesign: systemDesign))
    }

    /// The code font at a named step, for a call site that used to reach for
    /// `.monospaced()` on a semantic style.
    func themedCode(
        _ step: AppFontStep = .base,
        weight: Font.Weight? = nil,
        systemDesign: Font.Design? = nil
    ) -> some View {
        modifier(
            ThemedFontModifier(
                size: .step(step), weight: weight, isCode: true, systemDesign: systemDesign))
    }
}
