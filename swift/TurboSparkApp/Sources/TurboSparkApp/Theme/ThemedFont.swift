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
private struct ThemedFontModifier: ViewModifier {
    @Environment(\.appTheme) private var theme

    enum Size {
        case step(AppFontStep)
        case points(CGFloat, design: Font.Design?)
    }

    let size: Size
    let weight: Font.Weight?
    let isCode: Bool

    func body(content: Content) -> some View {
        content.font(resolvedFont)
    }

    private var resolvedFont: Font {
        switch size {
        case .step(let step):
            return isCode ? theme.code(step, weight: weight) : theme.ui(step, weight: weight)
        case .points(let points, let design):
            return isCode
                ? theme.code(points: points, weight: weight ?? .regular)
                : theme.ui(points: points, weight: weight ?? .regular, systemDesign: design)
        }
    }
}

public extension View {
    /// The UI font at a named step, e.g. `.themedFont(.small)` in place of
    /// `.font(.caption)`. See `AppFontStep` for the step-to-semantic-size map.
    func themedFont(_ step: AppFontStep = .base, weight: Font.Weight? = nil) -> some View {
        modifier(ThemedFontModifier(size: .step(step), weight: weight, isCode: false))
    }

    /// The UI font at an explicit point size, e.g. `.themedFont(points: 11)`
    /// in place of `.font(.system(size: 11))`. Scales with the user's base
    /// size the same way `ResolvedAppTheme.ui(points:weight:systemDesign:)`
    /// does.
    func themedFont(
        points: CGFloat, weight: Font.Weight = .regular, systemDesign: Font.Design? = nil
    ) -> some View {
        modifier(ThemedFontModifier(size: .points(points, design: systemDesign), weight: weight, isCode: false))
    }

    /// The code font at a named step, for a call site that used to reach for
    /// `.monospaced()` on a semantic style.
    func themedCode(_ step: AppFontStep = .base, weight: Font.Weight? = nil) -> some View {
        modifier(ThemedFontModifier(size: .step(step), weight: weight, isCode: true))
    }

    /// The code font at an explicit point size.
    func themedCode(points: CGFloat, weight: Font.Weight = .regular) -> some View {
        modifier(ThemedFontModifier(size: .points(points, design: nil), weight: weight, isCode: true))
    }
}
