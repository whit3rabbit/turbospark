import SwiftUI

/// Card, composer and glow treatments shared by workspace panes.
///
/// The chat composer and the send button drew the app's only focus glow and
/// accent glow, each spelled inline, and every workspace card picked its own
/// radius (8, 10, 12). These are the shared spellings, so a new surface reads
/// as part of the welcome screen instead of a fourth variant. Increase
/// Contrast swaps every glow for a solid ring: a soft accent halo is exactly
/// the low-contrast cue that setting asks the app to stop relying on.
extension AppMetrics {
    /// Corner radius of a composer-style surface (the chat prompt box).
    static let composerRadius: CGFloat = 20

    /// How far the UI font sits above the default. Never below 1, so a
    /// smaller font keeps the designed proportions. Frames that hold text
    /// and width breakpoints multiply by this so Extra Large text wraps or
    /// changes layout instead of clipping.
    static func textScale(_ theme: ResolvedAppTheme) -> CGFloat {
        max(1, theme.uiFontDescriptor.size / AppFontCatalog.defaultUISize)
    }
}

extension View {
    /// A grouped card: the raised theme surface, the theme hairline and the
    /// card radius.
    func appCardSurface(padding: CGFloat = 18) -> some View {
        modifier(AppCardSurfaceModifier(padding: padding))
    }

    /// The chat composer's surface: radius 20, the composer fill, a soft
    /// shadow and the focus edge while `focused`. `halo` adds the accent glow
    /// outside that edge; the chat and Humanize composers keep their
    /// original edge-only look.
    func appComposerSurface(focused: Bool, halo: Bool = true) -> some View {
        modifier(AppComposerSurfaceModifier(focused: focused, halo: halo))
    }

    /// An accent edge, and with `halo` an accent glow outside it, while
    /// `focused`, for an editor that has no native focus ring (`TextEditor`).
    /// A `TextField` keeps its system ring.
    func appFocusGlow(_ focused: Bool, radius: CGFloat = AppMetrics.controlRadius, halo: Bool = true) -> some View {
        modifier(AppFocusGlowModifier(focused: focused, radius: radius, halo: halo))
    }

    /// The accent halo under the one primary action of a surface while it
    /// can run, as the chat send button draws it.
    func appPrimaryGlow(_ active: Bool) -> some View {
        modifier(AppPrimaryGlowModifier(active: active))
    }
}

private struct AppCardSurfaceModifier: ViewModifier {
    let padding: CGFloat

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 1)
            )
    }
}

private struct AppComposerSurfaceModifier: ViewModifier {
    let focused: Bool
    let halo: Bool
    @Environment(\.appTheme) private var theme

    func body(content: Content) -> some View {
        content
            .background {
                RoundedRectangle(cornerRadius: AppMetrics.composerRadius, style: .continuous)
                    .fill(theme.composerBackground)
                    .shadow(color: .black.opacity(theme.isDark ? 0.28 : 0.08), radius: focused ? 10 : 6, x: 0, y: 2)
            }
            .appFocusGlow(focused, radius: AppMetrics.composerRadius, halo: halo)
    }
}

private struct AppFocusGlowModifier: ViewModifier {
    let focused: Bool
    let radius: CGFloat
    let halo: Bool
    @Environment(\.appTheme) private var theme

    func body(content: Content) -> some View {
        content
            // The halo is cast by an edge drawn BEHIND the surface, not by
            // the view itself: a shadow on the view would also make any text
            // over a see-through surface glow.
            .background {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(theme.accent.opacity(0.5), lineWidth: 1)
                    .shadow(color: theme.accent.opacity(0.35), radius: 10, x: 0, y: 0)
                    .opacity(focused && halo && !theme.isHighContrast ? 1 : 0)
                    .allowsHitTesting(false)
            }
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(strokeColor, lineWidth: lineWidth)
                    .allowsHitTesting(false)
            }
            .tsAnimation(TSMotion.hover, value: focused)
    }

    private var strokeColor: Color {
        if theme.isHighContrast { return focused ? theme.accent : theme.border }
        return focused ? theme.accent.opacity(0.55) : Color.primary.opacity(theme.isDark ? 0.18 : 0.14)
    }

    private var lineWidth: CGFloat {
        if theme.isHighContrast { return focused ? 2.5 : 1.5 }
        return focused ? 1.5 : 1
    }
}

private struct AppPrimaryGlowModifier: ViewModifier {
    let active: Bool
    @Environment(\.appTheme) private var theme

    func body(content: Content) -> some View {
        content
            .shadow(color: active && !theme.isHighContrast ? theme.accent.opacity(0.45) : .clear, radius: 8, x: 0, y: 2)
            .tsAnimation(TSMotion.hover, value: active)
    }
}
