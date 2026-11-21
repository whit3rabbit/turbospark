import SwiftUI

/// A soft highlight that sweeps across a view's own shape, for "something
/// is on its way" states (a skeleton before text arrives, a live status
/// label). Extracted from the Humanize skeleton so every site shares one
/// curve and one set of motion rules.
///
/// The sweep runs only while `active`, the window is key and motion is
/// allowed (the app setting first, the system one as its fallback, the same
/// two-source check as `TSMotionContext`). Otherwise the view draws as is,
/// static: Reduce Motion asks for that, and a background window should not
/// spend GPU frames on decoration while a model is generating on that GPU.
///
/// The sweep is masked by the content drawn opaque, so set the content's own
/// `foregroundStyle` OUTSIDE this modifier: a translucent style applied
/// inside it would dim the mask, and the sweep with it.
struct AppShimmerModifier: ViewModifier {
    let active: Bool
    /// Nil uses the theme accent at a quarter strength.
    let tint: Color?

    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @Environment(\.controlActiveState) private var controlActiveState

    /// Pure so the pause rules are testable without a window.
    static func animates(active: Bool, reduceMotion: Bool, windowIsKey: Bool) -> Bool {
        active && !reduceMotion && windowIsKey
    }

    /// Fraction of the view's width the bright band covers.
    static let bandFraction: CGFloat = 0.4

    func body(content: Content) -> some View {
        let animates = Self.animates(
            active: active,
            reduceMotion: appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion),
            windowIsKey: controlActiveState == .key)
        content.overlay {
            if animates {
                GeometryReader { geometry in
                    let width = geometry.size.width
                    Color.clear
                        .overlay(alignment: .leading) {
                            LinearGradient(
                                colors: [.clear, tint ?? theme.accent.opacity(0.25), .clear],
                                startPoint: .leading, endPoint: .trailing)
                            .frame(width: width * Self.bandFraction)
                            .phaseAnimator([false, true]) { view, moved in
                                view.offset(x: moved ? width : -width * Self.bandFraction)
                            } animation: { moved in
                                moved ? TSMotion.sweep : TSMotion.snap
                            }
                        }
                }
                // A mask reads only alpha; the theme's opaque foreground
                // stands in for "fully shown".
                .mask { content.foregroundStyle(theme.foreground) }
                .allowsHitTesting(false)
                .accessibilityHidden(true)
            }
        }
    }
}

extension View {
    /// Sweeps a soft highlight across this view's shape while `active`.
    /// See `AppShimmerModifier` for when it stays static.
    func appShimmer(active: Bool = true, tint: Color? = nil) -> some View {
        modifier(AppShimmerModifier(active: active, tint: tint))
    }
}
