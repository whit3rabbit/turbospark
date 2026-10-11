import SwiftUI

/// The status dot in front of the alias.
///
/// Grey when there is nothing to run, accent when a session is live, and
/// pulsing accent while one is being built. The pulse is the ONLY animated
/// thing in this control and it is state-driven rather than per token, which
/// is what lets it sit directly under a streaming transcript without paying a
/// redraw per decoded token.
struct ModelStatusDot: View {
    let state: ModelLoaderControl.LoaderState
    let accent: Color

    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    private var isWorking: Bool {
        state == .installing || state == .opening
    }

    private var tint: Color {
        switch state {
        case .ready: return accent
        case .installing, .opening: return accent
        case .notLoaded: return theme.secondaryText
        case .noneInstalled, .noneSelected: return theme.secondaryText
        }
    }

    var body: some View {
        Circle()
            .fill(tint)
            .frame(width: 7, height: 7)
            .opacity(state == .ready || isWorking ? 1 : 0.5)
            .modifier(PulseIfWorking(active: isWorking && !reduceMotion))
            .accessibilityHidden(true)
    }

    /// Applied as a modifier rather than branching in `body`, so the
    /// `phaseAnimator` is torn down with the view when the state leaves
    /// `installing`/`opening` instead of being left running behind a static
    /// dot.
    private struct PulseIfWorking: ViewModifier {
        let active: Bool

        func body(content: Content) -> some View {
            if active {
                content.phaseAnimator([0.4, 1.0]) { view, opacity in
                    view.opacity(opacity)
                } animation: { _ in
                    .easeInOut(duration: 0.85)
                }
            } else {
                content
            }
        }
    }
}

/// Determinate or indeterminate hairline bar, sized by its caller.
///
/// Not `ProgressView`: the linear style on the 14 SDK brings its own vertical
/// padding and a minimum width that both break a 34pt pill, and the
/// indeterminate variant cannot be tinted to `theme.accent` without an
/// appearance override.
struct ModelLoadProgressBar: View {
    /// `nil` renders the indeterminate sweep.
    let fraction: Double?
    let tint: Color

    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(.appText.opacity(0.12))

                if let fraction {
                    Capsule()
                        .fill(tint)
                        .frame(width: max(3, geometry.size.width * min(1, max(0, fraction))))
                        .tsAnimation(TSMotion.hover, value: fraction)
                } else if reduceMotion {
                    // A static half bar rather than a sweep: still reads as
                    // "working", with nothing in motion.
                    Capsule()
                        .fill(tint.opacity(0.5))
                        .frame(width: geometry.size.width * 0.5)
                } else {
                    Capsule()
                        .fill(tint)
                        .frame(width: geometry.size.width * 0.4)
                        .phaseAnimator([0.0, 1.0]) { view, phase in
                            view.offset(x: phase == 0 ? 0 : geometry.size.width * 0.6)
                        } animation: { _ in
                            .easeInOut(duration: 0.95)
                        }
                }
            }
        }
        .frame(height: 3)
        .accessibilityHidden(true)
    }
}
