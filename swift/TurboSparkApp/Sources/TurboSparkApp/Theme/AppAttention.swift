import SwiftUI

/// The "this needs you" signal: a persistent edge on whatever still waits on
/// the user, plus a sonar ping that locates it once when they try to go on.
///
/// The edge is the state and the ping is only the pointer. A user who looks
/// away during the ping still finds the edge, and Reduce Motion and Increase
/// Contrast lose nothing but the animation. Status is never color alone, so
/// the surface also shows a glyph and words (`AppAttentionNote`).
enum AppAttention {
    /// The warning hue the steps strip already uses for "Needs attention".
    /// A theme style rather than `Color.orange`, which is 2.1:1 on white.
    static var color: AppThemeColor { .appWarning }

    /// How long the Reduce Motion fallback holds its static ring.
    static let holdSeconds: Double = 2.7

    /// How far a sonar ring travels beyond the surface's edge.
    static let spread: CGFloat = 24
}

extension View {
    /// Marks a surface that still needs the user.
    ///
    /// - Parameters:
    ///   - isOn: draws the persistent attention edge.
    ///   - ping: a counter; every change sends sonar rings out of the edge. Bump
    ///     it when the user tries to continue, never on a timer.
    ///   - radius: the surface's corner radius, so the rings follow its shape.
    func appAttention(_ isOn: Bool, ping: Int = 0, radius: CGFloat = AppMetrics.cardRadius) -> some View {
        modifier(AppAttentionModifier(isOn: isOn, ping: ping, radius: radius))
    }
}

private struct AppAttentionModifier: ViewModifier {
    let isOn: Bool
    let ping: Int
    let radius: CGFloat

    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var holding = false

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    func body(content: Content) -> some View {
        content
            .overlay { edge }
            .overlay { rings }
            .onChange(of: ping) { _, _ in
                guard reduceMotion else { return }
                holding = true
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(AppAttention.holdSeconds))
                    holding = false
                }
            }
            .tsAnimation(TSMotion.hover, value: isOn)
    }

    private var edge: some View {
        RoundedRectangle(cornerRadius: radius, style: .continuous)
            .stroke(AppAttention.color, lineWidth: theme.isHighContrast ? 2.5 : 1.5)
            .opacity(isOn ? 1 : 0)
            .allowsHitTesting(false)
    }

    /// Three rings leave the edge one after another, twice. Under Reduce
    /// Motion they become one thick ring that holds, then fades with the
    /// hold timer.
    @ViewBuilder private var rings: some View {
        if reduceMotion {
            RoundedRectangle(cornerRadius: radius, style: .continuous)
                .stroke(AppAttention.color, lineWidth: 4)
                .opacity(holding ? 1 : 0)
                .allowsHitTesting(false)
        } else {
            ZStack {
                ring(delay: 0)
                ring(delay: 0.3)
                ring(delay: 0.6)
            }
            .allowsHitTesting(false)
            .accessibilityHidden(true)
        }
    }

    private func ring(delay: Double) -> some View {
        // Captured here: the animator's closures are Sendable and cannot read
        // the main-actor environment.
        let width: CGFloat = theme.isHighContrast ? 3 : 2
        let radius = self.radius
        return Color.clear.keyframeAnimator(initialValue: AppPingFrame(), trigger: ping) { _, frame in
            RoundedRectangle(cornerRadius: radius + frame.spread, style: .continuous)
                .stroke(AppAttention.color, lineWidth: width)
                .padding(-frame.spread)
                .opacity(frame.opacity)
        } keyframes: { _ in
            KeyframeTrack(\.spread) {
                LinearKeyframe(0, duration: max(0.001, delay))
                LinearKeyframe(AppAttention.spread, duration: 1.25, timingCurve: .easeOut)
                LinearKeyframe(0, duration: 0.001)
                LinearKeyframe(AppAttention.spread, duration: 1.25, timingCurve: .easeOut)
            }
            KeyframeTrack(\.opacity) {
                LinearKeyframe(0, duration: max(0.001, delay))
                LinearKeyframe(0.7, duration: 0.001)
                LinearKeyframe(0, duration: 1.25, timingCurve: .easeOut)
                LinearKeyframe(0.7, duration: 0.001)
                LinearKeyframe(0, duration: 1.25, timingCurve: .easeOut)
            }
        }
    }
}

private struct AppPingFrame {
    var spread: CGFloat = 0
    var opacity: Double = 0
}

/// The words and glyph that go with an attention edge: a warning triangle,
/// "Needs attention", and what to do about it.
struct AppAttentionNote: View {
    let message: Text

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .themedFont(.small)
                .foregroundStyle(AppAttention.color)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text("Needs attention", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(AppAttention.color)
                message
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .transition(.opacity)
    }
}
