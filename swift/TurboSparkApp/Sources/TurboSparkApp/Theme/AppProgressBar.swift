import SwiftUI

/// The themed progress bar: an accent gradient fill with a soft glow and a
/// sheen that sweeps across the completed part, or an indeterminate sweep
/// when the work has no total.
///
/// Not `ProgressView`: its linear style cannot be tinted with the theme
/// accent on the macOS 14 SDK, brings its own vertical padding and minimum
/// width, and its indeterminate variant is a spinner-like stripe that does
/// not match the app's surfaces. VoiceOver still gets a native progress
/// indicator through `accessibilityRepresentation`, so the role, label and
/// value read exactly as a system bar's would.
///
/// Size it with `minWidth`/`idealWidth`/`maxWidth`, never a fixed width.
struct AppProgressBar: View {
    enum Size {
        /// A status strip or an in-row bar.
        case compact
        /// The main progress of a card.
        case regular
    }

    /// The completed fraction, or nil for work with no known total.
    let value: Double?
    let label: Text
    /// Replaces the default percent value for VoiceOver (a counter, a time).
    var valueText: Text?
    var size: Size = .regular
    /// False for work that is paused: the bar holds still, because a moving
    /// sheen or sweep reads as progress being made.
    var animated = true

    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    init(value: Double?, label: Text, valueText: Text? = nil, size: Size = .regular, animated: Bool = true) {
        self.value = value
        self.label = label
        self.valueText = valueText
        self.size = size
        self.animated = animated
    }

    /// Nothing moves under Reduce Motion or for paused work.
    private var reduceMotion: Bool {
        !animated || appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    private var height: CGFloat {
        ((size == .compact ? 4 : 8) * AppMetrics.textScale(theme)).rounded()
    }

    private var fraction: Double? {
        value.map { $0.isFinite ? min(1, max(0, $0)) : 0 }
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                track
                if let fraction {
                    fill(width: max(height, geometry.size.width * fraction))
                        .tsAnimation(TSMotion.select, value: fraction)
                } else {
                    indeterminate(width: geometry.size.width)
                }
            }
            // The geometry is drawn left to right and mirrored as a whole,
            // so the sheen and sweep offsets never fight the layout flip.
            .environment(\.layoutDirection, .leftToRight)
            .flipsForRightToLeftLayoutDirection(true)
        }
        .frame(height: height)
        .accessibilityRepresentation {
            representation.modifier(OptionalAccessibilityValue(text: valueText))
        }
    }

    @ViewBuilder private var representation: some View {
        if let fraction {
            ProgressView(value: fraction) { label }
        } else {
            ProgressView { label }
        }
    }

    private var track: some View {
        Capsule()
            .fill(theme.foreground.opacity(0.10))
            .overlay {
                if theme.isHighContrast { Capsule().stroke(theme.border, lineWidth: 1) }
            }
    }

    private var fillStyle: AnyShapeStyle {
        if theme.isHighContrast { return AnyShapeStyle(theme.accent) }
        return AnyShapeStyle(LinearGradient(
            colors: [theme.accent.opacity(0.7), theme.accent], startPoint: .leading, endPoint: .trailing))
    }

    private func fill(width: CGFloat) -> some View {
        Capsule()
            .fill(fillStyle)
            .overlay { if !reduceMotion && !theme.isHighContrast { sheen(width: width) } }
            .clipShape(Capsule())
            .frame(width: width)
            .shadow(color: theme.isHighContrast ? .clear : theme.accent.opacity(0.35), radius: height * 0.75)
    }

    /// A bright band crossing the completed part, then an invisible reset.
    private func sheen(width: CGFloat) -> some View {
        let band = max(height * 4, width * 0.35)
        return LinearGradient(colors: [.clear, .white.opacity(0.35), .clear], startPoint: .leading, endPoint: .trailing)
            .frame(width: band)
            .phaseAnimator([false, true]) { view, moved in
                view.offset(x: moved ? width : -band)
            } animation: { moved in
                moved ? TSMotion.sweep : TSMotion.snap
            }
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private func indeterminate(width: CGFloat) -> some View {
        if reduceMotion {
            // A static half bar still reads as working, with nothing moving.
            Capsule().fill(fillStyle).opacity(0.5).frame(width: width * 0.5)
        } else {
            Capsule()
                .fill(fillStyle)
                .frame(width: width * 0.4)
                .phaseAnimator([false, true]) { view, moved in
                    view.offset(x: moved ? width * 0.6 : 0)
                } animation: { _ in
                    TSMotion.sweep
                }
        }
    }
}

private struct OptionalAccessibilityValue: ViewModifier {
    let text: Text?

    func body(content: Content) -> some View {
        if let text {
            content.accessibilityValue(text)
        } else {
            content
        }
    }
}
