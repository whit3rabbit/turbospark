import SwiftUI

/// The waveform is the pointer target. The native slider below it owns
/// keyboard and VoiceOver seeking, including the time value.
@MainActor
struct AudioPlaybackWaveform: View {
    let peaks: [Float]
    let progress: Double
    let seek: (Double) -> Void

    @Environment(\.appTheme) private var theme
    @Environment(\.layoutDirection) private var direction
    @Environment(\.isEnabled) private var isEnabled
    @GestureState private var scrub: Double? = nil

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                let position = scrub ?? progress
                let slotWidth = size.width / CGFloat(peaks.count)
                let barWidth = min(4, slotWidth * 0.62)
                let availableHeight = max(0, size.height - 6)
                let remaining = theme.secondaryText.opacity(theme.isHighContrast ? 1 : 0.6)
                for (index, peak) in peaks.enumerated() {
                    let fraction = (Double(index) + 0.5) / Double(peaks.count)
                    let x = CGFloat(visualFraction(fraction)) * size.width
                    let height = max(2, CGFloat(peak) * availableHeight)
                    let bar = CGRect(x: x - barWidth / 2, y: (size.height - height) / 2,
                                     width: barWidth, height: height)
                    let color = fraction <= position ? theme.accent : remaining
                    context.fill(Path(roundedRect: bar, cornerRadius: barWidth / 2), with: .color(color))
                }
                let x = CGFloat(visualFraction(position)) * max(0, size.width - 2)
                let playhead = CGRect(x: x, y: 0, width: 2, height: size.height)
                context.fill(Path(playhead), with: .color(theme.accentText))
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .updating($scrub) { value, state, _ in
                        if isEnabled { state = fraction(at: value.location.x, width: geometry.size.width) }
                    }
                    .onEnded { value in
                        if isEnabled { seek(fraction(at: value.location.x, width: geometry.size.width)) }
                    }
            )
        }
        .frame(height: 54 * AppMetrics.textScale(theme))
        .accessibilityHidden(true)
    }

    private func visualFraction(_ fraction: Double) -> Double {
        direction == .rightToLeft ? 1 - fraction : fraction
    }

    private func fraction(at x: CGFloat, width: CGFloat) -> Double {
        visualFraction(Double(min(1, max(0, x / max(1, width)))))
    }
}
