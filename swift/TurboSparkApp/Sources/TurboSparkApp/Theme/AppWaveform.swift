import SwiftUI

/// Bar waveforms for live audio work: a rolling input level while recording,
/// and a picture that fills in while speech is generated.
///
/// Both are decorative for VoiceOver (the progress bar or meter beside them
/// carries the value), and both are plain `HStack`s of capsules rather than a
/// `Canvas`, so they mirror in a right-to-left language like the reference
/// waveform. Bar height scales with `AppMetrics.textScale` so the frame
/// follows the text size setting.

/// Recent input levels, newest at the trailing edge, fading in from the
/// leading edge so the eye follows the signal as it scrolls.
struct AppLevelWaveform: View {
    /// Levels in 0...1, oldest first.
    let levels: [Float]
    var height: CGFloat = 56

    @Environment(\.appTheme) private var theme

    private var full: CGFloat { (height * AppMetrics.textScale(theme)).rounded() }

    var body: some View {
        HStack(alignment: .center, spacing: 2) {
            ForEach(Array(levels.enumerated()), id: \.offset) { index, level in
                let age = levels.count > 1 ? Double(index) / Double(levels.count - 1) : 1
                Capsule()
                    .fill(theme.accent.opacity(theme.isHighContrast ? 1 : 0.3 + 0.7 * age))
                    .frame(maxWidth: .infinity)
                    .frame(height: max(3, full * CGFloat(min(1, max(0, level)))))
            }
        }
        .frame(height: full)
        .accessibilityHidden(true)
    }

    /// A fixed-length history with `level` appended, for a caller that
    /// samples on a timer.
    static func pushed(_ level: Float, onto history: [Float], count: Int) -> [Float] {
        var next = history
        next.append(level.isFinite ? min(1, max(0, level)) : 0)
        if next.count > count { next.removeFirst(next.count - count) }
        return next
    }
}

/// A waveform that fills in as generation proceeds. Bars up to `fill` take
/// their full height in the accent; the rest are low and dim. While
/// `rippling` (the model is loading or preparing, so nothing is drawn yet)
/// the dim bars breathe in a slow wave. Static under Reduce Motion.
struct AppFillWaveform: View {
    /// How much of the speech exists, 0...1.
    let fill: Double
    var rippling = false
    var barCount = 48
    var height: CGFloat = 56

    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    private var full: CGFloat { (height * AppMetrics.textScale(theme)).rounded() }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: reduceMotion || !rippling)) { context in
            let phase = context.date.timeIntervalSinceReferenceDate
            HStack(alignment: .center, spacing: 2) {
                ForEach(0..<barCount, id: \.self) { index in
                    Capsule()
                        .fill(color(index))
                        .frame(maxWidth: .infinity)
                        .frame(height: barHeight(index, phase: phase))
                }
            }
            .frame(height: full)
        }
        .accessibilityHidden(true)
    }

    private func isFilled(_ index: Int) -> Bool {
        (Double(index) + 0.5) / Double(barCount) <= fill
    }

    private func color(_ index: Int) -> Color {
        if isFilled(index) { return theme.accent }
        return theme.isHighContrast ? theme.foreground.opacity(0.45) : theme.accent.opacity(0.3)
    }

    private func barHeight(_ index: Int, phase: Double) -> CGFloat {
        if isFilled(index) { return max(3, full * CGFloat(Self.shape(index))) }
        guard rippling, !reduceMotion else { return 3 }
        let wave = 0.5 + 0.5 * sin(phase * 6 + Double(index) * 0.7)
        return max(3, full * CGFloat(0.12 + 0.12 * wave))
    }

    /// A stable speech-like silhouette in 0.14...1, the same for every run so
    /// the picture does not jump between redraws.
    static func shape(_ index: Int) -> Double {
        let i = Double(index)
        let envelope = abs(sin(i * 0.37 + 7.7)) * 0.6 + abs(sin(i * 0.11 + 7.7 * 1.7)) * 0.4
        let grain = abs(sin(i * 12.9898 + 7.7 * 78.233) * 43758.5453).truncatingRemainder(dividingBy: 1)
        return 0.14 + (envelope * 0.75 + grain * 0.25) * 0.86
    }
}
