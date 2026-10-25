import SwiftUI

/// A picture of the reference: the trimmed region's waveform with a
/// playhead while it is previewed, and a range bar showing where that
/// region sits in the whole source.
///
/// Decorative for VoiceOver: the Start slider and the Duration field beside
/// it carry the same values. Bars are an `HStack` of capsules rather than a
/// `Canvas`, so they mirror with the slider in a right-to-left language.
@MainActor
struct VoiceReferenceWaveformView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let reference: AudioVoiceReference

    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var peaks: [Float] = []

    private static let barCount = 64

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        VStack(spacing: 8) {
            TimelineView(.animation(minimumInterval: reduceMotion ? 0.5 : 1.0 / 30, paused: !controller.referencePlaying)) { _ in
                bars(played: playedFraction)
            }
            .frame(height: (52 * AppMetrics.textScale(theme)).rounded())
            overview.frame(height: 5)
        }
        .accessibilityHidden(true)
        .task(id: reference) { await loadPeaks() }
    }

    /// How much of the preview has played, or nil while it is stopped.
    private var playedFraction: Double? {
        guard controller.referencePlaying, let player = controller.referencePlayer, player.duration > 0 else { return nil }
        return min(1, max(0, player.currentTime / player.duration))
    }

    private func bars(played: Double?) -> some View {
        HStack(alignment: .center, spacing: 2) {
            ForEach(0..<Self.barCount, id: \.self) { index in
                Capsule()
                    .fill(color(index: index, played: played))
                    .frame(maxWidth: .infinity)
                    .frame(height: barHeight(index))
            }
        }
        .frame(maxHeight: .infinity)
        .tsAnimation(TSMotion.pane, value: peaks)
    }

    private func barHeight(_ index: Int) -> CGFloat {
        let full = (52 * AppMetrics.textScale(theme)).rounded()
        guard index < peaks.count else { return 3 }
        return max(3, full * CGFloat(peaks[index]))
    }

    private func color(index: Int, played: Double?) -> Color {
        guard !peaks.isEmpty else { return theme.foreground.opacity(0.12) }
        guard let played else { return theme.accent.opacity(theme.isHighContrast ? 1 : 0.75) }
        let isPlayed = Double(index) / Double(Self.barCount) < played
        if theme.isHighContrast { return isPlayed ? theme.accent : theme.foreground.opacity(0.45) }
        return isPlayed ? theme.accent : theme.accent.opacity(0.35)
    }

    private var overview: some View {
        GeometryReader { geometry in
            let total = max(reference.totalDuration, reference.duration, 0.001)
            let width = geometry.size.width
            HStack(spacing: 0) {
                Color.clear.frame(width: width * CGFloat(min(1, reference.start / total)))
                Capsule()
                    .fill(theme.accent)
                    .frame(width: max(4, width * CGFloat(min(1, reference.duration / total))))
                Spacer(minLength: 0)
            }
            .background(Capsule().fill(theme.foreground.opacity(0.10)))
            .tsAnimation(TSMotion.select, value: reference.start)
            .tsAnimation(TSMotion.select, value: reference.duration)
        }
    }

    private func loadPeaks() async {
        // Dragging the Start slider changes the reference many times a
        // second; draw only once it settles.
        try? await Task.sleep(for: .milliseconds(150))
        guard !Task.isCancelled else { return }
        guard let loaded = try? await controller.referencePeaks(for: reference, count: Self.barCount),
              !Task.isCancelled else { return }
        peaks = loaded
    }
}
