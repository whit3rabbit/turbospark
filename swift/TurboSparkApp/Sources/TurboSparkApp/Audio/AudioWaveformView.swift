import SwiftUI

/// Bar waveform drawn in one `Canvas` pass.
///
/// Native on purpose (docs/AUDIO_UI.md, "Dependencies"): DSWaveformImage
/// would add a package for what is a resample plus a loop of rounded
/// rectangles, and a `Canvas` redraws a 200-bar live strip at meter rate
/// without allocating a view per bar.
///
/// One view serves every audio surface:
/// - the composer's live recording strip (`samples` is a level ring buffer),
/// - the chip's 40x16 thumbnail (static, no interaction),
/// - the bubble and preview pane (static, playhead, seek and trim).
struct AudioWaveformView: View {
    /// 0...1 values, oldest first. Resampled to fit the width.
    let samples: [Float]
    /// Playback position 0...1. Bars left of it draw in the accent.
    var progress: Double? = nil
    /// Selected range in 0...1 units, drawn as a tinted band.
    var selection: ClosedRange<Double>? = nil
    var barWidth: CGFloat = 2
    var barSpacing: CGFloat = 1
    /// Smallest drawn bar, so silence still reads as a line, not a gap.
    var minimumBarHeight: CGFloat = 1.5
    /// Overrides the accent for the played or live portion (recording red).
    var activeTint: Color? = nil
    /// Draws every bar in the active tint (the live strip has no "unplayed").
    var allActive = false
    /// Click to seek. Nil makes the view inert to the pointer.
    var onSeek: ((Double) -> Void)? = nil
    /// Drag to select. Nil leaves drags as seeks.
    var onSelect: ((ClosedRange<Double>?) -> Void)? = nil
    /// The VoiceOver value, e.g. "0:12 of 0:42".
    var accessibilityValueText: String? = nil

    @Environment(\.appTheme) private var theme

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                draw(in: &context, size: size)
            }
            .contentShape(Rectangle())
            .gesture(
                dragGesture(width: geometry.size.width),
                including: onSeek == nil && onSelect == nil ? .none : .all)
        }
        .accessibilityElement()
        .accessibilityLabel(Text("Waveform", bundle: .module))
        .accessibilityValue(accessibilityValueText ?? "")
        .accessibilityAdjustableAction { direction in
            guard let onSeek else { return }
            let current = progress ?? 0
            switch direction {
            case .increment: onSeek(min(1, current + 0.05))
            case .decrement: onSeek(max(0, current - 0.05))
            @unknown default: break
            }
        }
    }

    private var activeColor: Color { activeTint ?? theme.accent }

    private var inactiveColor: Color {
        // Increase Contrast gets a solid secondary instead of a wash.
        theme.isHighContrast ? Color.secondary : Color.secondary.opacity(0.45)
    }

    private func draw(in context: inout GraphicsContext, size: CGSize) {
        let step = barWidth + barSpacing
        let count = max(1, Int((size.width + barSpacing) / step))
        let bars = WaveformMath.resample(samples, to: count)

        if let selection {
            let rect = CGRect(
                x: size.width * selection.lowerBound, y: 0,
                width: size.width * (selection.upperBound - selection.lowerBound),
                height: size.height)
            context.fill(
                Path(roundedRect: rect, cornerRadius: 3), with: .color(activeColor.opacity(0.14)))
        }

        let playedBars = progress.map { Int((Double(count) * $0).rounded(.down)) } ?? -1
        for (index, value) in bars.enumerated() {
            let height = max(minimumBarHeight, CGFloat(value) * size.height)
            let rect = CGRect(
                x: CGFloat(index) * step,
                y: (size.height - height) / 2,
                width: barWidth,
                height: height)
            let isActive = allActive || index < playedBars
            context.fill(
                Path(roundedRect: rect, cornerRadius: barWidth / 2),
                with: .color(isActive ? activeColor : inactiveColor))
        }

        if let progress, onSeek != nil {
            let x = size.width * progress
            var line = Path()
            line.move(to: CGPoint(x: x, y: 0))
            line.addLine(to: CGPoint(x: x, y: size.height))
            context.stroke(line, with: .color(activeColor), lineWidth: 1.5)
        }
    }

    /// Click seeks; a drag longer than 1% of the width selects. Positions
    /// are fractions of the width, so the callers never see points.
    private func dragGesture(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard let onSelect, width > 0 else { return }
                let start = Self.fraction(value.startLocation.x, width)
                let current = Self.fraction(value.location.x, width)
                guard abs(current - start) > 0.01 else { return }
                onSelect(min(start, current)...max(start, current))
            }
            .onEnded { value in
                guard width > 0 else { return }
                let start = Self.fraction(value.startLocation.x, width)
                let end = Self.fraction(value.location.x, width)
                if let onSelect, abs(end - start) > 0.01 {
                    onSelect(min(start, end)...max(start, end))
                } else {
                    onSelect?(nil)
                    onSeek?(end)
                }
            }
    }

    static func fraction(_ x: CGFloat, _ width: CGFloat) -> Double {
        Double(max(0, min(1, x / width)))
    }
}

/// Compact RMS meter with a peak-hold tick.
///
/// Used where a waveform would be noise: the reduce-motion fallback for the
/// live strip, the Settings microphone test, and the system-capture sheet.
struct AudioLevelMeter: View {
    /// 0...1 display level from the engine (`ts_audio_display_level`).
    let level: Float
    var peak: Float? = nil
    var tint: Color? = nil

    @Environment(\.appTheme) private var theme

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.08))
                Capsule()
                    .fill(tint ?? theme.accent)
                    .frame(width: width * CGFloat(max(0, min(1, level))))
                if let peak {
                    Rectangle()
                        .fill(Color.primary.opacity(0.6))
                        .frame(width: 2)
                        .offset(x: max(0, width * CGFloat(min(1, peak)) - 2))
                }
            }
        }
        .frame(height: 6)
        .accessibilityElement()
        .accessibilityLabel(Text("Input level", bundle: .module))
        .accessibilityValue("\(Int(level * 100))%")
    }
}
