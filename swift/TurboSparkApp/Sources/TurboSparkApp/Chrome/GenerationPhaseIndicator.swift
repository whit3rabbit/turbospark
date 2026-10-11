import SwiftUI

/// Compact prefill/decode indicator shown to the right of the model loader.
struct GenerationPhaseIndicator: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    /// App preference over system value; see `ToastOverlayView`.
    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        Group {
            if let statusText {
                HStack(spacing: 5) {
                    if model.opening {
                        ProgressView().controlSize(.mini).scaleEffect(0.7)
                    } else if model.phase == .prefill, !reduceMotion {
                        PulsingIndicatorDot()
                    } else {
                        Circle()
                            .fill(.appAccent)
                            .frame(width: 6, height: 6)
                    }
                    statusText
                        .font(theme.ui(.tiny, weight: .medium))
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(TurboSparkTheme.surfaceColor, in: Capsule())
                .help(statusText)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(Text("Model status", bundle: .module))
                .accessibilityValue(statusText)
            }
        }
    }

    /// Nil while idle: a bar that shows "Ready" forever is chrome that never
    /// changes, and the status strip already carries the loaded/not signal.
    private var statusText: Text? {
        if model.opening { return Text("Loading model", bundle: .module) }
        switch model.phase {
        case .prefill:
            return Text("Prefill \(model.livePrefillDone)/\(max(model.livePrefillTotal, 1))", bundle: .module)
        case .decode:
            return Text("Generating", bundle: .module)
        case .idle:
            return nil
        }
    }
}

private struct PulsingIndicatorDot: View {
    var body: some View {
        Circle()
            .fill(.appAccent)
            .frame(width: 6, height: 6)
            .phaseAnimator([0.4, 1.0]) { dot, opacity in
                dot.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.7)
            }
    }
}
