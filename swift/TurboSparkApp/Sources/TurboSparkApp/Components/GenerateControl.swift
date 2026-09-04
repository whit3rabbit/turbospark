import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Primary generation action button and live progress/cancellation pill.
///
/// Switches between "Generate" (Cmd+Return) when idle and a live progress pill
/// showing tokens/sec or prefill counts with a Stop button (Cmd+.) while running.
@MainActor
struct GenerateControl: View {
    @ObservedObject var model: AppModel
    @ScaledMetric private var controlHeight: CGFloat = 34
    @ScaledMetric private var circularButtonSize: CGFloat = 30
    @ScaledMetric private var pillMinWidth: CGFloat = 140
    @ScaledMetric private var stopIconSize: CGFloat = 28

    var body: some View {
        if model.isRunning {
            runningPill
        } else {
            generateButton
        }
    }

    private var generateButton: some View {
        let fgColor = model.canRun ? TurboSparkTheme.accentColor.contrastForeground : Color.secondary.opacity(0.6)
        return Button {
            model.run()
        } label: {
            Image(systemName: "arrow.up")
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(fgColor)
                .frame(width: circularButtonSize, height: circularButtonSize)
                .background(
                    Circle()
                        .fill(model.canRun ? TurboSparkTheme.accentColor : Color.primary.opacity(0.08))
                )
                .overlay {
                    Circle().stroke(model.canRun ? TurboSparkTheme.accentColor.contrastForeground.opacity(0.16) : Color.primary.opacity(0.06), lineWidth: 0.5)
                }
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canRun)
        .help(model.canRun ? "Generate (Cmd+Return)" : "Generate (disabled)")
        .accessibilityLabel("Generate")
    }

    private var runningPill: some View {
        let contrastFg = TurboSparkTheme.accentColor.contrastForeground
        return Button {
            model.cancel()
        } label: {
            HStack(spacing: 8) {
                TaskProgressFlameIcon(size: 14)
                if model.isCancellationPending {
                    Text("Stopping")
                        .font(.callout.weight(.medium))
                } else if model.phase == .prefill {
                    Text("Reading \(model.livePrefillDone)/\(max(model.livePrefillTotal, 1))")
                        .font(.callout.weight(.medium))
                        .monospacedDigit()
                } else {
                    Text(String(format: "%.1f tok/s", model.liveTokensPerSecond))
                        .font(.callout.weight(.semibold))
                        .monospacedDigit()
                }
                ZStack {
                    Circle()
                        .fill(contrastFg.opacity(0.16))
                        .frame(width: 22, height: 22)
                    Image(systemName: "stop.fill")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(contrastFg)
                }
                .accessibilityHidden(true)
            }
            .padding(.leading, 14)
            .padding(.trailing, 6)
            .frame(minWidth: pillMinWidth, minHeight: controlHeight)
            .contentShape(Capsule())
        }

        .buttonStyle(.plain)
        .foregroundStyle(contrastFg)
        .background(TurboSparkTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(contrastFg.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(".", modifiers: .command)
        .disabled(!model.canCancel)
        .help("Stop generation (Cmd+.)")
        // The pill combines live status text and a stop icon. VoiceOver
        // should hear the live value as the action's current state and
        // know the button is "Stop", not whatever symbol happens to be on it.
        .accessibilityLabel("Stop generation")
        .accessibilityValue(stopButtonStatusText)
    }

    private var stopButtonStatusText: String {
        if model.isCancellationPending { return "Stopping" }
        switch model.phase {
        case .prefill:
            return "Reading \(model.livePrefillDone) of \(max(model.livePrefillTotal, 1))"
        case .decode:
            return String(format: "%.1f tokens per second", model.liveTokensPerSecond)
        case .idle:
            return "Running"
        }
    }
}
