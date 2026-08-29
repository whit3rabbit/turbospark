import SwiftUI

struct GenerateControl: View {
    @ObservedObject var model: AppModel
    @ScaledMetric private var controlHeight: CGFloat = 34
    @ScaledMetric private var generateMinWidth: CGFloat = 124
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
        Button {
            model.run()
        } label: {
            Label("Generate", systemImage: "arrow.up")
                .font(.callout.weight(.semibold))
                .padding(.horizontal, 24)
                .frame(minWidth: generateMinWidth, minHeight: controlHeight)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(TurboSparkTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canRun)
        .opacity(model.canRun ? 1 : 0.62)
        .help(model.canRun ? "Generate (⌘↩)" : "Generate (disabled)")
    }

    private var runningPill: some View {
        Button {
            model.cancel()
        } label: {
            HStack(spacing: 10) {
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
                Label("Stop generation", systemImage: "stop.fill")
                    .labelStyle(.iconOnly)
                    .font(.callout)
                    .frame(width: stopIconSize, height: stopIconSize)
                    .accessibilityHidden(true)
            }
            .padding(.leading, 18)
            .padding(.trailing, 4)
            .frame(minWidth: pillMinWidth, minHeight: controlHeight)
            .contentShape(Capsule())
        }

        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(TurboSparkTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(".", modifiers: .command)
        .disabled(!model.canCancel)
        .help("Stop generation (⌘.)")
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
