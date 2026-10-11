import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Primary generation action and live progress/cancellation pill.
///
/// Two changes from the previous version:
///
/// 1. The send button's fill is the THEME accent, not a hardcoded
///    `Color(red: 0.08, green: 0.65, blue: 0.50)`. That literal was the one
///    place in the composer that ignored the user's accent choice, so a
///    green send button sat inside an otherwise monochrome (or blue, or
///    amber) app. `contrastForeground` picks the glyph colour, so a light
///    accent still gets dark ink.
/// 2. The running pill carries the prefill fraction as a ring rather than as
///    "Reading 41/512" text alone. The number is still there for anyone
///    reading it, but progress you can see at a glance is what the state
///    actually needs.
///
/// The disabled state keeps its low-contrast fill on purpose: `canRun` is
/// false when there is no draft, and a fully saturated button that does
/// nothing is worse than a quiet one.
@MainActor
struct GenerateControl: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @ScaledMetric private var controlHeight: CGFloat = 34
    @ScaledMetric private var circularButtonSize: CGFloat = 31
    @ScaledMetric private var pillMinWidth: CGFloat = 140

    var body: some View {
        Group {
            if model.isRunning {
                runningPill
            } else {
                generateButton
            }
        }
        // The swap between the two is a real change of state, not a redraw:
        // animating it stops the footer from flickering at the moment a turn
        // starts or ends.
        .tsAnimation(TSMotion.select, value: model.isRunning)
    }

    // MARK: - Idle

    private var generateButton: some View {
        let ready = model.canRun
        let fill = ready
            ? AnyShapeStyle(theme.accent.gradient)
            : AnyShapeStyle(theme.foreground.opacity(0.08))
        let ink = ready ? theme.accent.contrastForeground : theme.secondaryText.opacity(0.5)

        return Button {
            model.run()
        } label: {
            Image(systemName: "arrow.up")
                .themedFont(.callout, weight: .bold)
                .foregroundStyle(ink)
                .frame(width: circularButtonSize, height: circularButtonSize)
                .background(Circle().fill(fill))
                .overlay {
                    Circle().stroke(
                        ready ? Color.white.opacity(0.2) : theme.foreground.opacity(0.06),
                        lineWidth: 0.5)
                }
                // Only the enabled button gets a shadow. It is the affordance
                // that says "this will do something".
                .shadow(
                    color: ready ? theme.accent.opacity(0.45) : .clear,
                    radius: 8, y: 2)
                .contentShape(Circle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!ready)
        .tsAnimation(TSMotion.hover, value: ready)
        .help(ready
            ? Text("Generate (Cmd+Return)", bundle: .module)
            : Text("Type a message first", bundle: .module))
        .accessibilityLabel(Text("Generate", bundle: .module))
    }

    // MARK: - Running

    private var runningPill: some View {
        let ink = theme.accent.contrastForeground
        return Button {
            model.cancel()
        } label: {
            HStack(spacing: 9) {
                TaskProgressFlameIcon(size: 14)

                statusLabel
                    .foregroundStyle(ink)

                Spacer(minLength: 2)

                ZStack {
                    // The stop target doubles as the prefill meter. During
                    // decode there is no honest fraction to draw, so the ring
                    // is simply absent rather than faked.
                    if let fraction = prefillFraction {
                        Circle()
                            .stroke(ink.opacity(0.22), lineWidth: 2)
                            .frame(width: 22, height: 22)
                        Circle()
                            .trim(from: 0, to: fraction)
                            .stroke(ink, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                            .frame(width: 22, height: 22)
                            .tsAnimation(TSMotion.select, value: fraction)
                    } else {
                        Circle()
                            .fill(ink.opacity(0.16))
                            .frame(width: 22, height: 22)
                    }
                    Image(systemName: "stop.fill")
                        .themedFont(.micro, weight: .bold)
                        .foregroundStyle(ink)
                }
                .accessibilityHidden(true)
            }
            .padding(.leading, 14)
            .padding(.trailing, 6)
            .frame(minWidth: pillMinWidth, minHeight: controlHeight)
            .contentShape(Capsule())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.97))
        .background(theme.accent, in: .capsule)
        .overlay { Capsule().stroke(ink.opacity(0.16), lineWidth: 0.5) }
        .keyboardShortcut(".", modifiers: .command)
        .disabled(!model.canCancel)
        .help(Text("Stop generation (Cmd+.)", bundle: .module))
        // The pill combines live status text and a stop icon. VoiceOver
        // should hear the live value as the action's current state and know
        // the button is "Stop", not whatever symbol happens to be on it.
        .accessibilityLabel(Text("Stop generation", bundle: .module))
        .accessibilityValue(stopButtonStatusText)
    }

    @ViewBuilder
    private var statusLabel: some View {
        if model.isCancellationPending {
            Text("Stopping", bundle: .module)
                .themedFont(.base, weight: .medium)
        } else if model.imageModeEnabled, let job = model.imageJob {
            if job.status == .waiting {
                Text("Waiting for image slot", bundle: .module)
                    .themedFont(.base, weight: .medium)
                    .lineLimit(1)
            } else {
                Group {
                    if let stage = job.stage {
                        Self.imageStageText(stage)
                            + Text(verbatim: " \(job.completed)/\(max(job.total, 1))")
                    } else {
                        Text("Generating image", bundle: .module)
                    }
                }
                .themedFont(.base, weight: .medium)
                .lineLimit(1)
            }
        } else if model.phase == .prefill {
            Text("Reading \(model.livePrefillDone)/\(max(model.livePrefillTotal, 1))", bundle: .module)
                .themedFont(.base, weight: .medium)
                .monospacedDigit()
                .contentTransition(.numericText())
        } else {
            (Text(verbatim: String(format: "%.1f ", model.liveTokensPerSecond)) + Text("tok/s", bundle: .module))
                .themedFont(.base, weight: .semibold)
                .monospacedDigit()
                // Digits roll rather than jump. This is the one per-token
                // animation in the app and it is on a 3-character label, not
                // on layout.
                .contentTransition(.numericText())
        }
    }

    /// Nil unless prefill is running against a known total.
    private var prefillFraction: CGFloat? {
        if model.imageModeEnabled, let fraction = model.imageProgressFraction {
            return CGFloat(fraction)
        }
        guard model.phase == .prefill, model.livePrefillTotal > 0 else { return nil }
        return min(1, CGFloat(model.livePrefillDone) / CGFloat(model.livePrefillTotal))
    }

    private var stopButtonStatusText: Text {
        if model.isCancellationPending { return Text("Stopping", bundle: .module) }
        if model.imageModeEnabled, let job = model.imageJob {
            if job.status == .waiting {
                return Text("Waiting for image slot", bundle: .module)
            }
            if let stage = job.stage {
                return Self.imageStageText(stage) + Text(verbatim: ", ")
                    + Text("\(job.completed) of \(max(job.total, 1))", bundle: .module)
            }
            return Text("Generating image", bundle: .module)
        }
        switch model.phase {
        case .prefill:
            return Text("Reading \(model.livePrefillDone) of \(max(model.livePrefillTotal, 1))", bundle: .module)
        case .decode:
            return Text("\(String(format: "%.1f", model.liveTokensPerSecond)) tokens per second", bundle: .module)
        case .idle:
            return Text("Running", bundle: .module)
        }
    }

    /// `AppImageJob.stage` holds the pipeline's snake_case stage id, which
    /// this pill used to print raw ("transformer 3/20"). Same names as the
    /// Images pane's progress card, so the two surfaces agree.
    static func imageStageText(_ stage: String) -> Text {
        switch stage {
        case "downloading_model": return Text("Downloading model files", bundle: .module)
        case "loading_tokenizer": return Text("Loading tokenizer", bundle: .module)
        case "loading_text_encoder": return Text("Loading text encoder", bundle: .module)
        case "loading_transformer": return Text("Loading transformer", bundle: .module)
        case "loading_vae": return Text("Loading VAE", bundle: .module)
        case "text_encoder": return Text("Encoding prompt", bundle: .module)
        case "transformer": return Text("Denoising", bundle: .module)
        case "vae_decoder": return Text("Decoding image", bundle: .module)
        case "png_encode": return Text("Saving image", bundle: .module)
        default: return Text("Loading model", bundle: .module)
        }
    }
}
