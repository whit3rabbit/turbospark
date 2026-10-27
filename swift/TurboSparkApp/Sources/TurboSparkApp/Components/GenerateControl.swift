import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Primary generation action: Send while idle, Stop while a reply runs.
///
/// Both are the same circle, so the composer toolbar never reflows when a
/// turn starts. The old running state was a wide accent pill that repeated
/// the phase in text ("Reading 41/512", "38.0 tok/s"); the transcript's
/// activity line now owns that text, and Stop carries only a ring: the
/// prefill (or image) fraction when there is one, a slow sweep otherwise.
///
/// The send fill is the THEME accent, never a literal, and
/// `contrastForeground` picks the glyph colour so a light accent still gets
/// dark ink. The disabled state keeps its low-contrast fill on purpose:
/// `canRun` is false when there is no draft, and a fully saturated button
/// that does nothing is worse than a quiet one.
@MainActor
struct GenerateControl: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @ScaledMetric private var circularButtonSize: CGFloat = 31
    /// How far the progress ring sits outside the Stop circle. Drawn as an
    /// overlay, so it never changes the control's frame.
    @ScaledMetric private var ringOutset: CGFloat = 4

    private var buttonSize: CGFloat {
        (circularButtonSize * AppMetrics.textScale(theme)).rounded()
    }

    var body: some View {
        Group {
            if model.isRunning {
                stopButton
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
                .frame(width: buttonSize, height: buttonSize)
                .background(Circle().fill(fill))
                .overlay {
                    Circle().stroke(
                        ready ? Color.white.opacity(0.2) : theme.foreground.opacity(0.06),
                        lineWidth: 0.5)
                }
                .appPrimaryGlow(ready)
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

    private var stopButton: some View {
        let ink = theme.accent.contrastForeground
        return Button {
            model.cancel()
        } label: {
            Image(systemName: "stop.fill")
                .themedFont(.tiny, weight: .bold)
                .foregroundStyle(ink)
                .frame(width: buttonSize, height: buttonSize)
                .background(Circle().fill(theme.accent))
                .overlay {
                    // Waiting on the user (an approval) or on the stop is
                    // not progress, so the sweep holds still.
                    StopProgressRing(
                        fraction: prefillFraction,
                        isMoving: !model.isCancellationPending && model.pendingToolCall == nil)
                        .padding(-ringOutset)
                }
                .contentShape(Circle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .keyboardShortcut(".", modifiers: .command)
        .disabled(!model.canCancel)
        .help(Text("Stop generation (Cmd+.)", bundle: .module))
        // VoiceOver hears the action as "Stop" and the live phase as its
        // value, since the ring alone says nothing to a listener.
        .accessibilityLabel(Text("Stop generation", bundle: .module))
        .accessibilityValue(stopButtonStatusText)
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

/// The ring around Stop. A known fraction fills it; otherwise a quarter arc
/// sweeps once per `TSMotion.sweep` and snaps back invisibly (a full turn
/// looks the same as none). Under Reduce Motion, or while nothing is
/// progressing, the arc holds still as a half ring, so a moving ring always
/// means work is moving.
private struct StopProgressRing: View {
    let fraction: CGFloat?
    let isMoving: Bool
    @Environment(\.appTheme) private var theme

    var body: some View {
        ZStack {
            Circle()
                .stroke(theme.foreground.opacity(0.10), lineWidth: 2)
            if let fraction {
                arc(to: fraction)
                    .tsAnimation(TSMotion.select, value: fraction)
            } else {
                TSMotionContext { reduceMotion in
                    if reduceMotion || !isMoving {
                        arc(to: 0.5)
                    } else {
                        arc(to: 0.25)
                            .phaseAnimator([false, true]) { view, turned in
                                view.rotationEffect(.degrees(turned ? 360 : 0))
                            } animation: { turned in
                                turned ? TSMotion.sweep : TSMotion.snap
                            }
                    }
                }
            }
        }
        // Drawn clockwise from the top and mirrored as a whole, like
        // `AppProgressBar`, so right-to-left reads the fill the other way.
        .environment(\.layoutDirection, .leftToRight)
        .flipsForRightToLeftLayoutDirection(true)
        .accessibilityHidden(true)
    }

    private func arc(to end: CGFloat) -> some View {
        Circle()
            .trim(from: 0, to: end)
            .stroke(theme.accent, style: StrokeStyle(lineWidth: 2, lineCap: .round))
            .rotationEffect(.degrees(-90))
    }
}
