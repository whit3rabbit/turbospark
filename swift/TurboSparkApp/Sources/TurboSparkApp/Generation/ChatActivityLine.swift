import SwiftUI

/// What a running reply is doing, in the order the activity line checks it.
///
/// Pure so the priority order is testable without a model: the old footer
/// spread the same decision over three views and the composer pill, and they
/// disagreed (the pill read "0.0 tok/s" while the footer said "Running
/// tools...").
enum ChatActivityStatus: Equatable {
    case stopping
    case escArmed
    case needsConfirmation(tool: String)
    case imageWaiting
    case imageStage(stage: String?, completed: Int, total: Int, fraction: Double?)
    /// `total` is 0 until the first prefill event lands, which is what the
    /// old pill printed as "0/1". The line shows an indeterminate bar and no
    /// count until then.
    case readingPrompt(done: Int, total: Int)
    case runningTool(name: String)
    case task(String)
    case thinking(latestLine: String?)
    case generating

    struct Inputs {
        var isCancellationPending = false
        var isEscCancelArmed = false
        var pendingToolName: String?
        var imageJob: (waiting: Bool, stage: String?, completed: Int, total: Int, fraction: Double?)?
        var isPrefill = false
        var prefillDone = 0
        var prefillTotal = 0
        var runningToolName: String?
        var activeTask: String?
        var reasoning = ""
        var hasOutput = false
    }

    static func resolve(_ inputs: Inputs) -> ChatActivityStatus {
        if inputs.isCancellationPending { return .stopping }
        if inputs.isEscCancelArmed { return .escArmed }
        if let tool = inputs.pendingToolName { return .needsConfirmation(tool: tool) }
        if let job = inputs.imageJob {
            if job.waiting { return .imageWaiting }
            return .imageStage(
                stage: job.stage, completed: job.completed, total: job.total, fraction: job.fraction)
        }
        if inputs.isPrefill {
            return .readingPrompt(done: inputs.prefillDone, total: inputs.prefillTotal)
        }
        if let name = inputs.runningToolName { return .runningTool(name: name) }
        if let task = inputs.activeTask, !task.isEmpty { return .task(task) }
        if !inputs.reasoning.isEmpty && !inputs.hasOutput {
            return .thinking(latestLine: latestLine(of: inputs.reasoning))
        }
        return .generating
    }

    /// The last non-blank line of a reasoning trace, for the one-line preview.
    static func latestLine(of text: String) -> String? {
        text.split(whereSeparator: \.isNewline)
            .last { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }

    /// Elapsed run time as the line shows it: "7s", then "1m 5s". Units are
    /// verbatim, like the other machine readouts.
    static func elapsedText(seconds: Int) -> String {
        let seconds = max(0, seconds)
        if seconds < 60 { return "\(seconds)s" }
        return "\(seconds / 60)m \(seconds % 60)s"
    }
}

/// The one-line header of an assistant reply: the model, what the reply is
/// doing, and how long it has run.
///
/// While a turn runs this single row replaces three that said the same
/// thing: the waiting row, the footer under it with its rotating phrase, and
/// the composer's wide "Reading N/M" pill. When the turn ends the same view
/// stays as the reply's header, so nothing changes shape as the answer lands.
///
/// When the reply has reasoning the row is a disclosure whose chevron opens
/// the trace below it. Live reasoning starts closed, so the transcript shows
/// one line while the model thinks, with the latest line of the trace as
/// `detail`.
struct ChatActivityLine: View {
    enum Tone {
        case normal
        /// Needs the user (an approval, a second Esc).
        case attention
    }

    @Environment(\.appTheme) private var theme

    /// The model alias. Nil for a reply row whose turn header already names
    /// the model (a tool-call step).
    var modelName: String?
    /// The mark breathes and the elapsed time ticks only while live.
    var isLive = false
    var label: Text?
    var tone: Tone = .normal
    /// One truncated line after the label. Model output or a tool id, so it
    /// renders verbatim.
    var detail: String?
    var detailIsCode = false
    /// `.some(nil)` draws an indeterminate bar, nil draws none.
    var progress: Double?? = nil
    /// Trailing machine facts (token counts), set in the code face.
    var counts: Text?
    var startedAt: Date?
    /// Set when there is reasoning to reveal.
    var reasoningExpanded: Binding<Bool>?

    var body: some View {
        if let reasoningExpanded {
            Button {
                TSMotion.animate(TSMotion.select) { reasoningExpanded.wrappedValue.toggle() }
            } label: {
                row(chevronOpen: reasoningExpanded.wrappedValue)
            }
            .buttonStyle(.plain)
            .help(Text("Model thought process and reasoning trace", bundle: .module))
            .accessibilityValue(reasoningExpanded.wrappedValue
                ? Text("Expanded", bundle: .module) : Text("Collapsed", bundle: .module))
            .accessibilityHint(Text("Expands to reveal the model's chain-of-thought reasoning", bundle: .module))
        } else {
            row(chevronOpen: nil)
                .accessibilityElement(children: .combine)
        }
    }

    private var labelStyle: AnyShapeStyle {
        tone == .attention ? AnyShapeStyle(Color.orange) : AnyShapeStyle(.appText)
    }

    private func row(chevronOpen: Bool?) -> some View {
        HStack(spacing: 6) {
            Image(systemName: tone == .attention ? "exclamationmark.triangle.fill" : "sparkles")
                .themedFont(.small, weight: .bold)
                .foregroundStyle(tone == .attention ? AnyShapeStyle(Color.orange) : AnyShapeStyle(.appAccent))
                .tsPulse(active: isLive && tone == .normal)
                .accessibilityHidden(true)

            if let modelName {
                Text(verbatim: modelName)
                    .themedCode(.small)
                    .foregroundStyle(theme.metadataForeground)
                    .lineLimit(1)
                    .layoutPriority(2)
            }

            if let label {
                if modelName != nil {
                    Text(verbatim: "\u{00B7}")
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                }
                label
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(labelStyle)
                    .lineLimit(1)
                    .layoutPriority(2)
            }

            if let progress, let label {
                AppProgressBar(value: progress, label: label, size: .compact)
                    .frame(minWidth: 32, idealWidth: 72, maxWidth: 72 * AppMetrics.textScale(theme))
            }

            if let detail, !detail.isEmpty {
                Group {
                    if detailIsCode {
                        Text(verbatim: detail).themedCode(.small).foregroundStyle(.appText)
                    } else {
                        Text(verbatim: detail).themedFont(.small).foregroundStyle(.appSecondary)
                    }
                }
                .lineLimit(1)
                .truncationMode(.tail)
            }

            if let counts {
                counts
                    .themedCode(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                    .layoutPriority(1)
            }

            if isLive, let startedAt {
                TimelineView(.periodic(from: startedAt, by: 1)) { context in
                    Text(verbatim: ChatActivityStatus.elapsedText(
                        seconds: Int(context.date.timeIntervalSince(startedAt))))
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                        .monospacedDigit()
                }
                .layoutPriority(1)
                .accessibilityLabel(Text("Elapsed time", bundle: .module))
            }

            if let chevronOpen {
                Image(systemName: "chevron.right")
                    .themedFont(.micro, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                    .rotationEffect(.degrees(chevronOpen ? 90 : 0))
                    .accessibilityHidden(true)
            }

            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

/// The reasoning trace a reply's activity line reveals.
struct ReasoningTraceView: View {
    @Environment(\.appTheme) private var theme
    let reasoning: String

    var body: some View {
        Text(reasoning)
            .font(theme.code(.large))
            .foregroundStyle(theme.metadataForeground)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.03))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .textSelection(.enabled)
            .transition(.opacity)
    }
}
