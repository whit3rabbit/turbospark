import SwiftUI

/// Live card for one subagent run: foreground runs stream in the active
/// turn's row, background runs persist in a strip under the transcript
/// until dismissed. The transcript's own `ToolCallCardView` remains the
/// durable record for foreground runs; a background run's card (with its
/// final result) is the only record its chat has.
struct SubagentLiveCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var state: SubagentRunState
    /// Set only for background runs: asks the model layer to stop the task.
    var onStop: (() -> Void)? = nil
    /// Set only for finished background runs: clears the card.
    var onDismiss: (() -> Void)? = nil

    @State private var isExpanded: Bool = true

    private var isRunning: Bool { state.status == "running" }

    /// The run's status in the UI language. The engine reports a small set
    /// of lowercase states; one it adds later shows as sent rather than
    /// vanishing.
    private var statusText: Text {
        if isRunning { return Text("Turn \(state.turns)", bundle: .module) }
        switch state.status {
        case "completed": return Text("Completed", bundle: .module)
        case "killed": return Text("Killed", bundle: .module)
        case "cancelled": return Text("Cancelled", bundle: .module)
        case "error": return Text("Error", bundle: .module)
        case "failed": return Text("Failed", bundle: .module)
        default: return Text(verbatim: state.status)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header

            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    if !state.promptHead.isEmpty {
                        Text(state.promptHead)
                            .font(theme.code(.small))
                            .foregroundStyle(theme.metadataForeground)
                            .lineLimit(2)
                    }

                    if !state.toolRows.isEmpty {
                        ForEach(state.toolRows) { row in
                            HStack(alignment: .top, spacing: 6) {
                                if row.isRunning {
                                    TaskProgressFlameIcon(size: 11)
                                        .accessibilityHidden(true)
                                } else {
                                    Image(systemName: row.isError
                                            ? "xmark.circle.fill" : "checkmark.circle.fill")
                                        .themedFont(.tiny)
                                        .foregroundStyle(row.isError ? AnyShapeStyle(AppAttention.color) : AnyShapeStyle(.appSuccess))
                                        .accessibilityHidden(true)
                                }
                                Text(row.name)
                                    .font(theme.code(.small))
                                Text(row.summary)
                                    .themedFont(.small)
                                    .foregroundStyle(theme.metadataForeground)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                            }
                        }
                    }

                    let text = streamedTail
                    if !text.isEmpty {
                        Text(text)
                            .font(theme.code(.small))
                            .foregroundStyle(.appText)
                            .lineLimit(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }

                    if !isRunning, let result = state.result, state.mode == .background {
                        Text(result.finalResponse)
                            .font(theme.code(.small))
                            .foregroundStyle(.appText)
                            .lineLimit(20)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                }
                .padding(.top, 2)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.appSurface)
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(isRunning
                    ? theme.accent.opacity(0.45)
                    : theme.border,
                    lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text("Subagent \(state.displayName), \(statusText)", bundle: .module))
    }

    private var header: some View {
        HStack(spacing: 6) {
            Button {
                TSMotion.animate(TSMotion.select) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "person.2.wave.2")
                        .themedFont(.small, weight: .bold)
                        .foregroundStyle(.appAccent)
                        .accessibilityHidden(true)
                    Text(state.displayName)
                        .themedFont(.small, weight: .semibold)
                    if !state.taskDescription.isEmpty {
                        Text(state.taskDescription)
                            .themedFont(.small)
                            .foregroundStyle(theme.metadataForeground)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel(isExpanded
                                ? Text("Collapse subagent \(state.displayName)", bundle: .module)
                                : Text("Expand subagent \(state.displayName)", bundle: .module))

            Spacer(minLength: 8)

            if isRunning {
                TaskProgressFlameIcon(size: 14)
                statusText
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(theme.metadataForeground)
                Text(state.startedAt, style: .timer)
                    .themedFont(.small)
                    .foregroundStyle(theme.metadataForeground)
                if let onStop {
                    Button(action: onStop) {
                        Image(systemName: "stop.fill")
                            .themedFont(.tiny)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(AppAttention.color)
                    .help(Text("Stop this background subagent", bundle: .module))
                    .accessibilityLabel(Text("Stop subagent \(state.displayName)", bundle: .module))
                }
            } else {
                statusText
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(statusColor)
                if let onDismiss {
                    Button(action: onDismiss) {
                        Image(systemName: "xmark")
                            .themedFont(.tiny)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(theme.metadataForeground)
                    .help(Text("Dismiss this finished subagent", bundle: .module))
                    .accessibilityLabel(Text("Dismiss subagent \(state.displayName)", bundle: .module))
                }
            }
        }
    }

    private var statusColor: Color {
        switch state.status {
        case "completed": return theme.success
        case "killed", "cancelled": return theme.warning
        case "error", "failed": return theme.danger
        default: return theme.metadataForeground
        }
    }

    /// The tail of the streamed text, so a long subagent turn shows its
    /// current end rather than its beginning.
    private var streamedTail: String {
        // utf8.count is O(1); only pay the grapheme walk when it could matter.
        guard state.streamedText.utf8.count > 4_000, state.streamedText.count > 4_000 else {
            return state.streamedText
        }
        return String(state.streamedText.suffix(4_000))
    }
}
