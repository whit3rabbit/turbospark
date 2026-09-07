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

    private var statusText: String {
        if isRunning { return "Turn \(state.turns)" }
        return state.status
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
                                        .foregroundStyle(row.isError ? Color.red : Color.green)
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
                            .foregroundStyle(.primary.opacity(0.85))
                            .lineLimit(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }

                    if !isRunning, let result = state.result, state.mode == .background {
                        Text(result.finalResponse)
                            .font(theme.code(.small))
                            .foregroundStyle(.primary.opacity(0.85))
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
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(isRunning
                    ? TurboSparkTheme.accentColor.opacity(0.45)
                    : Color(nsColor: .separatorColor).opacity(0.35),
                    lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Subagent \(state.displayName), \(statusText)")
    }

    private var header: some View {
        HStack(spacing: 6) {
            Button {
                withAnimation(.easeInOut(duration: 0.12)) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "person.2.wave.2")
                        .themedFont(.small, weight: .bold)
                        .foregroundStyle(TurboSparkTheme.accentColor)
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
            .accessibilityLabel("\(isExpanded ? "Collapse" : "Expand") subagent \(state.displayName)")

            Spacer(minLength: 8)

            if isRunning {
                TaskProgressFlameIcon(size: 14)
                Text(statusText)
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
                    .foregroundStyle(Color.red.opacity(0.8))
                    .help("Stop this background subagent")
                    .accessibilityLabel("Stop subagent \(state.displayName)")
                }
            } else {
                Text(statusText)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(statusColor)
                if let onDismiss {
                    Button(action: onDismiss) {
                        Image(systemName: "xmark")
                            .themedFont(.tiny)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(theme.metadataForeground)
                    .help("Dismiss this finished subagent")
                    .accessibilityLabel("Dismiss subagent \(state.displayName)")
                }
            }
        }
    }

    private var statusColor: Color {
        switch state.status {
        case "completed": return .green
        case "killed", "cancelled": return .orange
        case "error", "failed": return .red
        default: return theme.metadataForeground
        }
    }

    /// The tail of the streamed text, so a long subagent turn shows its
    /// current end rather than its beginning.
    private var streamedTail: String {
        guard state.streamedText.count > 4_000 else { return state.streamedText }
        return String(state.streamedText.suffix(4_000))
    }
}
