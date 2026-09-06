import AppKit
import SwiftUI

/// Coalesced group view displaying multiple sequential tool executions in Unsloth Studio style.
@MainActor
struct ToolGroupView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let toolCalls: [AppToolCall]
    let toolResults: [AppToolResult]

    @State private var isManuallyExpanded: Bool? = nil

    private var hasRunningTool: Bool {
        toolCalls.contains(where: { $0.status == .running })
    }

    private var hasPendingApproval: Bool {
        toolCalls.contains(where: { call in
            call.status == .pendingApproval || (model.pendingToolCall?.id == call.id)
        })
    }

    private var hasFailedTool: Bool {
        toolCalls.contains(where: { $0.status == .failed }) ||
            toolResults.contains(where: { $0.isError })
    }

    private var isExpanded: Bool {
        // Auto-expand during active approval, overriding prior manual collapse.
        if hasPendingApproval {
            return true
        }
        if let manual = isManuallyExpanded {
            return manual
        }
        // Auto-expand during active work, collapse when done.
        return hasRunningTool
    }

    private var groupLabel: String {
        let count = toolCalls.count
        return "\(count) tool \(count == 1 ? "call" : "calls")"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            triggerButton

            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(toolCalls, id: \.id) { (call: AppToolCall) in
                        let matchResult = toolResults.first(where: { $0.callID == call.id })
                        ToolCallCardView(
                            model: model,
                            call: call,
                            result: matchResult,
                            isNestedInGroup: true
                        )
                    }
                }
                .padding(.leading, 12)
                .overlay(
                    Rectangle()
                        .fill(Color.primary.opacity(0.12))
                        .frame(width: 1.5),
                    alignment: .leading
                )
                .padding(.top, 2)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var triggerButton: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.18)) {
                isManuallyExpanded = !isExpanded
            }
        } label: {
            HStack(spacing: 8) {
                if hasRunningTool {
                    TaskProgressFlameIcon(size: 13)
                } else if hasFailedTool {
                    Image(systemName: "exclamationmark.circle")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Color.red)
                } else {
                    Image(systemName: "wrench.and.screwdriver")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(TurboSparkTheme.accentColor)
                }

                Text(groupLabel)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.primary)

                if hasRunningTool {
                    Text("running...")
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(TurboSparkTheme.accentColor)
                } else if hasPendingApproval {
                    Text("approval required")
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(Color.orange)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.tertiary)
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.2), lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(groupLabel)")
        .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
    }
}
