import AppKit
import SwiftUI

/// Coalesced group view displaying multiple sequential tool executions in Unsloth Studio style.
@MainActor
struct ToolGroupView: View {
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
                            result: matchResult
                        )
                    }
                }
                .padding(.top, 2)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var triggerButton: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.18)) {
                // The approval force-open overrides the manual flag, so
                // recording a toggle here would store an invisible value
                // (the click looks dead, the group stays open) that then
                // wins the moment the approval resolves. Ignore clicks for
                // as long as the force holds.
                if !hasPendingApproval {
                    isManuallyExpanded = !isExpanded
                }
            }
        } label: {
            HStack(spacing: 7) {
                Group {
                    if hasRunningTool {
                        TaskProgressFlameIcon(size: 13)
                    } else if hasFailedTool {
                        Image(systemName: "exclamationmark.circle")
                            .themedFont(.small, weight: .semibold)
                            .foregroundStyle(Color.red)
                    } else {
                        Image(systemName: "wrench.and.screwdriver")
                            .themedFont(.small, weight: .semibold)
                            .foregroundStyle(.appAccent)
                    }
                }
                .frame(width: ConversationLayout.activityIconWidth)

                Text(groupLabel)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appText)

                if hasRunningTool {
                    Text("running...", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(.appAccent)
                } else if hasPendingApproval {
                    Text("approval required", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(Color.orange)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .themedFont(.tiny, weight: .bold)
                    .foregroundStyle(.tertiary)
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
            }
            .frame(minHeight: ConversationLayout.activityHeight)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: ConversationLayout.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: ConversationLayout.cardRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(groupLabel)")
        .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
    }
}
