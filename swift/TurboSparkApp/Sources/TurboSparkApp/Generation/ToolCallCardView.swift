import AppKit
import SwiftUI

struct ToolCallCardView: View {
    @ObservedObject var model: AppModel
    let call: AppToolCall
    let result: AppToolResult?

    @State private var isOutputExpanded: Bool = true
    @State private var isCopied: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            argumentsPreview
            if let risk = call.riskAssessment, !risk.reasons.isEmpty, isPendingApproval {
                riskWarningBox(risk)
            }
            if isPendingApproval {
                approvalPrompt
            }
            if let result {
                outputSection(result)
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.8))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(borderColor, lineWidth: 1)
        )
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var isPendingApproval: Bool {
        call.status == .pendingApproval || (model.pendingToolCall?.id == call.id)
    }

    private var isHighRisk: Bool {
        call.riskAssessment?.isHighRisk ?? false
    }

    private var borderColor: Color {
        if isPendingApproval {
            return isHighRisk ? Color.red.opacity(0.7) : Color.orange.opacity(0.6)
        }
        if let result, result.isError {
            return Color.red.opacity(0.4)
        }
        return Color(nsColor: .separatorColor).opacity(0.4)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: call.category.systemImage)
                .font(.callout)
                .foregroundStyle(TurboSparkTheme.accentColor)
                .accessibilityHidden(true)

            Text(call.name)
                .font(.callout.weight(.semibold).monospaced())

            if let risk = call.riskAssessment, risk.level != .safe {
                riskBadge(risk)
            }

            Spacer()

            statusBadge
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Tool call \(call.name)")
        .accessibilityValue(statusLabel)
    }

    private func riskBadge(_ risk: ToolRiskAssessment) -> some View {
        HStack(spacing: 4) {
            Image(systemName: risk.level.systemImage)
                .font(.caption2)
            Text(risk.level.label)
                .font(.caption2.weight(.semibold))
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(risk.level == .high ? Color.red.opacity(0.15) : Color.orange.opacity(0.12), in: Capsule())
        .foregroundStyle(risk.level == .high ? Color.red : Color.orange)
    }

    private func riskWarningBox(_ risk: ToolRiskAssessment) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(risk.level == .high ? Color.red : Color.orange)
                    .font(.caption)
                Text(risk.level == .high ? "High-Risk Operation Detected" : "Security Notice")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(risk.level == .high ? Color.red : Color.orange)
            }
            ForEach(risk.reasons, id: \.self) { reason in
                Text("• \(reason)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            (risk.level == .high ? Color.red : Color.orange).opacity(0.08),
            in: RoundedRectangle(cornerRadius: 6)
        )
    }

    private var statusBadge: some View {
        HStack(spacing: 4) {
            if call.status == .running {
                ProgressView().controlSize(.small)
            }
            Text(statusLabel)
                .font(.caption2.weight(.medium))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(statusBackground, in: Capsule())
        .foregroundStyle(statusForeground)
    }

    private var statusLabel: String {
        if isPendingApproval { return "Needs Approval" }
        switch call.status {
        case .pendingApproval: return "Needs Approval"
        case .running: return "Executing..."
        case .completed: return "Completed"
        case .denied: return "Denied"
        case .failed: return "Failed"
        }
    }

    private var statusBackground: Color {
        if isPendingApproval { return isHighRisk ? Color.red.opacity(0.15) : Color.orange.opacity(0.15) }
        switch call.status {
        case .pendingApproval: return Color.orange.opacity(0.15)
        case .running: return Color.blue.opacity(0.15)
        case .completed: return Color.green.opacity(0.15)
        case .denied: return Color.secondary.opacity(0.15)
        case .failed: return Color.red.opacity(0.15)
        }
    }

    private var statusForeground: Color {
        if isPendingApproval { return isHighRisk ? .red : .orange }
        switch call.status {
        case .pendingApproval: return .orange
        case .running: return .blue
        case .completed: return .green
        case .denied: return .secondary
        case .failed: return .red
        }
    }

    private var argumentsPreview: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(call.arguments.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                HStack(alignment: .top, spacing: 6) {
                    Text("\(key):")
                        .font(.caption.monospaced().weight(.medium))
                        .foregroundStyle(.secondary)
                    Text(value)
                        .font(.caption.monospaced())
                        .foregroundStyle(.primary)
                        .lineLimit(4)
                }
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 6))
    }

    private var approvalPrompt: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("This action requires your confirmation to execute.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                Button {
                    model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                } label: {
                    Label("Approve Once", systemImage: "checkmark")
                }
                .buttonStyle(.borderedProminent)
                .tint(isHighRisk ? Color.orange : TurboSparkTheme.accentColor)
                .accessibilityLabel("Approve once \(call.name)")
                .accessibilityHint("Allows this single tool call to execute")

                Button {
                    model.approvePendingToolCall(id: call.id, alwaysAllowSession: true)
                } label: {
                    Label("Always Allow", systemImage: "checkmark.circle")
                }
                .buttonStyle(.bordered)
                .accessibilityLabel("Always allow \(call.name) in this session")
                .accessibilityHint("Allows this tool and command prefix to run without asking for the rest of this session")

                Button("Deny", role: .cancel) {
                    model.denyPendingToolCall(id: call.id)
                }
                .buttonStyle(.bordered)
                .accessibilityLabel("Deny \(call.name)")
                .accessibilityHint("Rejects this tool call and tells the model to continue without it")
            }
            .controlSize(.small)
        }
        .padding(.top, 4)
    }

    private func outputSection(_ res: AppToolResult) -> some View {
        DisclosureGroup(isExpanded: $isOutputExpanded) {
            VStack(alignment: .leading, spacing: 6) {
                ScrollView(.horizontal) {
                    Text(res.output)
                        .font(.caption.monospaced())
                        .foregroundStyle(res.isError ? .red : .primary)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: 180)
                .padding(8)
                .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 6))

                HStack {
                    if res.durationSeconds > 0 {
                        Text(String(format: "Execution: %.2fs", res.durationSeconds))
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    Spacer()
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(res.output, forType: .string)
                        isCopied = true
                    } label: {
                        Label(isCopied ? "Copied" : "Copy Output", systemImage: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.caption2)
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel(isCopied ? "Copied output" : "Copy tool output")
                    .accessibilityHint("Copies the tool's output text to the clipboard")
                }
            }
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Text(res.isError ? "Error output" : "Tool output")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(res.isError ? .red : .secondary)
            }
        }
        .padding(.top, 2)
        .accessibilityLabel(res.isError ? "Tool output (error)" : "Tool output")
        .accessibilityHint("Expands to reveal the tool's output text")
    }
}
