import AppKit
import SwiftUI

/// Interactive message card component rendering agent tool executions in Unsloth Studio style.
@MainActor
struct ToolCallCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearance = AppearanceManager.shared

    private var diffAdditionColor: Color {
        appearance.diffMarkers == .color ? Color.green : Color.primary
    }

    private var diffDeletionColor: Color {
        appearance.diffMarkers == .color ? Color.red : Color.secondary
    }

    @ObservedObject var model: AppModel
    let call: AppToolCall
    let result: AppToolResult?
    var isNestedInGroup: Bool = false

    @State private var isExpanded: Bool = false

    private var summary: ToolCallSummaryInfo {
        ToolCallDiffFormatter.summarize(callName: call.name, arguments: call.arguments)
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
        return Color(nsColor: .separatorColor).opacity(isNestedInGroup ? 0.2 : 0.35)
    }

    private var terminalCommand: String? {
        call.arguments["command"] ?? call.arguments["cmd"]
    }

    private var isTodoCall: Bool {
        call.name.lowercased().contains("todo")
    }

    private var parsedTodos: [TodoItem]? {
        try? TodoWriteExecutor.parseTodos(from: call.arguments)
    }

    private var isQuestionCall: Bool {
        call.name.lowercased().contains("question")
    }

    private var parsedQuestions: [UserQuestionItem]? {
        try? AskUserQuestionExecutor.parseQuestions(from: call.arguments)
    }

    private var isFindingsCall: Bool {
        call.name.lowercased().contains("findings")
    }

    private var parsedFindings: [CodeFindingItem]? {
        (try? ReportFindingsExecutor.parseFindings(from: call.arguments))?.findings
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            summaryHeaderButton

            if isExpanded || isPendingApproval {
                VStack(alignment: .leading, spacing: 8) {
                    if isTodoCall, let todos = parsedTodos, !todos.isEmpty {
                        todoChecklistPreview(todos)
                    } else if isQuestionCall, let questions = parsedQuestions, !questions.isEmpty {
                        questionPreview(questions)
                    } else if isFindingsCall, let findings = parsedFindings, !findings.isEmpty {
                        findingsPreview(findings)
                    } else if let cmd = terminalCommand {
                        ToolCodeCellView(
                            label: "command",
                            code: cmd,
                            language: "bash",
                            downloadFilename: "command.sh"
                        )
                    } else {
                        argumentsPreview
                    }

                    if let risk = call.riskAssessment, !risk.reasons.isEmpty, isPendingApproval {
                        riskWarningBox(risk)
                    }

                    if isPendingApproval {
                        approvalPrompt
                    }

                    if let result {
                        ToolResultOutputView(
                            output: result.output,
                            isError: result.isError,
                            durationSeconds: result.durationSeconds
                        )
                    }
                }
                .padding(.top, 2)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            isNestedInGroup
                ? Color(nsColor: .controlBackgroundColor).opacity(0.5)
                : Color(nsColor: .controlBackgroundColor).opacity(0.85)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(borderColor, lineWidth: 1)
        )
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Unsloth Studio-Style Summary Row

    private var summaryHeaderButton: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.18)) {
                isExpanded.toggle()
            }
        } label: {
            HStack(spacing: 7) {
                statusLeadingIcon

                headerLabelView

                if let additions = summary.additions {
                    Text("+\(additions)")
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(diffAdditionColor)
                }

                if let deletions = summary.deletions {
                    Text("-\(deletions)")
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(diffDeletionColor)
                }

                if let range = summary.lineRange {
                    Text(range)
                        .font(theme.code(.tiny))
                        .foregroundStyle(.secondary)
                }

                if let risk = call.riskAssessment, risk.level != .safe {
                    riskBadge(risk)
                }

                Spacer()

                statusBadge

                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .rotationEffect(.degrees((isExpanded || isPendingApproval) ? 90 : 0))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(summary.action) \(summary.target)")
        .accessibilityValue(statusLabel)
    }

    private var statusLeadingIcon: some View {
        Group {
            if isPendingApproval {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(isHighRisk ? Color.red : Color.orange)
            } else {
                switch call.status {
                case .running:
                    TaskProgressFlameIcon(size: 12)
                case .completed:
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(Color.green)
                case .denied:
                    Image(systemName: "minus.circle.fill")
                        .font(.caption)
                        .foregroundStyle(Color.secondary)
                case .failed:
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(Color.red)
                case .pendingApproval:
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(Color.orange)
                }
            }
        }
    }

    @ViewBuilder
    private var headerLabelView: some View {
        let isCancelled = call.status == .denied
        if let cmd = terminalCommand {
            Text("$ \(cmd)")
                .font(theme.code(.small, weight: .semibold))
                .foregroundStyle(isCancelled ? .secondary : .primary)
                .strikethrough(isCancelled, color: .secondary)
                .lineLimit(1)
        } else if call.category == .web, let query = call.arguments["query"] {
            HStack(spacing: 4) {
                Text("Search:")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.secondary)
                Text("\"\(query)\"")
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.primary)
            }
            .strikethrough(isCancelled, color: .secondary)
            .lineLimit(1)
        } else {
            HStack(spacing: 5) {
                Text(summary.action)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(isCancelled ? .secondary : .primary)

                Text(summary.target)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(isCancelled ? .secondary : .primary)
            }
            .strikethrough(isCancelled, color: .secondary)
            .lineLimit(1)
        }
    }

    private func riskBadge(_ risk: ToolRiskAssessment) -> some View {
        HStack(spacing: 4) {
            Image(systemName: risk.level.systemImage)
                .font(.system(size: 9))
            Text(risk.level.label)
                .font(.system(size: 10, weight: .semibold))
        }
        .padding(.horizontal, 5)
        .padding(.vertical, 1)
        .background(risk.level == .high ? Color.red.opacity(0.15) : Color.orange.opacity(0.12), in: Capsule())
        .foregroundStyle(risk.level == .high ? Color.red : Color.orange)
        .help("Security risk: \(risk.level.label)")
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
                Text("- \(reason)")
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
        Text(statusLabel)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(statusBackground, in: Capsule())
            .foregroundStyle(statusForeground)
    }

    private var statusLabel: String {
        if isPendingApproval { return "Needs Approval" }
        switch call.status {
        case .pendingApproval: return "Needs Approval"
        case .running: return "Running..."
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

    private func todoChecklistPreview(_ todos: [TodoItem]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(todos) { item in
                HStack(alignment: .center, spacing: 8) {
                    if item.isCompleted {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(Color.green)
                            .font(.caption)
                    } else if item.isInProgress {
                        TaskProgressFlameIcon(size: 12)
                    } else if item.isCancelled {
                        Image(systemName: "minus.circle.fill")
                            .foregroundStyle(Color.secondary)
                            .font(.caption)
                    } else {
                        Image(systemName: "circle")
                            .foregroundStyle(Color.secondary.opacity(0.7))
                            .font(.caption)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.content)
                            .font(.callout.weight(item.isInProgress ? .semibold : .regular))
                            .foregroundStyle(item.isCompleted ? .secondary : .primary)
                            .strikethrough(item.isCompleted || item.isCancelled, color: .secondary)

                        if item.isInProgress && !item.activeForm.isEmpty && item.activeForm != item.content {
                            Text(item.activeForm)
                                .font(.caption2)
                                .foregroundStyle(TurboSparkTheme.accentColor)
                        }
                    }

                    Spacer()

                    if item.isInProgress {
                        Text("In Progress")
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.orange.opacity(0.15), in: Capsule())
                            .foregroundStyle(Color.orange)
                    }
                }
                .padding(.vertical, 2)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private func questionPreview(_ questions: [UserQuestionItem]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(questions.enumerated()), id: \.offset) { idx, q in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(q.header)
                            .font(.caption2.weight(.bold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(TurboSparkTheme.accentColor.opacity(0.15), in: Capsule())
                            .foregroundStyle(TurboSparkTheme.accentColor)

                        Text(q.question)
                            .font(.callout.weight(.medium))
                            .foregroundStyle(.primary)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(q.options.enumerated()), id: \.offset) { _, opt in
                            HStack(alignment: .top, spacing: 6) {
                                Image(systemName: "circle")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .padding(.top, 2)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(opt.label)
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.primary)
                                    Text(opt.description)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 1)
                        }
                    }
                    .padding(.leading, 8)
                }
                if idx < questions.count - 1 {
                    Divider()
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private func findingsPreview(_ findings: [CodeFindingItem]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(findings.enumerated()), id: \.offset) { idx, f in
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.orange)

                        Text(f.file + (f.line.map { ":\($0)" } ?? ""))
                            .font(theme.code(.small, weight: .bold))
                            .foregroundStyle(.primary)

                        if let cat = f.category {
                            Text(cat)
                                .font(.caption2)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Color.secondary.opacity(0.12), in: Capsule())
                                .foregroundStyle(.secondary)
                        }
                    }

                    Text(f.summary)
                        .font(.callout)
                        .foregroundStyle(.primary)

                    Text("Scenario: " + f.failureScenario)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)
                if idx < findings.count - 1 {
                    Divider()
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private var argumentsPreview: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ARGUMENTS")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 3) {
                ForEach(call.arguments.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                    HStack(alignment: .top, spacing: 6) {
                        Text("\(key):")
                            .font(theme.code(.small, weight: .medium))
                            .foregroundStyle(.secondary)
                        Text(value)
                            .font(theme.code(.small))
                            .foregroundStyle(.primary)
                            .lineLimit(6)
                    }
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .textBackgroundColor).opacity(0.4))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .padding(.leading, 8)
        .overlay(
            Rectangle()
                .fill(Color.primary.opacity(0.15))
                .frame(width: 2),
            alignment: .leading
        )
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
                .help("Approve this tool call once")
                .accessibilityLabel("Approve once \(call.name)")

                if call.category == .mcp,
                   let target = McpPermissionRule.targetOfCall(name: call.name, arguments: call.arguments) {
                    // MCP calls can persist the grant on the PROJECT, not
                    // just this session: the rule the engine matches is
                    // written with the same parse that reads the call, so
                    // "this tool" and "this server" mean exactly what the
                    // evaluation will compare.
                    Menu {
                        Button("Always Allow This Tool") {
                            model.addMcpPermissionRule(serverName: target.server, toolName: target.tool, allow: true)
                            model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                        }
                        Button("Always Allow This Server") {
                            model.addMcpPermissionRule(serverName: target.server, toolName: nil, allow: true)
                            model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                        }
                    } label: {
                        Label("Always Allow", systemImage: "checkmark.seal")
                    }
                    .buttonStyle(.bordered)
                    .help("Persist an allow rule for this MCP tool or server in project settings")
                    .accessibilityLabel("Always allow \(call.name) in project settings")
                } else {
                    Button {
                        model.approvePendingToolCall(id: call.id, alwaysAllowSession: true)
                    } label: {
                        Label("Always Allow", systemImage: "checkmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .help("Always allow this tool and command in this session")
                    .accessibilityLabel("Always allow \(call.name) in this session")
                }

                Button("Deny", role: .cancel) {
                    model.denyPendingToolCall(id: call.id)
                }
                .buttonStyle(.bordered)
                .help("Deny this tool call execution")
                .accessibilityLabel("Deny \(call.name)")
            }
            .controlSize(.small)
        }
        .padding(.top, 4)
    }
}
