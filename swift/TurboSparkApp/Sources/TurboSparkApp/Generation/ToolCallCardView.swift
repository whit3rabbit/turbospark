import AppKit
import SwiftUI

/// Interactive message card component rendering agent tool executions in Unsloth Studio style.
@MainActor
struct ToolCallCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearance = AppearanceManager.shared

    private var diffAdditionColor: Color {
        appearance.diffMarkers == .color ? theme.success : theme.foreground
    }

    private var diffDeletionColor: Color {
        appearance.diffMarkers == .color ? theme.danger : theme.secondaryText
    }

    @ObservedObject var model: AppModel
    let call: AppToolCall
    let result: AppToolResult?

    @State private var isExpanded: Bool = false

    private var browserCardPresentation: BrowserToolCardPresentation? {
        guard BrowserToolCardPresentation.isBrowserTool(call.name) else { return nil }
        return BrowserToolCardPresentation(
            call: call,
            result: result,
            pendingOrigin: model.browserPermissionApprovalRequest(for: call)?.origin
        )
    }

    private var summary: ToolCallSummaryInfo {
        if let browserCardPresentation {
            return browserCardPresentation.summary
        }
        let presentation = ToolPresentation.resolve(call.name)
        if let url = ToolPresentation.webURL(for: call) {
            return ToolCallSummaryInfo(action: presentation.localizedLabel, target: url.absoluteString)
        }
        if !presentation.isBuiltIn {
            return ToolCallSummaryInfo(action: presentation.localizedLabel,
                target: call.arguments["path"] ?? call.arguments["query"] ?? "")
        }
        // The body reads `summary` several times and re-runs for every
        // streamed token; summarizing a large write splits its whole content.
        return ToolCallSummaryCache.summary(for: call)
    }

    private var isPendingApproval: Bool {
        // A parked agent batch persists `.pendingApproval` on EVERY member,
        // but one verdict covers the whole batch and only the primary call's
        // buttons are wired to it. Showing controls on the others would be
        // buttons that look live and do nothing.
        if model.pendingToolCall?.id != call.id,
            model.pendingBatchCalls?.contains(where: { $0.id == call.id }) == true
        {
            return false
        }
        return call.status == .pendingApproval || (model.pendingToolCall?.id == call.id)
    }

    /// Whether THIS call is the one the turn is parked on, waiting for the
    /// user's answers (the interactive AskUserQuestion flow). Its options
    /// render tappable while true, and the card holds itself open.
    private var isActiveQuestionSet: Bool {
        isQuestionCall && model.pendingUserQuestions?.toolCallID == call.id
    }

    private var isHighRisk: Bool {
        call.riskAssessment?.isHighRisk ?? false
    }

    private var borderColor: Color {
        if isPendingApproval {
            return isHighRisk ? theme.warning : theme.warning.opacity(0.8)
        }
        if let result, result.isError {
            return theme.warning.opacity(0.6)
        }
        return theme.border
    }

    private var terminalCommand: String? {
        ToolPresentation.resolve(call.name).icon == "terminal" ? call.shellCommand : nil
    }

    private var submittedPlan: String? {
        guard ["exitplanmode", "exit_plan_mode"].contains(call.name.lowercased()) else { return nil }
        let text = call.arguments["plan"] ?? call.arguments["summary"] ?? call.arguments["content"]
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return text
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

    private var isBrowserCall: Bool {
        browserCardPresentation != nil
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

    private var isAgentCall: Bool {
        AppModel.isAgentFamilyToolName(call.name)
    }

    private var isStopAgentCall: Bool {
        let n = call.name.lowercased()
        return ToolPresentation.resolve(n).icon == "bot-off"
    }

    private var isWriteCall: Bool {
        let n = call.name.lowercased()
        return ToolPresentation.resolve(n).icon == "file-plus"
    }

    private var isEditCall: Bool {
        let n = call.name.lowercased()
        return ["file-pen", "files"].contains(ToolPresentation.resolve(n).icon)
    }

    private var isMultiEditCall: Bool {
        ["multiedit", "multi_edit"].contains(call.name.lowercased())
    }

    private var isHttpRequestCall: Bool {
        ["http_request", "httprequest"].contains(call.name.lowercased())
    }

    private var isPatchCall: Bool {
        let n = call.name.lowercased()
        return n == "apply_patch" || n == "applypatch"
    }

    private var isReadCall: Bool {
        let n = call.name.lowercased()
        return ToolPresentation.resolve(n).icon == "file-text"
    }

    private var isWebFetchCall: Bool {
        let n = call.name.lowercased()
        return ToolPresentation.resolve(n).icon == "globe" || ToolPresentation.resolve(n).icon == "network"
    }

    private var isSkillCall: Bool {
        call.name.lowercased() == "skill"
    }

    private var isMcpCall: Bool {
        let n = call.name.lowercased()
        return n.contains("mcp")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            summaryHeaderButton

            if isExpanded || isPendingApproval || isActiveQuestionSet {
                VStack(alignment: .leading, spacing: 8) {
                    if isBrowserCall {
                        EmptyView()
                    } else if let submittedPlan {
                        CollapsibleMessageContentView(text: submittedPlan, maxHeight: 260)
                            .padding(6)
                    } else if isTodoCall, let todos = parsedTodos, !todos.isEmpty {
                        todoChecklistPreview(todos)
                    } else if isQuestionCall, let questions = parsedQuestions, !questions.isEmpty {
                        if isActiveQuestionSet {
                            // The tappable card: the turn is parked on these
                            // questions right now.
                            InteractiveQuestionCardView(model: model, questions: questions)
                        } else {
                            questionPreview(questions)
                        }
                    } else if isFindingsCall, let findings = parsedFindings, !findings.isEmpty {
                        findingsPreview(findings)
                    } else if isAgentCall {
                        agentPreview
                    } else if isStopAgentCall {
                        stopAgentPreview
                    } else if let cmd = terminalCommand {
                        ToolCodeCellView(
                            label: "command",
                            code: cmd,
                            language: "bash",
                            downloadFilename: "command.sh"
                        )
                    } else if isPatchCall {
                        patchPreview
                    } else if isWriteCall {
                        fileWritePreview
                    } else if isMultiEditCall, let summary = ToolApprovalPreviewSummary.multiEditSummary(call.arguments) {
                        ToolCodeCellView(label: "edits", code: summary, language: "diff")
                    } else if isHttpRequestCall {
                        ToolCodeCellView(
                            label: "request",
                            code: ToolApprovalPreviewSummary.httpRequestSummary(call.arguments),
                            language: "http")
                    } else if isEditCall, !isMultiEditCall {
                        fileEditPreview
                    } else if isWebFetchCall {
                        // Before the read arm: a fetch tool named
                        // `read_url_content` contains "read", and matched
                        // first it rendered the file-read layout under a
                        // "Fetched <host>" header.
                        webFetchPreview
                    } else if isReadCall {
                        fileReadPreview
                    } else if isSkillCall {
                        skillPreview
                    } else if isMcpCall {
                        mcpPreview
                    } else {
                        argumentsPreview
                    }

                    if !isBrowserCall, let risk = call.riskAssessment, !risk.reasons.isEmpty, isPendingApproval {
                        riskWarningBox(risk)
                    }

                    if isPendingApproval {
                        approvalPrompt
                    }

                    if !isBrowserCall, let result, submittedPlan == nil || result.isError {
                        if let outcome = result.efficiency, outcome.hasAnyOutcome {
                            efficiencyOutcome(outcome)
                        }
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
        .padding(.vertical, 4)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: ConversationLayout.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: ConversationLayout.cardRadius, style: .continuous)
                .stroke(borderColor, lineWidth: 1)
        )
        .frame(maxWidth: .infinity, alignment: .leading)
        .onChange(of: submittedPlan, initial: true) { _, plan in
            if plan != nil { isExpanded = true }
        }
    }

    // MARK: - Unsloth Studio-Style Summary Row

    private var summaryHeaderButton: some View {
        Button {
            TSMotion.animate(TSMotion.select) {
                isExpanded.toggle()
            }
        } label: {
            HStack(spacing: 7) {
                Group {
                    if isBrowserCall {
                        BundledToolIcon(name: ToolPresentation.resolve(call.name).icon)
                    } else if let url = ToolPresentation.webURL(for: call) {
                        OfflineSiteIcon(url: url)
                    } else {
                        BundledToolIcon(name: ToolPresentation.resolve(call.name).icon)
                    }
                }
                .frame(width: ConversationLayout.activityIconWidth)
                headerLabelView
                if let browserCardPresentation {
                    BrowserToolCardView(presentation: browserCardPresentation)
                }

                if let additions = summary.additions {
                    Text(verbatim: "+\(additions)")
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(diffAdditionColor)
                }

                if let deletions = summary.deletions {
                    Text(verbatim: "-\(deletions)")
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(diffDeletionColor)
                }

                if let range = summary.lineRange {
                    Text(range)
                        .font(theme.code(.tiny))
                        .foregroundStyle(.appSecondary)
                }

                if let risk = call.riskAssessment, risk.level != .safe {
                    riskBadge(risk)
                }

                if result?.efficiency?.validation != nil {
                    Text("validated", bundle: .module)
                        .font(theme.code(.tiny, weight: .semibold))
                        .foregroundStyle(.appSecondary)
                }

                Spacer()

                statusLeadingIcon
                statusBadge

                Image(systemName: "chevron.down")
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                    .rotationEffect(.degrees((isExpanded || isPendingApproval || isActiveQuestionSet) ? 180 : 0))
            }
            .frame(minHeight: ConversationLayout.activityHeight)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        // Mirrors the visible header (localized tool label, then target)
        // rather than `summary.action`, an English verb.
        .accessibilityLabel(Text(verbatim: "\(ToolPresentation.resolve(call.name).localizedLabel) \(summary.target)"))
        .accessibilityValue(statusAccessibilityLabel + ", " + ((isExpanded || isPendingApproval || isActiveQuestionSet) ? AppStrings.localized("Expanded") : AppStrings.localized("Collapsed")))
    }

    @ViewBuilder
    private func efficiencyOutcome(_ outcome: ToolEfficiencyOutcome) -> some View {
        HStack(spacing: 4) {
            if let validation = outcome.validation {
                Text("Validation: \(validation)", bundle: .module)
            }
            if outcome.archivedOutput {
                if outcome.validation != nil { Text(verbatim: "|") }
                Text("Full output archived and available on request", bundle: .module)
            }
            if outcome.verifiedReduction {
                if outcome.validation != nil || outcome.archivedOutput { Text(verbatim: "|") }
                Text("Verified local reduction", bundle: .module)
            }
            if outcome.boundaryCompaction {
                if outcome.validation != nil || outcome.archivedOutput || outcome.verifiedReduction { Text(verbatim: "|") }
                Text("Completed-todo boundary compacted", bundle: .module)
            }
        }
        .font(theme.ui(.tiny))
        .foregroundStyle(.appSecondary)
    }

    private var statusLeadingIcon: some View {
        Group {
            if isPendingApproval {
                Image(systemName: "exclamationmark.triangle.fill")
                    .themedFont(.small)
                    .foregroundStyle(isHighRisk ? AppAttention.color : AppAttention.color.opacity(0.8))
            } else {
                switch ToolPresentation.status(call: call, result: result) {
                case .running:
                    TaskProgressFlameIcon(size: 12)
                case .completed:
                    Image(systemName: "checkmark.circle.fill")
                        .themedFont(.small)
                        .foregroundStyle(.appSuccess)
                case .denied:
                    Image(systemName: "minus.circle.fill")
                        .themedFont(.small)
                        .foregroundStyle(theme.secondaryText)
                case .failed:
                    Image(systemName: "xmark.circle.fill")
                        .themedFont(.small)
                        .foregroundStyle(AppAttention.color)
                case .pendingApproval:
                    Image(systemName: "exclamationmark.triangle.fill")
                        .themedFont(.small)
                        .foregroundStyle(AppAttention.color)
                }
            }
        }
    }

    @ViewBuilder
    private var headerLabelView: some View {
        let isCancelled = call.status == .denied
        if let browserCardPresentation {
            HStack(spacing: 5) {
                Text(browserCardPresentation.action)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(isCancelled ? theme.secondaryText : theme.foreground)

                if let target = browserCardPresentation.target {
                    Text(verbatim: target)
                        .font(theme.code(.base, weight: .semibold))
                        .foregroundStyle(isCancelled ? theme.secondaryText : theme.foreground)
                }
            }
            .strikethrough(isCancelled, color: .secondary)
            .lineLimit(1)
        } else if submittedPlan != nil {
            Text("Plan", bundle: .module)
                .themedFont(.small, weight: .medium)
                .foregroundStyle(.appText)
        } else if let cmd = terminalCommand {
            HStack(spacing: 5) {
                Text(ToolPresentation.resolve(call.name).localizedLabel)
                    .themedFont(.small, weight: .medium)
                Text(verbatim: "$ " + cmd)
                    .font(theme.code(.base, weight: .semibold))
            }
            .foregroundStyle(isCancelled ? theme.secondaryText : theme.foreground)
            .strikethrough(isCancelled, color: .secondary)
            .lineLimit(1)
        } else if call.category == .web, let query = call.arguments["query"] {
            HStack(spacing: 4) {
                Text(ToolPresentation.resolve(call.name).localizedLabel)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appSecondary)
                Text(verbatim: "\"\(query)\"")
                    .font(theme.code(.base, weight: .semibold))
                    .foregroundStyle(.appText)
            }
            .strikethrough(isCancelled, color: .secondary)
            .lineLimit(1)
        } else {
            HStack(spacing: 5) {
                Text(ToolPresentation.resolve(call.name).localizedLabel)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(isCancelled ? theme.secondaryText : theme.foreground)

                Text(summary.target)
                    .font(theme.code(.base, weight: .semibold))
                    .foregroundStyle(isCancelled ? theme.secondaryText : theme.foreground)
            }
            .strikethrough(isCancelled, color: .secondary)
            .lineLimit(1)
        }
    }

    private func riskBadge(_ risk: ToolRiskAssessment) -> some View {
        HStack(spacing: 4) {
            Image(systemName: risk.level.systemImage)
                .themedFont(.micro)
            Text(LocalizedStringKey(risk.level.label), bundle: .module)
                .themedFont(.tiny, weight: .semibold)
        }
        .padding(.horizontal, 5)
        .padding(.vertical, 1)
        .background(risk.level == .high ? AppAttention.color.opacity(0.15) : AppAttention.color.opacity(0.12), in: Capsule())
        .foregroundStyle(risk.level == .high ? AppAttention.color : AppAttention.color.opacity(0.9))
        .help(Text("Security risk: \(Text(LocalizedStringKey(risk.level.label), bundle: .module))", bundle: .module))
    }

    private func riskWarningBox(_ risk: ToolRiskAssessment) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(AppAttention.color)
                    .themedFont(.small)
                (risk.level == .high
                    ? Text("High-Risk Operation Detected", bundle: .module)
                    : Text("Security Notice", bundle: .module))
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(AppAttention.color)
            }
            ForEach(risk.reasons, id: \.self) { reason in
                Text(verbatim: "- \(reason)")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            AppAttention.color.opacity(0.08),
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
    }

    @ViewBuilder
    private var statusBadge: some View {
        statusText
            .themedFont(.tiny, weight: .medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(statusBackground, in: Capsule())
            .foregroundStyle(statusForeground)
    }

    @ViewBuilder
    private var statusText: some View {
        if isPendingApproval {
            Text("Needs Approval", bundle: .module)
        } else if call.autoApprovedBy == "classifier" && (ToolPresentation.status(call: call, result: result) == .running || ToolPresentation.status(call: call, result: result) == .completed) {
            Text("Classifier Approved", bundle: .module)
        } else {
            switch ToolPresentation.status(call: call, result: result) {
            case .pendingApproval:
                Text("Needs Approval", bundle: .module)
            case .running:
                Text("Running...", bundle: .module)
            case .completed:
                Text("Completed", bundle: .module)
            case .denied:
                Text("Denied", bundle: .module)
            case .failed:
                Text("Failed", bundle: .module)
            }
        }
    }

    private var statusAccessibilityLabel: String {
        if isPendingApproval { return AppStrings.localized("Needs Approval") }
        if call.autoApprovedBy == "classifier" {
            switch ToolPresentation.status(call: call, result: result) {
            case .running, .completed: return AppStrings.localized("Classifier Approved")
            default: break
            }
        }
        switch ToolPresentation.status(call: call, result: result) {
        case .pendingApproval: return AppStrings.localized("Needs Approval")
        case .running: return AppStrings.localized("Running...")
        case .completed: return AppStrings.localized("Completed")
        case .denied: return AppStrings.localized("Denied")
        case .failed: return AppStrings.localized("Failed")
        }
    }

    private var statusBackground: Color {
        if isPendingApproval { return isHighRisk ? theme.warning.opacity(0.15) : theme.warning.opacity(0.12) }
        switch ToolPresentation.status(call: call, result: result) {
        case .pendingApproval: return theme.warning.opacity(0.12)
        case .running: return theme.accent.opacity(0.15)
        case .completed: return theme.successTint
        case .denied: return theme.secondaryText.opacity(0.15)
        case .failed: return theme.warning.opacity(0.15)
        }
    }

    private var statusForeground: Color {
        if isPendingApproval { return isHighRisk ? theme.warning : theme.warning.opacity(0.9) }
        switch ToolPresentation.status(call: call, result: result) {
        case .pendingApproval: return theme.warning
        case .running: return theme.accent
        case .completed: return theme.success
        case .denied: return theme.secondaryText
        case .failed: return theme.warning
        }
    }

    private func todoChecklistPreview(_ todos: [TodoItem]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(todos) { item in
                TodoItemRow(item: item)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private func questionPreview(_ questions: [UserQuestionItem]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(questions.enumerated()), id: \.offset) { idx, q in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(q.header)
                            .themedFont(.tiny, weight: .bold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.appAccent.opacity(0.15), in: Capsule())
                            .foregroundStyle(.appAccent)

                        Text(q.question)
                            .themedFont(.base, weight: .medium)
                            .foregroundStyle(.appText)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(q.options.enumerated()), id: \.offset) { _, opt in
                            HStack(alignment: .top, spacing: 6) {
                                Image(systemName: "circle")
                                    .themedFont(.tiny)
                                    .foregroundStyle(.appSecondary)
                                    .padding(.top, 2)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(opt.label)
                                        .themedFont(.small, weight: .semibold)
                                        .foregroundStyle(.appText)
                                    Text(opt.description)
                                        .themedFont(.tiny)
                                        .foregroundStyle(.appSecondary)
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
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private func findingsPreview(_ findings: [CodeFindingItem]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(findings.enumerated()), id: \.offset) { idx, f in
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.circle.fill")
                            .themedFont(.tiny)
                            .foregroundStyle(AppAttention.color)

                        Text(f.file + (f.line.map { ":\($0)" } ?? ""))
                            .font(theme.code(.small, weight: .bold))
                            .foregroundStyle(.appText)

                        if let cat = f.category {
                            Text(cat)
                                .themedFont(.tiny)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(.appSecondary.opacity(0.15), in: Capsule())
                                .foregroundStyle(.appSecondary)
                        }
                    }

                    Text(f.summary)
                        .themedFont(.base)
                        .foregroundStyle(.appText)

                    Text("Scenario: \(f.failureScenario)", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                .padding(.vertical, 2)
                if idx < findings.count - 1 {
                    Divider()
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var agentPreview: some View {
        let agentType = call.arguments["subagent_type"]
            ?? call.arguments["subagentType"]
            ?? call.arguments["agent"]
            ?? call.arguments["agent_name"]
            ?? "general"
        let description = call.arguments["description"]
            ?? call.arguments["task"]
            ?? ""
        let prompt = call.arguments["prompt"]
            ?? (description.isEmpty ? "" : description)
        let isBackground = call.arguments["run_in_background"]?.lowercased() == "true"
            || call.arguments["runInBackground"]?.lowercased() == "true"
            || call.arguments["background"]?.lowercased() == "true"

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "person.2.wave.2")
                    .themedFont(.tiny, weight: .bold)
                    .foregroundStyle(.appAccent)
                Text(agentType)
                    .themedFont(.tiny, weight: .bold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.appAccent.opacity(0.15), in: Capsule())
                    .foregroundStyle(.appAccent)

                if isBackground {
                    HStack(spacing: 3) {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .themedFont(.micro)
                        Text("Background", bundle: .module)
                            .themedFont(.micro, weight: .medium)
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.appAccent.opacity(0.15), in: Capsule())
                    .foregroundStyle(.appAccent)
                }

                if !description.isEmpty && description != prompt {
                    Text(description)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                }
            }

            if !prompt.isEmpty {
                Text(prompt)
                    .font(theme.code(.small))
                    .foregroundStyle(.appText)
                    .lineLimit(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var stopAgentPreview: some View {
        let taskId = call.arguments["task_id"] ?? call.arguments["taskId"] ?? call.arguments["id"] ?? "agent"
        return HStack(spacing: 6) {
            Image(systemName: "stop.circle.fill")
                .themedFont(.small)
                .foregroundStyle(AppAttention.color)
            Text("Stop Background Agent:", bundle: .module)
                .themedFont(.small, weight: .medium)
                .foregroundStyle(.appSecondary)
            Text(taskId)
                .font(theme.code(.small, weight: .bold))
                .foregroundStyle(.appText)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var fileWritePreview: some View {
        // The executor's own resolver: what the card shows is what runs, and
        // an ambiguous call shows a refusal marker instead of one candidate.
        let resolvedWrite = ToolArgumentResolver.fileArguments(call.arguments)
        let targetPath = ToolArgumentResolver.displayPath(call.arguments)
        let content = resolvedWrite.content ?? ""
        let fileName = (targetPath as NSString).lastPathComponent
        let lang = detectLanguage(from: fileName)

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "doc.badge.plus")
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
                Text(targetPath)
                    .font(theme.code(.small, weight: .bold))
                    .foregroundStyle(.appText)
                Spacer()
                let lineCount = ToolCallDiffFormatter.countLines(content)
                Text("\(lineCount) lines", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }

            if !content.isEmpty {
                ToolCodeCellView(
                    label: fileName,
                    code: content,
                    language: lang,
                    downloadFilename: fileName,
                    labelIsVerbatim: true
                )
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var fileEditPreview: some View {
        let resolvedEdit = ToolArgumentResolver.fileArguments(call.arguments)
        let targetPath = ToolArgumentResolver.displayPath(call.arguments)
        let oldStr = resolvedEdit.oldString ?? ""
        let newStr = resolvedEdit.newString ?? ""

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "doc.badge.ellipsis")
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
                Text(targetPath)
                    .font(theme.code(.small, weight: .bold))
                    .foregroundStyle(.appText)
            }

            if !oldStr.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 4) {
                        Image(systemName: "minus")
                            .themedFont(.micro, weight: .bold)
                            .foregroundStyle(diffDeletionColor)
                        Text("ORIGINAL", bundle: .module)
                            .themedCode(.callout, weight: .bold)
                            .foregroundStyle(diffDeletionColor)
                    }
                    Text(oldStr)
                        .font(theme.code(.small))
                        .foregroundStyle(.appText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(6)
                        .background(.appDanger.opacity(0.06), in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                        .textSelection(.enabled)
                }
            }

            if !newStr.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 4) {
                        Image(systemName: "plus")
                            .themedFont(.micro, weight: .bold)
                            .foregroundStyle(diffAdditionColor)
                        Text("REPLACEMENT", bundle: .module)
                            .themedCode(.callout, weight: .bold)
                            .foregroundStyle(diffAdditionColor)
                    }
                    Text(newStr)
                        .font(theme.code(.small))
                        .foregroundStyle(.appText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(6)
                        .background(.appSuccess.opacity(0.06), in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                        .textSelection(.enabled)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var patchPreview: some View {
        let patch = call.arguments["patch_text"]
            ?? call.arguments["patchText"]
            ?? call.arguments["patch"]
            ?? ""
        return ToolCodeCellView(
            label: "patch",
            code: patch,
            language: "diff",
            downloadFilename: "change.patch"
        )
    }

    private var fileReadPreview: some View {
        let targetPath = ToolArgumentResolver.displayPath(call.arguments, forRead: true)
        let start = call.arguments["StartLine"] ?? call.arguments["start_line"] ?? call.arguments["startLine"] ?? call.arguments["start"]
        let end = call.arguments["EndLine"] ?? call.arguments["end_line"] ?? call.arguments["endLine"] ?? call.arguments["end"]

        return HStack(spacing: 8) {
            Image(systemName: "doc.text")
                .themedFont(.small)
                .foregroundStyle(.appAccent)
            Text(targetPath)
                .font(theme.code(.small, weight: .semibold))
                .foregroundStyle(.appText)

            if let start, let end, !start.isEmpty, !end.isEmpty {
                Text("Lines \(start)-\(end)", bundle: .module)
                    .themedFont(.tiny, weight: .medium)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.appSecondary.opacity(0.15), in: Capsule())
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var webFetchPreview: some View {
        let urlString = call.arguments["url"]
            ?? call.arguments["uri"]
            ?? call.arguments["Url"]
            ?? call.arguments["URL"]
            ?? call.arguments["href"]
            ?? ""
        let format = call.arguments["format"] ?? "markdown"

        return HStack(spacing: 8) {
            Image(systemName: "globe")
                .themedFont(.small)
                .foregroundStyle(.appAccent)

            if let url = ToolPresentation.webLink(urlString) {
                Link(urlString, destination: url)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.appAccentText)
                    .lineLimit(1)
            } else {
                Text(urlString)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.appText)
                    .lineLimit(1)
            }

            Spacer()

            Text(format.uppercased())
                .themedFont(.tiny, weight: .bold)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.appSecondary.opacity(0.15), in: Capsule())
                .foregroundStyle(.appSecondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var skillPreview: some View {
        let skillName = call.arguments["name"] ?? call.arguments["skill_name"] ?? "skill"
        let otherArgs = call.arguments.filter { $0.key != "name" && $0.key != "skill_name" }

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "puzzlepiece.extension")
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
                Text(skillName)
                    .themedFont(.tiny, weight: .bold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.appAccent.opacity(0.15), in: Capsule())
                    .foregroundStyle(.appAccent)
            }

            if !otherArgs.isEmpty {
                ForEach(otherArgs.sorted(by: { $0.key < $1.key }), id: \.key) { key, val in
                    HStack(alignment: .top, spacing: 6) {
                        Text(verbatim: "\(key):")
                            .font(theme.code(.small, weight: .medium))
                            .foregroundStyle(.appSecondary)
                        Text(val)
                            .font(theme.code(.small))
                            .foregroundStyle(.appText)
                    }
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private var mcpPreview: some View {
        let server = call.arguments["server"] ?? call.arguments["server_name"] ?? call.arguments["ServerName"] ?? ""
        let tool = call.arguments["toolName"] ?? call.arguments["tool"] ?? call.arguments["name"] ?? call.arguments["ToolName"] ?? ""
        let otherArgs = call.arguments.filter {
            !["server", "server_name", "ServerName", "toolName", "tool", "name", "ToolName"].contains($0.key)
        }

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "shippingbox")
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
                if !server.isEmpty {
                    Text(server)
                        .themedFont(.tiny, weight: .bold)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.appAccent.opacity(0.15), in: Capsule())
                        .foregroundStyle(.appAccent)
                }
                if !tool.isEmpty {
                    Text(tool)
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(.appText)
                }
            }

            if !otherArgs.isEmpty {
                ForEach(otherArgs.sorted(by: { $0.key < $1.key }), id: \.key) { key, val in
                    HStack(alignment: .top, spacing: 6) {
                        Text(verbatim: "\(key):")
                            .font(theme.code(.small, weight: .medium))
                            .foregroundStyle(.appSecondary)
                        Text(val)
                            .font(theme.code(.small))
                            .foregroundStyle(.appText)
                            .lineLimit(6)
                    }
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private func detectLanguage(from path: String) -> String {
        let ext = (path as NSString).pathExtension.lowercased()
        switch ext {
        case "swift": return "swift"
        case "rs": return "rust"
        case "py": return "python"
        case "js", "mjs", "cjs": return "javascript"
        case "ts", "tsx": return "typescript"
        case "json": return "json"
        case "md": return "markdown"
        case "sh", "zsh", "bash": return "bash"
        case "html": return "html"
        case "css": return "css"
        case "yml", "yaml": return "yaml"
        case "toml": return "toml"
        case "c", "h", "cpp", "hpp": return "c"
        case "diff", "patch": return "diff"
        default: return "plaintext"
        }
    }

    private var argumentsPreview: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ARGUMENTS", bundle: .module)
                .themedCode(.callout, weight: .bold)
                .foregroundStyle(.appSecondary)

            VStack(alignment: .leading, spacing: 3) {
                ForEach(call.arguments.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                    HStack(alignment: .top, spacing: 6) {
                        Text(verbatim: "\(key):")
                            .font(theme.code(.small, weight: .medium))
                            .foregroundStyle(.appSecondary)
                        Text(value)
                            .font(theme.code(.small))
                            .foregroundStyle(.appText)
                            .lineLimit(6)
                    }
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appElevated.opacity(0.4))
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        .padding(.leading, 8)
        .overlay(
            Rectangle()
                .fill(theme.border)
                .frame(width: 2),
            alignment: .leading
        )
    }

    private var approvalPrompt: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let notice = model.pendingToolCallClassifierNotice, isPendingApproval {
                // Agent mode punted this call to a human: the classifier
                // was unavailable or is being skipped. Qwen Code renders
                // the same notice and pairs it with the suspend option.
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "bolt.slash.fill")
                        .themedFont(.tiny, weight: .semibold)
                        .foregroundStyle(AppAttention.color)
                    Text(notice)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    AppAttention.color.opacity(0.08),
                    in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            }

            if isPendingApproval, let request = model.browserPermissionApprovalRequest(for: call) {
                BrowserPermissionApprovalView(
                    request: request,
                    onAllowOnce: {
                        model.approvePendingBrowserToolCall(id: call.id, alwaysAllowOrigin: false)
                    },
                    onAlwaysAllow: {
                        model.approvePendingBrowserToolCall(id: call.id, alwaysAllowOrigin: true)
                    },
                    onDeny: {
                        model.denyPendingBrowserToolCall(id: call.id)
                    }
                )
            } else if call.category == .browser {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Browser origin unavailable. This action cannot be approved safely.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    Button(role: .cancel) {
                        model.denyPendingBrowserToolCall(id: call.id)
                    } label: {
                        Text("Deny", bundle: .module)
                    }
                    .buttonStyle(.bordered)
                }
            } else if !isHighRisk, let host = AppNetworkPolicy.host(forWebCall: call) {
                // A URL tool remembers its HOST, never the tool name: one
                // approved fetch must not open every domain. Hidden for high
                // risk, where the engine checks risk before a host grant, so
                // the grant would never apply.
                NetworkAccessApprovalView(
                    host: host,
                    toolLabel: ToolPresentation.resolve(call.name).localizedLabel,
                    onAllowForChat: { model.approvePendingToolCall(id: call.id, alwaysAllowSession: true) },
                    onAllowOnce: { model.approvePendingToolCall(id: call.id, alwaysAllowSession: false) },
                    onAlwaysAllow: {
                        model.addNetworkDomain(host, allow: true)
                        model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                    },
                    onAlwaysBlock: {
                        model.addNetworkDomain(host, allow: false)
                        model.denyPendingToolCall(id: call.id)
                    },
                    onDeny: { model.denyPendingToolCall(id: call.id) }
                )
            } else {
                Text("This action requires your confirmation to execute.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)

                HStack(spacing: 8) {
                    Button {
                        model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                    } label: {
                        Label { Text("Approve Once", bundle: .module) } icon: { Image(systemName: "checkmark") }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(isHighRisk ? theme.warning : theme.accent)
                    .help(Text("Approve this tool call once", bundle: .module))
                    .accessibilityLabel(Text("Approve once \(call.name)", bundle: .module))

                    if model.pendingToolCallClassifierNotice != nil, isPendingApproval {
                        Button {
                            model.suspendAgentModeForSession()
                            model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                        } label: {
                            Label { Text("Suspend Agent Mode", bundle: .module) } icon: { Image(systemName: "pause.circle") }
                        }
                        .buttonStyle(.bordered)
                        .help(Text("Approve once and stop using the classifier for the rest of this session", bundle: .module))
                        .accessibilityLabel(Text("Suspend Agent mode for this session and approve once", bundle: .module))
                    }

                    if call.category == .mcp,
                       let target = McpPermissionRule.targetOfCall(name: call.name, arguments: call.arguments) {
                        // MCP calls can persist the grant on the PROJECT, not
                        // just this session: the rule the engine matches is
                        // written with the same parse that reads the call, so
                        // "this tool" and "this server" mean exactly what the
                        // evaluation will compare.
                        Menu {
                            Button {
                                model.addMcpPermissionRule(serverName: target.server, toolName: target.tool, allow: true)
                                model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                            } label: { Text("Always Allow This Tool", bundle: .module) }
                            Button {
                                model.addMcpPermissionRule(serverName: target.server, toolName: nil, allow: true)
                                model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                            } label: { Text("Always Allow This Server", bundle: .module) }
                        } label: {
                            Label { Text("Always Allow", bundle: .module) } icon: { Image(systemName: "checkmark.seal") }
                        }
                        .buttonStyle(.bordered)
                        .help(Text("Persist an allow rule for this MCP tool or server in project settings", bundle: .module))
                        .accessibilityLabel(Text("Always allow \(call.name) in project settings", bundle: .module))
                    } else if !isHighRisk {
                        // Hidden for high risk: the engine checks risk before
                        // a session grant, so the grant would never apply and
                        // every repeat would ask again.
                        Button {
                            model.approvePendingToolCall(id: call.id, alwaysAllowSession: true)
                        } label: {
                            Label { Text("Always Allow", bundle: .module) } icon: { Image(systemName: "checkmark.circle") }
                        }
                        .buttonStyle(.bordered)
                        .help(call.shellCommand != nil
                            ? Text("Always allow this exact command in this session", bundle: .module)
                            : Text("Always allow this tool and command in this session", bundle: .module))
                        .accessibilityLabel(Text("Always allow \(call.name) in this session", bundle: .module))
                    }

                    Button(role: .cancel) {
                        model.denyPendingToolCall(id: call.id)
                    } label: { Text("Deny", bundle: .module) }
                    .buttonStyle(.bordered)
                    .help(Text("Deny this tool call execution", bundle: .module))
                    .accessibilityLabel(Text("Deny \(call.name)", bundle: .module))
                }
                .controlSize(.small)
            }
        }
        .padding(.top, 4)
    }
}

/// Memoizes `ToolCallDiffFormatter.summarize` per call id. An entry is reused
/// only while the call's name and arguments are equal to what produced it, so
/// a streaming call whose arguments grow recomputes once per change instead
/// of once per read, and a finished call never recomputes.
@MainActor
enum ToolCallSummaryCache {
    private struct Entry {
        let name: String
        let arguments: [String: String]
        let summary: ToolCallSummaryInfo
    }
    private static var entries: [UUID: Entry] = [:]
    private static let maxEntries = 256
    static private(set) var computeCount = 0

    static func summary(for call: AppToolCall) -> ToolCallSummaryInfo {
        if let hit = entries[call.id], hit.name == call.name, hit.arguments == call.arguments {
            return hit.summary
        }
        computeCount += 1
        let summary = ToolCallDiffFormatter.summarize(callName: call.name, arguments: call.arguments)
        if entries.count >= maxEntries { entries.removeAll() }
        entries[call.id] = Entry(name: call.name, arguments: call.arguments, summary: summary)
        return summary
    }

    static func reset() { entries.removeAll(); computeCount = 0 }
}
