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
    @State private var showsFailingTests: Bool = false
    @State private var showsSourceViewer: Bool = false
    /// Edit-before-run: the pending shell command is being edited in place.
    @State private var isEditingCommand: Bool = false
    @State private var commandDraft: String = ""

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

    /// Nil while running (the runner returns only at exit, so there is no
    /// live output to tail) and for results it cannot read with certainty.
    private var terminalSummary: TerminalResultSummary? {
        guard terminalCommand != nil, !isPendingApproval, let result else { return nil }
        return TerminalResultSummary.make(
            output: result.output, isError: result.isError, durationSeconds: result.durationSeconds)
    }

    /// Counts and failing tests when the finished command was a `cargo test`
    /// or `swift test` run the parser can vouch for, else nil (raw output
    /// only). Gated on the terminal summary so a denial or a background
    /// launch never parses.
    private var testRunSummary: TestRunSummary? {
        guard terminalSummary != nil, let result else { return nil }
        return TestRunSummary.parse(result.output)
    }

    /// Hidden once the call failed: no job exists, and the error says why.
    /// Shown while approval is pending, which is when the runs matter most.
    private func cronSummary(now: Date) -> CronScheduleSummary? {
        guard CronScheduleSummary.isCronCreate(call.name), result?.isError != true else { return nil }
        return CronScheduleSummary.make(arguments: call.arguments, hasRun: result != nil, now: now)
    }

    private var submittedPlan: String? {
        guard ["exitplanmode", "exit_plan_mode"].contains(call.name.lowercased()) else { return nil }
        let text = call.arguments["plan"] ?? call.arguments["summary"] ?? call.arguments["content"]
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return text
    }

    /// Whether the agent stopped on this plan, so its card offers Start and
    /// Request changes and holds itself open. Read from the selected chat:
    /// the transcript only draws that chat's rows.
    private var isOfferedPlan: Bool {
        submittedPlan != nil
            && PlanCardActions.isOffered(
                callID: call.id, in: model.selectedChat.messages, isIdle: model.isIdleForPlanCard,
                awaitingCallID: model.pendingPlanApproval?.toolCallID)
    }

    /// Whether the turn stopped on this failed call, so the card offers
    /// Retry, Skip and Tell the model. The status check runs first so the
    /// transcript scan happens only for failed cards.
    private var isOfferedRecovery: Bool {
        call.status == .failed
            && RunRecoveryActions.isOffered(
                callID: call.id, in: model.selectedChat.messages, isIdle: model.isIdleForPlanCard)
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

    /// The validated card for a finished `gen_ui_render` call. Parsed only once
    /// a non-error result exists, so streamed partial arguments never parse.
    private var genUiSpec: GenUiSpec? {
        guard call.name.lowercased() == "gen_ui_render", let result, !result.isError,
            let raw = call.arguments["spec"], case .success(let spec) = GenUiParser.parse(raw)
        else { return nil }
        return spec
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

    /// Arguments are parsed once the call exists in the transcript: the
    /// agent loop appends a call only after the reply finished streaming,
    /// so there is no live argument stream to tail here yet.
    private var fileEditPeek: FileEditPeek? {
        guard !isBrowserCall else { return nil }
        let kind: FileEditPeek.ToolKind
        if isMultiEditCall {
            kind = .multiEdit
        } else if isWriteCall {
            kind = .write
        } else if isEditCall {
            kind = .edit
        } else {
            return nil
        }
        return FileEditPeek.make(kind, arguments: call.arguments)
    }

    /// Header counts. The summary's own for most tools; multiedit's summary
    /// has none (its edits are a JSON list), so they come from the peek.
    private var headerAdditions: Int? {
        if summary.additions != nil || summary.deletions != nil || !isMultiEditCall {
            return summary.additions
        }
        return fileEditPeek.flatMap { $0.additions > 0 ? $0.additions : nil }
    }

    private var headerDeletions: Int? {
        if summary.additions != nil || summary.deletions != nil || !isMultiEditCall {
            return summary.deletions
        }
        return fileEditPeek.flatMap { $0.deletions > 0 ? $0.deletions : nil }
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

            // Outside the expanded section on purpose: the record of how a
            // card was answered stays visible on the collapsed row.
            if !isPendingApproval, let record = call.approvalRecord {
                approvalRecordLine(record)
            }

            // A finished command reads at a glance: the last lines while
            // collapsed (the expanded section shows the full output), and
            // the exit, duration, and line count either way.
            if let terminalSummary {
                terminalSummaryView(terminalSummary, showsTail: !isExpanded)
            }

            if let testRunSummary {
                testRunSummaryView(testRunSummary)
            }

            // The first changed lines of an edit or write without expanding.
            // Hidden once the card opens, where the full diff takes over.
            if !isExpanded, !isPendingApproval, let fileEditPeek {
                fileEditPeekView(fileEditPeek)
            }

            // A schedule reads in words with its next runs, on the collapsed
            // row so it confirms what was set up without opening the card.
            // The minute timeline keeps the runs from going stale on screen.
            if CronScheduleSummary.isCronCreate(call.name) {
                TimelineView(.everyMinute) { context in
                    if let cronSummary = cronSummary(now: context.date) {
                        cronSummaryView(cronSummary)
                    }
                }
            }

            if let genUiSpec {
                // Gen UI (ZP-06): a native card drawn from a validated closed
                // schema. Its follow-up fills the composer draft, never sends.
                GenUiCardView(spec: genUiSpec) { text in
                    model.composeInChat(
                        AppStrings.localized("From the interactive card \(genUiSpec.title):") + "\n\n" + text)
                }
                .id(call.id)
            }

            if isExpanded || isPendingApproval || isActiveQuestionSet || isOfferedPlan {
                VStack(alignment: .leading, spacing: 8) {
                    if isBrowserCall {
                        EmptyView()
                    } else if let submittedPlan {
                        PlanCardView(model: model, plan: submittedPlan, isOffered: isOfferedPlan)
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
                        if isEditingCommand, isPendingApproval {
                            PendingCommandEditorView(
                                text: $commandDraft,
                                onSubmit: submitCommandEdit,
                                onCancel: cancelCommandEdit)
                        } else {
                            ToolCodeCellView(
                                label: "command",
                                code: cmd,
                                language: "bash",
                                downloadFilename: "command.sh"
                            )
                        }
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

            // Outside the expanded section, like the approval record: the
            // way forward from a failure stays on the collapsed row.
            if isOfferedRecovery {
                recoveryFooter
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
        // OC-09: "Reveal" in the sources popover. `initial` so a card created
        // after the request (lazy row, group that was closed) still sees it.
        .onChange(of: model.toolCardReveal, initial: true) { _, reveal in
            guard reveal?.callID == call.id,
                  model.consumeToolCardReveal(forCallID: call.id)
            else { return }
            TSMotion.animate(TSMotion.select) { isExpanded = true }
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

                if let additions = headerAdditions {
                    Text(verbatim: "+\(additions)")
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(diffAdditionColor)
                }

                if let deletions = headerDeletions {
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
            .strikethrough(isCancelled, color: theme.secondaryText)
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
            .strikethrough(isCancelled, color: theme.secondaryText)
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
            .strikethrough(isCancelled, color: theme.secondaryText)
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
            .strikethrough(isCancelled, color: theme.secondaryText)
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

            if let viewer = sourceViewerRequest {
                Spacer(minLength: 0)
                Button { showsSourceViewer = true } label: {
                    Text("View lines \(viewer.range.lowerBound)-\(viewer.range.upperBound)", bundle: .module)
                }
                .appAction(.quiet, size: .small)
                .sheet(isPresented: $showsSourceViewer) {
                    SourceLineViewerSheet(
                        displayPath: viewer.path,
                        resolution: SourceLineViewer.resolve(
                            path: viewer.path, projectRoot: model.selectedProject?.rootDirectoryURL),
                        requested: viewer.range,
                        readAt: viewer.readAt)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    /// What the read-only source viewer needs, offered only after a read
    /// that succeeded with a numeric range and one unambiguous path.
    private var sourceViewerRequest: (path: String, range: ClosedRange<Int>, readAt: Date)? {
        guard let result, !result.isError,
            let range = SourceLineViewer.requestedRange(call.arguments)
        else { return nil }
        let resolved = ToolArgumentResolver.fileArguments(call.arguments, forRead: true)
        guard resolved.conflict == nil, let path = resolved.path, !path.isEmpty else { return nil }
        return (path, range, call.createdAt.addingTimeInterval(result.durationSeconds))
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

    /// "Allowed once by you, 3:42 PM": glyph plus words, never color alone
    /// (StatusLabel). The header above already shows the command, struck
    /// through when denied.
    private func approvalRecordLine(_ record: ToolApprovalRecord) -> some View {
        let line = ToolApprovalRecordLine(record)
        let time = record.decidedAt.formatted(
            Calendar.current.isDateInToday(record.decidedAt)
                ? .dateTime.hour().minute() : .dateTime.month(.abbreviated).day().hour().minute())
        return HStack(spacing: 5) {
            Image(systemName: line.isEdited ? "pencil.circle" : (line.isAllowed ? "checkmark.circle" : "xmark.octagon"))
                .themedFont(.tiny, weight: .semibold)
                .foregroundStyle(line.isEdited
                    ? AppThemeColor.appSecondary
                    : (line.isAllowed ? AppThemeColor.appSuccess : AppThemeColor.appDanger))
                .accessibilityHidden(true)
            Text(LocalizedStringKey(line.labelKey), bundle: .module)
                .themedFont(.tiny, weight: .medium)
                .foregroundStyle(.appText)
            Text(verbatim: time)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
        .lineLimit(1)
        .padding(.leading, ConversationLayout.activityIconWidth + 7)
        .accessibilityElement(children: .combine)
    }

    /// Retry and Skip send a reply, Tell the model drafts one; none runs the
    /// call itself (`RunRecoveryActions`). No keyboard shortcut: Return
    /// belongs to the composer, and inline strips must not steal it.
    private var recoveryFooter: some View {
        HStack(spacing: 8) {
            Button {
                model.retryFailedCallFromCard(call)
            } label: {
                Text("Retry", bundle: .module)
            }
            .appAction(.secondary, size: .small)
            .help(Text(
                "Sends a reply asking the agent to try this call again. The new call is approved like any other.",
                bundle: .module))

            Button {
                model.skipFailedCallFromCard(call)
            } label: {
                Text("Skip", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            .help(Text("Sends a reply asking the agent to continue without this call", bundle: .module))

            Button {
                model.explainFailedCallFromCard(call)
            } label: {
                Text("Tell the model", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            .help(Text("Drafts a reply in the composer so you can tell the agent what went wrong", bundle: .module))
        }
        .padding(.leading, ConversationLayout.activityIconWidth + 7)
        .padding(.bottom, 2)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Recover from the failed call", bundle: .module))
    }

    /// Up to six lines, each led by a + or - glyph so the kind never rests on
    /// color alone; the tint follows the diff-markers setting. The overflow
    /// row expands the card to the full diff.
    private func fileEditPeekView(_ peek: FileEditPeek) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(peek.lines.enumerated()), id: \.offset) { _, line in
                let isAdded = line.kind == .added
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(verbatim: isAdded ? "+" : "-")
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(isAdded ? diffAdditionColor : diffDeletionColor)
                    // A blank line still takes its row.
                    Text(verbatim: line.text.isEmpty ? " " : line.text)
                        .font(theme.code(.small))
                        .foregroundStyle(.appText)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                .padding(.horizontal, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background {
                    if appearance.diffMarkers == .color {
                        Rectangle().fill(
                            isAdded ? AnyShapeStyle(.appSuccess.opacity(0.08))
                                : AnyShapeStyle(.appDanger.opacity(0.08)))
                    }
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(
                    isAdded
                        ? Text("Added line: \(line.text)", bundle: .module)
                        : Text("Removed line: \(line.text)", bundle: .module))
            }
            if peek.hiddenLineCount > 0 {
                Button {
                    TSMotion.animate(TSMotion.select) { isExpanded = true }
                } label: {
                    Text("Show \(peek.hiddenLineCount) more lines", bundle: .module)
                        .font(theme.code(.tiny))
                        .foregroundStyle(.appAccentText)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .padding(.leading, ConversationLayout.activityIconWidth + 7)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// "Weekdays at 09:00" beside the raw expression (the expression alone
    /// when the shape has no name or does not parse), then the next runs in
    /// the UI locale, under a heading that names the time zone.
    private func cronSummaryView(_ summary: CronScheduleSummary) -> some View {
        let locale = AppStrings.currentLanguage.locale
        let zone = TimeZone.current
        let zoneName = CronScheduleSummary.zoneLabel(zone, locale: locale)
        return VStack(alignment: .leading, spacing: 1) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                if let description = summary.description {
                    Text(verbatim: description)
                        .font(theme.code(.small, weight: .semibold))
                        .foregroundStyle(.appText)
                }
                Text(verbatim: summary.expression)
                    .font(theme.code(.small))
                    .foregroundStyle(summary.description == nil ? AnyShapeStyle(.appText) : AnyShapeStyle(.appSecondary))
                    .textSelection(.enabled)
                if !summary.recurring {
                    Text(verbatim: "\u{00B7}").foregroundStyle(.appSecondary).accessibilityHidden(true)
                    Text("Runs once", bundle: .module)
                        .font(theme.code(.tiny))
                        .foregroundStyle(.appSecondary)
                }
            }
            .lineLimit(1)
            .truncationMode(.tail)
            if !summary.nextRuns.isEmpty {
                Group {
                    if summary.recurring {
                        Text("Next runs (\(zoneName))", bundle: .module)
                    } else {
                        Text("Next run (\(zoneName))", bundle: .module)
                    }
                }
                .font(theme.code(.tiny))
                .foregroundStyle(.appSecondary)
                .padding(.top, 2)
                ForEach(summary.nextRuns, id: \.self) { run in
                    Text(verbatim: CronScheduleSummary.runLabel(run, locale: locale, timeZone: zone))
                        .font(theme.code(.small))
                        .foregroundStyle(.appText)
                        .lineLimit(1)
                }
            }
        }
        .padding(.leading, ConversationLayout.activityIconWidth + 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    /// Up to three tail lines, then "Exit 0, 1.2 s, 48 lines" (dot separated). A failure is
    /// named in words ("Exit 2", "Timed out"), with color only as a second
    /// cue; a benign exit 1 (grep with no match) is not an error and stays
    /// secondary.
    private func terminalSummaryView(_ summary: TerminalResultSummary, showsTail: Bool) -> some View {
        let isFailure = result?.isError == true
        return VStack(alignment: .leading, spacing: 1) {
            if showsTail {
                ForEach(Array(summary.tail.enumerated()), id: \.offset) { _, line in
                    Text(verbatim: line)
                        .font(theme.code(.small))
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
            HStack(spacing: 5) {
                Group {
                    switch summary.outcome {
                    case .exited(let code):
                        Text("Exit \(code)", bundle: .module)
                    case .timedOut:
                        Text("Timed out", bundle: .module)
                    }
                }
                .foregroundStyle(isFailure ? AnyShapeStyle(.appDanger) : AnyShapeStyle(.appSecondary))
                if let duration = summary.durationText {
                    Text(verbatim: "\u{00B7}").foregroundStyle(.appSecondary).accessibilityHidden(true)
                    Text(verbatim: duration).foregroundStyle(.appSecondary)
                }
                // "%lld lines" has no plural forms in the catalog, so a single
                // line (already visible in the tail) is not counted.
                if let count = summary.lineCount, count > 1 {
                    Text(verbatim: "\u{00B7}").foregroundStyle(.appSecondary).accessibilityHidden(true)
                    Text("\(count) lines", bundle: .module).foregroundStyle(.appSecondary)
                }
            }
            .font(theme.code(.tiny))
            .lineLimit(1)
        }
        .padding(.leading, ConversationLayout.activityIconWidth + 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    /// "12 passed, 2 failed, 1 skipped, 3.4 s" behind a pass or fail glyph,
    /// then a collapsed list of the failing tests, each with its first
    /// failure line. The status is the glyph and the words; color is the
    /// second cue.
    private func testRunSummaryView(_ summary: TestRunSummary) -> some View {
        let separator = Text(verbatim: "\u{00B7}").foregroundStyle(.appSecondary).accessibilityHidden(true)
        return VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 5) {
                Image(systemName: summary.allPassed ? "checkmark.circle.fill" : "xmark.octagon.fill")
                    .foregroundStyle(summary.allPassed ? AnyShapeStyle(.appSuccess) : AnyShapeStyle(.appDanger))
                    .accessibilityHidden(true)
                Text("\(summary.passed) passed", bundle: .module)
                    .foregroundStyle(summary.allPassed ? AnyShapeStyle(.appSuccess) : AnyShapeStyle(.appSecondary))
                if summary.failed > 0 {
                    separator
                    Text("\(summary.failed) failed", bundle: .module).foregroundStyle(.appDanger)
                }
                if summary.skipped > 0 {
                    separator
                    Text("\(summary.skipped) skipped", bundle: .module).foregroundStyle(.appSecondary)
                }
                if let seconds = summary.durationSeconds,
                   let duration = TerminalResultSummary.durationText(seconds: seconds) {
                    separator
                    Text(verbatim: duration).foregroundStyle(.appSecondary)
                }
            }
            .font(theme.code(.tiny))
            .lineLimit(1)
            .accessibilityElement(children: .combine)

            if !summary.failures.isEmpty {
                Button {
                    TSMotion.animate(TSMotion.select) { showsFailingTests.toggle() }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.right")
                            .rotationEffect(.degrees(showsFailingTests ? 90 : 0))
                            .accessibilityHidden(true)
                        Text("Failing tests", bundle: .module)
                    }
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityValue(
                    showsFailingTests ? AppStrings.localized("Expanded") : AppStrings.localized("Collapsed"))

                if showsFailingTests {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(summary.failures.enumerated()), id: \.offset) { _, failure in
                            VStack(alignment: .leading, spacing: 1) {
                                HStack(spacing: 4) {
                                    Image(systemName: "xmark.octagon")
                                        .foregroundStyle(.appDanger)
                                        .accessibilityHidden(true)
                                    Text(verbatim: failure.name)
                                        .foregroundStyle(.appText)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                }
                                if let message = failure.message {
                                    Text(verbatim: message)
                                        .foregroundStyle(.appSecondary)
                                        .lineLimit(2)
                                        .truncationMode(.tail)
                                        .padding(.leading, 16)
                                }
                            }
                            .font(theme.code(.tiny))
                            .textSelection(.enabled)
                            .accessibilityElement(children: .combine)
                        }
                    }
                    .padding(.leading, 8)
                }
            }
        }
        .padding(.leading, ConversationLayout.activityIconWidth + 7)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// The model's stated reason and the directory a shell command starts
    /// in, so the user can judge the call without expanding anything else.
    @ViewBuilder
    private var approvalContext: some View {
        let reason = ToolApprovalContext.reason(for: call)
        let cwd: String? = {
            guard terminalCommand != nil, model.pendingToolCall?.id == call.id,
                  let root = model.pendingToolCallProject?.rootDirectoryURL
            else { return nil }
            // The tracker is where the runner will start, which can differ
            // from the root after an earlier `cd` in this chat.
            let start = ShellCwdTracker.shared.startDirectory(
                for: root, chatID: model.pendingToolCallChatID)
            return ToolApprovalContext.displayPath(start)
        }()
        if reason != nil || cwd != nil {
            VStack(alignment: .leading, spacing: 3) {
                if let reason {
                    Text(verbatim: reason)
                        .themedFont(.small)
                        .foregroundStyle(.appText)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityLabel(Text("Reason: \(reason)", bundle: .module))
                }
                if let cwd {
                    HStack(spacing: 4) {
                        Image(systemName: "folder")
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .accessibilityHidden(true)
                        Text(verbatim: cwd)
                            .font(theme.code(.tiny))
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                            .truncationMode(.head)
                    }
                    .help(Text("Runs in \(cwd)", bundle: .module))
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel(Text("Runs in \(cwd)", bundle: .module))
                }
            }
        }
    }

    /// On an edited call's card: the command the model proposed, so the
    /// user approves the edit knowing what it replaced.
    @ViewBuilder
    private var editedFromLine: some View {
        if let proposed = call.editedFromCommand {
            VStack(alignment: .leading, spacing: 2) {
                Label {
                    Text("You edited this command. The model proposed:", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(.appText)
                } icon: {
                    Image(systemName: "pencil")
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                }
                Text(verbatim: proposed)
                    .font(theme.code(.tiny))
                    .foregroundStyle(.appSecondary)
                    .lineLimit(3)
                    .truncationMode(.tail)
                    .textSelection(.enabled)
            }
            .accessibilityElement(children: .combine)
        }
    }

    private func beginCommandEdit() {
        commandDraft = call.shellCommand ?? ""
        isEditingCommand = true
    }

    /// Esc or Cancel: the draft is dropped and the model's command shows
    /// again, untouched.
    private func cancelCommandEdit() {
        isEditingCommand = false
        commandDraft = ""
    }

    /// The model decides; an unchanged or empty draft submits nothing and
    /// reads as a cancel.
    private func submitCommandEdit() {
        model.submitEditedPendingToolCall(id: call.id, command: commandDraft)
        cancelCommandEdit()
    }

    private func commandEditHelp(_ unavailable: ToolCallCommandEdit.Unavailable?) -> Text {
        switch unavailable {
        case .batch:
            return Text("Editing is not available for a batch of calls. Approve or deny the batch.", bundle: .module)
        case .compound:
            return Text("Editing is not available for a command with a validation step.", bundle: .module)
        case nil:
            return Text("Edit this command. The edit is checked again and needs its own approval.", bundle: .module)
        }
    }

    private var approvalPrompt: some View {
        VStack(alignment: .leading, spacing: 8) {
            if isPendingApproval {
                approvalContext
                editedFromLine
            }

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
                    .appAction(.secondary, size: .small)
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
                        model.approvePendingToolCall(
                            id: call.id, alwaysAllowSession: false, recordAs: .allowedAlways)
                    },
                    onAlwaysBlock: {
                        model.addNetworkDomain(host, allow: false)
                        model.denyPendingToolCall(id: call.id)
                    },
                    onDeny: { model.denyPendingToolCall(id: call.id) }
                )
            } else if isEditingCommand, isPendingApproval {
                // The editor above carries its own Use and Cancel; Approve
                // stays out of reach so the unedited command cannot be
                // approved while the user is mid-edit.
                EmptyView()
            } else {
                Text("This action requires your confirmation to execute.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)

                HStack(spacing: 8) {
                    // The approval row speaks the shared action vocabulary,
                    // like the network approval card beside it: primary is
                    // the one approve, secondary everything else.
                    Button {
                        model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                    } label: {
                        Label { Text("Approve Once", bundle: .module) } icon: { Image(systemName: "checkmark") }
                    }
                    .appAction(.primary, size: .small)
                    .tint(isHighRisk ? theme.warning : theme.accent)
                    .help(Text("Approve this tool call once", bundle: .module))
                    .accessibilityLabel(Text("Approve once \(call.name)", bundle: .module))

                    if model.showsPendingCommandEdit(for: call) {
                        let unavailable = model.pendingCommandEditUnavailableReason(for: call)
                        Button {
                            beginCommandEdit()
                        } label: {
                            Label { Text("Edit", bundle: .module) } icon: { Image(systemName: "pencil") }
                        }
                        .appAction(.secondary, size: .small)
                        .disabled(unavailable != nil)
                        .help(commandEditHelp(unavailable))
                        .accessibilityLabel(Text("Edit the command before approving", bundle: .module))
                    }

                    if model.pendingToolCallClassifierNotice != nil, isPendingApproval {
                        Button {
                            model.suspendAgentModeForSession()
                            model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                        } label: {
                            Label { Text("Suspend Agent Mode", bundle: .module) } icon: { Image(systemName: "pause.circle") }
                        }
                        .appAction(.secondary, size: .small)
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
                                model.approvePendingToolCall(
                                    id: call.id, alwaysAllowSession: false, recordAs: .allowedAlways)
                            } label: { Text("Always Allow This Tool", bundle: .module) }
                            Button {
                                model.addMcpPermissionRule(serverName: target.server, toolName: nil, allow: true)
                                model.approvePendingToolCall(
                                    id: call.id, alwaysAllowSession: false, recordAs: .allowedAlways)
                            } label: { Text("Always Allow This Server", bundle: .module) }
                        } label: {
                            Label { Text("Always Allow", bundle: .module) } icon: { Image(systemName: "checkmark.seal") }
                        }
                        .appMenuAction(size: .small)
                        .help(Text("Persist an allow rule for this MCP tool or server in project settings", bundle: .module))
                        .accessibilityLabel(Text("Always allow \(call.name) in project settings", bundle: .module))
                    } else if !isHighRisk, ToolCallCommandEdit.allowsStandingGrant(call) {
                        // Hidden for high risk: the engine checks risk before
                        // a session grant, so the grant would never apply and
                        // every repeat would ask again. Hidden for an edited
                        // command too: the user's own text is approved once.
                        Button {
                            model.approvePendingToolCall(id: call.id, alwaysAllowSession: true)
                        } label: {
                            Label { Text("Always Allow", bundle: .module) } icon: { Image(systemName: "checkmark.circle") }
                        }
                        .appAction(.secondary, size: .small)
                        .help(call.shellCommand != nil
                            ? Text("Always allow this exact command in this session", bundle: .module)
                            : Text("Always allow this tool and command in this session", bundle: .module))
                        .accessibilityLabel(Text("Always allow \(call.name) in this session", bundle: .module))
                    }

                    Button(role: .cancel) {
                        model.denyPendingToolCall(id: call.id)
                    } label: { Text("Deny", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .help(Text("Deny this tool call execution", bundle: .module))
                    .accessibilityLabel(Text("Deny \(call.name)", bundle: .module))
                }
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
