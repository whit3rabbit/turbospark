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

    private var isAgentCall: Bool {
        AppModel.isAgentFamilyToolName(call.name)
    }

    private var isStopAgentCall: Bool {
        let n = call.name.lowercased()
        return n == "stop_agent" || n == "stopagent"
    }

    private var isWriteCall: Bool {
        let n = call.name.lowercased()
        return n.contains("write") || n.contains("create")
    }

    private var isEditCall: Bool {
        let n = call.name.lowercased()
        return n.contains("edit") || n.contains("replace")
    }

    private var isPatchCall: Bool {
        let n = call.name.lowercased()
        return n == "apply_patch" || n == "applypatch"
    }

    private var isReadCall: Bool {
        let n = call.name.lowercased()
        return n.contains("read") || n.contains("view")
    }

    private var isWebFetchCall: Bool {
        let n = call.name.lowercased()
        return n.contains("fetch") || n.contains("read_url")
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
                    if isTodoCall, let todos = parsedTodos, !todos.isEmpty {
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
                    } else if isEditCall {
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
                    Text("+\(additions)", bundle: .module)
                        .font(theme.code(.small, weight: .bold))
                        .foregroundStyle(diffAdditionColor)
                }

                if let deletions = summary.deletions {
                    Text("-\(deletions)", bundle: .module)
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

                Spacer()

                statusBadge

                Image(systemName: "chevron.right")
                    .themedFont(.tiny, weight: .semibold)
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
                    .themedFont(.small)
                    .foregroundStyle(isHighRisk ? Color.red : Color.orange)
            } else {
                switch call.status {
                case .running:
                    TaskProgressFlameIcon(size: 12)
                case .completed:
                    Image(systemName: "checkmark.circle.fill")
                        .themedFont(.small)
                        .foregroundStyle(Color.green)
                case .denied:
                    Image(systemName: "minus.circle.fill")
                        .themedFont(.small)
                        .foregroundStyle(Color.secondary)
                case .failed:
                    Image(systemName: "xmark.circle.fill")
                        .themedFont(.small)
                        .foregroundStyle(Color.red)
                case .pendingApproval:
                    Image(systemName: "exclamationmark.triangle.fill")
                        .themedFont(.small)
                        .foregroundStyle(Color.orange)
                }
            }
        }
    }

    @ViewBuilder
    private var headerLabelView: some View {
        let isCancelled = call.status == .denied
        if let cmd = terminalCommand {
            Text("$ \(cmd)", bundle: .module)
                .font(theme.code(.small, weight: .semibold))
                .foregroundStyle(isCancelled ? .secondary : .primary)
                .strikethrough(isCancelled, color: .secondary)
                .lineLimit(1)
        } else if call.category == .web, let query = call.arguments["query"] {
            HStack(spacing: 4) {
                Text("Search:", bundle: .module)
                    .themedFont(.base, weight: .medium)
                    .foregroundStyle(.appSecondary)
                Text("\"\(query)\"", bundle: .module)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.appText)
            }
            .strikethrough(isCancelled, color: .secondary)
            .lineLimit(1)
        } else {
            HStack(spacing: 5) {
                Text(summary.action)
                    .themedFont(.base, weight: .medium)
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
                .themedFont(.micro)
            Text(risk.level.label)
                .themedFont(.tiny, weight: .semibold)
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
                    .themedFont(.small)
                Text(risk.level == .high ? "High-Risk Operation Detected" : "Security Notice")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(risk.level == .high ? Color.red : Color.orange)
            }
            ForEach(risk.reasons, id: \.self) { reason in
                Text("- \(reason)", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
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
            .themedFont(.tiny, weight: .medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(statusBackground, in: Capsule())
            .foregroundStyle(statusForeground)
    }

    private var statusLabel: String {
        if isPendingApproval { return "Needs Approval" }
        // A classifier-approved call is marked so "who decided this" is
        // always on the card (`swift/docs/SWIFT_AGENT_MODE.md`).
        if call.autoApprovedBy == "classifier" {
            switch call.status {
            case .running: return "Classifier Approved"
            case .completed: return "Classifier Approved"
            default: break
            }
        }
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
                TodoItemRow(item: item)
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
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private func findingsPreview(_ findings: [CodeFindingItem]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(findings.enumerated()), id: \.offset) { idx, f in
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.circle.fill")
                            .themedFont(.tiny)
                            .foregroundStyle(.orange)

                        Text(f.file + (f.line.map { ":\($0)" } ?? ""))
                            .font(theme.code(.small, weight: .bold))
                            .foregroundStyle(.appText)

                        if let cat = f.category {
                            Text(cat)
                                .themedFont(.tiny)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Color.secondary.opacity(0.12), in: Capsule())
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
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
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
                    .background(Color.purple.opacity(0.15), in: Capsule())
                    .foregroundStyle(Color.purple)
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
                    .foregroundStyle(.primary.opacity(0.9))
                    .lineLimit(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 6))
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private var stopAgentPreview: some View {
        let taskId = call.arguments["task_id"] ?? call.arguments["taskId"] ?? call.arguments["id"] ?? "agent"
        return HStack(spacing: 6) {
            Image(systemName: "stop.circle.fill")
                .themedFont(.small)
                .foregroundStyle(Color.red)
            Text("Stop Background Agent:", bundle: .module)
                .themedFont(.small, weight: .medium)
                .foregroundStyle(.appSecondary)
            Text(taskId)
                .font(theme.code(.small, weight: .bold))
                .foregroundStyle(.appText)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private var fileWritePreview: some View {
        let targetPath = call.arguments["TargetFile"]
            ?? call.arguments["AbsolutePath"]
            ?? call.arguments["path"]
            ?? call.arguments["file_path"]
            ?? call.arguments["filePath"]
            ?? call.arguments["file"]
            ?? "file"
        let content = call.arguments["CodeContent"]
            ?? call.arguments["content"]
            ?? call.arguments["text"]
            ?? call.arguments["code"]
            ?? ""
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
                    downloadFilename: fileName
                )
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private var fileEditPreview: some View {
        let targetPath = call.arguments["TargetFile"]
            ?? call.arguments["AbsolutePath"]
            ?? call.arguments["path"]
            ?? call.arguments["file_path"]
            ?? call.arguments["filePath"]
            ?? call.arguments["file"]
            ?? "file"
        let oldStr = call.arguments["TargetContent"]
            ?? call.arguments["old_string"]
            ?? call.arguments["oldString"]
            ?? call.arguments["target"]
            ?? ""
        let newStr = call.arguments["ReplacementContent"]
            ?? call.arguments["new_string"]
            ?? call.arguments["newString"]
            ?? call.arguments["replacement"]
            ?? ""

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
                        .foregroundStyle(.primary.opacity(0.9))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(6)
                        .background(Color.red.opacity(0.06), in: RoundedRectangle(cornerRadius: 6))
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
                        .foregroundStyle(.primary.opacity(0.9))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(6)
                        .background(Color.green.opacity(0.06), in: RoundedRectangle(cornerRadius: 6))
                        .textSelection(.enabled)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
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
        let targetPath = call.arguments["AbsolutePath"]
            ?? call.arguments["path"]
            ?? call.arguments["file_path"]
            ?? call.arguments["filePath"]
            ?? call.arguments["file"]
            ?? call.arguments["TargetFile"]
            ?? "file"
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
                    .background(Color.secondary.opacity(0.12), in: Capsule())
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
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

            if let url = URL(string: urlString), url.scheme != nil {
                Link(urlString, destination: url)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.appAccent)
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
                .background(Color.secondary.opacity(0.12), in: Capsule())
                .foregroundStyle(.appSecondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
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
                        Text("\(key):", bundle: .module)
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
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
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
                        .background(Color.indigo.opacity(0.15), in: Capsule())
                        .foregroundStyle(Color.indigo)
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
                        Text("\(key):", bundle: .module)
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
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
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
                        Text("\(key):", bundle: .module)
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
            if let notice = model.pendingToolCallClassifierNotice, isPendingApproval {
                // Agent mode punted this call to a human: the classifier
                // was unavailable or is being skipped. Qwen Code renders
                // the same notice and pairs it with the suspend option.
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "bolt.slash.fill")
                        .themedFont(.tiny, weight: .semibold)
                        .foregroundStyle(Color.orange)
                    Text(notice)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    Color.orange.opacity(0.08),
                    in: RoundedRectangle(cornerRadius: 6))
            }

            Text("This action requires your confirmation to execute.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)

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

                if model.pendingToolCallClassifierNotice != nil, isPendingApproval {
                    Button {
                        model.suspendAgentModeForSession()
                        model.approvePendingToolCall(id: call.id, alwaysAllowSession: false)
                    } label: {
                        Label("Suspend Agent Mode", systemImage: "pause.circle")
                    }
                    .buttonStyle(.bordered)
                    .help("Approve once and stop using the classifier for the rest of this session")
                    .accessibilityLabel("Suspend Agent mode for this session and approve once")
                }

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
