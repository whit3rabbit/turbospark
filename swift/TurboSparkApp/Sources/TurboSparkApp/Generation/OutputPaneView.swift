import AppKit
import SwiftUI
import TurboSpark

/// Main view displaying generated responses, conversation transcripts, or empty-state guidance.
struct OutputPaneView: View {
    @ObservedObject var model: AppModel
    @State private var responseCopyFeedbackID: UUID?
    @State private var lastRunningState = false

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .task(id: responseCopyFeedbackID) {
            guard responseCopyFeedbackID != nil else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.15)) {
                responseCopyFeedbackID = nil
            }
        }
        .contextMenu {
            Button("Copy response") {
                copyResponse()
            }
            .disabled(model.outputResponsePlainText.isEmpty)

            Button("Copy conversation") {
                copy(model.outputConversationPlainText)
            }
            .disabled(model.outputConversationPlainText.isEmpty)

            Divider()

            Button("Session Stats...") {
                model.showSessionStats = true
            }
            Button("Export Conversation...") {
                model.exportSelectedChat(format: .markdown)
            }
            Button("Help...") {
                model.showHelpSheet = true
            }

            Divider()

            Button("Background Tasks...") {
                model.showTasksSheet = true
            }
            Button("Context Usage...") {
                model.showContextSheet = true
            }

            Divider()

            if model.collapsedTurnAnchors.isEmpty {
                Button("Collapse finished turns") {
                    model.collapseAllTurns()
                }
                .disabled(!model.canCollapseTurns)
            } else {
                Button("Expand collapsed turns") {
                    model.collapsedTurnAnchors.removeAll()
                }
            }

            Button("Clear chat history") { model.clearOutput() }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
        .sheet(isPresented: $model.showSessionStats) {
            SessionStatsSheet(model: model)
        }
        .sheet(isPresented: $model.showHelpSheet) {
            HelpSheetView(model: model)
        }
        // The qwen-code local-command sheets (`/context`, `/tasks`,
        // `/tools`, `/status`, `/rewind`, `/recap`).
        .sheet(isPresented: $model.showContextSheet) {
            ContextUsageSheet(model: model)
        }
        .sheet(isPresented: $model.showTasksSheet) {
            TasksStatusSheet(model: model)
        }
        .sheet(isPresented: $model.showToolsSheet) {
            ToolsListSheet(model: model)
        }
        .sheet(isPresented: $model.showStatusSheet) {
            StatusSheet(model: model)
        }
        .sheet(isPresented: $model.showRewindSheet) {
            RewindSheet(model: model)
        }
        .sheet(isPresented: $model.showRecapSheet) {
            RecapSheet(model: model)
        }
        // The qwen-code git-command sheet (`/diff`, `/log`, `/prs`).
        .sheet(isPresented: $model.showGitSheet) {
            GitInfoSheet(model: model)
        }
        // `/delete` asks first, exactly like the sidebar's Delete action.
        .alert(
            "Delete this chat?",
            isPresented: $model.confirmDeleteChat
        ) {
            Button("Delete", role: .destructive) {
                model.deleteChat(id: model.selectedChatID)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The conversation '\(model.selectedChat.title)' will be removed. This cannot be undone.", bundle: .module)
        }
        .onChange(of: model.isRunning) { wasRunning, isRunning in
            // Announce when generation finishes so a screen-reader user knows
            // they can read the response. Streaming text itself is too
            // granular to announce per token, but the completion moment is.
            // This app is macOS-only (.macOS(.v14) in Package.swift), so the
            // static AccessibilityNotification.Announcement.post is the
            // cross-platform API to use here; the iOS-only
            // \.accessibilityAnnouncementQueue environment value is not.
            if wasRunning && !isRunning {
                // A turn that ends by itself disarms a two-stage Esc that
                // never got its second press.
                model.disarmEscCancel()
                let count = model.liveTokenCount
                let message = count > 0
                    ? "Generation finished. \(count) tokens."
                    : "Generation finished."
                _ = AccessibilityNotification.Announcement.post(.init(message))
            }
            lastRunningState = isRunning
        }
    }

    private var placeholder: some View {
        ScrollView {
            WelcomeHeroView(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.vertical, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var transcript: some View {
        ChatTranscriptView(model: model)
            .id(model.selectedChatID)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func copyResponse() {
        copy(model.outputResponsePlainText)
        withAnimation(.easeIn(duration: 0.15)) {
            responseCopyFeedbackID = UUID()
        }
    }
}
/// Native SwiftUI transcript view rendering multi-turn conversations with Markdown formatting.
private struct ChatTranscriptView: View {
    @ObservedObject var model: AppModel

    /// Transcript turns (qwen-code collapse parity): a user prompt anchors
    /// each turn; a collapsed turn renders prompt + final answer with the
    /// hidden rows behind a toggle. Anchors are UUIDs, so a stale anchor
    /// from another chat simply matches nothing.
    private var transcriptTurns: [(offset: Int, element: TurnCollapseModel.Turn)] {
        Array(TurnCollapseModel.turns(in: model.selectedTurnMessages).enumerated())
    }

    private var messagesByID: [UUID: AppChatMessage] {
        Dictionary(uniqueKeysWithValues: model.selectedTurnMessages.map { ($0.id, $0) })
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 20) {
                    // The compaction divider sits above the boundary rows,
                    // which are a PREFIX of the transcript.
                    let compaction = model.selectedCompactionState
                    if compaction.boundary > 0 {
                        CompactionDividerView(
                            summary: compaction.summary ?? "",
                            summarizedRows: compaction.boundary)
                    }
                    let turns = transcriptTurns
                    let byID = messagesByID
                    ForEach(turns, id: \.offset) { _, turn in
                        turnRows(turn, byID: byID)
                    }

                    NewChatSuggestionBanner(model: model)

                    // The live task checklist sits OUTSIDE the streaming row
                    // (same rationale as BackgroundAgentsStripView below): it
                    // must stay visible while the turn runs AND after it ends.
                    TaskChecklistPanelView(model: model)

                    // The active-goal banner, same rationale: the goal must
                    // stay visible (and stoppable) through its whole loop.
                    GoalBannerView(model: model)

                    // The interactive AskUserQuestion card (qwen-code
                    // parity). It lives OUTSIDE the streaming row like the
                    // checklist and the goal banner: the tool call's own
                    // transcript row does not exist until its result lands,
                    // and the whole point is that the result waits for the
                    // user's pick. Scoped to the asking chat -- the same
                    // visibility rule as the strips below -- so a question
                    // parked by one chat does not render inside another.
                    if let pendingQuestions = model.pendingUserQuestions,
                        pendingQuestions.chatID == nil
                            || pendingQuestions.chatID == model.selectedChatID
                    {
                        InteractiveQuestionCardView(
                            model: model,
                            questions: pendingQuestions.items)
                    }

                    if model.isRunning || !model.outputText.isEmpty || !model.outputReasoningText.isEmpty {
                        ActiveStreamingRowView(
                            model: model,
                            output: model.outputText,
                            reasoning: model.outputReasoningText,
                            isRunning: model.isRunning
                        )
                    }

                    BackgroundAgentsStripView(model: model)

                    BackgroundShellsStripView(model: model)

                    Color.clear
                        .frame(height: 1)
                        .id("bottom")
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
            }
            .onChange(of: model.outputText) {
                if model.isRunning {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onChange(of: model.outputReasoningText) {
                if model.isRunning {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            // A parked question set scrolls itself into view: mid-turn it is
            // the one thing the user has to act on.
            .onChange(of: model.pendingUserQuestions) { _, pending in
                if pending != nil {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            .onAppear {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            // Turn navigation (qwen-code parity): the menu commands and
            // chords bump a token; this is where the jump actually lands.
            .onChange(of: model.turnNavigationToken) { _, _ in
                guard let target = model.turnNavigationTargetID else { return }
                withAnimation(.easeInOut(duration: 0.2)) {
                    proxy.scrollTo(target, anchor: .top)
                }
            }
        }
    }

    /// The rows of one turn: every message when expanded, prompt + toggle +
    /// final answer when collapsed. A collapse that would hide nothing
    /// renders expanded.
    @ViewBuilder
    private func turnRows(
        _ turn: TurnCollapseModel.Turn, byID: [UUID: AppChatMessage]
    ) -> some View {
        let hidden = TurnCollapseModel.hiddenCount(for: turn, messagesByID: byID)
        let isCollapsed = turn.isCollapsible && hidden > 0
            && model.collapsedTurnAnchors.contains(turn.anchorID!)
        let visibleIDs = isCollapsed
            ? TurnCollapseModel.visibleIDs(collapsedFor: turn, messagesByID: byID)
            : turn.messageIDs
        VStack(alignment: .leading, spacing: 20) {
            ForEach(visibleIDs, id: \.self) { messageID in
                if let message = byID[messageID] {
                    MessageRowView(model: model, message: message)
                        .id(message.id)
                    // The toggle sits under the PROMPT of a collapsed turn
                    // (qwen-code's "expand the middle steps" affordance);
                    // only rendered when something is actually hidden.
                    if isCollapsed && messageID == visibleIDs.first {
                        TurnCollapseToggleRow(count: hidden) {
                            withAnimation(.easeInOut(duration: 0.15)) {
                                _ = model.collapsedTurnAnchors.remove(turn.anchorID!)
                            }
                        }
                    }
                }
            }
            if !isCollapsed && turn.isCollapsible && hidden > 0 && !model.isRunning {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        _ = model.collapsedTurnAnchors.insert(turn.anchorID!)
                    }
                } label: {
                    Label("Collapse turn", systemImage: "rectangle.compress.vertical")
                        .themedFont(.tiny)
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Fold this turn down to the prompt and its final answer")
            }
        }
    }
}

/// The "N hidden steps" toggle a collapsed turn shows under its prompt.
private struct TurnCollapseToggleRow: View {
    let count: Int
    let onExpand: () -> Void

    var body: some View {
        Button(action: onExpand) {
            HStack(spacing: 5) {
                Image(systemName: "chevron.down")
                    .themedFont(.tiny)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text(
                    count == 1
                        ? "1 hidden step from this turn"
                        : "\(count) hidden steps from this turn")
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.appSecondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.primary.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Expand the steps this turn took")
        .accessibilityLabel("Show \(count) hidden steps")
    }
}

/// View displaying a committed conversation message turn with Claude-style layout and hover actions.
private struct MessageRowView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let message: AppChatMessage
    @State private var isHovered = false
    @State private var branchTarget: AppModel.BranchTarget?
    @ObservedObject private var speechManager = AppSpeechSynthesizer.shared

    private var isCurrentlySpeakingThis: Bool {
        speechManager.isSpeaking && speechManager.speakingMessageID == message.id
    }

    /// Variant navigation reads; the bar hides the switcher on "1 / 1".
    private var variantPosition: (position: Int, count: Int) {
        model.variantPosition(of: message)
    }

    private func stepVariant(_ delta: Int) {
        model.stepVariant(of: message.id, delta: delta)
    }

    private var canEditThisInPlace: Bool {
        model.canEditInPlace(message)
    }

    private var canBranchThis: Bool {
        model.canBranch(message)
    }

    /// Opens an ```html fence in the sandboxed preview panel. Inline bytes:
    /// the fence is not a file, so nothing joins the artifact archive.
    private var previewHTML: (String) -> Void {
        { model.openHTMLPreview(title: "HTML preview", html: $0) }
    }

    /// One chip per artifact this message's tool calls produced, anchored by
    /// `AppArtifact.toolCallID` -- the field exists for exactly this row.
    /// Clicking reopens the panel, which is the only manual way back to a
    /// file artifact the auto-open already showed once.
    @ViewBuilder
    private var artifactChips: some View {
        let callIDs = Set(message.toolCalls.map(\.id))
        let rows = model.selectedChat.artifacts.filter { artifact in
            artifact.toolCallID.map(callIDs.contains) == true
        }
        ForEach(rows) { artifact in
            Button {
                model.openArtifact(id: artifact.id)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: artifact.symbolName)
                        .themedFont(.tiny, weight: .semibold)
                        .foregroundStyle(.appAccent)
                        .accessibilityHidden(true)
                    Text(artifact.fileName)
                        .themedFont(.tiny, weight: .medium)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(artifact.formatLabel)
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                    Spacer(minLength: 4)
                    Image(systemName: "sidebar.right")
                        .themedFont(.micro)
                        .foregroundStyle(.tertiary)
                        .accessibilityHidden(true)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color.primary.opacity(0.04))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .stroke(.appBorder.opacity(0.4), lineWidth: 0.5)
                )
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Open \(artifact.fileName) in the artifact panel")
            .accessibilityLabel("Open artifact \(artifact.fileName)")
        }
    }

    /// The ordinary user bubble. Split out of `body` when `#` quick-saves
    /// arrived: a transcript row holding a `<user-memory-input>` wrap is a
    /// user message too, and rendering it as a bubble showed raw tags.
    private var standardUserMessageRow: some View {
        HStack(alignment: .top, spacing: 0) {
            Spacer(minLength: 48)
            VStack(alignment: .trailing, spacing: 6) {
                CollapsibleMessageContentView(
                    text: message.content,
                    isUser: true
                )
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(.appSurface.opacity(0.85))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .shadow(color: Color.black.opacity(0.03), radius: 3, x: 0, y: 1)

                if isHovered || isCurrentlySpeakingThis {
                    MessageActionBarView(
                        text: message.content,
                        messageID: message.id,
                        date: message.createdAt,
                        variantStep: variantPosition.count > 1 ? { stepVariant($0) } : nil,
                        variantPosition: variantPosition.position,
                        variantCount: variantPosition.count,
                        editAction: canEditThisInPlace ? { _ = model.beginEdit(messageID: message.id) } : nil,
                        branchAction: canBranchThis
                            ? { branchTarget = AppModel.BranchTarget(id: message.id, originalText: message.content) }
                            : nil,
                        actionsDisabled: model.isRunning
                    )
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
                }
            }
        }
    }

    /// The row while its in-place editor is open. The composer holds its
    /// own draft text, so a Cancelled commit leaves the row as it was.
    private var editingUserMessageRow: some View {
        HStack(alignment: .top, spacing: 0) {
            Spacer(minLength: 48)
            MessageEditComposerView(
                initialText: message.content,
                onSave: { model.commitEdit(messageID: message.id, newText: $0) },
                onCancel: { model.cancelEdit() }
            )
        }
    }

    /// The qwen-code UserShellMessage parity: a `!command` bang run rendered
    /// as the command plus its captured output, not as a chat bubble.
    private func shellMessageRow(_ shell: ShellMessageContent) -> some View {
        HStack(alignment: .top, spacing: 0) {
            Spacer(minLength: 32)
            VStack(alignment: .trailing, spacing: 6) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Image(systemName: "terminal")
                            .themedFont(.tiny, weight: .semibold)
                            .foregroundStyle(.appAccent)
                            .accessibilityHidden(true)
                        Text(shell.command)
                            .themedCode(.small, weight: .medium)
                            .textSelection(.enabled)
                    }
                    if let output = shell.output {
                        Text(Self.attributedShellOutput(output, base: theme.code(.small)))
                            .themedCode(.small)
                            .foregroundStyle(theme.metadataForeground)
                            .frame(maxWidth: .infinity, maxHeight: 220, alignment: .topLeading)
                            .padding(8)
                            .background(Color.primary.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .textSelection(.enabled)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.appSurface.opacity(0.85))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(.appAccent.opacity(0.25), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                if isHovered || isCurrentlySpeakingThis {
                    MessageActionBarView(
                        text: message.content,
                        messageID: message.id,
                        date: message.createdAt,
                        actionsDisabled: model.isRunning
                    )
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
                }
            }
        }
    }

    /// Bang-command output as attributed text: ANSI-colored when the output
    /// carries escape codes, plain otherwise (the same split
    /// `ToolResultOutputView` draws). `base` is the themed font the runs
    /// inherit, so the block renders at one size instead of a hardcoded
    /// literal fighting the `.themedCode` modifier on the Text.
    private static func attributedShellOutput(
        _ output: String,
        base: Font
    ) -> AttributedString {
        guard output.contains("\u{1B}") else { return AttributedString(output) }
        var attributed = AttributedString()
        for segment in ANSIColorizer.segments(in: output) {
            var run = AttributedString(segment.text)
            if segment.bold || segment.italic {
                run.font = segment.bold && segment.italic
                    ? base.bold().italic() : segment.bold ? base.bold() : base.italic()
            }
            if segment.underline { run.underlineStyle = .single }
            if segment.foreground != .default {
                run.foregroundColor = ansiColor(segment.foreground)
            }
            if segment.faint {
                run.foregroundColor = ansiColor(segment.foreground).opacity(0.5)
            }
            attributed += run
        }
        return attributed
    }

    /// The 16-color palette, the same RGB values `ToolResultOutputView`
    /// draws with so a bang row and a tool card agree on what "red" is.
    private static func ansiColor(_ palette: ANSIColorizer.Palette) -> Color {
        switch palette {
        case .black: return Color(red: 0, green: 0, blue: 0)
        case .red: return Color(red: 0.8, green: 0, blue: 0)
        case .green: return Color(red: 0, green: 0.8, blue: 0)
        case .yellow: return Color(red: 0.8, green: 0.8, blue: 0)
        case .blue: return Color(red: 0, green: 0, blue: 0.93)
        case .magenta: return Color(red: 0.8, green: 0, blue: 0.8)
        case .cyan: return Color(red: 0, green: 0.8, blue: 0.8)
        case .white: return Color(red: 0.9, green: 0.9, blue: 0.9)
        case .brightBlack: return Color(white: 0.5)
        case .brightRed: return Color(red: 1, green: 0.25, blue: 0.25)
        case .brightGreen: return Color(red: 0.25, green: 1, blue: 0.25)
        case .brightYellow: return Color(red: 1, green: 1, blue: 0.25)
        case .brightBlue: return Color(red: 0.35, green: 0.35, blue: 1)
        case .brightMagenta: return Color(red: 1, green: 0.25, blue: 1)
        case .brightCyan: return Color(red: 0.25, green: 1, blue: 1)
        case .brightWhite: return Color(white: 1)
        case .default: return .primary
        }
    }

    /// A `#` quick-save, shown as the memory note it is (Claude Code renders
    /// the same shape as a badged memory row, not as a chat bubble).
    private func memoryQuickSaveRow(_ memoryText: String) -> some View {
        HStack {
            Spacer(minLength: 48)
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "brain")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Saved to memory", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Text(memoryText)
                    .themedFont(.small)
                    .foregroundStyle(theme.metadataForeground)
                    .lineLimit(3)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.appSurface.opacity(0.85))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(.appAccent.opacity(0.25), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if message.role == .user {
                if let memoryText = UserMemoryInputMessage.parse(message.content) {
                    memoryQuickSaveRow(memoryText)
                } else if let shell = ShellMessageContent.parse(message.content) {
                    shellMessageRow(shell)
                } else if model.editingMessageID == message.id {
                    editingUserMessageRow
                } else {
                    standardUserMessageRow
                }
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 6) {
                        Image(systemName: "sparkles")
                            .themedFont(.small, weight: .bold)
                            .foregroundStyle(.appAccent)
                            .accessibilityHidden(true)
                        Text(model.selected?.alias ?? "TurboSpark")
                            .themedFont(.small, weight: .semibold)
                            .foregroundStyle(theme.metadataForeground)
                    }
                    .padding(.bottom, -2)

                    if !message.reasoning.isEmpty {
                        ReasoningDisclosureView(reasoning: message.reasoning)
                    }

                    if !message.toolCalls.isEmpty {
                        if message.toolCalls.count > 1 {
                            ToolGroupView(
                                model: model,
                                toolCalls: message.toolCalls,
                                toolResults: message.toolResults
                            )
                        } else if let call = message.toolCalls.first {
                            let matchResult = message.toolResults.first(where: { $0.callID == call.id })
                            ToolCallCardView(model: model, call: call, result: matchResult)
                        }
                        artifactChips
                    }

                    if !message.content.isEmpty && message.toolCalls.isEmpty {
                        CollapsibleMessageContentView(
                            text: message.content,
                            isUser: false,
                            maxHeight: 380,
                            onPreviewHTML: previewHTML)
                    } else if !message.content.isEmpty {
                        let trimmed = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmed.starts(with: "<tool_call>") && !trimmed.starts(with: "Invoking tool") {
                            CollapsibleMessageContentView(
                                text: message.content,
                                isUser: false,
                                maxHeight: 380,
                                onPreviewHTML: previewHTML)
                        }
                    }

                    if isHovered || isCurrentlySpeakingThis {
                        HStack(spacing: 6) {
                            MessageActionBarView(
                                text: message.content,
                                messageID: message.id,
                                date: message.createdAt,
                                variantStep: variantPosition.count > 1 ? { stepVariant($0) } : nil,
                                variantPosition: variantPosition.position,
                                variantCount: variantPosition.count,
                                retryAction: model.canRetry(response: message)
                                    ? { _ = model.regenerateResponse() } : nil,
                                actionsDisabled: model.isRunning
                            )
                            Spacer()
                        }
                        .padding(.top, 2)
                        .transition(.opacity.combined(with: .scale(scale: 0.98)))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
        .sheet(item: $branchTarget) { target in
            BranchEditSheet(
                originalText: target.originalText,
                chatTitle: model.selectedChat.title,
                onCreate: { newText in
                    branchTarget = nil
                    _ = model.branchFrom(messageID: target.id, editedText: newText)
                },
                onCancel: { branchTarget = nil }
            )
        }
    }
}

/// View rendering live streaming output and prefill animations.
private struct ActiveStreamingRowView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let output: String
    let reasoning: String
    let isRunning: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .themedFont(.small, weight: .bold)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text(model.selected?.alias ?? "TurboSpark")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(theme.metadataForeground)
            }
            .padding(.bottom, -2)

            if isRunning && output.isEmpty && reasoning.isEmpty {
                HStack(spacing: 8) {
                    TaskProgressFlameIcon(size: 16)
                    Text(waitingStatusText)
                        .themedFont(.base)
                        .foregroundStyle(theme.metadataForeground)
                }
                .padding(.vertical, 8)
                // Combine the spinner and status into one announcement.
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(model.reasoning != .off ? "Thinking" : "Generating response")
                .accessibilityHint(model.reasoning != .off ? "The model is reasoning before producing a response" : "The model is preparing a response")
            } else {
                if !reasoning.isEmpty {
                    ReasoningDisclosureView(reasoning: reasoning, defaultExpanded: true)
                }
                if let pending = model.pendingToolCall {
                    ToolCallCardView(model: model, call: pending, result: nil)
                }
                // Foreground subagent runs proposed by THIS turn stream
                // here, the same place the approval card goes up. The run's
                // own chat is what filters, never a selection made later.
                ForEach(liveRunsForChat) { state in
                    SubagentLiveCardView(state: state)
                }
                if !output.isEmpty {
                    ChatMessageMarkdownView(output)
                }
            }

            if isRunning {
                StreamingStatusFooterView(model: model)
                    .padding(.top, 2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Copy shown while nothing has streamed yet: which of the waiting
    /// states (tool approval, prefill, reasoning, or plain decode) the turn is in.
    private var waitingStatusText: String {
        if model.pendingToolCall != nil {
            "Awaiting tool confirmation..."
        } else if model.phase == .prefill {
            "Reading prompt..."
        } else if model.reasoning != .off {
            "Thinking..."
        } else {
            "Generating response..."
        }
    }

    /// Live foreground subagent runs belonging to this chat, oldest first.
    private var liveRunsForChat: [SubagentRunState] {
        model.liveSubagentRuns.values
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
            .sorted { $0.startedAt < $1.startedAt }
    }
}

/// The strip of background subagent runs for the selected chat, running or
/// finished. It renders OUTSIDE the streaming row so a background agent is
/// visible while nothing else is happening in the conversation -- which is
/// most of a background run's life.
private struct BackgroundAgentsStripView: View {
    @ObservedObject var model: AppModel

    private var runsForChat: [SubagentRunState] {
        model.backgroundAgentRuns.values
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
            .sorted { $0.startedAt < $1.startedAt }
    }

    var body: some View {
        let runs = runsForChat
        if !runs.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(runs) { state in
                    SubagentLiveCardView(
                        state: state,
                        onStop: state.status == "running" ? { stop(state.id) } : nil,
                        onDismiss: state.status != "running" && state.isRecordedComplete
                            ? { model.dismissBackgroundAgent(state.id) } : nil)
                }
            }
        }
    }

    private func stop(_ id: String) {
        Task { @MainActor in
            do {
                let message = try await model.stopBackgroundAgent(id)
                model.showToast(message, style: .info)
            } catch {
                model.showToast(error.localizedDescription, style: .error)
            }
        }
    }
}

/// The strip of RUNNING background shells for the selected chat, one row
/// each with a Kill button. `KillShell` has always existed for the model;
/// this is the user's equivalent -- a hung `yes` loop or dev server is
/// visible and endable without asking the assistant or quitting the app.
/// Finished shells are deliberately absent: their output reaches the model
/// through `BashOutput`, and a transcript row per finished background
/// command would grow without bound. Killed shells' descendants die with
/// them (`ProcessExecutor.terminateAndReap` is a tree kill).
private struct BackgroundShellsStripView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    /// Same visibility rule as the agent strip: unscoped shells (a run
    /// whose chat was captured as nil) show everywhere; a chat-scoped shell
    /// shows only in its own chat.
    private var shellsForChat: [BackgroundShellSummary] {
        model.backgroundShellSummaries
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
    }

    var body: some View {
        let shells = shellsForChat
        if !shells.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(shells) { shell in
                    shellRow(shell)
                }
            }
        }
    }

    private func shellRow(_ shell: BackgroundShellSummary) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "terminal")
                .themedFont(.small, weight: .bold)
                .foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                if let description = shell.description, !description.isEmpty {
                    Text(description)
                        .themedFont(.small, weight: .semibold)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Text(shell.commandHead)
                        .font(theme.code(.small))
                        .foregroundStyle(theme.metadataForeground)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                } else {
                    Text(shell.commandHead)
                        .font(theme.code(.small))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
            }
            Spacer(minLength: 8)
            TaskProgressFlameIcon(size: 14)
            Text(shell.id)
                .themedFont(.small, weight: .medium)
                .foregroundStyle(theme.metadataForeground)
            Text(shell.startedAt, style: .timer)
                .themedFont(.small)
                .foregroundStyle(theme.metadataForeground)
            Button {
                model.killBackgroundShell(id: shell.id)
            } label: {
                Image(systemName: "stop.fill")
                    .themedFont(.tiny)
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.red.opacity(0.8))
            .help("Kill this background shell (its child processes are killed with it)")
            .accessibilityLabel("Kill background shell \(shell.id)")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.appSurface.opacity(0.7))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(.appAccent.opacity(0.45), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Background shell \(shell.id) running, \(shell.commandHead)")
    }
}

/// The line marking where compaction summarized the transcript's prefix.
///
/// The rows below it are exactly the ones the prompt no longer carries; the
/// summary that replaced them expands in place. A disclosure rather than a
/// toast because a summary a user can never read is a summary they cannot
/// correct.
private struct CompactionDividerView: View {
    @Environment(\.appTheme) private var theme
    let summary: String
    let summarizedRows: Int

    @State private var isExpanded: Bool = false

    var body: some View {
        DisclosureGroup(
            isExpanded: $isExpanded
        ) {
            Text(summary)
                .font(theme.code(.small))
                .foregroundStyle(theme.metadataForeground)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color.primary.opacity(0.03))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .textSelection(.enabled)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "rectangle.compress.vertical")
                    .themedFont(.tiny)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text(
                    summarizedRows == 1
                        ? "Earlier message compacted into a summary"
                        : "\(summarizedRows) earlier messages compacted into a summary"
                )
                .themedFont(.small, weight: .medium)
                .foregroundStyle(theme.metadataForeground)
            }
        }
        .padding(.vertical, 2)
        .accessibilityLabel("Earlier conversation compacted into a summary")
        .accessibilityHint("Expands to show the summary the model sees instead of those messages")
    }
}

/// Collapsible disclosure view for model reasoning and chain of thought.
private struct ReasoningDisclosureView: View {
    @Environment(\.appTheme) private var theme
    let reasoning: String
    var defaultExpanded: Bool = false

    @State private var isExpanded: Bool = false
    /// Whether the user has clicked the disclosure. Until they do, the
    /// panel shows `defaultExpanded`; with a `true` default the old
    /// `defaultExpanded || isExpanded` binding could never read false, so
    /// the live trace was permanently pinned open.
    @State private var hasUserOverride: Bool = false

    var body: some View {
        DisclosureGroup(
            isExpanded: Binding(
                get: { hasUserOverride ? isExpanded : defaultExpanded },
                set: { newValue in
                    hasUserOverride = true
                    isExpanded = newValue
                }
            )
        ) {
            Text(reasoning)
                .font(theme.code(.large))
                .foregroundStyle(theme.metadataForeground)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.03))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .textSelection(.enabled)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "brain")
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text("Thought process", bundle: .module)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(theme.metadataForeground)
            }
        }
        .padding(.vertical, 2)
        .help("Model thought process and reasoning trace")
        .accessibilityLabel("Thought process")
        .accessibilityHint("Expands to reveal the model's chain-of-thought reasoning")
    }
}
