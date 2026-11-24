import AppKit
import SwiftUI
import TurboSpark

/// View displaying a committed conversation message turn with Claude-style layout and hover actions.
struct MessageRowView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let message: AppChatMessage
    /// The web sources of the turn this row closes (OC-09). Set by the
    /// transcript only on a turn's final reply row; nil draws no chip.
    var sources: TurnSources?
    /// The files the turn this row closes touched (component audit,
    /// file-tree). Set on the same final reply row as `sources`.
    var touchedFiles: TurnFileTree?
    /// What conversation recall did for this turn, in memory only
    /// (`RecallInspection`). Set on the same final reply row as `sources`.
    var recallInspection: RecallInspection?
    /// Keeps the action bar up without hover. The transcript sets it on the
    /// newest reply only (`MessageActionBarPin`).
    var pinsActions = false
    /// Shows the opt-in follow-up chips. The transcript sets it on one row
    /// at most (`FollowUpSuggestions.rowID`).
    var showsFollowUps = false
    @State private var isHovered = false
    @State private var showsReasoning = false
    @State private var branchTarget: AppModel.BranchTarget?
    @ObservedObject private var speechManager = AppSpeechSynthesizer.shared

    private var isCurrentlySpeakingThis: Bool {
        speechManager.isSpeaking && speechManager.speakingMessageID == message.id
    }

    private var showsMessageActions: Bool {
        isHovered || isCurrentlySpeakingThis || pinsActions
    }

    private var showsAssistantContent: Bool {
        guard !message.content.isEmpty else { return false }
        guard !message.toolCalls.isEmpty else { return true }
        let trimmed = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.starts(with: "<tool_call>") && !trimmed.starts(with: "Invoking tool")
    }

    /// Variant navigation reads; the bar hides the switcher on "1 / 1".
    private var variantPosition: (position: Int, count: Int) {
        model.variantPosition(of: message)
    }

    private func stepVariant(_ delta: Int) {
        model.stepVariant(of: message.id, delta: delta)
    }

    /// Memory proposals staged by this step's `memory` calls. None while
    /// model memory is off or in a Ghost chat: the note must not advertise
    /// memory where it is not in use.
    private var memoryProposalNotes: [MemoryProposalNote] {
        guard model.memoryEnabled, !model.selectedChat.isGhost else { return [] }
        return MemoryProposalNote.notes(calls: message.toolCalls, results: message.toolResults)
    }

    private var canEditThisInPlace: Bool {
        model.canEditInPlace(message)
    }

    private var canBranchThis: Bool {
        model.canBranch(message)
    }

    private var standingInstructionAction: (() -> Void)? {
        guard AppChatInstructionPin.canPin(message) else { return nil }
        return {
            model.setMessageStandingInstruction(
                id: message.id,
                pinned: !message.isStandingInstruction,
                chatID: model.selectedChatID)
        }
    }

    /// "Quote in reply" (component audit, selection-toolbar). The whole
    /// message, because SwiftUI's selectable `Text` does not expose the
    /// selected range. Nil when there is no prose to quote: a tool-call
    /// step's raw call text, a `#` memory save, or a `!` shell run.
    private var quoteAction: (() -> Void)? {
        if message.role == .user {
            guard UserMemoryInputMessage.parse(message.content) == nil,
                  ShellMessageContent.parse(message.content) == nil
            else { return nil }
        } else if !showsAssistantContent {
            return nil
        }
        guard MessageQuote.blockquote(message.content) != nil else { return nil }
        return { _ = model.quoteInReply(message.content) }
    }

    private var transcriptPresentation: MidTurnTranscriptRowPresentation {
        MidTurnTranscriptRowPresentation(message: message)
    }

    /// Opens an ```html fence in the sandboxed preview panel. Inline bytes:
    /// the fence is not a file, so nothing joins the artifact archive.
    private var previewHTML: (String) -> Void {
        { model.openHTMLPreview(title: AppStrings.localized("HTML preview"), html: $0) }
    }

    /// A reply whose `[n]` markers name the turn's sources (component audit,
    /// inline-citation): markers render as links and the cited sources as
    /// chips below. Only this row, which `sources` marks as the committed
    /// reply, takes the rewrite, so the streaming row is untouched. The
    /// `openURL` override is scoped to this branch so other rows keep the
    /// default link handling, and any URL that is not a citation still goes
    /// to the system.
    @ViewBuilder
    private func citedAssistantContent(_ citations: InlineCitations.Rendering, sources: TurnSources) -> some View {
        MarkdownContentWithTablesView(text: citations.markdown, onPreviewHTML: previewHTML)
            .fixedSize(horizontal: false, vertical: true)
            .environment(\.openURL, OpenURLAction { url in
                guard let number = InlineCitations.number(from: url, sourceCount: sources.sources.count)
                else { return .systemAction }
                model.revealToolCard(for: sources.sources[number - 1])
                return .handled
            })
        InlineCitationChipsView(model: model, sources: sources, numbers: citations.numbers)
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
            ArtifactOpenChip(artifact: artifact) {
                model.openArtifact(id: artifact.id)
            }
        }
        // A call still executing has no artifact row yet; the real chip
        // replaces this one when `registerProducedArtifacts` anchors on it.
        ForEach(ArtifactChipState.writing(
            calls: message.toolCalls, results: message.toolResults,
            artifacts: model.selectedChat.artifacts)
        ) { writing in
            ArtifactWritingChip(fileName: writing.fileName)
        }
    }

    @ViewBuilder
    private var imageAttachments: some View {
        ForEach(message.imagePaths, id: \.self) { path in
            if let image = NSImage(contentsOfFile: AppStorageRoot.resolveStoredPath(path)) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 420)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .accessibilityLabel(Text("Attached image", bundle: .module))
            } else {
                Label {
                    Text("Image is no longer at its saved path", bundle: .module)
                } icon: {
                    Image(systemName: "photo.badge.exclamationmark")
                }
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
        }
    }

    /// The collapsed reasoning header: "Thought for 12s" when the row's
    /// duration was measured, else the plain label. Rows from before the
    /// duration was stored show no time rather than an estimate.
    private var reasoningLabel: Text? {
        guard !message.reasoning.isEmpty else { return nil }
        if let duration = ReasoningDurationText.text(seconds: message.reasoningSeconds) {
            return Text("Thought for \(duration)", bundle: .module)
        }
        return Text("Thought process", bundle: .module)
    }

    /// The ordinary user bubble. Split out of `body` when `#` quick-saves
    /// arrived: a transcript row holding a `<user-memory-input>` wrap is a
    /// user message too, and rendering it as a bubble showed raw tags.
    private var standardUserMessageRow: some View {
        HStack(alignment: .top, spacing: 0) {
            Spacer(minLength: 48)
            VStack(alignment: .trailing, spacing: 6) {
                if transcriptPresentation.role == .user,
                   let label = transcriptPresentation.label
                {
                    Text(LocalizedStringKey(label), bundle: .module)
                        .themedFont(.small, weight: .semibold)
                        .foregroundStyle(theme.metadataForeground)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4)
                        .background(.appSurface, in: Capsule())
                }

                CollapsibleMessageContentView(
                    text: message.content,
                    isUser: true
                )
                .accessibilityHeading(.h2)
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(.appSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: AppMetrics.composerRadius, style: .continuous)
                        .stroke(.appBorder, lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.composerRadius, style: .continuous))
                .shadow(color: Color.black.opacity(0.03), radius: 3, x: 0, y: 1)

                imageAttachments

                MessageActionBarView(
                    text: message.content,
                    messageID: message.id,
                    date: message.createdAt,
                    isVisible: showsMessageActions,
                    variantStep: variantPosition.count > 1 ? { stepVariant($0) } : nil,
                    variantPosition: variantPosition.position,
                    variantCount: variantPosition.count,
                    editAction: canEditThisInPlace ? { _ = model.beginEdit(messageID: message.id) } : nil,
                    branchAction: canBranchThis
                        ? { branchTarget = AppModel.BranchTarget(id: message.id, originalText: message.content) }
                        : nil,
                    standingInstructionAction: standingInstructionAction,
                    isStandingInstruction: message.isStandingInstruction,
                    quoteAction: quoteAction,
                    actionsDisabled: model.isRunning
                )
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
                            .background(.appSurface)
                            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                            .textSelection(.enabled)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.appSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: AppMetrics.composerRadius, style: .continuous)
                        .stroke(theme.accent.opacity(0.25), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.composerRadius, style: .continuous))

                MessageActionBarView(
                    text: message.content,
                    messageID: message.id,
                    date: message.createdAt,
                    isVisible: showsMessageActions,
                    standingInstructionAction: standingInstructionAction,
                    isStandingInstruction: message.isStandingInstruction,
                    actionsDisabled: model.isRunning
                )
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
            .background(.appSurface)
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.composerRadius, style: .continuous)
                    .stroke(theme.accent.opacity(0.25), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.composerRadius, style: .continuous))
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
                    // One line for the model and its reasoning, the same row
                    // the reply showed while it ran. A tool-call step keeps
                    // no model header (the turn already has one), so it shows
                    // the line only when there is reasoning to reveal.
                    if message.toolCalls.isEmpty || !message.reasoning.isEmpty {
                        ChatActivityLine(
                            modelName: message.toolCalls.isEmpty
                                ? (model.selected?.alias ?? "TurboSpark") : nil,
                            label: reasoningLabel,
                            // The summary stands in for the folded trace, so
                            // it hides once the full trace is open below.
                            detail: showsReasoning ? nil : ReasoningSummary.headline(of: message.reasoning),
                            reasoningExpanded: message.reasoning.isEmpty ? nil : $showsReasoning)
                            .accessibilityHeading(.h2)
                            .padding(.bottom, -6)
                        if showsReasoning && !message.reasoning.isEmpty {
                            ReasoningTraceView(reasoning: message.reasoning)
                        }
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
                        ForEach(memoryProposalNotes, id: \.callID) { note in
                            MemoryProposalNoteRow(
                                note: note,
                                resolve: { model.memoryProposalResolution(callID: note.callID) },
                                act: { model.actOnMemoryProposal(callID: note.callID, $0) },
                                onReview: { model.openSettings(tab: .memory) })
                        }
                    }

                    if showsAssistantContent {
                        if let sources, let citations = InlineCitations.apply(
                            to: message.content, sourceCount: sources.sources.count) {
                            citedAssistantContent(citations, sources: sources)
                        } else {
                            MarkdownContentWithTablesView(text: message.content, onPreviewHTML: previewHTML)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    imageAttachments

                    if let sources, !sources.isEmpty {
                        SourcesChipView(model: model, sources: sources, inspection: recallInspection)
                    } else if let recallInspection {
                        SourcesChipView(model: model, sources: TurnSources(), inspection: recallInspection)
                    }

                    if let touchedFiles {
                        TurnFileTreeCardView(model: model, tree: touchedFiles)
                    }

                    if let notice = MessageStopNotice.resolve(message), let reason = message.stopReason {
                        MessageStopNoticeView(
                            notice: notice,
                            stopReason: reason,
                            retryAction: model.canRetry(response: message)
                                ? { _ = model.regenerateResponse() } : nil,
                            actionsDisabled: model.isRunning,
                            compactAction: notice == .contextFull ? { model.compactKeepingDraft() } : nil,
                            compactBlocker: model.contextMeterCompactBlocker)
                    }

                    MessageActionBarView(
                        text: message.content,
                        messageID: message.id,
                        date: message.createdAt,
                        isVisible: showsMessageActions,
                        variantStep: variantPosition.count > 1 ? { stepVariant($0) } : nil,
                        variantPosition: variantPosition.position,
                        variantCount: variantPosition.count,
                        retryAction: model.canRetry(response: message)
                            ? { _ = model.regenerateResponse() } : nil,
                        quoteAction: quoteAction,
                        actionsDisabled: model.isRunning
                    )
                    .padding(.top, 2)

                    if showsFollowUps {
                        FollowUpSuggestionsRow { model.draftFollowUp($0) }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
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
        .accessibilityElement(children: .contain)
        .accessibilityAction(named: Text("Quote in reply", bundle: .module)) {
            quoteAction?()
        }
        .accessibilityAction(named: Text("Copy message", bundle: .module)) {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(message.content, forType: .string)
        }
        .accessibilityAction(
            named: isCurrentlySpeakingThis
                ? Text("Stop reading message out loud", bundle: .module)
                : Text("Read message out loud", bundle: .module)
        ) {
            if isCurrentlySpeakingThis {
                speechManager.stop()
            } else {
                speechManager.speak(text: message.content, messageID: message.id)
            }
        }
        .accessibilityAction(named: Text("Edit message", bundle: .module)) {
            if canEditThisInPlace {
                _ = model.beginEdit(messageID: message.id)
            }
        }
        .accessibilityAction(named: Text("Regenerate response", bundle: .module)) {
            if model.canRetry(response: message) {
                _ = model.regenerateResponse()
            }
        }
        .accessibilityAction(named: Text("Branch conversation", bundle: .module)) {
            if canBranchThis {
                branchTarget = AppModel.BranchTarget(id: message.id, originalText: message.content)
            }
        }
    }
}
