import SwiftUI

/// Native SwiftUI transcript view rendering multi-turn conversations with Markdown formatting.
@MainActor
struct ChatTranscriptView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    /// The chat this view was built for. The parent keys the view by chat, so
    /// on teardown this is still the chat being left, not the new selection.
    let chatID: UUID

    init(model: AppModel, chatID: UUID? = nil) {
        self.model = model
        self.chatID = chatID ?? model.selectedChatID
    }
    /// Whether the reader is at the bottom. Streaming output scrolls the
    /// transcript only while pinned, so scrolling up to reread is not undone
    /// by the next token. The parent keys this view by chat, so a chat switch
    /// starts pinned.
    @StateObject private var follower = TranscriptFollower()

    /// Scrolls to the newest row if the reader is following. `force` re-pins
    /// first, for what the reader just asked for (a send, a question to answer).
    private func followBottom(_ proxy: ScrollViewProxy, force: Bool = false) {
        if force { follower.pin() }
        guard follower.isPinned else { return }
        proxy.scrollTo("bottom", anchor: .bottom)
    }

    /// Id of the newest row the human wrote, so a send re-pins the transcript.
    private var lastHumanMessageID: UUID? {
        model.selectedTurnMessages.last { $0.role == .user && !$0.isSynthetic }?.id
    }

    /// The one row whose action bar stays visible without hover.
    private var pinnedActionsID: UUID? {
        MessageActionBarPin.pinnedMessageID(in: model.selectedTurnMessages, isRunning: model.isRunning)
    }

    /// The row that shows the opt-in follow-up chips, if any. Busy also
    /// covers a pending approval or submission, where a drafted ask would
    /// land behind the card the user must answer first.
    private var followUpRowID: UUID? {
        FollowUpSuggestions.rowID(
            in: model.selectedTurnMessages,
            isBusy: model.isRunning || !model.isIdleForPlanCard,
            isEnabled: model.followUpSuggestionsEnabled)
    }

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

    private var selectedRecoveryEvents: [StreamRecoveryEvent] {
        model.streamRecoveryEvents[model.selectedChatID] ?? []
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
                    let pinnedID = pinnedActionsID
                    let followUpID = followUpRowID
                    ForEach(turns, id: \.offset) { index, turn in
                        if index > 0 { ConversationDivider().padding(.vertical, 8) }
                        turnRows(turn, byID: byID, pinnedID: pinnedID, followUpID: followUpID)
                    }

                    if let event = model.selectedCompactionBoundaryEvent {
                        CompactionBoundaryNoticeView(event: event)
                            .id(event.id)
                    }

                    ForEach(Array(selectedRecoveryEvents.enumerated()), id: \.offset) { entry in
                        StreamRecoveryEventNoticeView(event: entry.element, occurrence: entry.offset)
                            .id("recovery-\(model.selectedChatID.uuidString)-\(entry.offset)")
                    }

                    NewChatSuggestionBanner(model: model)

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

                    // The parked ExitPlanMode gate, placed and scoped like the
                    // question card for the same reason: its tool row does
                    // not exist until the user decides.
                    if let pendingPlan = model.pendingPlanApproval,
                        pendingPlan.chatID == nil
                            || pendingPlan.chatID == model.selectedChatID
                    {
                        PlanCardView(
                            model: model, plan: pendingPlan.plan,
                            isOffered: false, isAwaitingDecision: true)
                            .id(pendingPlan.id)
                    }

                    // The live buffer belongs to the running turn's chat
                    // (ZP-23): another chat's reply must not stream here.
                    if model.isRunningInSelectedChat
                        || (model.liveOutputBelongsToSelectedChat
                            && (!model.outputText.isEmpty || !model.outputReasoningText.isEmpty))
                    {
                        ActiveStreamingRowView(
                            model: model,
                            output: model.outputText,
                            reasoning: model.outputReasoningText,
                            isRunning: model.isRunningInSelectedChat
                        )
                    }

                    BackgroundAgentsStripView(model: model)

                    BackgroundShellsStripView(model: model)

                    Color.clear
                        .frame(height: 1)
                        .id("bottom")
                }
                .conversationColumn()
                .padding(.vertical, 24)
                .background(TranscriptScrollObserver(follower: follower))
            }
            .overlay(alignment: .bottom) {
                if !follower.isPinned {
                    JumpToLatestButton {
                        TSMotion.animate(TSMotion.pane) {
                            followBottom(proxy, force: true)
                        }
                    }
                    .padding(.bottom, 12)
                    .transition(.opacity)
                }
            }
            .overlay(alignment: .trailing) {
                scrollRail(proxy)
            }
            .tsAnimation(TSMotion.hover, value: follower.isPinned)
            .onChange(of: model.outputText) {
                if model.isRunningInSelectedChat {
                    followBottom(proxy)
                }
            }
            .onChange(of: model.outputReasoningText) {
                if model.isRunningInSelectedChat {
                    followBottom(proxy)
                }
            }
            // Sending is the reader asking to see the answer, wherever they
            // had scrolled to.
            .onChange(of: lastHumanMessageID) { _, id in
                if id != nil { followBottom(proxy, force: true) }
            }
            .onChange(of: model.selectedCompactionBoundaryEvent) { _, event in
                if event != nil {
                    followBottom(proxy)
                }
            }
            .onChange(of: selectedRecoveryEvents) { _, events in
                guard !events.isEmpty else { return }
                TSMotion.animate(TSMotion.pane) {
                    followBottom(proxy)
                }
            }
            // A parked question set scrolls itself into view: mid-turn it is
            // the one thing the user has to act on.
            .onChange(of: model.pendingUserQuestions) { _, pending in
                if let pending {
                    let count = pending.items.count
                    let announcement = count == 1
                        ? AppStrings.localized("Assistant is asking a question. Please choose an answer.")
                        : AppStrings.localized("Assistant is asking \(count) questions. Please choose an answer.")
                    _ = AccessibilityNotification.Announcement.post(.init(announcement))
                    // Forced: the user has to answer, so it outranks a
                    // reader who scrolled up.
                    TSMotion.animate(TSMotion.pane) {
                        followBottom(proxy, force: true)
                    }
                }
            }
            // The plan gate is announced and revealed the same way: the turn
            // waits on it.
            .onChange(of: model.pendingPlanApproval) { _, pending in
                guard pending != nil else { return }
                _ = AccessibilityNotification.Announcement.post(.init(
                    AppStrings.localized("Assistant is waiting for your decision on the plan.")))
                TSMotion.animate(TSMotion.pane) {
                    followBottom(proxy, force: true)
                }
            }
            .onAppear {
                // OC-04: a tab returning to a remembered spot lands there. No
                // memory, or a pending reveal, keeps the old behavior.
                switch model.transcriptScrollRestore(for: chatID) {
                case .offset(let offset):
                    Task { @MainActor in await follower.restore(offset: offset) }
                case .followLatest, nil:
                    followBottom(proxy, force: true)
                    follower.markSettled()
                }
            }
            .onDisappear {
                if let snapshot = follower.memorySnapshot {
                    model.recordTranscriptScroll(
                        chatID: chatID, pinned: snapshot.pinned, offset: snapshot.offset)
                }
            }
            // Turn navigation (qwen-code parity): the menu commands and
            // chords bump a token; this is where the jump actually lands.
            .onChange(of: model.turnNavigationToken) { _, _ in
                guard let target = model.turnNavigationTargetID else { return }
                TSMotion.animate(TSMotion.pane) {
                    proxy.scrollTo(target, anchor: .top)
                }
            }
            // OC-09: the sources popover's Reveal. The turn may have just
            // been opened, so its rows do not exist until the next pass;
            // yield once before asking the proxy to scroll.
            .onChange(of: model.toolCardReveal) { _, reveal in
                guard let reveal else { return }
                follower.cancelRestore()
                Task { @MainActor in
                    await Task.yield()
                    TSMotion.animate(TSMotion.pane) {
                        proxy.scrollTo(reveal.messageID, anchor: .top)
                    }
                }
            }
            .accessibilityRotor(Text("Messages", bundle: .module)) {
                ForEach(model.selectedTurnMessages) { msg in
                    let prefix = msg.role == .user
                        ? AppStrings.localized("User: \(String(msg.content.prefix(40)))")
                        : AppStrings.localized("Assistant: \(String(msg.content.prefix(40)))")
                    AccessibilityRotorEntry(prefix, id: msg.id)
                }
            }
        }
    }

    /// scroll-rail: tick per human prompt along the trailing edge, shown only
    /// with 3+ prompts and a gutter beside the reading column. The current
    /// tick is the last jump target; nothing here measures rows.
    @ViewBuilder
    private func scrollRail(_ proxy: ScrollViewProxy) -> some View {
        let items = TranscriptScrollRail.items(in: model.selectedTurnMessages)
        if !items.isEmpty {
            GeometryReader { geometry in
                let column = (ConversationLayout.readingWidth * AppMetrics.textScale(theme)).rounded()
                if TranscriptScrollRail.fitsBeside(columnWidth: column, containerWidth: geometry.size.width) {
                    TranscriptScrollRailView(
                        items: items,
                        currentIndex: TranscriptScrollRail.currentIndex(
                            of: model.turnNavigationTargetID, in: items)
                    ) { id, animated in
                        jumpToTurn(id, proxy: proxy, animated: animated)
                    }
                    .frame(height: min(TranscriptScrollRail.maxHeight, max(0, geometry.size.height - 96)))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
                    .padding(.trailing, TranscriptScrollRail.edgeInset)
                }
            }
        }
    }

    /// A rail jump is the reader moving away from the bottom, so it unpins
    /// (a streaming reply then stops pulling the viewport back) and cancels a
    /// pending scroll restore. Recording the target keeps Previous/Next Turn
    /// stepping from here. Landing at the bottom re-pins through the probe.
    private func jumpToTurn(_ id: UUID, proxy: ScrollViewProxy, animated: Bool) {
        follower.cancelRestore()
        follower.unpin()
        model.turnNavigationTargetID = id
        if animated {
            TSMotion.animate(TSMotion.pane) { proxy.scrollTo(id, anchor: .top) }
        } else {
            proxy.scrollTo(id, anchor: .top)
        }
    }

    /// The rows of one turn: every message when expanded, prompt + toggle +
    /// final answer when collapsed. A collapse that would hide nothing
    /// renders expanded.
    @ViewBuilder
    private func turnRows(
        _ turn: TurnCollapseModel.Turn, byID: [UUID: AppChatMessage], pinnedID: UUID?,
        followUpID: UUID?
    ) -> some View {
        let hidden = TurnCollapseModel.hiddenCount(for: turn, messagesByID: byID)
        let isCollapsed = turn.isCollapsible && hidden > 0
            && model.collapsedTurnAnchors.contains(turn.anchorID!)
        let visibleIDs = isCollapsed
            ? TurnCollapseModel.visibleIDs(collapsedFor: turn, messagesByID: byID)
            : turn.messageIDs
        // OC-09: derived per render from the turn's own tool results, and
        // only for a turn that made a web call, so other turns never parse.
        let turnMessages = turn.messageIDs.compactMap { byID[$0] }
        let turnSources = TurnSources.hasWebCall(in: turnMessages)
            ? TurnSources.collect(in: turnMessages) : TurnSources()
        let sourcesRowID = turnSources.isEmpty ? nil : TurnSources.replyRowID(in: turnMessages)
        // file-tree: same reply row, and only for a turn that changed a file.
        let touchedFiles = TurnFileTree.hasFileChange(in: turnMessages)
            ? TurnFileTree.make(messages: turnMessages, rootPath: model.selectedProject?.rootDirectoryURL?.path)
            : nil
        let filesRowID = touchedFiles == nil ? nil : TurnSources.replyRowID(in: turnMessages)
        // retrieval-inspector: in memory only, keyed by the person's message
        // that opens the turn, on the same reply row.
        let recallInspection = turn.anchorID.flatMap {
            model.recallInspection(forTurnAnchor: $0, chatID: chatID)
        }
        let recallRowID = recallInspection == nil ? nil : TurnSources.replyRowID(in: turnMessages)
        VStack(alignment: .leading, spacing: 18) {
            // checkpoint: above the prompt, keyed by it; renders nothing for
            // a turn that changed no files.
            if let anchor = turn.anchorID {
                TurnCheckpointDividerSlot(
                    index: model.turnCheckpoints, model: model, chatID: chatID, turnID: anchor,
                    isRunning: model.isRunning)
            }
            ForEach(visibleIDs, id: \.self) { messageID in
                if let message = byID[messageID] {
                    MessageRowView(
                        model: model, message: message,
                        sources: messageID == sourcesRowID ? turnSources : nil,
                        touchedFiles: messageID == filesRowID ? touchedFiles : nil,
                        recallInspection: messageID == recallRowID ? recallInspection : nil,
                        pinsActions: messageID == pinnedID,
                        showsFollowUps: messageID == followUpID)
                        .id(message.id)
                    if messageID == turn.anchorID && hidden > 0 && !model.isRunning {
                        TurnActivityDisclosure(count: hidden, isExpanded: !isCollapsed) {
                            if let anchor = turn.anchorID {
                                if isCollapsed { model.collapsedTurnAnchors.remove(anchor) }
                                else { model.collapsedTurnAnchors.insert(anchor) }
                            }
                        }
                    }
                }
            }
        }
        .padding(.bottom, 8)
    }
}

private struct CompactionBoundaryNoticeView: View {
    let event: CompactionBoundaryEvent

    var body: some View {
        Label {
            Text("Some older tool output was shortened for this request.", bundle: .module)
        } icon: {
            Image(systemName: "info.circle")
        }
        .foregroundStyle(.appSecondary)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("microcompact-boundary-notice-\(event.id.uuidString)")
    }
}

private struct TurnActivityDisclosure: View {
    @Environment(\.appTheme) private var theme
    let count: Int
    let isExpanded: Bool
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(spacing: 10) {
                Image(systemName: "terminal")
                    .frame(width: (ConversationLayout.activityIconWidth * AppMetrics.textScale(theme)).rounded())
                Text("Activity", bundle: .module)
                Text(verbatim: "\(count)").monospacedDigit()
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.micro, weight: .semibold)
                ConversationDivider()
            }
            .themedFont(.small, weight: .medium)
            .foregroundStyle(.appSecondary)
            .frame(minHeight: (ConversationLayout.activityHeight * AppMetrics.textScale(theme)).rounded())
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityValue(isExpanded
            ? Text("Expanded", bundle: .module) : Text("Collapsed", bundle: .module))
    }
}
