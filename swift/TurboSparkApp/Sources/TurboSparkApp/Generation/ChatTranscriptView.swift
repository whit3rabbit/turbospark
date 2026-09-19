import SwiftUI

/// Native SwiftUI transcript view rendering multi-turn conversations with Markdown formatting.
@MainActor
struct ChatTranscriptView: View {
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
                    ForEach(turns, id: \.offset) { index, turn in
                        if index > 0 { ConversationDivider().padding(.vertical, 8) }
                        turnRows(turn, byID: byID)
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
                .conversationColumn()
                .padding(.vertical, 24)
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
                if let pending {
                    let count = pending.items.count
                    let announcement = count == 1
                        ? "Assistant is asking a question. Please choose an answer."
                        : "Assistant is asking \(count) questions. Please choose your answers."
                    _ = AccessibilityNotification.Announcement.post(.init(announcement))
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
            .accessibilityRotor("Messages") {
                ForEach(model.selectedTurnMessages) { msg in
                    AccessibilityRotorEntry(
                        msg.role == .user ? "User: \(msg.content.prefix(40))" : "Assistant: \(msg.content.prefix(40))",
                        id: msg.id
                    )
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
        VStack(alignment: .leading, spacing: 18) {
            ForEach(visibleIDs, id: \.self) { messageID in
                if let message = byID[messageID] {
                    MessageRowView(model: model, message: message)
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

private struct TurnActivityDisclosure: View {
    let count: Int
    let isExpanded: Bool
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(spacing: 10) {
                Image(systemName: "terminal")
                    .frame(width: ConversationLayout.activityIconWidth)
                Text("Activity", bundle: .module)
                Text(verbatim: "\(count)").monospacedDigit()
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.micro, weight: .semibold)
                ConversationDivider()
            }
            .themedFont(.small, weight: .medium)
            .foregroundStyle(.appSecondary)
            .frame(minHeight: ConversationLayout.activityHeight)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityValue(isExpanded
            ? Text("Expanded", bundle: .module) : Text("Collapsed", bundle: .module))
    }
}
