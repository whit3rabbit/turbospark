import Foundation
import TurboSpark

/// The turn itself: the window fit, the options, the event stream, and what
/// the reply turns out to be.
///
/// The phases either side of it are their own files -- `AppModel+Submission`
/// gets as far as an appended user turn, `AppModel+Cancellation` handles a
/// turn that was stopped, `AppModel+History` assembles what is sent, and
/// `AppModel+AgentLoop` takes over once a tool call is parsed out of the
/// reply.
extension AppModel {
    /// The APP-WIDE sampling preferences (temperature, top-k, top-p,
    /// repetition penalty, seed, stop sequences), as a fresh `GenerateOptions`
    /// with everything else at its default.
    ///
    /// The main chat loop no longer reads this: `executeGenerationTurn` uses
    /// `samplingOptions(chatID:)` (`AppModel+Sampling.swift`), which honors a
    /// chat's own override. What remains here is the subagent surface --
    /// `SubagentRunner`, an `enum` with no `AppModel` to read
    /// (`swift/CLAUDE.md` Gotcha 46), is handed these same values through
    /// `AppToolRegistry.subagentSamplingOptionsProvider` rather than running
    /// every subagent turn at a hardcoded `temperature: 0.2` regardless of
    /// what the user set (`swift/docs/SWIFT_SETTINGS_AUDIT.md`). A subagent launch
    /// carries no chat id, so it deliberately reads the app-wide settings and
    /// not the originating chat's override. Deliberately excludes `reasoning`
    /// and `maxNewTokens`: both callers set those themselves, since a
    /// subagent's own turn budget is an architectural choice about its tool
    /// loop rather than a sampling preference.
    func samplingOptions() -> GenerateOptions {
        var options = GenerateOptions()
        options.temperature = temperature
        if topKEnabled {
            options.topK = UInt32(topK)
        }
        if topPEnabled {
            options.topP = topP
        }
        if repetitionPenaltyEnabled {
            options.repetitionPenalty = repetitionPenalty
        }
        if seedEnabled {
            options.seed = seed
        }
        let customStops = stopSequences
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !customStops.isEmpty {
            options.stop = customStops
        }
        return options
    }

    func beginRecoveryEventHistoryForGenerationStart(step: Int, chatID: UUID) {
        streamRecoveryEvents[chatID] = AppStreamRecovery.recoveryEventsAtGenerationStart(
            existing: streamRecoveryEvents[chatID, default: []],
            step: step,
            isAnchoredRecoveryRetry: pendingRecoveryRetryCounts[chatID] != nil)
    }

    /// Executes a generation step for the conversation identified by `chatID`.
    ///
    /// **THE CHAT IS AN ARGUMENT, NEVER `selectedChatIndex`** (state#17). A turn's
    /// continuation is started from the agent loop long after the user became
    /// free to click another chat (`generating` goes false as soon as a call
    /// is proposed, state#9), so resolving the chat here would build the next
    /// step's prompt from a DIFFERENT conversation's history and put the reply
    /// there. Every caller has the originating chat id in hand.
    func executeGenerationTurn(step: Int, chatID: UUID) {
        guard let session, let chatIndex = chats.firstIndex(where: { $0.id == chatID }) else { return }
        clearCompactionBoundaryEvent(for: chatID)
        beginRecoveryEventHistoryForGenerationStart(step: step, chatID: chatID)

        // Captured once, up front: every append this turn produces (prose,
        // tool call, denial, pending-approval) targets THIS chat, never
        // whatever `selectedChatID` resolves to at the moment of appending.
        let turnChatID = chatID
        // And every POLICY this turn reads comes from the chat's own project
        // rather than from the selection (state#30): system prompt, workspace
        // root, agent type, step cap, guardrails mode, skill-state toggle,
        // and the permission evaluation of whatever call the turn proposes.
        let turnProject = self.turnProject(chatID: chatID)
        let memoryTurnQuery = turnMessages(for: chatID).last(where: { $0.role == .user })?.content ?? ""
        let usesSkillState = turnProject?.skillStateEnabled ?? false
        let turnMediaCapability = AppToolMediaCapability(
            supportsImageBearingToolResults: session.info.vision.active,
            maximumPixelCount: session.info.vision.maxPixels)
        let turnAvailableTools = AppToolCatalog.captureTurnAvailableTools(
            for: turnProject,
            globalMcpServers: globalMcpServers,
            contextTokens: maxContextTokens > 0 ? maxContextTokens : nil,
            webToolsEnabled: webSearchEnabled,
            browserAvailability: browserToolAvailability(mediaCapability: turnMediaCapability))
        let turnAllowsToolCalls = interactionMode == .projects && turnProject != nil

        generationEpoch += 1
        let myEpoch = generationEpoch

        outputPromptText = turnMessages(for: chatID).last(where: { $0.role == .user })?.content ?? ""
        outputText = ""
        outputReasoningText = ""
        generating = true
        phase = .prefill
        livePrefillDone = 0
        livePrefillTotal = 0
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        isCancellationPending = false
        error = nil
        decodeStartTime = nil

        // Two prompt shapes. The bounded-state one is O(1) in step count; the
        // append-only one below grows with every turn and every tool result,
        // and on a 4,096-context install stops the run outright between step
        // 30 and 35 (docs/SKILL_STATE.md). Opt-in per project, default off, so
        // the append-only path is byte-identical when the toggle is not set.
        if !usesSkillState {
            // This advances the two-full-send observation policy only for a
            // generation that is about to be started, never for a context
            // meter refresh or an ordinary transcript redraw.
            prepareToolOutputProjectionsForPrompt(chatID: chatID)
        }
        let turnReasoning = reasoning

        // Per-chat sampling: the chat's own override when it carries one,
        // the app-wide settings otherwise. Resolved from `chatID`, never the
        // selection (state#17's rule for every per-turn read), and captured
        // ONCE here so a scope edit mid-turn cannot move a running turn.
        // `maxNewTokens` arrives folded in, already clamped against the
        // `UInt32` conversion this read used to do by hand (state#35).
        var options = samplingOptions(chatID: chatID)
        options.reasoning = turnReasoning
        let requestedNewTokens = options.maxNewTokens
        let continuationEnabled = autoContinuationEnabled

        runTask = Task {
            var generationRequestIndex = 0
            var continuationRequestsStarted = 0
            do {
                await self.interruptTitleGenerationForForeground()
                await self.interruptMemoryCaptureForForeground()
                await self.prepareSemanticMemoryRecall(
                    chatID: turnChatID, project: turnProject, query: memoryTurnQuery)
                try Task.checkCancellation()
                let rawHistoryProjection: AppChatHistoryProjection
                if usesSkillState {
                    let skillHistory = self.buildSkillStateHistory(
                        chatIndex: chatIndex, project: turnProject,
                        availableTools: turnAvailableTools,
                        mediaCapability: turnMediaCapability)
                    rawHistoryProjection = AppChatHistoryProjection(
                        messages: skillHistory,
                        sourceRowIndexByMessage: Array(repeating: nil, count: skillHistory.count),
                        instructionPinBlockIndex: nil,
                        sourceTranscriptRowCount: 0)
                } else {
                    rawHistoryProjection = self.buildAppendOnlyHistoryProjection(
                        chatIndex: chatIndex, project: turnProject,
                        availableTools: turnAvailableTools,
                        reasoning: turnReasoning, mediaCapability: turnMediaCapability)
                }
                // **AUTO-COMPACTION, BEFORE THE FIT.** Near the window the
                // choice used to be the no-room error below or `fitWindow`'s
                // silent drop of older turns; summarizing them first is the
                // path that keeps the conversation going. The bounded
                // SKILL.state path is excluded: its prompt is O(1) in step
                // count and never needs it. On any failure this falls
                // through to the unchanged fit below, so compaction adds no
                // new way for a turn to die.
                var turnHistoryProjection = rawHistoryProjection
                if !usesSkillState {
                    let compaction = self.compactionState(chatID: turnChatID)
                    await self.refreshInstructionPinningCache(
                        chatID: turnChatID,
                        boundary: compaction.boundary,
                        messages: self.turnMessages(for: turnChatID),
                        session: session,
                        reasoning: turnReasoning)
                    guard let currentChatIndex = self.chats.firstIndex(where: {
                        $0.id == turnChatID
                    }) else { return }
                    turnHistoryProjection = self.buildAppendOnlyHistoryProjection(
                        chatIndex: currentChatIndex, project: turnProject,
                        availableTools: turnAvailableTools, reasoning: turnReasoning,
                        mediaCapability: turnMediaCapability)

                    let compacted = await self.runAutoCompactionIfNeeded(
                        chatID: turnChatID, project: turnProject,
                        rawHistory: turnHistoryProjection.messages,
                        maxContext: session.info.maxContext,
                        reservedForNew: requestedNewTokens, reasoning: turnReasoning)
                    if compacted {
                        // The boundary moved; assemble again from it rather
                        // than reuse a history built against the old one.
                        // By ID, not the captured index: the array can have
                        // changed while the summarizer ran.
                        guard let refreshed = self.chats.firstIndex(where: { $0.id == turnChatID })
                        else { return }
                        turnHistoryProjection = self.buildAppendOnlyHistoryProjection(
                            chatIndex: refreshed, project: turnProject,
                            availableTools: turnAvailableTools, reasoning: turnReasoning,
                            mediaCapability: turnMediaCapability)
                    }
                }
                // **THE PROMPT BUDGET IS THE WINDOW MINUS WHAT GENERATION
                // NEEDS** (state#36). This called `fitWindow` at its default
                // bound, which is the whole `maxContext` -- so a history that
                // "fits" leaves no room at all for the reply, and the three
                // things the outcome reports were all discarded: the turn was
                // sent truncated with no word to the user, and a no-room
                // verdict still called `generate`, which then clamps to one
                // token or throws a context overflow naming neither cause.
                let promptBudget = session.info.maxContext > requestedNewTokens
                    ? session.info.maxContext - requestedNewTokens
                    : session.info.maxContext
                let fitted = try await self.fitRequestHistoryAfterMicrocompact(
                    chatID: turnChatID,
                    history: turnHistoryProjection,
                    countTokens: { messages in
                        try await session.countTokens(messages, reasoning: turnReasoning)
                    },
                    fitWindow: { messages in
                        await self.fitRequestHistoryPreservingInstructionPins(
                            messages,
                            injectedBlockIndex: turnHistoryProjection.instructionPinBlockIndex,
                            chatID: turnChatID,
                            maxTokens: promptBudget,
                            session: session,
                            reasoning: turnReasoning)
                    })
                let fittedInstructionPinBlockIndex: Int?
                if let projectionIndex = turnHistoryProjection.instructionPinBlockIndex,
                    turnHistoryProjection.messages.indices.contains(projectionIndex)
                {
                    let pinBlock = turnHistoryProjection.messages[projectionIndex]
                    fittedInstructionPinBlockIndex = fitted.retained.firstIndex(of: pinBlock)
                } else {
                    fittedInstructionPinBlockIndex = nil
                }
                let requestCompactionBoundaryEvent = self.compactionBoundaryEvents[turnChatID]
                guard fitted.hasRoomForGeneration else {
                    self.error =
                        "This conversation no longer fits: \(fitted.measuredTokens) prompt tokens "
                        + "against a \(session.info.maxContext)-token window with "
                        + "\(requestedNewTokens) reserved for the reply. Clear the conversation, "
                        + "shorten the attachments, or lower Max New Tokens."
                    self.finishCancelled(chatID: turnChatID, reason: "context_overflow")
                    // Same epoch guard the tail below carries (state#10):
                    // this early exit must not clobber a newer turn either.
                    if self.generationEpoch == myEpoch {
                        self.generating = false
                        self.phase = .idle
                        self.isCancellationPending = false
                        self.runTask = nil
                        self.updateTokenEstimate()
                        self.drainPendingUserMessagesIfIdle(chatID: turnChatID)
                        self.drainPendingTaskNotificationsIfIdle(chatID: turnChatID)
                        self.drainPendingTitleGenerationIfIdle(session: session)
                    }
                    return
                }
                if fitted.removedTurnCount > 0 {
                    self.showToast(
                        "Dropped \(fitted.removedTurnCount) older "
                            + "\(fitted.removedTurnCount == 1 ? "turn" : "turns") to fit the "
                            + "context window.",
                        style: .warning)
                }
                let completed = try await AppStreamRecovery.generateUntilComplete(
                    baseMessages: fitted.retained,
                    options: options,
                    enabled: continuationEnabled,
                    prepareContinuationMessages: { messages, reservedOutputTokens in
                        let continuationPromptBudget = session.info.maxContext > reservedOutputTokens
                            ? session.info.maxContext - reservedOutputTokens
                            : session.info.maxContext
                        let continuationFit = await self.fitRequestHistoryPreservingInstructionPins(
                            messages,
                            injectedBlockIndex: fittedInstructionPinBlockIndex,
                            chatID: turnChatID,
                            maxTokens: continuationPromptBudget,
                            session: session,
                            reasoning: turnReasoning)
                        guard continuationFit.hasRoomForGeneration else {
                            throw ContinuationGenerationError.continuationContextDoesNotFit
                        }
                        return continuationFit.retained
                    },
                    onRecoveryEvent: { event in
                        await MainActor.run {
                            self.streamRecoveryEvents[turnChatID, default: []].append(event)
                        }
                    }
                ) { requestMessages, requestOptions in
                    if generationRequestIndex > 0 {
                        continuationRequestsStarted = min(
                            continuationRequestsStarted + 1,
                            AppStreamRecovery.maxContinuationsPerTurn)
                    }
                    generationRequestIndex += 1
                    var segmentResult: GenerationResult?
                    for try await event in session.generate(requestMessages, options: requestOptions) {
                        try Task.checkCancellation()
                        switch event {
                        case .prefill(let done, let total):
                            self.phase = .prefill
                            self.livePrefillDone = done
                            self.livePrefillTotal = total
                        case .content(let chunk):
                            if self.phase != .decode {
                                self.phase = .decode
                                self.decodeStartTime = Date()
                            }
                            self.outputText += chunk
                            self.liveTokenCount += 1
                            if let start = self.decodeStartTime {
                                self.liveElapsedDecodeSeconds = Date().timeIntervalSince(start)
                            }
                        case .reasoning(let chunk):
                            // Reasoning shares the live decode clock and stays
                            // display-only, matching the main assistant text.
                            if self.phase != .decode {
                                self.phase = .decode
                                self.decodeStartTime = Date()
                            }
                            self.outputReasoningText += chunk
                            self.liveTokenCount += 1
                            if let start = self.decodeStartTime {
                                self.liveElapsedDecodeSeconds = Date().timeIntervalSince(start)
                            }
                        case .toolCall, .stopped:
                            // The binding does not offer tools yet. `.stopped`
                            // precedes the result that ends this segment.
                            break
                        case .finished(let result):
                            self.phase = .idle
                            let phaseReport = try? await session.phases()
                            self.diagnostics = AppDiagnostics(
                                result: result,
                                peakMemory: TurboSparkSession.peakFootprintBytes,
                                phases: phaseReport,
                                compactionBoundaryEvent: requestCompactionBoundaryEvent
                            )
                            // Account for every model request, including each
                            // continuation segment, exactly once.
                            self.recordUsage(
                                promptTokens: result.promptTokens,
                                outputTokens: result.newTokens,
                                chatID: turnChatID)
                            self.recordGoalTurnTokens(
                                chatID: turnChatID, tokens: result.newTokens)
                            segmentResult = result
                        }
                    }
                    guard let segmentResult else {
                        throw ContinuationGenerationError.missingResult
                    }
                    return segmentResult
                }
                let result = completed.result
                // Merge the state patch BEFORE parsing tool calls, so the
                // parser never mistakes patch JSON for a tool call.
                var generatedContent = completed.content
                if usesSkillState,
                    let idx = self.chats.firstIndex(where: { $0.id == turnChatID }) {
                    generatedContent = self.applySkillStatePatch(
                        from: generatedContent, chatIndex: idx, project: turnProject)
                }
                let generatedReasoning = completed.reasoning
                let streamState: ToolCallStreamState
                if case .cancelled = result.stopReason {
                    streamState = .failed
                } else {
                    streamState = .completed
                }
                let dispatchGate = ToolCallDispatchGate.evaluate(
                    content: generatedContent,
                    streamState: streamState,
                    availableTools: turnAvailableTools,
                    forgeGuardrailsEnabled: self.forgeGuardrailsEnabled(for: turnProject),
                    allowsParsing: turnAllowsToolCalls,
                    projectURL: turnProject?.rootDirectoryURL)
                if !dispatchGate.refusals.isEmpty {
                    self.showToast(
                        dispatchGate.refusals
                            .map(\.userFacingSummary)
                            .joined(separator: " "),
                        style: .warning)
                }
                // Ordinary Retry/Edit variants describe prose and cannot be
                // attached to a chain of tool rows. Recovery retries keep the
                // partial variant parked until their eventual prose reply.
                if !dispatchGate.dispatchableCalls.isEmpty,
                    self.pendingRecoveryRetryCounts[turnChatID] == nil
                {
                    self.pendingResponseVariants[turnChatID] = nil
                }

                // One helper for the four dispatch sites below. An all-agent
                // batch runs its calls concurrently; everything else keeps
                // the first-call-only rule (state#37).
                func dispatch(_ calls: [AppToolCall], content: String) async {
                    if calls.isEmpty {
                        await self.finishProseTurn(
                            content: content,
                            reasoning: generatedReasoning,
                            result: result,
                            chatID: turnChatID,
                            step: step,
                            project: turnProject,
                            continuationsUsed: completed.continuationsUsed)
                    } else {
                        await self.handleExtractedToolCalls(
                            calls,
                            fullContent: content,
                            reasoning: generatedReasoning,
                            result: result,
                            currentStep: step,
                            chatID: turnChatID,
                            project: turnProject)
                    }
                }

                let maxSteps = turnProject?.maxAutonomousSteps ?? 5
                let resolution = ToolCallDispatchResolution.resolve(
                    gateResult: dispatchGate,
                    originalContent: generatedContent,
                    completedAttempts: step + 1,
                    maximumAttempts: maxSteps)
                let handlingResult = await ToolCallDispatchResolution.handle(
                    resolution,
                    retry: { nudge, assistantContent in
                        self.mutateTurnMessages(for: turnChatID) { messages in
                            messages.append(AppChatMessage(
                                role: .assistant,
                                content: assistantContent,
                                reasoning: generatedReasoning,
                                stopReason: "guardrail_retry"
                            ))
                            messages.append(AppChatMessage(
                                role: .user,
                                content: nudge
                            ))
                        }
                        self.outputText = ""
                        self.outputReasoningText = ""
                        self.continueAgentLoop(step: step + 1, chatID: turnChatID)
                    },
                    finishProse: { content in
                        await dispatch([], content: content)
                    },
                    dispatch: { calls, content in
                        await dispatch(calls, content: content)
                    })
                if handlingResult == .retried { return }

                // **UNDER THE EPOCH GUARD** (state#98). `dispatch` above can
                // re-enter `executeGenerationTurn` through the agent loop.
                if self.generationEpoch == myEpoch {
                    self.outputText = ""
                    self.outputReasoningText = ""
                }
            } catch {
                let interruptionReason = AppStreamRecovery.interruptionReason(
                    taskIsCancelled: Task.isCancelled,
                    cancellationPending: self.isCancellationPending)
                if interruptionReason == .error {
                    self.error = error.localizedDescription
                }
                self.finishCancelled(
                    chatID: turnChatID,
                    reason: interruptionReason.rawValue,
                    continuationsUsed: continuationRequestsStarted)
            }
            guard self.generationEpoch == myEpoch else { return }
            self.generating = false
            self.phase = .idle
            self.isCancellationPending = false
            self.runTask = nil
            self.updateTokenEstimate()
            // **A BACKGROUND AGENT MAY HAVE FINISHED MID-TURN.** Its
            // notification parked in `pendingTaskNotifications` because the
            // chat was busy; this tail is the first idle moment after, under
            // the same epoch guard that says no newer turn owns the state.
            // The drain re-checks idleness and may start the next turn here.
            //
            // **QUEUED USER PROMPTS DRAIN FIRST.** A prompt the user typed
            // mid-turn is the older intent; the notification waits one more
            // tail rather than reverse the order the user saw. (The
            // notification drain re-checks idleness, so a queue drain that
            // started a turn parks it cleanly.)
            self.drainPendingUserMessagesIfIdle(chatID: turnChatID)
            self.drainPendingTaskNotificationsIfIdle(chatID: turnChatID)
            self.drainPendingTitleGenerationIfIdle(session: session)
        }
    }

    private func finishProseTurn(
        content: String,
        reasoning: String,
        result: GenerationResult,
        chatID: UUID,
        step: Int,
        project: AppProject?,
        continuationsUsed: Int
    ) async {
        if case .cancelled = result.stopReason {
            outputText = content
            outputReasoningText = reasoning
            finishCancelled(
                chatID: chatID,
                reason: "cancelled",
                continuationsUsed: continuationsUsed)
            return
        }
        // Variants parked by Retry/Edit land on the first committed prose
        // reply of the turn they belong to, and are consumed exactly once:
        // removeValue here is what keeps a stale entry from grafting old
        // text onto some later turn.
        let parkedVariants = self.pendingResponseVariants.removeValue(forKey: chatID) ?? []
        if let recoveryRetryAttempt = self.pendingRecoveryRetryCounts.removeValue(forKey: chatID) {
            self.mutateRecoveryState(for: chatID) { messages, anchor in
                AppStreamRecovery.commitSuccessfulRecoveryRetry(
                    into: &messages,
                    recoveryAnchor: &anchor,
                    content: content,
                    reasoning: reasoning,
                    result: result,
                    alternates: parkedVariants)
            }
            var recoveryEvents = self.streamRecoveryEvents[chatID, default: []]
            AppStreamRecovery.recordSuccessfulRecoveryRetry(
                attempt: recoveryRetryAttempt,
                into: &recoveryEvents)
            self.streamRecoveryEvents[chatID] = recoveryEvents
        } else {
            self.mutateTurnMessages(for: chatID) { messages in
                AppStreamRecovery.commitAssistantRow(
                    into: &messages,
                    content: content,
                    reasoning: reasoning,
                    result: result,
                    alternates: parkedVariants)
            }
        }
        _ = await self.dispatchStopAndContinueIfBlocked(
            chatID: chatID, resumeStep: step + 1, project: project)
    }
}
