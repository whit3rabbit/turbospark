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

        // Captured once, up front: every append this turn produces (prose,
        // tool call, denial, pending-approval) targets THIS chat, never
        // whatever `selectedChatID` resolves to at the moment of appending.
        let turnChatID = chatID
        // And every POLICY this turn reads comes from the chat's own project
        // rather than from the selection (state#30): system prompt, workspace
        // root, agent type, step cap, guardrails mode, skill-state toggle,
        // and the permission evaluation of whatever call the turn proposes.
        let turnProject = self.turnProject(chatID: chatID)
        let usesSkillState = turnProject?.skillStateEnabled ?? false

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
        let rawHistory: [ChatMessage] =
            usesSkillState
            ? buildSkillStateHistory(chatIndex: chatIndex, project: turnProject)
            : buildAppendOnlyHistory(chatIndex: chatIndex, project: turnProject)

        var options = GenerateOptions()
        options.reasoning = reasoning
        options.temperature = temperature
        // `UInt32(clamping:)`, never `UInt32(_:)`: the plain conversion TRAPS
        // (kills the process, no error, no log) on any value above
        // `UInt32.max`, and this one is loaded straight out of a JSON file a
        // user or a future release can write (state#35). `clampedSetting` on
        // load is the other half; this is the one that cannot trap whatever
        // reaches it.
        let requestedNewTokens = UInt32(clamping: max(1, maxNewTokens))
        options.maxNewTokens = requestedNewTokens
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

        runTask = Task {
            do {
                // **AUTO-COMPACTION, BEFORE THE FIT.** Near the window the
                // choice used to be the no-room error below or `fitWindow`'s
                // silent drop of older turns; summarizing them first is the
                // path that keeps the conversation going. The bounded
                // SKILL.state path is excluded: its prompt is O(1) in step
                // count and never needs it. On any failure this falls
                // through to the unchanged fit below, so compaction adds no
                // new way for a turn to die.
                var turnHistory = rawHistory
                if !usesSkillState {
                    let compacted = await self.runAutoCompactionIfNeeded(
                        chatID: turnChatID, project: turnProject, rawHistory: rawHistory,
                        maxContext: session.info.maxContext,
                        reservedForNew: requestedNewTokens, reasoning: self.reasoning)
                    if compacted {
                        // The boundary moved; assemble again from it rather
                        // than reuse a history built against the old one.
                        // By ID, not the captured index: the array can have
                        // changed while the summarizer ran.
                        guard let refreshed = self.chats.firstIndex(where: { $0.id == turnChatID })
                        else { return }
                        turnHistory = self.buildAppendOnlyHistory(
                            chatIndex: refreshed, project: turnProject)
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
                let fitted = try await session.fitWindow(
                    turnHistory, maxTokens: promptBudget, reasoning: self.reasoning)
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
                for try await event in session.generate(fitted.retained, options: options) {
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
                        // **REASONING STARTS THE CLOCK AND MUST THEREFORE BE
                        // COUNTED** (state#100). This set `decodeStartTime`
                        // and incremented nothing, so on a thinking turn the
                        // HUD divided a growing elapsed time by a token count
                        // that stayed at zero until the answer began -- the
                        // rate read 0 tok/s through the whole reasoning
                        // phase and then jumped. Same caveat as every other
                        // number here: this counts non-empty EVENTS rather
                        // than tokens (`swift/CLAUDE.md` Gotcha 7), and the
                        // authoritative figure is `GenerationResult`.
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
                        // No binding surface offers tools yet, and `.stopped`
                        // precedes the `.finished` this loop finalizes on.
                        break
                    case .finished(let result):
                        self.phase = .idle
                        let phaseReport = try? await session.phases()
                        self.diagnostics = AppDiagnostics(
                            result: result,
                            peakMemory: TurboSparkSession.peakFootprintBytes,
                            phases: phaseReport
                        )

                        // Merge the state patch BEFORE parsing tool calls, so
                        // the tool parser never sees the patch JSON and cannot
                        // mistake it for a call. Re-resolve the chat index
                        // from THIS TURN's chat, not from the selection: the
                        // selection can move while a turn is in flight, and
                        // merging into the chat the user switched to writes
                        // one conversation's bookkeeping into another's.
                        var generatedContent = self.outputText
                        if usesSkillState,
                            let idx = self.chats.firstIndex(where: { $0.id == turnChatID }) {
                            generatedContent = self.applySkillStatePatch(
                                from: generatedContent, chatIndex: idx, project: turnProject)
                        }
                        let generatedReasoning = self.outputReasoningText
                        let parsedCalls = self.extractToolCalls(
                            from: generatedContent, project: turnProject)

                        // One helper for the four dispatch sites below. An
                        // all-agent batch runs its calls concurrently
                        // (handleExtractedToolCalls); everything else keeps
                        // the "first call runs, the rest are recorded as
                        // refused" rule (state#37).
                        func dispatch(_ calls: [AppToolCall], content: String) async {
                            if calls.isEmpty {
                                await self.finishProseTurn(
                                    content: content,
                                    reasoning: generatedReasoning,
                                    result: result,
                                    chatID: turnChatID,
                                    step: step,
                                    project: turnProject)
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

                        if self.forgeGuardrailsEnabled(for: turnProject) {
                            let availableSpecs = AppToolCatalog.tools(
                                for: self.agentType(for: turnProject))
                            let verdict = ForgeGuardrailsEngine.inspect(
                                text: generatedContent,
                                parsedCalls: parsedCalls,
                                availableTools: availableSpecs,
                                requiresCall: false
                            )

                            switch verdict {
                            case .accept:
                                await dispatch(parsedCalls, content: generatedContent)
                            case .rescued(let rescuedCalls, let sanitizedText):
                                await dispatch(rescuedCalls, content: sanitizedText)
                            case .retry(let nudge):
                                let maxSteps = turnProject?.maxAutonomousSteps ?? 5
                                if step + 1 < maxSteps && !nudge.isEmpty {
                                    self.mutateTurnMessages(for: turnChatID) { messages in
                                        messages.append(AppChatMessage(
                                            role: .assistant,
                                            content: generatedContent,
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
                                    return
                                } else {
                                    await dispatch(parsedCalls, content: generatedContent)
                                }
                            }
                        } else {
                            await dispatch(parsedCalls, content: generatedContent)
                        }

                        // **UNDER THE EPOCH GUARD** (state#98). `dispatch`
                        // above can re-enter `executeGenerationTurn` through
                        // the agent loop, which sets up the NEXT turn's
                        // output state; clearing here unconditionally is the
                        // same clobber state#10 added the guard at this
                        // function's tail to prevent, one block earlier.
                        if self.generationEpoch == myEpoch {
                            self.outputText = ""
                            self.outputReasoningText = ""
                        }
                    }
                }
            } catch is CancellationError {
                self.finishCancelled(chatID: turnChatID, reason: "cancelled")
            } catch {
                self.error = error.localizedDescription
                self.finishCancelled(chatID: turnChatID, reason: "error")
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
            self.drainPendingTaskNotificationsIfIdle(chatID: turnChatID)
        }
    }

    private func finishProseTurn(content: String, reasoning: String, result: GenerationResult, chatID: UUID, step: Int, project: AppProject?) async {
        self.mutateTurnMessages(for: chatID) { messages in
            messages.append(AppChatMessage(
                role: .assistant,
                content: content,
                reasoning: reasoning,
                stopReason: result.stopReason.rawValue
            ))
        }
        _ = await self.dispatchStopAndContinueIfBlocked(
            chatID: chatID, resumeStep: step + 1, project: project)
    }
}
