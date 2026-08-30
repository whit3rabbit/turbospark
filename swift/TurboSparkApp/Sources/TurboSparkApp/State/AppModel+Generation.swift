import Foundation
import TurboSpark

extension AppModel {
    public func run() {
        guard canRun, session != nil else { return }
        let userDraft = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachments = promptAttachments

        var fullUserContent = userDraft
        if !attachments.isEmpty {
            let docsText = attachments.map { doc in
                "--- Attachment: \(doc.fileName) (\(doc.formatLabel)) ---\n\(doc.extractedText)\n--- End of \(doc.fileName) ---"
            }.joined(separator: "\n\n")
            if fullUserContent.isEmpty {
                fullUserContent = docsText
            } else {
                fullUserContent = "\(fullUserContent)\n\n\(docsText)"
            }
        }

        guard !fullUserContent.isEmpty else { return }

        // `UserPromptSubmit` has to be awaited BEFORE the message is
        // appended to the chat, or a hook cannot actually stop the turn: by
        // the time this app used to fire it (inside `executeGenerationTurn`,
        // fire-and-forget), the user's message was already in the
        // transcript and prefill was already starting. A block now simply
        // never appends the message and leaves the draft intact, which
        // needs no "remove/annotate" step because there is nothing to undo.
        Task {
            self.stopHookReentryCount = 0
            let verdict = await self.evaluateUserPromptSubmit(prompt: fullUserContent)
            if verdict.isBlocked {
                self.error = verdict.blockReason ?? "Prompt blocked by a UserPromptSubmit hook."
                return
            }

            let chatIndex: Int
            if let existing = self.selectedChatIndex {
                chatIndex = existing
            } else {
                let newChat = AppChat(id: self.selectedChatID, projectID: self.selectedProjectID)
                self.chats.insert(newChat, at: 0)
                chatIndex = 0
                self.selectedChatID = newChat.id
                // `createChat()` dispatches `SessionStart` on its own path;
                // this is the OTHER place a chat gets created lazily
                // (the very first prompt in a fresh window), which had no
                // dispatch at all.
                Task {
                    _ = await self.dispatchLifecycleHook(event: .sessionStart, source: "startup")
                }
            }

            // Clear draft & attachments
            self.chats[chatIndex].draft = ""
            self.chats[chatIndex].draftAttachments = []
            self.chats[chatIndex].updatedAt = Date()

            var contentForModel = fullUserContent
            if let context = verdict.additionalContext, !context.isEmpty {
                contentForModel += "\n\n<hook_context>\n\(context)\n</hook_context>"
            }

            // Append user turn
            let userMessage = AppChatMessage(role: .user, content: contentForModel)
            self.chats[chatIndex].messages.append(userMessage)
            if (self.chats[chatIndex].title == "New Chat" || self.chats[chatIndex].title.isEmpty) && !userDraft.isEmpty {
                self.chats[chatIndex].title = String(userDraft.prefix(40)).replacingOccurrences(of: "\n", with: " ")
            }
            self.persistChats()

            self.executeGenerationTurn(step: 0)
        }
    }

    /// Executes a generation step for the active conversation.
    func executeGenerationTurn(step: Int) {
        guard let session, let chatIndex = selectedChatIndex else { return }

        // Captured once, up front: every append this turn produces (prose,
        // tool call, denial, pending-approval) targets THIS chat, never
        // whatever `selectedChatID` resolves to at the moment of appending.
        // `generating` goes false as soon as a tool call is proposed
        // (state#9), so the UI treats a chat switch as legal in between;
        // without this capture, an approval or a prose reply lands in
        // whatever chat the user has since switched to.
        let turnChatID = chats[chatIndex].id

        generationEpoch += 1
        let myEpoch = generationEpoch

        outputPromptText = chats[chatIndex].messages.last(where: { $0.role == .user })?.content ?? ""
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

        // Build system prompt if project / agent is configured
        var rawHistory: [ChatMessage] = []
        let systemContent = buildSystemPrompt(for: selectedProject)
        if !systemContent.isEmpty {
            rawHistory.append(ChatMessage(role: .system, content: systemContent))
        }

        for msg in chats[chatIndex].messages {
            guard !msg.content.isEmpty else { continue }
            rawHistory.append(ChatMessage(role: msg.role, content: msg.content))
            // If message contained tool execution results, inject them as system/environment responses
            for res in msg.toolResults {
                let tag = res.isError ? "tool_error" : "tool_response"
                rawHistory.append(ChatMessage(role: .system, content: "<\(tag)>\n\(res.output)\n</\(tag)>"))
            }
        }

        var options = GenerateOptions()
        options.reasoning = reasoning
        options.temperature = temperature
        options.maxNewTokens = UInt32(max(1, maxNewTokens))
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
                let fitted = try await session.fitWindow(rawHistory, reasoning: self.reasoning)
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
                        if self.phase != .decode {
                            self.phase = .decode
                            self.decodeStartTime = Date()
                        }
                        self.outputReasoningText += chunk
                    case .finished(let result):
                        self.phase = .idle
                        let phaseReport = try? await session.phases()
                        self.diagnostics = AppDiagnostics(
                            result: result,
                            peakMemory: TurboSparkSession.peakFootprintBytes,
                            phases: phaseReport
                        )

                        let generatedContent = self.outputText
                        let generatedReasoning = self.outputReasoningText
                        let parsedCalls = self.extractToolCalls(from: generatedContent)

                        if self.effectiveForgeGuardrailsEnabled {
                            let availableSpecs = AppToolCatalog.tools(for: self.activeAgentType)
                            let verdict = ForgeGuardrailsEngine.inspect(
                                text: generatedContent,
                                parsedCalls: parsedCalls,
                                availableTools: availableSpecs,
                                requiresCall: false
                            )

                            switch verdict {
                            case .accept:
                                if let firstCall = parsedCalls.first {
                                    self.handleExtractedToolCall(firstCall, fullContent: generatedContent, reasoning: generatedReasoning, result: result, currentStep: step, chatID: turnChatID)
                                } else {
                                    await self.finishProseTurn(content: generatedContent, reasoning: generatedReasoning, result: result, chatID: turnChatID, step: step)
                                }
                            case .rescued(let rescuedCalls, let sanitizedText):
                                if let firstCall = rescuedCalls.first {
                                    self.handleExtractedToolCall(firstCall, fullContent: sanitizedText, reasoning: generatedReasoning, result: result, currentStep: step, chatID: turnChatID)
                                } else {
                                    await self.finishProseTurn(content: sanitizedText, reasoning: generatedReasoning, result: result, chatID: turnChatID, step: step)
                                }
                            case .retry(let nudge):
                                let maxSteps = self.selectedProject?.maxAutonomousSteps ?? 5
                                if step + 1 < maxSteps && !nudge.isEmpty {
                                    if let idx = self.chats.firstIndex(where: { $0.id == turnChatID }) {
                                        self.chats[idx].messages.append(AppChatMessage(
                                            role: .assistant,
                                            content: generatedContent,
                                            reasoning: generatedReasoning,
                                            stopReason: "guardrail_retry"
                                        ))
                                        self.chats[idx].messages.append(AppChatMessage(
                                            role: .user,
                                            content: nudge
                                        ))
                                        self.chats[idx].updatedAt = Date()
                                        self.persistChats()
                                    }
                                    self.outputText = ""
                                    self.outputReasoningText = ""
                                    self.continueAgentLoop(step: step + 1)
                                    return
                                } else {
                                    if let firstCall = parsedCalls.first {
                                        self.handleExtractedToolCall(firstCall, fullContent: generatedContent, reasoning: generatedReasoning, result: result, currentStep: step, chatID: turnChatID)
                                    } else {
                                        await self.finishProseTurn(content: generatedContent, reasoning: generatedReasoning, result: result, chatID: turnChatID, step: step)
                                    }
                                }
                            }
                        } else {
                            if let firstCall = parsedCalls.first {
                                self.handleExtractedToolCall(firstCall, fullContent: generatedContent, reasoning: generatedReasoning, result: result, currentStep: step, chatID: turnChatID)
                            } else {
                                await self.finishProseTurn(content: generatedContent, reasoning: generatedReasoning, result: result, chatID: turnChatID, step: step)
                            }
                        }

                        self.outputText = ""
                        self.outputReasoningText = ""
                    }
                }
            } catch is CancellationError {
                self.finishCancelled(chatID: turnChatID)
            } catch {
                self.error = error.localizedDescription
                self.finishCancelled(chatID: turnChatID)
            }
            guard self.generationEpoch == myEpoch else { return }
            self.generating = false
            self.phase = .idle
            self.isCancellationPending = false
            self.runTask = nil
            self.updateTokenEstimate()
        }
    }

    private func finishProseTurn(content: String, reasoning: String, result: GenerationResult, chatID: UUID, step: Int) async {
        if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
            self.chats[idx].messages.append(AppChatMessage(
                role: .assistant,
                content: content,
                reasoning: reasoning,
                stopReason: result.stopReason.rawValue
            ))
            self.chats[idx].updatedAt = Date()
            self.persistChats()
        }
        _ = await self.dispatchStopAndContinueIfBlocked(chatID: chatID, resumeStep: step + 1)
    }

    /// Dispatches `Stop` and, if a hook blocks, re-enters the agent loop
    /// with the hook's reason folded in as the next user turn -- Claude
    /// Code's own "a Stop hook can force the turn to continue" behavior.
    /// Capped at 8 consecutive re-entries so a hook that always blocks
    /// cannot loop forever. Returns `true` when it re-entered (the caller
    /// should treat the turn as still in progress, not finished).
    @discardableResult
    func dispatchStopAndContinueIfBlocked(chatID: UUID, resumeStep: Int) async -> Bool {
        let wasActive = stopHookReentryCount > 0
        let verdict = await evaluateStop(stopHookActive: wasActive)
        guard verdict.isBlocked, stopHookReentryCount < 8 else {
            stopHookReentryCount = 0
            return false
        }
        stopHookReentryCount += 1
        let reason = verdict.blockReason ?? "A Stop hook requested the turn continue."
        if let idx = chats.firstIndex(where: { $0.id == chatID }) {
            chats[idx].messages.append(AppChatMessage(role: .user, content: reason))
            chats[idx].updatedAt = Date()
            persistChats()
        }
        continueAgentLoop(step: resumeStep)
        return true
    }

    /// Continues the agent loop if steps remain, otherwise asks `Stop`
    /// hooks whether to keep going anyway (bounded by the 8-cap above,
    /// independent of `maxAutonomousSteps`).
    func continueOrStop(afterStep currentStep: Int, chatID: UUID) async {
        let maxSteps = selectedProject?.maxAutonomousSteps ?? 5
        if currentStep + 1 < maxSteps {
            continueAgentLoop(step: currentStep + 1)
        } else {
            _ = await dispatchStopAndContinueIfBlocked(chatID: chatID, resumeStep: currentStep + 1)
        }
    }

    // `internal` rather than `private`: exercised directly by
    // AgentLoopRoutingTests / HookDecisionRoutingTests via `@testable
    // import`, since it is otherwise reachable only from inside a live
    // generation `Task` that needs a real model session.
    func handleExtractedToolCall(_ call: AppToolCall, fullContent: String, reasoning: String, result: GenerationResult, currentStep: Int, chatID: UUID) {
        let sessionID = chatID.uuidString

        Task { @MainActor in
            // Evaluate PreToolUse lifecycle hooks first
            let hookDecision = await self.evaluatePreToolUseHooks(toolName: call.name, toolArguments: call.arguments)

            // `updatedInput` replaces the corresponding argument keys before
            // anything downstream (permission evaluation, execution, the
            // approval card) sees the call.
            var call = call
            if let updated = hookDecision.updatedInput {
                for (key, value) in updated { call.arguments[key] = value }
            }
            let cmd = call.arguments["command"] ?? call.arguments["cmd"]

            if hookDecision.behavior == .deny {
                let reason = hookDecision.reason ?? "Blocked by PreToolUse hook"
                let deniedResult = AppToolResult(
                    callID: call.id,
                    output: "Tool execution blocked by hook: \(reason)",
                    isError: true,
                    durationSeconds: 0.0
                )
                var deniedCall = call
                deniedCall.status = .denied
                if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
                    self.chats[idx].messages.append(AppChatMessage(
                        role: .assistant,
                        content: fullContent,
                        reasoning: reasoning,
                        stopReason: "tool_use",
                        toolCalls: [deniedCall],
                        toolResults: [deniedResult]
                    ))
                    self.chats[idx].updatedAt = Date()
                    self.persistChats()
                }
                await self.continueOrStop(afterStep: currentStep, chatID: chatID)
                return
            }

            // A hook that asked for confirmation was previously ignored
            // outright -- only `.deny` was checked above, so `.ask` fell
            // through to the ordinary permission evaluation below, which
            // could return `.allow` and run the call with no prompt at all
            // (T10). Route it into the same pending-approval UI the normal
            // engine's own `.ask` uses, folding the hook's reason into the
            // call's risk assessment rather than replacing it.
            if hookDecision.behavior == .ask {
                var pending = call
                pending.status = .pendingApproval
                let baseAssessment = call.riskAssessment ?? ToolRiskClassifier.assessRisk(name: call.name, arguments: call.arguments)
                let hookReason = hookDecision.reason ?? "A PreToolUse hook requested confirmation before this call runs."
                pending.riskAssessment = ToolRiskAssessment(
                    level: baseAssessment.level,
                    category: baseAssessment.category,
                    reasons: baseAssessment.reasons + [hookReason]
                )
                self.pendingToolCall = pending
                self.pendingToolCallChatID = chatID
                self.pendingToolCallStep = currentStep
                if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
                    self.chats[idx].messages.append(AppChatMessage(
                        role: .assistant,
                        content: fullContent,
                        reasoning: reasoning,
                        stopReason: "tool_use",
                        toolCalls: [pending]
                    ))
                    self.chats[idx].updatedAt = Date()
                    self.persistChats()
                }
                return
            }

            let sessionApproved = await SessionApprovalStore.shared.isApproved(sessionID: sessionID, toolName: call.name, command: cmd)
            let decision = AppToolPermissionEngine.evaluate(call: call, project: self.selectedProject, sessionApproved: sessionApproved)

            switch decision {
            case .ask(let assessment, _):
                // `PermissionRequest` fires exactly where this app would
                // otherwise show the approval card, so a hook can resolve
                // `allow`/`deny` without ever surfacing the UI.
                let permVerdict = await self.evaluatePermissionRequest(toolName: call.name, toolArguments: call.arguments)

                if permVerdict.permissionDecision == .allow {
                    await self.runApprovedCall(call, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep)
                } else if permVerdict.permissionDecision == .deny {
                    let reason = permVerdict.permissionReason ?? "Denied by PermissionRequest hook"
                    await self.recordDeniedCall(call, reason: reason, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep)
                } else {
                    var pending = call
                    pending.status = .pendingApproval
                    pending.riskAssessment = assessment
                    self.pendingToolCall = pending
                    self.pendingToolCallChatID = chatID
                    self.pendingToolCallStep = currentStep
                    if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
                        self.chats[idx].messages.append(AppChatMessage(
                            role: .assistant,
                            content: fullContent,
                            reasoning: reasoning,
                            stopReason: "tool_use",
                            toolCalls: [pending]
                        ))
                        self.chats[idx].updatedAt = Date()
                        self.persistChats()
                    }
                }

            case .allow:
                await self.runApprovedCall(call, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep)

            case .deny(let reason):
                await self.recordDeniedCall(call, reason: reason, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep)
            }
        }
    }

    /// Runs an approved call, folds `PostToolUse`'s feedback/context into the
    /// result, appends the turn, and continues the loop. Shared by the
    /// ordinary `.allow` path and a `PermissionRequest` hook resolving
    /// `allow` in place of the approval card.
    private func runApprovedCall(_ call: AppToolCall, fullContent: String, reasoning: String, chatID: UUID, currentStep: Int) async {
        var runningCall = call
        runningCall.status = .running
        var toolResult = await AppToolRegistry.execute(call: runningCall, in: self.selectedProject)
        runningCall.status = toolResult.isError ? .failed : .completed

        let postVerdict = await self.dispatchPostToolUseVerdict(
            toolName: runningCall.name,
            toolArguments: runningCall.arguments,
            toolOutput: toolResult.output,
            toolDurationSeconds: toolResult.durationSeconds,
            isError: toolResult.isError
        )
        // Exit-2 stderr (or a `decision: "block"`) from a PostToolUse hook is
        // feedback, never a block -- the tool already ran. `additionalContext`
        // is folded in the same way. Both flow to the model on the NEXT turn
        // through the `<tool_response>`/`<tool_error>` tags built from
        // `toolResults` in `executeGenerationTurn`.
        if let note = postVerdict.blockReason ?? postVerdict.feedbackMessage, !note.isEmpty {
            toolResult.output += "\n\n<hook_feedback>\n\(note)\n</hook_feedback>"
        }
        if let ctx = postVerdict.additionalContext, !ctx.isEmpty {
            toolResult.output += "\n\n<hook_context>\n\(ctx)\n</hook_context>"
        }

        if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
            self.chats[idx].messages.append(AppChatMessage(
                role: .assistant,
                content: fullContent,
                reasoning: reasoning,
                stopReason: "tool_use",
                toolCalls: [runningCall],
                toolResults: [toolResult]
            ))
            self.chats[idx].updatedAt = Date()
            self.persistChats()
        }

        await self.continueOrStop(afterStep: currentStep, chatID: chatID)
    }

    /// Records a denied call (permission engine or `PermissionRequest` hook)
    /// and continues the loop.
    private func recordDeniedCall(_ call: AppToolCall, reason: String, fullContent: String, reasoning: String, chatID: UUID, currentStep: Int) async {
        let deniedResult = AppToolResult(
            callID: call.id,
            output: "Tool execution denied: \(reason)",
            isError: true,
            durationSeconds: 0.0
        )
        var deniedCall = call
        deniedCall.status = .denied
        if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
            self.chats[idx].messages.append(AppChatMessage(
                role: .assistant,
                content: fullContent,
                reasoning: reasoning,
                stopReason: "tool_use",
                toolCalls: [deniedCall],
                toolResults: [deniedResult]
            ))
            self.chats[idx].updatedAt = Date()
            self.persistChats()
        }
        await self.continueOrStop(afterStep: currentStep, chatID: chatID)
    }

    /// Continues multi-turn autonomous loop after tool execution.
    ///
    /// Does NOT gate on `!generating`: by the time anything calls this
    /// (the allow/deny paths above, or an approved/denied pending call),
    /// the CURRENT turn's token stream has already finished -- `generating`
    /// reads `true` here only because the ORIGINAL turn's own `runTask`
    /// tail (which resets it) has not yet run, a race against this very
    /// call (state#10). The `generationEpoch` guard on that tail is what
    /// keeps it from clobbering the state a reentrant call here is about
    /// to set up.
    public func continueAgentLoop(step: Int) {
        guard session != nil else { return }
        executeGenerationTurn(step: step)
    }

    func finishCancelled(chatID: UUID? = nil) {
        let targetID = chatID ?? selectedChatID
        if !outputText.isEmpty || !outputReasoningText.isEmpty {
            if let idx = chats.firstIndex(where: { $0.id == targetID }) {
                chats[idx].messages.append(AppChatMessage(
                    role: .assistant,
                    content: outputText,
                    reasoning: outputReasoningText,
                    stopReason: "cancelled"
                ))
                chats[idx].updatedAt = Date()
                persistChats()
            }
            outputText = ""
            outputReasoningText = ""
        }
        // `Stop` fires for observability even on a cancelled or errored
        // turn, but its block-and-continue capability does not apply here
        // on purpose: cancellation is user-initiated (the Stop button
        // should really stop), and re-entering after an error risks looping
        // straight back into the same failure. Fire-and-forget rather than
        // awaited, so a slow hook cannot delay the cancel/error UI update.
        Task {
            _ = await self.evaluateStop(stopHookActive: false)
        }
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        pendingToolCall = nil
        session?.cancel()
        runTask?.cancel()
    }

    public func updateTokenEstimate() {
        guard let session else {
            estimatedPromptTokens = 0
            return
        }
        var history = selectedChat.messages.compactMap { msg -> ChatMessage? in
            guard !msg.content.isEmpty else { return nil }
            return ChatMessage(role: msg.role, content: msg.content)
        }
        if !promptText.isEmpty {
            history.append(ChatMessage(role: .user, content: promptText))
        }

        tokenEstimateTask?.cancel()
        tokenEstimateTask = Task {
            if let count = try? await session.countTokens(history, reasoning: self.reasoning) {
                if !Task.isCancelled {
                    self.estimatedPromptTokens = count
                }
            }
        }
    }
}
