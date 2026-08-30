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

        let chatIndex: Int
        if let existing = selectedChatIndex {
            chatIndex = existing
        } else {
            let newChat = AppChat(id: selectedChatID, projectID: selectedProjectID)
            chats.insert(newChat, at: 0)
            chatIndex = 0
            selectedChatID = newChat.id
        }

        // Clear draft & attachments
        chats[chatIndex].draft = ""
        chats[chatIndex].draftAttachments = []
        chats[chatIndex].updatedAt = Date()

        // Append user turn
        let userMessage = AppChatMessage(role: .user, content: fullUserContent)
        chats[chatIndex].messages.append(userMessage)
        if (chats[chatIndex].title == "New Chat" || chats[chatIndex].title.isEmpty) && !userDraft.isEmpty {
            chats[chatIndex].title = String(userDraft.prefix(40)).replacingOccurrences(of: "\n", with: " ")
        }
        persistChats()

        executeGenerationTurn(step: 0)
    }

    /// Executes a generation step for the active conversation.
    func executeGenerationTurn(step: Int) {
        guard let session, let chatIndex = selectedChatIndex else { return }

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

        // Two prompt shapes. The bounded-state one is O(1) in step count; the
        // append-only one below grows with every turn and every tool result,
        // and on a 4,096-context install stops the run outright between step
        // 30 and 35 (docs/SKILL_STATE.md). Opt-in per project, default off, so
        // the append-only path is byte-identical when the toggle is not set.
        var rawHistory: [ChatMessage] = []
        if skillStateEnabled {
            rawHistory = buildSkillStateHistory(chatIndex: chatIndex)
        } else {
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

        if step == 1 {
            Task {
                _ = await self.dispatchLifecycleHook(event: .userPromptSubmit)
            }
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

                        // Merge the state patch BEFORE parsing tool calls, so
                        // the tool parser never sees the patch JSON and cannot
                        // mistake it for a call. Re-resolve the chat index:
                        // the selection can move while a turn is in flight.
                        var generatedContent = self.outputText
                        if self.skillStateEnabled, let idx = self.selectedChatIndex {
                            generatedContent = self.applySkillStatePatch(
                                from: generatedContent, chatIndex: idx)
                        }
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
                                    self.handleExtractedToolCall(firstCall, fullContent: generatedContent, reasoning: generatedReasoning, result: result, currentStep: step)
                                } else {
                                    self.finishProseTurn(content: generatedContent, reasoning: generatedReasoning, result: result)
                                }
                            case .rescued(let rescuedCalls, let sanitizedText):
                                if let firstCall = rescuedCalls.first {
                                    self.handleExtractedToolCall(firstCall, fullContent: sanitizedText, reasoning: generatedReasoning, result: result, currentStep: step)
                                } else {
                                    self.finishProseTurn(content: sanitizedText, reasoning: generatedReasoning, result: result)
                                }
                            case .retry(let nudge):
                                let maxSteps = self.selectedProject?.maxAutonomousSteps ?? 5
                                if step + 1 < maxSteps && !nudge.isEmpty {
                                    if let idx = self.selectedChatIndex {
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
                                        self.handleExtractedToolCall(firstCall, fullContent: generatedContent, reasoning: generatedReasoning, result: result, currentStep: step)
                                    } else {
                                        self.finishProseTurn(content: generatedContent, reasoning: generatedReasoning, result: result)
                                    }
                                }
                            }
                        } else {
                            if let firstCall = parsedCalls.first {
                                self.handleExtractedToolCall(firstCall, fullContent: generatedContent, reasoning: generatedReasoning, result: result, currentStep: step)
                            } else {
                                self.finishProseTurn(content: generatedContent, reasoning: generatedReasoning, result: result)
                            }
                        }

                        self.outputText = ""
                        self.outputReasoningText = ""
                    }
                }
            } catch is CancellationError {
                self.finishCancelled()
            } catch {
                self.error = error.localizedDescription
                self.finishCancelled()
            }
            self.generating = false
            self.phase = .idle
            self.isCancellationPending = false
            self.runTask = nil
            self.updateTokenEstimate()
        }
    }

    private func finishProseTurn(content: String, reasoning: String, result: GenerationResult) {
        if let idx = self.selectedChatIndex {
            self.chats[idx].messages.append(AppChatMessage(
                role: .assistant,
                content: content,
                reasoning: reasoning,
                stopReason: result.stopReason.rawValue
            ))
            self.chats[idx].updatedAt = Date()
            self.persistChats()
        }
        Task {
            _ = await self.dispatchLifecycleHook(event: .stop)
        }
    }

    private func handleExtractedToolCall(_ call: AppToolCall, fullContent: String, reasoning: String, result: GenerationResult, currentStep: Int) {
        let sessionID = selectedChatID.uuidString
        let cmd = call.arguments["command"] ?? call.arguments["cmd"]

        Task { @MainActor in
            // Evaluate PreToolUse lifecycle hooks first
            let hookDecision = await self.evaluatePreToolUseHooks(toolName: call.name, toolArguments: call.arguments)
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
                if let idx = self.selectedChatIndex {
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
                let maxSteps = self.selectedProject?.maxAutonomousSteps ?? 5
                if currentStep + 1 < maxSteps {
                    self.continueAgentLoop(step: currentStep + 1)
                }
                return
            }

            let sessionApproved = await SessionApprovalStore.shared.isApproved(sessionID: sessionID, toolName: call.name, command: cmd)
            let decision = AppToolPermissionEngine.evaluate(call: call, project: self.selectedProject, sessionApproved: sessionApproved)

            switch decision {
            case .ask(let assessment, _):
                var pending = call
                pending.status = .pendingApproval
                pending.riskAssessment = assessment
                self.pendingToolCall = pending
                if let idx = self.selectedChatIndex {
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

            case .allow:
                var runningCall = call
                runningCall.status = .running
                let toolResult = await AppToolRegistry.execute(call: runningCall, in: self.selectedProject)
                runningCall.status = toolResult.isError ? .failed : .completed

                // Dispatch PostToolUse & PostToolUseFailure lifecycle hooks
                await self.dispatchLifecycleHook(
                    event: .postToolUse,
                    toolName: runningCall.name,
                    toolArguments: runningCall.arguments,
                    toolOutput: toolResult.output,
                    toolDurationSeconds: toolResult.durationSeconds,
                    isError: toolResult.isError
                )
                if toolResult.isError {
                    await self.dispatchLifecycleHook(
                        event: .postToolUseFailure,
                        toolName: runningCall.name,
                        toolArguments: runningCall.arguments,
                        toolOutput: toolResult.output,
                        toolDurationSeconds: toolResult.durationSeconds,
                        isError: true
                    )
                }

                if let idx = self.selectedChatIndex {
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

                let maxSteps = self.selectedProject?.maxAutonomousSteps ?? 5
                if currentStep + 1 < maxSteps {
                    self.continueAgentLoop(step: currentStep + 1)
                }

            case .deny(let reason):
                let deniedResult = AppToolResult(
                    callID: call.id,
                    output: "Tool execution denied: \(reason)",
                    isError: true,
                    durationSeconds: 0.0
                )
                var deniedCall = call
                deniedCall.status = .denied
                if let idx = self.selectedChatIndex {
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
                let maxSteps = self.selectedProject?.maxAutonomousSteps ?? 5
                if currentStep + 1 < maxSteps {
                    self.continueAgentLoop(step: currentStep + 1)
                }
            }
        }
    }

    /// Continues multi-turn autonomous loop after tool execution.
    public func continueAgentLoop(step: Int = 1) {
        guard !generating, session != nil else { return }
        executeGenerationTurn(step: step)
    }

    func finishCancelled() {
        if !outputText.isEmpty || !outputReasoningText.isEmpty {
            if let idx = selectedChatIndex {
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
