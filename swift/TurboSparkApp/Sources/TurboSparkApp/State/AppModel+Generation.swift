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
                        let toolCalls = self.extractToolCalls(from: generatedContent)

                        if let firstCall = toolCalls.first {
                            self.handleExtractedToolCall(firstCall, fullContent: generatedContent, reasoning: generatedReasoning, result: result, currentStep: step)
                        } else {
                            if let idx = self.selectedChatIndex {
                                self.chats[idx].messages.append(AppChatMessage(
                                    role: .assistant,
                                    content: generatedContent,
                                    reasoning: generatedReasoning,
                                    stopReason: result.stopReason.rawValue
                                ))
                                self.chats[idx].updatedAt = Date()
                                self.persistChats()
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

    private func handleExtractedToolCall(_ call: AppToolCall, fullContent: String, reasoning: String, result: GenerationResult, currentStep: Int) {
        let sessionID = selectedChatID.uuidString
        let cmd = call.arguments["command"] ?? call.arguments["cmd"]

        Task { @MainActor in
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
