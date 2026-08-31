import Foundation
import TurboSpark

extension AppModel {
    public func run() {
        guard canRun, session != nil else { return }
        let userDraft = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachments = promptAttachments

        // **A PICTURE IS NOT AN ATTACHMENT WITH EMPTY TEXT.** Images go to
        // the vision tower by path and are injected at the marker the
        // template renders; inlining them here would produce an empty
        // "--- Attachment ---" block and send the model nothing at all.
        // Only a picture whose source is still on disk can be sent, because
        // the engine reads the file itself.
        let (imageDocs, textDocs) = attachments.reduce(
            into: ([AppPromptAttachment](), [AppPromptAttachment]())
        ) { split, doc in
            if doc.isSendableImage {
                split.0.append(doc)
            } else {
                split.1.append(doc)
            }
        }
        let promptImages: [ChatImage] =
            visionIsActive ? imageDocs.compactMap { $0.sourcePath.map(ChatImage.path) } : []

        var fullUserContent = userDraft
        if !textDocs.isEmpty {
            let docsText = textDocs.map { doc in
                "--- Attachment: \(doc.fileName) (\(doc.formatLabel)) ---\n\(doc.extractedText)\n--- End of \(doc.fileName) ---"
            }.joined(separator: "\n\n")
            if fullUserContent.isEmpty {
                fullUserContent = docsText
            } else {
                fullUserContent = "\(fullUserContent)\n\n\(docsText)"
            }
        }

        // A turn carrying only a picture has no text and is still a turn.
        guard !fullUserContent.isEmpty || !promptImages.isEmpty else { return }

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
            let userMessage = AppChatMessage(
                role: .user,
                content: contentForModel,
                imagePaths: promptImages.compactMap {
                    if case .path(let p) = $0 { return p } else { return nil }
                })
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
            // **AN IMAGE-ONLY TURN HAS NO TEXT AND IS STILL A TURN.** This
            // guard predates images and would drop one entirely, leaving the
            // model to answer a question whose picture was never sent -- the
            // emptiness-guard failure in its usual shape.
            guard !msg.content.isEmpty || !msg.imagePaths.isEmpty else { continue }
            rawHistory.append(
                ChatMessage(
                    role: msg.role,
                    content: msg.content,
                    images: msg.imagePaths.map(ChatImage.path)))
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
        // **THIS ESTIMATE IS A FLOOR ON A TURN CARRYING IMAGES, NOT A
        // COUNT.** `countTokens` renders the template and encodes it, and the
        // template emits ONE marker per image whatever its size -- the
        // expansion to that page's merged-token count happens later, in the
        // engine's splice, which needs the preprocessed grid. So an image
        // turn is undercounted by roughly a page's worth of positions. The
        // images are passed anyway so the estimate tracks what is actually
        // sent rather than a different conversation.
        var history = selectedChat.messages.compactMap { msg -> ChatMessage? in
            guard !msg.content.isEmpty || !msg.imagePaths.isEmpty else { return nil }
            return ChatMessage(
                role: msg.role,
                content: msg.content,
                images: msg.imagePaths.map(ChatImage.path))
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
