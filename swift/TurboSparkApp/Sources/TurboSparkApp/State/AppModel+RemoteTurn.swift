import Foundation
import TurboSpark

/// The model a chat is pointed at when it is not the local one (ZP-22).
struct RemoteModelSelection: Equatable, Sendable {
    var providerID: UUID
    var modelID: String
}

/// A pending "content leaves this Mac" confirmation. The picker raises it on
/// selecting a remote model, so nothing is sent before the person says yes.
struct RemoteConfirmationRequest: Identifiable, Equatable {
    let id = UUID()
    var chatID: UUID
    var providerID: UUID
    var modelID: String
    var providerName: String
    var host: String
}

/// The pre-send size estimate for a remote chat: characters over a fixed
/// constant, labeled "estimate" wherever it is shown. A remote chat has no
/// local tokenizer for its model, so `session.countTokens` does not apply.
enum RemoteContextEstimate {
    static let charactersPerToken = 4

    static func tokens(messages: [ChatMessage], tools: [ToolSpec]) -> Int {
        var characters = 0
        for m in messages {
            characters += m.content.count
            for call in m.toolCalls { characters += call.name.count + call.argumentsJSON.count }
        }
        for t in tools {
            characters += t.name.count + (t.description?.count ?? 0)
                + (t.parameters?.jsonString.count ?? 0)
        }
        return characters / charactersPerToken + 1
    }
}

/// What the context ring shows for a remote chat.
struct RemoteContextReading: Equatable {
    var usedTokens: Int
    var windowTokens: Int
    /// True when no API usage has been reported yet and the number is the
    /// characters-over-four estimate.
    var isEstimate: Bool
}

extension AppModel {
    // MARK: - Selection

    func remoteSelection(for chatID: UUID) -> RemoteModelSelection? { remoteSelections[chatID] }

    var selectedChatRemoteSelection: RemoteModelSelection? { remoteSelections[selectedChatID] }

    /// A remote chat needs no loaded local model.
    var hasTurnModel: Bool { session != nil || selectedChatRemoteSelection != nil }

    /// The label the composer and the indicator show for the selection.
    func remoteSelectionLabel(for chatID: UUID) -> (provider: String, model: String, host: String)? {
        guard let sel = remoteSelections[chatID], let provider = remoteProviders.provider(id: sel.providerID)
        else { return nil }
        let model = provider.models.first { $0.id == sel.modelID }?.displayName ?? sel.modelID
        let host = URL(string: provider.baseURL)?.host ?? provider.baseURL
        return (provider.name, model, host)
    }

    /// Points a chat at a remote model. Content leaving the Mac is explicit:
    /// the first selection per provider per chat raises the confirmation, and
    /// only its Confirm button selects. Already confirmed selects at once.
    func requestRemoteModel(providerID: UUID, modelID: String, chatID: UUID) {
        guard !(generating && generatingChatID == chatID) else { return }
        switch remoteProviders.sendDecision(provider: providerID, model: modelID, chat: chatID) {
        case .proceed:
            remoteSelections[chatID] = RemoteModelSelection(providerID: providerID, modelID: modelID)
        case .needsConfirmation(let host):
            let name = remoteProviders.provider(id: providerID)?.name ?? host
            remoteConfirmation = RemoteConfirmationRequest(
                chatID: chatID, providerID: providerID, modelID: modelID, providerName: name, host: host)
        case .refused(let error):
            self.error = remoteErrorText(error)
        }
    }

    func confirmRemoteSelection() {
        guard let request = remoteConfirmation else { return }
        remoteConfirmation = nil
        remoteProviders.confirmConsent(provider: request.providerID, chat: request.chatID)
        remoteSelections[request.chatID] = RemoteModelSelection(
            providerID: request.providerID, modelID: request.modelID)
    }

    func cancelRemoteConfirmation() { remoteConfirmation = nil }

    /// Back to the local model. The default for every chat.
    func useLocalModel(chatID: UUID) {
        remoteSelections[chatID] = nil
        remoteUsage[chatID] = nil
    }

    // MARK: - Mapping

    /// System and developer rows become the request's system text; the rest
    /// map one to one. Images are not sent: the remote request is text only.
    nonisolated static func remoteMessages(
        from messages: [ChatMessage]
    ) -> (system: String?, messages: [RemoteChatMessage]) {
        var system: [String] = []
        var out: [RemoteChatMessage] = []
        for m in messages {
            switch m.role {
            case .system, .developer:
                if !m.content.isEmpty { system.append(m.content) }
            case .user:
                out.append(RemoteChatMessage(role: .user, content: m.content))
            case .assistant:
                out.append(RemoteChatMessage(
                    role: .assistant, content: m.content,
                    toolCalls: m.toolCalls.map {
                        RemoteToolCall(id: $0.id, name: $0.name,
                                       argumentsJSON: $0.argumentsJSON.isEmpty ? "{}" : $0.argumentsJSON)
                    }))
            case .tool:
                out.append(RemoteChatMessage(role: .tool, content: m.content, toolCallID: m.toolCallId))
            }
        }
        return (system.isEmpty ? nil : system.joined(separator: "\n\n"), out)
    }

    nonisolated static func remoteToolSpecs(from specs: [ToolSpec]) -> [RemoteToolSpec] {
        specs.map {
            RemoteToolSpec(
                name: $0.name, description: $0.description ?? "",
                parametersJSON: $0.parameters?.jsonString ?? #"{"type":"object","properties":{}}"#)
        }
    }

    /// `GenerationResult` has only a decoding initializer, so a remote reply
    /// is built through it. Timings are zero: nothing local was prefilled.
    nonisolated static func makeGenerationResult(
        promptTokens: Int, newTokens: Int, decodeSeconds: Double, finish: RemoteFinishReason,
        content: String, reasoning: String, toolCalls: [RemoteToolCall]
    ) -> GenerationResult? {
        let stop: String
        switch finish {
        case .endOfTurn: stop = "endOfTurn"
        case .toolCalls: stop = "toolCalls"
        case .maxTokens: stop = "maxTokens"
        case .other: stop = "unknown"
        }
        let object: [String: Any] = [
            "promptTokens": promptTokens, "newTokens": newTokens,
            "prefillSeconds": 0.0, "decodeSeconds": decodeSeconds,
            "stopReason": stop, "content": content, "reasoning": reasoning,
            "toolCalls": toolCalls.map { ["id": $0.id, "name": $0.name, "arguments": $0.argumentsJSON] },
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return nil }
        return try? JSONDecoder().decode(GenerationResult.self, from: data)
    }

    // MARK: - Context

    func remoteContextReading(chatID: UUID) -> RemoteContextReading? {
        guard let sel = remoteSelections[chatID],
            let model = remoteProviders.provider(id: sel.providerID)?.models.first(where: { $0.id == sel.modelID })
        else { return nil }
        if let used = remoteUsage[chatID] {
            return RemoteContextReading(usedTokens: used, windowTokens: model.contextWindow, isEstimate: false)
        }
        let rows = turnMessages(for: chatID)
        let chars = rows.reduce(0) { $0 + $1.content.count }
        return RemoteContextReading(
            usedTokens: chars / RemoteContextEstimate.charactersPerToken,
            windowTokens: model.contextWindow, isEstimate: true)
    }

    /// Compaction for a remote chat triggers on the larger of the last
    /// API-reported count and the characters-over-four estimate, against the
    /// model entry's window. Same threshold as the local path. Fail-open.
    func runRemoteAutoCompactionIfNeeded(
        chatID: UUID, project: AppProject?, rawHistory: [ChatMessage], tools: [ToolSpec],
        window: Int, reservedForNew: UInt32
    ) async -> Bool {
        guard autoCompactEnabled, window > 0 else { return false }
        let estimate = RemoteContextEstimate.tokens(messages: rawHistory, tools: tools)
        let measured = max(estimate, remoteUsage[chatID] ?? 0)
        guard AppChatCompaction.shouldAutoCompact(
            measuredTokens: measured, maxContext: UInt32(clamping: window), reservedForNew: reservedForNew)
        else { return false }
        switch await performCompaction(
            chatID: chatID, project: project, focus: nil, trigger: "auto",
            keepRecent: compactionKeepRecentTurns, requestReasoning: nil)
        {
        case .compacted: return true
        case .nothingToCompact, .cancelled: return false
        case .failed(let message):
            showToast(AppStrings.localized("Auto-compaction failed: \(message)"), style: .warning)
            return false
        }
    }

    /// One non-streaming-to-the-user remote call for the compaction summary,
    /// run on the model the chat selected.
    func remoteSummary(chatID: UUID, messages: [ChatMessage], options: GenerateOptions) async throws -> String {
        guard let sel = remoteSelections[chatID] else { return "" }
        let mapped = Self.remoteMessages(from: messages)
        let request = RemoteChatRequest(
            model: sel.modelID, system: mapped.system, messages: mapped.messages,
            maxTokens: Int(options.maxNewTokens), temperature: options.temperature)
        let service = RemoteProviderService(store: remoteProviders, transport: remoteTransport)
        var text = ""
        for try await event in try service.stream(providerID: sel.providerID, chat: chatID, request: request) {
            try Task.checkCancellation()
            if case .content(let chunk) = event { text += chunk }
        }
        return text
    }

    // MARK: - The turn

    /// The remote sibling of `executeGenerationTurn`. Same lane, epoch, cancel
    /// flags and tail as a local turn; only the model call differs. A remote
    /// reply is handed to `processCompletedTurn`, so its tool calls meet the
    /// dispatch gate and the permission ladder exactly as a local call does.
    func executeRemoteGenerationTurn(step: Int, chatID: UUID, selection: RemoteModelSelection) {
        guard let chatIndex = chats.firstIndex(where: { $0.id == chatID }) else { return }
        clearCompactionBoundaryEvent(for: chatID)
        beginRecoveryEventHistoryForGenerationStart(step: step, chatID: chatID)

        let turnChatID = chatID
        let turnProject = self.turnProject(chatID: chatID)
        let provider = remoteProviders.provider(id: selection.providerID)
        let modelEntry = provider?.models.first { $0.id == selection.modelID }
        let window = modelEntry?.contextWindow ?? 0
        let turnAllowsToolCalls = (interactionMode == .projects && turnProject != nil)
            || (interactionMode == .chat && chatToolsEnabled)
        let mediaCapability = AppToolMediaCapability(
            supportsImageBearingToolResults: false, maximumPixelCount: 0)
        let turnAvailableTools = captureTurnTools(
            project: turnProject, mediaCapability: mediaCapability,
            contextTokens: window > 0 ? window : nil)
        // Both families take native tool definitions, so the prompt does not
        // also describe them. An empty offer is a plain chat turn.
        let nativeToolSpecs = turnAllowsToolCalls ? Self.nativeToolSpecs(from: turnAvailableTools) : []
        let nativeLane = !nativeToolSpecs.isEmpty

        generationEpoch += 1
        let myEpoch = generationEpoch
        outputPromptText = turnMessages(for: chatID).last(where: { $0.role == .user })?.content ?? ""
        outputText = ""
        outputReasoningText = ""
        reasoningClock = ReasoningClock(chatID: chatID)
        generating = true
        claimLane(for: chatID)
        phase = .prefill
        livePrefillDone = 0
        livePrefillTotal = 0
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        isCancellationPending = false
        error = nil
        decodeStartTime = nil

        prepareToolOutputProjectionsForPrompt(chatID: chatID)
        let options = samplingOptions(chatID: chatID)
        let providerName = provider?.name ?? ""
        let host = provider.flatMap { URL(string: $0.baseURL)?.host } ?? ""
        _ = chatIndex

        runTask = Task {
            do {
                try Task.checkCancellation()
                @MainActor func history() -> AppChatHistoryProjection? {
                    guard let index = self.chats.firstIndex(where: { $0.id == turnChatID }) else { return nil }
                    return self.buildAppendOnlyHistoryProjection(
                        chatIndex: index, project: turnProject, availableTools: turnAvailableTools,
                        reasoning: nil, mediaCapability: mediaCapability, nativeToolLane: nativeLane)
                }
                guard var projection = history() else { return }
                if await self.runRemoteAutoCompactionIfNeeded(
                    chatID: turnChatID, project: turnProject, rawHistory: projection.messages,
                    tools: nativeToolSpecs, window: window, reservedForNew: options.maxNewTokens)
                {
                    guard let refreshed = history() else { return }
                    projection = refreshed
                }
                let mapped = Self.remoteMessages(from: projection.messages)
                let request = RemoteChatRequest(
                    model: selection.modelID, system: mapped.system, messages: mapped.messages,
                    tools: Self.remoteToolSpecs(from: nativeToolSpecs),
                    maxTokens: Int(options.maxNewTokens), temperature: options.temperature)
                let service = RemoteProviderService(
                    store: self.remoteProviders, transport: self.remoteTransport)
                let stream = try service.stream(
                    providerID: selection.providerID, chat: turnChatID, request: request)

                var content = ""
                var reasoning = ""
                var calls: [RemoteToolCall] = []
                var usage: (prompt: Int?, output: Int?) = (nil, nil)
                var finish: RemoteFinishReason = .endOfTurn
                let started = Date()
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .content(let chunk):
                        if self.phase != .decode { self.phase = .decode; self.decodeStartTime = Date() }
                        content += chunk
                        self.outputText += chunk
                        self.liveTokenCount += 1
                    case .reasoning(let chunk):
                        if self.phase != .decode { self.phase = .decode; self.decodeStartTime = Date() }
                        reasoning += chunk
                        self.outputReasoningText += chunk
                        self.reasoningClock.note(at: Date())
                        self.liveTokenCount += 1
                    case .toolCall(let call):
                        calls.append(call)
                    case .usage(let p, let o):
                        usage = (p, o)
                    case .finished(let reason):
                        finish = reason
                    }
                    if let start = self.decodeStartTime {
                        self.liveElapsedDecodeSeconds = Date().timeIntervalSince(start)
                    }
                }
                try Task.checkCancellation()
                self.phase = .idle
                let estimatedPrompt = RemoteContextEstimate.tokens(
                    messages: projection.messages, tools: nativeToolSpecs)
                let promptTokens = usage.prompt ?? estimatedPrompt
                let newTokens = usage.output ?? max(1, (content.count + reasoning.count) / RemoteContextEstimate.charactersPerToken)
                guard let result = Self.makeGenerationResult(
                    promptTokens: promptTokens, newTokens: newTokens,
                    decodeSeconds: Date().timeIntervalSince(started), finish: finish,
                    content: content, reasoning: reasoning, toolCalls: calls)
                else { throw RemoteProviderError.badStream("could not assemble the reply") }

                if usage.prompt != nil || usage.output != nil {
                    self.remoteUsage[turnChatID] = promptTokens + newTokens
                }
                self.recordUsage(
                    promptTokens: promptTokens, outputTokens: newTokens, chatID: turnChatID,
                    provider: providerName)
                self.logTrajectoryRemoteCall(
                    result: result, step: step, chatID: turnChatID, host: host,
                    model: selection.modelID, providerID: selection.providerID)
                self.recordGoalTurnTokens(chatID: turnChatID, tokens: newTokens)

                await self.processCompletedTurn(
                    completed: CompletedGenerationTurn(
                        result: result, content: content, reasoning: reasoning,
                        continuationsUsed: 0, outputTokensGenerated: newTokens, recoveryEvents: []),
                    nativeToolCalls: calls.map {
                        GenerationToolCall(id: $0.id, name: $0.name, argumentsJSON: $0.argumentsJSON)
                    },
                    step: step, turnChatID: turnChatID, turnProject: turnProject,
                    usesSkillState: false, turnAvailableTools: turnAvailableTools,
                    turnAllowsToolCalls: turnAllowsToolCalls, myEpoch: myEpoch)
            } catch {
                let interruptionReason = AppStreamRecovery.interruptionReason(
                    taskIsCancelled: Task.isCancelled,
                    cancellationPending: self.isCancellationPending)
                if interruptionReason == .error {
                    if let remote = error as? RemoteProviderError {
                        if remote != .cancelled { self.error = self.remoteErrorText(remote) }
                    } else {
                        self.error = RemoteProviderRedactor.redact(error.localizedDescription)
                    }
                }
                self.finishCancelled(chatID: turnChatID, reason: interruptionReason.rawValue)
            }
            self.finishTurnTail(myEpoch: myEpoch, turnChatID: turnChatID, session: self.session)
        }
    }

    // MARK: - Rollout log

    /// One finished remote model call. Only when logging is on and the chat is
    /// eligible (same gate as a local call), with the host marked and every
    /// text field redacted: the exact key first, then the shared patterns the
    /// store applies again on write. The request body is never logged.
    func logTrajectoryRemoteCall(
        result: GenerationResult, step: Int, chatID: UUID, host: String, model: String,
        providerID: UUID
    ) {
        guard trajectoryEligible(chatID: chatID) else { return }
        let keys = remoteProviders.keys.cachedKey(for: providerID).map { [$0] } ?? []
        let entry = RolloutEntry(
            kind: .modelCall, role: "assistant", step: step, model: model,
            promptTokens: result.promptTokens, completionTokens: result.newTokens,
            latencyMs: Int((result.decodeSeconds * 1_000).rounded()),
            finishReason: result.stopReason.rawValue, args: nil,
            result: result.content.isEmpty ? nil : RemoteProviderRedactor.redact(
                result.content, keys: keys, maxCharacters: 64_000),
            remoteHost: host)
        RolloutStore.shared.append(entry, chatID: chatID)
    }

    // MARK: - Errors

    /// The local engine's error row is `AppModel.error` plus an interrupted
    /// row (`finishCancelled(reason: "error")`); a remote failure lands in the
    /// same two places with a localized, key-free message.
    func remoteErrorText(_ error: RemoteProviderError) -> String {
        switch error {
        case .notConfigured:
            return AppStrings.localized("This remote provider needs an API key. Add one in Settings, Network, Remote providers.")
        case .disabled:
            return AppStrings.localized("This remote provider or model is turned off.")
        case .invalidBaseURL:
            return AppStrings.localized("The provider's base URL is not allowed. Use https, or http for localhost only.")
        case .keyUnavailable(let why):
            return AppStrings.localized("The Keychain could not be read: \(RemoteProviderRedactor.redact(why, maxCharacters: 200))")
        case .auth:
            return AppStrings.localized("The provider rejected the API key.")
        case .endpointNotFound:
            return AppStrings.localized("The provider endpoint was not found. Check the base URL.")
        case .rateLimited:
            return AppStrings.localized("The provider is rate limiting requests. Try again shortly.")
        case .overloaded:
            return AppStrings.localized("The provider is overloaded. Try again shortly.")
        case .contextOverflow:
            return AppStrings.localized("This conversation exceeds the model's context window. Compact the chat or start a new one.")
        case .server(let status):
            return AppStrings.localized("The provider returned an error (HTTP \(status)).")
        case .requestRefused(let status, let detail):
            return AppStrings.localized("The provider refused the request (HTTP \(status)): \(detail)")
        case .network(let why):
            return AppStrings.localized("Network error: \(why)")
        case .badStream:
            return AppStrings.localized("The provider's reply was cut off or malformed. Nothing was run.")
        case .redirectRefused:
            return AppStrings.localized("The provider tried to redirect the request, so it was blocked.")
        case .firstByteTimeout:
            return AppStrings.localized("The provider did not respond in time.")
        case .stalled:
            return AppStrings.localized("The reply stalled and was stopped.")
        case .responseTooLarge:
            return AppStrings.localized("The reply was too large and was stopped.")
        case .requestTooLarge:
            return AppStrings.localized("The request is too large to send. Compact the chat.")
        case .cancelled:
            return AppStrings.localized("Cancelled")
        }
    }
}
