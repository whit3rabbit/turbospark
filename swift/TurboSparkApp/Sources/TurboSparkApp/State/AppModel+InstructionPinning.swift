import Foundation
import TurboSpark

extension AppModel {
    func cachedInstructionPinningOutcome(
        chatID: UUID, boundary: Int, messages: [AppChatMessage],
        reasoning: GenerateOptions.Reasoning? = nil
    ) -> AppChatInstructionPin.PinningOutcome? {
        guard let entry = instructionPinningCache[chatID] else { return nil }
        let currentRows = AppChatInstructionPin.pinnedUserRows(messages, before: boundary)
        guard AppChatInstructionPin.cacheMatches(
            entry,
            activeSessionIdentity: session.map(ObjectIdentifier.init),
            ceilingTokens: instructionPinTokenCeiling,
            reasoning: reasoning ?? self.reasoning,
            sourceRows: currentRows),
            entry.boundary == boundary
        else {
            instructionPinningCache.removeValue(forKey: chatID)
            return nil
        }

        guard instructionPinningEnabled,
            instructionPinTokenCeiling > 0,
            entry.outcome.estimatedTokens >= 0,
            entry.outcome.estimatedTokens <= instructionPinTokenCeiling
        else { return nil }

        let currentPins = currentRows.map(\.text)
        var cursor = 0
        for group in entry.outcome.pinnedGroups {
            guard cursor < currentPins.count,
                let match = currentPins[cursor...].firstIndex(of: group)
            else { return nil }
            cursor = match + 1
        }
        return entry.outcome
    }

    func refreshInstructionPinningCache(
        chatID: UUID, boundary: Int, messages: [AppChatMessage],
        session: TurboSparkSession, reasoning: GenerateOptions.Reasoning
    ) async {
        guard instructionPinningEnabled, instructionPinTokenCeiling > 0,
            self.session === session
        else {
            instructionPinningCache.removeValue(forKey: chatID)
            return
        }
        let sessionIdentity = ObjectIdentifier(session)
        await rebuildInstructionPinningCache(
            chatID: chatID,
            boundary: boundary,
            messages: messages,
            sessionIdentity: sessionIdentity,
            reasoning: reasoning,
            countTokens: { content in
                let block = ChatMessage(role: .user, content: content)
                return try? await session.countTokens([block], reasoning: reasoning)
            })
        guard self.session === session else {
            if instructionPinningCache[chatID]?.sessionIdentity == sessionIdentity {
                instructionPinningCache.removeValue(forKey: chatID)
            }
            return
        }
    }

    func rebuildInstructionPinningCache(
        chatID: UUID, boundary: Int, messages: [AppChatMessage],
        sessionIdentity: ObjectIdentifier?, reasoning: GenerateOptions.Reasoning,
        countTokens: (String) async -> Int?
    ) async {
        guard instructionPinningEnabled, instructionPinTokenCeiling > 0 else {
            instructionPinningCache.removeValue(forKey: chatID)
            return
        }
        if cachedInstructionPinningOutcome(
            chatID: chatID, boundary: boundary, messages: messages,
            reasoning: reasoning) != nil
        {
            return
        }

        let ceilingTokens = instructionPinTokenCeiling
        let sourceRows = AppChatInstructionPin.pinnedUserRows(messages, before: boundary)
        let outcome = await AppChatInstructionPin.select(
            userRows: sourceRows,
            boundary: boundary,
            ceilingTokens: ceilingTokens,
            countTokens: countTokens)

        guard instructionPinningEnabled,
            instructionPinTokenCeiling == ceilingTokens,
            self.compactionState(chatID: chatID).boundary == boundary,
            AppChatInstructionPin.pinnedUserRows(
                self.turnMessages(for: chatID), before: boundary) == sourceRows
        else { return }

        instructionPinningCache[chatID] = AppChatInstructionPin.CacheEntry(
            boundary: boundary,
            outcome: outcome,
            sessionIdentity: sessionIdentity,
            ceilingTokens: ceilingTokens,
            reasoning: reasoning,
            sourceRows: sourceRows)
    }

    func fitRequestHistoryPreservingInstructionPins(
        _ messages: [ChatMessage], injectedBlockIndex: Int?, chatID: UUID, maxTokens: UInt32,
        session: TurboSparkSession, reasoning: GenerateOptions.Reasoning,
        tools: [ToolSpec] = []
    ) async -> AppChatInstructionPin.FitOutcome {
        let compaction = compactionState(chatID: chatID)
        let outcome = cachedInstructionPinningOutcome(
            chatID: chatID, boundary: compaction.boundary,
            messages: turnMessages(for: chatID), reasoning: reasoning)
        let pinnedBlock = outcome.flatMap { selected in
            selected.pinnedGroups.isEmpty
                ? nil : AppChatInstructionPin.injectionBlock(selected.pinnedGroups)
        }
        return await AppChatInstructionPin.fitPreservingBlock(
            messages,
            injectedBlockIndex: injectedBlockIndex,
            pinnedBlock: pinnedBlock,
            maximumTokens: Int(maxTokens),
            // Both measure the offered tool definitions too (empty on the text
            // lane): they sit in the same prompt as the messages.
            countTokens: { request in
                try? await session.countTokens(request, reasoning: reasoning, tools: tools)
            },
            fitWindow: { request, limit in
                guard let limit = UInt32(exactly: limit),
                    let fitted = try? await session.fitWindow(
                        request, maxTokens: limit, reasoning: reasoning, tools: tools)
                else { return nil }
                return AppChatInstructionPin.FitCandidate(
                    retained: fitted.retained,
                    measuredTokens: fitted.measuredTokens,
                    removedTurnCount: fitted.removedTurnCount,
                    hasRoomForGeneration: fitted.hasRoomForGeneration)
            })
    }

    /// Persists a user message's standing-instruction marker through the
    /// ordinary transcript store or the encrypted ghost-chat vault.
    public func setMessageStandingInstruction(id: UUID, pinned: Bool, chatID: UUID) {
        let messages = turnMessages(for: chatID)
        guard let currentIndex = messages.firstIndex(where: { $0.id == id }) else { return }
        let current = messages[currentIndex]
        guard AppChatInstructionPin.canPin(current),
            current.isStandingInstruction != pinned
        else { return }

        mutateTurnMessages(for: chatID) { messages in
            guard let index = messages.firstIndex(where: { $0.id == id }),
                AppChatInstructionPin.canPin(messages[index]),
                messages[index].isStandingInstruction != pinned
            else { return }
            messages[index].isStandingInstruction = pinned
        }

        if currentIndex < compactionState(chatID: chatID).boundary {
            instructionPinningCache.removeValue(forKey: chatID)
        }
    }
}
