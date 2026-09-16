import Foundation
import TurboSpark

extension AppModel {
    /// Runs a completed mutation and its optional terminal validation in
    /// order. The caller has already preflighted both legs through hooks and
    /// permissions before this method may mutate anything.
    func executeApprovedTool(
        _ call: AppToolCall, project: AppProject?, chatID: UUID,
        preMutationFingerprints: [ToolActionFusion.Fingerprint] = [],
        webToolsEnabled: Bool = true
    ) async -> (call: AppToolCall, result: AppToolResult, stopReason: String?) {
        var mutation = call
        var result = await AppToolRegistry.execute(
            call: mutation, in: project, chatID: chatID,
            webToolsEnabled: webToolsEnabled)
        var stopReason = result.continuationStopReason
        mutation.status = result.isError ? .failed : .completed
        let mutationPost = await applyPostToolUse(
            to: result, call: mutation, chatID: chatID, project: project)
        result = mutationPost.result
        if stopReason == nil { stopReason = mutationPost.stopReason }

        guard actionFusionEnabled, let validation = ToolActionFusion.validation(for: mutation) else {
            result = archiveToolObservation(result, chatID: chatID)
            result = await reduceToolOutputIfEligible(result, call: mutation, chatID: chatID)
            return (mutation, result, stopReason)
        }
        guard !result.isError else {
            var outcome = result.efficiency ?? ToolEfficiencyOutcome()
            outcome.validation = "not run because the mutation failed"
            result.efficiency = outcome
            result = archiveToolObservation(result, chatID: chatID)
            return (mutation, result, stopReason)
        }

        let root = project?.rootDirectoryURL ?? URL(fileURLWithPath: "/dev/null")
        // `before` was captured by preflight and then refreshed after the
        // mutation. A successful mutation must leave every declared target
        // in the state its operation requires before the terminal leg runs.
        let after = ToolActionFusion.fingerprints(for: mutation, rootURL: root)
        guard ToolActionFusion.postMutationVerified(
            before: preMutationFingerprints, after: after, for: mutation) else {
            result.isError = true
            result.output += "\n\nValidation was not run: post-mutation target fingerprints did not verify."
            var outcome = result.efficiency ?? ToolEfficiencyOutcome()
            outcome.validation = "blocked by target fingerprint verification"
            result.efficiency = outcome
            result = archiveToolObservation(result, chatID: chatID)
            return (mutation, result, stopReason)
        }

        var validationCall = ToolActionFusion.validationCall(for: validation)
        var validationResult = await AppToolRegistry.execute(
            call: validationCall, in: project, chatID: chatID,
            webToolsEnabled: webToolsEnabled)
        validationCall.status = validationResult.isError ? .failed : .completed
        let validationPost = await applyPostToolUse(
            to: validationResult, call: validationCall, chatID: chatID, project: project)
        validationResult = validationPost.result
        if stopReason == nil { stopReason = validationPost.stopReason }
        result.output += "\n\n<fused_validation>\n\(validationResult.output)\n</fused_validation>"
        var outcome = result.efficiency ?? ToolEfficiencyOutcome()
        outcome.validation = validationResult.isError ? "failed" : "passed"
        result.efficiency = outcome
        result = archiveToolObservation(result, chatID: chatID)
        result = await reduceToolOutputIfEligible(result, call: mutation, chatID: chatID)
        return (mutation, result, stopReason)
    }

    private func applyPostToolUse(
        to initial: AppToolResult, call: AppToolCall, chatID: UUID, project: AppProject?
    ) async -> (result: AppToolResult, stopReason: String?) {
        var result = initial
        let verdict = await dispatchPostToolUseVerdict(
            toolName: call.name, toolArguments: call.arguments, toolOutput: result.output,
            toolDurationSeconds: result.durationSeconds, isError: result.isError,
            chatID: chatID, project: project)
        if let note = verdict.blockReason ?? verdict.feedbackMessage, !note.isEmpty {
            let addition = "\n\n<hook_feedback>\n\(note)\n</hook_feedback>"
            result.output += addition
            result.archivalOutput? += addition
        }
        if let context = verdict.additionalContext, !context.isEmpty {
            let addition = "\n\n<hook_context>\n\(context)\n</hook_context>"
            result.output += addition
            result.archivalOutput? += addition
        }
        return (result, verdict.preventContinuation ? verdict.continuationStopReason : nil)
    }

    /// Archives large normal-chat output and builds the first model-facing
    /// projection. The UI and export continue reading `result.output`.
    func archiveToolObservation(_ result: AppToolResult, chatID: UUID) -> AppToolResult {
        guard observationPackEnabled else { return result }
        let source = result.archivalOutput ?? result.output
        let bytes = Data(source.utf8)
        guard bytes.count >= ToolOutputProjection.archiveThresholdBytes else { return result }

        var result = result
        let reference: ToolObservationRef
        if let index = chats.firstIndex(where: { $0.id == chatID }), chats[index].isGhost {
            reference = ToolObservationRef(
                sourceHash: ToolObservationStore.hash(bytes), byteCount: bytes.count)
            mutateGhostPayload(for: chatID) { payload in
                payload.toolObservations[reference.id] = GhostToolObservation(
                    reference: reference, bytes: bytes)
            }
        } else {
            guard let archived = try? ToolObservationStore.shared.archive(bytes, chatID: chatID) else {
                return result
            }
            reference = archived
        }

        result.observation = reference
        var outcome = result.efficiency ?? ToolEfficiencyOutcome()
        outcome.archivedOutput = true
        result.efficiency = outcome
        if fullObservationCannotFit(reference) {
            result.promptProjection = ToolOutputProjection.headTail(bytes, reference: reference)
            result.fullPromptSendCount = ToolOutputProjection.fullSendLimit
        } else {
            // This is kept in RAM for the first prompt send. Custom encoding
            // intentionally declines to put this potentially huge string in
            // the chat archive.
            result.promptProjection = source
        }
        result.archivalOutput = nil
        return result
    }

    /// Called exactly before history is rendered for a real generation, not
    /// by the context meter. That makes the two-full-sends rule independent
    /// of redraws and estimate refreshes.
    func prepareToolOutputProjectionsForPrompt(chatID: UUID) {
        guard observationPackEnabled else { return }
        mutateTurnMessages(for: chatID) { messages in
            for messageIndex in messages.indices {
                for resultIndex in messages[messageIndex].toolResults.indices {
                    guard let reference = messages[messageIndex].toolResults[resultIndex].observation else { continue }
                    var result = messages[messageIndex].toolResults[resultIndex]
                    guard result.fullPromptSendCount < ToolOutputProjection.fullSendLimit else {
                        if let bytes = self.observationBytes(reference, chatID: chatID) {
                            result.promptProjection = ToolOutputProjection.headTail(bytes, reference: reference)
                        }
                        messages[messageIndex].toolResults[resultIndex] = result
                        continue
                    }
                    guard let bytes = self.observationBytes(reference, chatID: chatID) else { continue }
                    if self.fullObservationCannotFit(reference) {
                        result.promptProjection = ToolOutputProjection.headTail(bytes, reference: reference)
                        result.fullPromptSendCount = ToolOutputProjection.fullSendLimit
                    } else {
                        result.promptProjection = String(decoding: bytes, as: UTF8.self)
                        result.fullPromptSendCount += 1
                    }
                    messages[messageIndex].toolResults[resultIndex] = result
                }
            }
        }
    }

    func recallToolOutput(
        chatID: UUID?, observationID: String, offsetBytes: Int, maxBytes: Int
    ) throws -> String {
        guard let chatID, let id = UUID(uuidString: observationID),
              let reference = observationReference(id: id, chatID: chatID)
        else {
            throw ToolObservationStore.ObservationError.invalidRange
        }
        if let index = chats.firstIndex(where: { $0.id == chatID }), chats[index].isGhost {
            guard let observation = ghostVault.payload(for: chatID).toolObservations[id],
                  observation.reference == reference,
                  offsetBytes >= 0, maxBytes > 0,
                  offsetBytes < observation.bytes.count || (offsetBytes == 0 && observation.bytes.isEmpty)
            else { throw ToolObservationStore.ObservationError.invalidRange }
            let end = min(observation.bytes.count, offsetBytes + maxBytes)
            return String(decoding: observation.bytes.subdata(in: offsetBytes..<end), as: UTF8.self)
        }
        return try ToolObservationStore.shared.recall(
            reference, chatID: chatID, offsetBytes: offsetBytes, maxBytes: maxBytes)
    }

    /// Fails open: malformed receipts, cancellation, unavailable sessions,
    /// and local generation errors leave the archived projection untouched.
    func reduceToolOutputIfEligible(
        _ result: AppToolResult, call: AppToolCall, chatID: UUID
    ) async -> AppToolResult {
        guard evidenceReducerEnabled, call.category == .terminal,
              let reference = result.observation,
              reference.byteCount >= ToolOutputProjection.archiveThresholdBytes,
              let bytes = observationBytes(reference, chatID: chatID), let session
        else { return result }

        let status = result.isError ? "error" : "success"
        let source = String(decoding: bytes, as: UTF8.self)
        let messages = ToolOutputReducer.messages(
            source: source, sourceHash: reference.sourceHash, status: status)
        var options = GenerateOptions()
        options.reasoning = .off
        options.temperature = 0
        options.maxNewTokens = 700
        var raw = ""
        do {
            for try await event in session.generate(messages, options: options) {
                try Task.checkCancellation()
                if case .content(let chunk) = event { raw += chunk }
            }
        } catch {
            return result
        }
        guard let receipt = ToolOutputReducer.validReceipt(
            raw, source: source, sourceHash: reference.sourceHash, status: status)
        else { return result }
        var result = result
        result.promptProjection = ToolOutputReducer.projection(receipt)
        var outcome = result.efficiency ?? ToolEfficiencyOutcome()
        outcome.verifiedReduction = true
        result.efficiency = outcome
        return result
    }

    private func observationBytes(_ reference: ToolObservationRef, chatID: UUID) -> Data? {
        if let index = chats.firstIndex(where: { $0.id == chatID }), chats[index].isGhost {
            let observation = ghostVault.payload(for: chatID).toolObservations[reference.id]
            guard observation?.reference == reference else { return nil }
            return observation?.bytes
        }
        return try? ToolObservationStore.shared.load(reference, chatID: chatID)
    }

    private func observationReference(id: UUID, chatID: UUID) -> ToolObservationRef? {
        for message in turnMessages(for: chatID) {
            if let reference = message.toolResults.first(where: { $0.observation?.id == id })?.observation {
                return reference
            }
        }
        return nil
    }

    private func fullObservationCannotFit(_ reference: ToolObservationRef) -> Bool {
        guard maxContextTokens > 0 else { return false }
        return reference.byteCount / 4 >= max(maxContextTokens / 2, 1)
    }

    func todos(for chatID: UUID) -> [TodoItem] {
        guard let index = chats.firstIndex(where: { $0.id == chatID }) else { return [] }
        return chats[index].isGhost ? ghostVault.payload(for: chatID).todos : chats[index].todos
    }

    func compactAtTodoBoundaryIfNeeded(
        call: AppToolCall, previousTodos: [TodoItem], result: AppToolResult,
        chatID: UUID, project: AppProject?
    ) async -> AppToolResult {
        guard todoBoundaryCompactionEnabled,
              ["todowrite", "todo_write"].contains(call.name.lowercased())
        else { return result }
        let current = todos(for: chatID)
        let state = compactionState(chatID: chatID)
        guard TodoBoundaryCompactionPolicy.shouldCompact(
            previous: previousTodos, current: current,
            messageCount: turnMessages(for: chatID).count, boundary: state.boundary,
            keepRecent: compactionKeepRecentTurns)
        else { return result }
        switch await performCompaction(
            chatID: chatID, project: project, focus: nil, trigger: "todo_completed",
            keepRecent: compactionKeepRecentTurns)
        {
        case .compacted:
            var updated = result
            var outcome = updated.efficiency ?? ToolEfficiencyOutcome()
            outcome.boundaryCompaction = true
            updated.efficiency = outcome
            return updated
        case .nothingToCompact, .cancelled, .failed:
            return result
        }
    }
}

enum ToolOutputReducer {
    struct Receipt: Codable, Equatable {
        var source_hash: String
        var status: String
        var summary: String
        var quotes: [String]
    }

    static func messages(source: String, sourceHash: String, status: String) -> [ChatMessage] {
        let instruction = """
        Return JSON only, with exactly source_hash, status, summary, and quotes. Do not call tools.
        source_hash must be \(sourceHash). status must be \(status). summary must be at most 4000 UTF-8 bytes.
        quotes must contain at most 12 exact substrings copied from the source, each at most 600 UTF-8 bytes.
        """
        return [
            ChatMessage(role: .system, content: "You reduce terminal diagnostics into verifiable JSON receipts."),
            ChatMessage(role: .user, content: instruction + "\n\nSOURCE:\n" + source),
        ]
    }

    static func validReceipt(_ raw: String, source: String, sourceHash: String, status: String) -> Receipt? {
        guard let data = raw.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let receipt = try? JSONDecoder().decode(Receipt.self, from: data),
              receipt.source_hash == sourceHash,
              receipt.status == status,
              !receipt.summary.isEmpty,
              receipt.summary.utf8.count <= 4_000,
              receipt.quotes.count <= 12,
              receipt.quotes.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 600 && source.contains($0) })
        else { return nil }
        return receipt
    }

    static func projection(_ receipt: Receipt) -> String {
        let quotes = receipt.quotes.map { "> \($0)" }.joined(separator: "\n")
        return "<verified_tool_reduction source_hash=\"\(receipt.source_hash)\" status=\"\(receipt.status)\">\n"
            + receipt.summary + (quotes.isEmpty ? "" : "\n\nExact evidence:\n" + quotes)
            + "\n</verified_tool_reduction>"
    }
}
