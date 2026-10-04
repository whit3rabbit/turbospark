import CryptoKit
import Foundation
import TurboSpark

enum ContinuationOutcome: Equatable, Sendable {
    case proceed(attempt: Int)
    case exhausted(limit: Int)
}

enum RecoveryRetryOutcome: Equatable, Sendable {
    case retry(attempt: Int)
    case conflict
    case exhausted(limit: Int)
}

struct RecoveryRetryPlan: Equatable, Sendable {
    let attempt: Int
    let transcript: [AppChatMessage]
    let alternates: [AppChatMessage]
}

enum RecoveryRetryPreparation: Equatable, Sendable {
    case retry(RecoveryRetryPlan)
    case conflict
    case exhausted(limit: Int)
}

struct ResponseRetryPlan: Equatable, Sendable {
    let transcript: [AppChatMessage]
    let alternates: [AppChatMessage]
    let recoveryAttempt: Int?
    let stagedRecoveryAnchor: RecoveryAnchor?
}

enum ResponseRetryPreparation: Equatable, Sendable {
    case retry(ResponseRetryPlan)
    case conflict
    case exhausted(limit: Int)
    case unavailable
}

enum StreamInterruptionReason: String, Equatable, Sendable {
    case cancelled
    case error
}

enum StreamRecoveryEvent: Equatable, Sendable {
    case continued(attempt: Int, tokensSoFar: Int)
    case partialPreserved(rows: Int)
    case tailDiscarded(description: String)
    case anchorRecorded(RecoveryAnchor)
    case retrySucceeded(attempt: Int)
    case retryConflict
    case retryExhausted
}

struct CompletedGenerationTurn: Sendable {
    let result: GenerationResult
    let content: String
    let reasoning: String
    let continuationsUsed: Int
    let outputTokensGenerated: Int
    let recoveryEvents: [StreamRecoveryEvent]
}

struct InterruptedStreamPreservation: Equatable, Sendable {
    let anchor: RecoveryAnchor
    let events: [StreamRecoveryEvent]
}

enum ContinuationGenerationError: Error, Equatable, LocalizedError {
    case exhausted(limit: Int, outputTokensGenerated: Int)
    case outputBudgetExhausted(totalBudget: Int)
    case continuationContextDoesNotFit
    case missingResult

    var errorDescription: String? {
        switch self {
        case .exhausted(let limit, _):
            return "The reply reached the limit of \(limit) automatic continuations."
        case .outputBudgetExhausted(let totalBudget):
            return "The reply reached its total output budget of \(totalBudget) tokens."
        case .continuationContextDoesNotFit:
            return "The continuation prompt no longer fits in the model context window."
        case .missingResult:
            return "Generation ended without a result."
        }
    }
}

enum AppStreamRecovery {
    static let maxContinuationsPerTurn = RecoveryAnchor.maximumContinuationsPerTurn
    static let maxRecoveryRetries = RecoveryAnchor.maximumRetriesPerInterruptedTurn
    static let incompleteToolCallLabel = ToolCallParser.incompleteToolCallLabel

    static func recoveryEventsAtGenerationStart(
        existing: [StreamRecoveryEvent],
        step: Int,
        isAnchoredRecoveryRetry: Bool
    ) -> [StreamRecoveryEvent] {
        guard step > 0 || isAnchoredRecoveryRetry else { return [] }
        return existing
    }

    static func recordSuccessfulRecoveryRetry(
        attempt: Int,
        into events: inout [StreamRecoveryEvent]
    ) {
        guard (1...maxRecoveryRetries).contains(attempt) else { return }
        events.append(.retrySucceeded(attempt: attempt))
    }

    static func interruptionReason(
        taskIsCancelled: Bool,
        cancellationPending: Bool
    ) -> StreamInterruptionReason {
        taskIsCancelled || cancellationPending ? .cancelled : .error
    }

    static func continuationOutcome(used: Int) -> ContinuationOutcome {
        guard (0..<maxContinuationsPerTurn).contains(used) else {
            return .exhausted(limit: maxContinuationsPerTurn)
        }
        return .proceed(attempt: used + 1)
    }

    static func shouldContinue(_ result: GenerationResult, used: Int) -> Bool {
        guard case .maxTokens = result.stopReason else { return false }
        guard case .proceed = continuationOutcome(used: used) else { return false }
        return true
    }

    /// Runs one request and any bounded max-token continuations as a single
    /// assistant turn. Each continuation receives the base history followed
    /// by the exact assistant text already generated. The configured token
    /// limit is per request; the complete turn is bounded to one initial
    /// request plus the fixed continuation count, and each result's
    /// authoritative `newTokens` reduces that remaining total budget.
    static func generateUntilComplete(
        baseMessages: [ChatMessage],
        options: GenerateOptions,
        enabled: Bool,
        prepareContinuationMessages: (([ChatMessage], UInt32) async throws -> [ChatMessage])? = nil,
        onRecoveryEvent: ((StreamRecoveryEvent) async -> Void)? = nil,
        generate: ([ChatMessage], GenerateOptions) async throws -> GenerationResult
    ) async throws -> CompletedGenerationTurn {
        let configuredRequestLimit = Int(options.maxNewTokens)
        let totalOutputBudget = configuredRequestLimit * (maxContinuationsPerTurn + 1)
        var remainingRequestMessages = baseMessages
        var requestOptions = options
        var continuationsUsed = 0
        var outputTokensGenerated = 0
        var combinedContent = ""
        var combinedReasoning = ""
        var recoveryEvents: [StreamRecoveryEvent] = []

        while true {
            let result = try await generate(remainingRequestMessages, requestOptions)
            combinedContent += result.content
            combinedReasoning += result.reasoning
            let tokenCount = max(result.newTokens, 0)
            let (sum, overflow) = outputTokensGenerated.addingReportingOverflow(tokenCount)
            outputTokensGenerated = overflow ? Int.max : sum

            guard enabled, case .maxTokens = result.stopReason else {
                return CompletedGenerationTurn(
                    result: result,
                    content: combinedContent,
                    reasoning: combinedReasoning,
                    continuationsUsed: continuationsUsed,
                    outputTokensGenerated: outputTokensGenerated,
                    recoveryEvents: recoveryEvents)
            }

            guard case .proceed(let attempt) = continuationOutcome(used: continuationsUsed) else {
                throw ContinuationGenerationError.exhausted(
                    limit: maxContinuationsPerTurn,
                    outputTokensGenerated: outputTokensGenerated)
            }

            let remainingOutputBudget = max(0, totalOutputBudget - min(outputTokensGenerated, totalOutputBudget))
            guard remainingOutputBudget > 0 else {
                throw ContinuationGenerationError.outputBudgetExhausted(
                    totalBudget: totalOutputBudget)
            }

            let nextRequestLimit = min(configuredRequestLimit, remainingOutputBudget)
            guard nextRequestLimit > 0 else {
                throw ContinuationGenerationError.outputBudgetExhausted(
                    totalBudget: totalOutputBudget)
            }

            let continuationMessages = baseMessages + [
                ChatMessage(role: .assistant, content: combinedContent),
            ]
            requestOptions.maxNewTokens = UInt32(nextRequestLimit)
            remainingRequestMessages = try await prepareContinuationMessages?(
                continuationMessages, requestOptions.maxNewTokens) ?? continuationMessages

            let event = StreamRecoveryEvent.continued(
                attempt: attempt,
                tokensSoFar: outputTokensGenerated)
            continuationsUsed = attempt
            recoveryEvents.append(event)
            await onRecoveryEvent?(event)
        }
    }

    /// Appends the one completed assistant row shared by prose generation
    /// and bounded multi-segment turns.
    static func commitAssistantRow(
        into transcript: inout [AppChatMessage],
        content: String,
        reasoning: String,
        result: GenerationResult,
        alternates: [AppChatMessage] = []
    ) {
        var message = AppChatMessage(
            role: .assistant,
            content: content,
            reasoning: reasoning,
            stopReason: result.stopReason.rawValue)
        if !alternates.isEmpty {
            message.alternates = alternates.map { alternate in
                var flat = alternate
                flat.alternates = []
                return flat
            }
        }
        transcript.append(message)
    }

    /// Commits the successful response for an anchored retry and clears its
    /// persisted anchor in the same caller-owned mutation.
    static func commitSuccessfulRecoveryRetry(
        into transcript: inout [AppChatMessage],
        recoveryAnchor: inout RecoveryAnchor?,
        content: String,
        reasoning: String,
        result: GenerationResult,
        alternates: [AppChatMessage]
    ) {
        commitAssistantRow(
            into: &transcript,
            content: content,
            reasoning: reasoning,
            result: result,
            alternates: alternates)
        recoveryAnchor = nil
    }

    /// Saves one interrupted assistant row and anchors the exact appended
    /// transcript content. Incomplete recognized call wrappers stay visible
    /// as text; they are never converted to `AppToolCall` values here.
    @discardableResult
    static func preserveInterruptedOutput(
        into transcript: inout [AppChatMessage],
        content: String,
        reasoning: String,
        stopReason: String,
        continuationsUsed: Int = 0,
        retriesUsed: Int = 0,
        alternates: [AppChatMessage] = [],
        toolCallLabel: String = incompleteToolCallLabel,
        messageID: UUID = UUID(),
        recordedAt: Date = Date()
    ) -> InterruptedStreamPreservation? {
        guard !content.isEmpty || !reasoning.isEmpty || !alternates.isEmpty else { return nil }

        let persistedContent = labelIncompleteToolCallFragments(in: content, label: toolCallLabel)
        let retainedMessageCount = transcript.count
        var message = AppChatMessage(
            id: messageID,
            role: .assistant,
            content: persistedContent,
            reasoning: reasoning,
            stopReason: stopReason)
        if !alternates.isEmpty {
            message.alternates = alternates.map { alternate in
                var flat = alternate
                flat.alternates = []
                return flat
            }
        }
        transcript.append(message)

        guard let anchor = anchor(
            retainedMessageCount: retainedMessageCount,
            messageID: message.id,
            persistedRowContent: message.content,
            continuationsUsed: min(max(continuationsUsed, 0), maxContinuationsPerTurn),
            retriesUsed: min(max(retriesUsed, 0), maxRecoveryRetries),
            recordedAt: recordedAt)
        else {
            return nil
        }
        return InterruptedStreamPreservation(
            anchor: anchor,
            events: [.partialPreserved(rows: 1), .anchorRecorded(anchor)])
    }

    /// Labels recognized wrappers that are unclosed or not valid complete
    /// calls. The wrapper and surrounding transcript bytes stay in place;
    /// only the visible inert label is inserted before each invalid wrapper.
    static func labelIncompleteToolCallFragments(
        in content: String,
        label: String = incompleteToolCallLabel
    ) -> String {
        let source = content as NSString
        guard source.length > 0,
            let xmlOpen = try? NSRegularExpression(pattern: "<tool_call>", options: []),
            let markdownOpen = try? NSRegularExpression(
                pattern: "```(?:tool_call|json_tool_call)\\s*\\n", options: [])
        else { return content }

        var cursor = 0
        var labeledContent = ""
        while cursor < source.length {
            let searchRange = NSRange(location: cursor, length: source.length - cursor)
            let xmlMatch = xmlOpen.firstMatch(in: content, options: [], range: searchRange)
            let markdownMatch = markdownOpen.firstMatch(in: content, options: [], range: searchRange)
            guard let next = [xmlMatch, markdownMatch].compactMap({ $0 })
                .min(by: { $0.range.location < $1.range.location })
            else {
                labeledContent += source.substring(from: cursor)
                return labeledContent
            }

            labeledContent += source.substring(
                with: NSRange(location: cursor, length: next.range.location - cursor))

            let isXML = xmlMatch?.range.location == next.range.location
            let closeMarker = isXML ? "</tool_call>" : "\n```"
            let openEnd = NSMaxRange(next.range)
            let closeSearchRange = NSRange(location: openEnd, length: source.length - openEnd)
            let closeRange = isXML
                ? matchingXMLToolCallClose(in: source, from: openEnd)
                    ?? NSRange(location: NSNotFound, length: 0)
                : source.range(of: closeMarker, options: [], range: closeSearchRange)
            let hasCloseMarker = closeRange.location != NSNotFound
            let fragmentEnd = hasCloseMarker ? NSMaxRange(closeRange) : source.length
            let fragmentRange = NSRange(
                location: next.range.location,
                length: fragmentEnd - next.range.location)
            let fragment = source.substring(with: fragmentRange)
            let fragmentLength = (fragment as NSString).length
            let isValidCompletedCall = hasCloseMarker
                && ToolCallParser.parseCandidates(from: fragment).contains(where: {
                    $0.call != nil && $0.refusal == nil
                        && $0.sourceRange == NSRange(location: 0, length: fragmentLength)
                })

            if isValidCompletedCall {
                labeledContent += fragment
            } else {
                if !ToolCallParser.isPrefixedByIncompleteToolCallLabel(next.range, in: content) {
                    let leadingNewline = next.range.location > 0
                        && source.character(at: next.range.location - 1) != 0x0A
                        ? "\n"
                        : ""
                    labeledContent += leadingNewline + label + "\n"
                }
                labeledContent += neutralizedToolCallFragment(fragment)
            }

            guard hasCloseMarker else { return labeledContent }
            cursor = fragmentEnd
        }
        return labeledContent
    }

    /// Finds the matching XML wrapper close without treating markup-shaped
    /// text inside valid JSON arguments as an XML delimiter. If an arguments
    /// boundary is ambiguous, the caller keeps the remainder inert.
    private static func matchingXMLToolCallClose(in source: NSString, from start: Int) -> NSRange? {
        var cursor = start
        var depth = 1
        var argumentsStart: Int?

        while cursor < source.length {
            if let payloadStart = argumentsStart {
                guard let argumentsEnd = matchingArgumentsClose(in: source, from: payloadStart) else {
                    return nil
                }
                argumentsStart = nil
                cursor = NSMaxRange(argumentsEnd)
                continue
            }

            let searchRange = NSRange(location: cursor, length: source.length - cursor)
            let tokens = ["<arguments>", "<tool_call>", "</tool_call>"]
                .compactMap { token -> (String, NSRange)? in
                    let range = source.range(of: token, options: [], range: searchRange)
                    return range.location == NSNotFound ? nil : (token, range)
                }
            guard let next = tokens.min(by: { $0.1.location < $1.1.location }) else {
                return nil
            }

            switch next.0 {
            case "<arguments>":
                argumentsStart = NSMaxRange(next.1)
                cursor = NSMaxRange(next.1)
            case "<tool_call>":
                depth += 1
                cursor = NSMaxRange(next.1)
            default:
                depth -= 1
                if depth == 0 { return next.1 }
                cursor = NSMaxRange(next.1)
            }
        }
        return nil
    }

    /// Returns a plausible `</arguments>` boundary. A valid JSON object is
    /// authoritative. For malformed JSON, only accept a boundary outside a
    /// string and without nested recognized call markup; otherwise fail closed.
    private static func matchingArgumentsClose(in source: NSString, from start: Int) -> NSRange? {
        let marker = "</arguments>"
        let tokenCallMarkers = [
            "<tool_call>",
            "</tool_call>",
            "```tool_call",
            "```json_tool_call",
        ]
        var cursor = start

        while cursor < source.length {
            let searchRange = NSRange(location: cursor, length: source.length - cursor)
            let candidate = source.range(of: marker, options: [], range: searchRange)
            guard candidate.location != NSNotFound else { return nil }

            let payload = source.substring(
                with: NSRange(location: start, length: candidate.location - start))
            if isJSONObject(payload) { return candidate }

            guard !isInsideJSONString(payload),
                !tokenCallMarkers.contains(where: payload.contains)
            else {
                cursor = NSMaxRange(candidate)
                continue
            }
            return candidate
        }
        return nil
    }

    private static func isJSONObject(_ value: String) -> Bool {
        guard let data = value.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data)
        else { return false }
        return object is [String: Any]
    }

    private static func isInsideJSONString(_ value: String) -> Bool {
        let source = value as NSString
        var insideString = false
        var escaped = false
        for index in 0..<source.length {
            let character = source.character(at: index)
            if escaped {
                escaped = false
            } else if insideString, character == 0x5C {
                escaped = true
            } else if character == 0x22 {
                insideString.toggle()
            }
        }
        return insideString || escaped
    }

    /// Persist malformed or interrupted wrappers as visible text with every
    /// current dispatch dialect neutralized, including a different wrapper
    /// nested inside the outer fragment.
    private static func neutralizedToolCallFragment(_ fragment: String) -> String {
        let mutable = NSMutableString(string: fragment)
        let literalMarkers = [
            ("<tool_call>", "&lt;tool_call&gt;"),
            ("<function=", "&lt;function="),
            ("<function_call", "&lt;function_call"),
            ("<invoke", "&lt;invoke"),
            ("<arg_key>", "&lt;arg_key&gt;"),
            ("</arg_key>", "&lt;/arg_key&gt;"),
            ("<arg_value>", "&lt;arg_value&gt;"),
            ("</arg_value>", "&lt;/arg_value&gt;"),
            ("<|tool_call_begin|>", "&lt;|tool_call_begin|&gt;"),
            ("<|tool_call_argument_begin|>", "&lt;|tool_call_argument_begin|&gt;"),
            ("<|tool_call_end|>", "&lt;|tool_call_end|&gt;"),
            ("<|tool_call>", "&lt;|tool_call&gt;"),
            ("<tool_call|>", "&lt;tool_call|&gt;"),
            ("<longcat_tool_call>", "&lt;longcat_tool_call&gt;"),
            ("[TOOL_CALLS]", "&#91;TOOL_CALLS&#93;"),
            ("```", "&#96;&#96;&#96;"),
        ]
        for (marker, replacement) in literalMarkers {
            var searchLocation = 0
            while searchLocation < mutable.length {
                let searchRange = NSRange(
                    location: searchLocation,
                    length: mutable.length - searchLocation)
                let range = mutable.range(of: marker, options: [], range: searchRange)
                guard range.location != NSNotFound else { break }
                mutable.replaceCharacters(in: range, with: replacement)
                searchLocation = range.location + (replacement as NSString).length
            }
        }

        let rescuePatterns = [("\\bcall:(?=[a-zA-Z0-9_\\-]+\\s*\\{)", "call&#58;")]
        for (pattern, replacement) in rescuePatterns {
            guard let rescueRegex = try? NSRegularExpression(pattern: pattern, options: []) else {
                continue
            }
            let matches = rescueRegex.matches(
                in: mutable as String,
                options: [],
                range: NSRange(location: 0, length: mutable.length))
            for match in matches.reversed() {
                mutable.replaceCharacters(in: match.range, with: replacement)
            }
        }
        return mutable as String
    }

    static func anchor(
        retainedMessageCount: Int,
        messageID: UUID,
        persistedRowContent: String,
        continuationsUsed: Int,
        retriesUsed: Int,
        recordedAt: Date = Date()
    ) -> RecoveryAnchor? {
        RecoveryAnchor(
            retainedMessageCount: retainedMessageCount,
            interruptedMessageID: messageID,
            interruptedContentHash: contentHash(forPersistedRowContent: persistedRowContent),
            continuationsUsed: continuationsUsed,
            retriesUsed: retriesUsed,
            recordedAt: recordedAt)
    }

    static func contentHash(forPersistedRowContent content: String) -> String {
        SHA256.hash(data: Data(content.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func validateAnchor(_ anchor: RecoveryAnchor, messages: [AppChatMessage]) -> Bool {
        guard anchor.retainedMessageCount >= 0,
            anchor.retainedMessageCount < messages.count,
            anchor.isValid(forMessageCount: messages.count)
        else { return false }

        let row = messages[anchor.retainedMessageCount]
        guard case .assistant = row.role else { return false }
        return row.id == anchor.interruptedMessageID
            && contentHash(forPersistedRowContent: row.content) == anchor.interruptedContentHash
    }

    static func retryOutcome(
        anchor: RecoveryAnchor,
        messages: [AppChatMessage]
    ) -> RecoveryRetryOutcome {
        guard validateAnchor(anchor, messages: messages) else { return .conflict }
        guard (0..<maxRecoveryRetries).contains(anchor.retriesUsed) else {
            return .exhausted(limit: maxRecoveryRetries)
        }
        return .retry(attempt: anchor.retriesUsed + 1)
    }

    static func lastPromptAnchorIndex(in messages: [AppChatMessage]) -> Int? {
        messages.lastIndex {
            $0.role == .user
                && $0.toolResults.isEmpty
                && UserMemoryInputMessage.parse($0.content) == nil
        }
    }

    /// Prepares a recovery retry without changing the caller's transcript.
    /// The anchored partial must still be the sole plain-prose response to
    /// the last prompt and must be the final row; later transcript activity
    /// therefore fails closed instead of being truncated by Retry.
    static func prepareAnchoredRetry(
        anchor: RecoveryAnchor,
        messages: [AppChatMessage]
    ) -> RecoveryRetryPreparation {
        var retryMessages = messages
        if !validateAnchor(anchor, messages: retryMessages) {
            guard anchor.retainedMessageCount == retryMessages.count,
                let interrupted = anchor.interruptedMessage,
                anchor.matchesInterruptedMessage(interrupted),
                !retryMessages.contains(where: { $0.id == interrupted.id })
            else { return .conflict }
            retryMessages.append(interrupted)
            guard validateAnchor(anchor, messages: retryMessages) else { return .conflict }
        }

        guard anchor.retainedMessageCount == retryMessages.count - 1,
            let promptIndex = lastPromptAnchorIndex(in: retryMessages),
            promptIndex + 1 == anchor.retainedMessageCount
        else { return .conflict }

        var interrupted = retryMessages[anchor.retainedMessageCount]
        guard isSingleProseResponse([interrupted]) else { return .conflict }
        guard (0..<maxRecoveryRetries).contains(anchor.retriesUsed) else {
            return .exhausted(limit: maxRecoveryRetries)
        }
        interrupted.alternates = []
        return .retry(RecoveryRetryPlan(
            attempt: anchor.retriesUsed + 1,
            transcript: Array(messages[..<anchor.retainedMessageCount]),
            alternates: [interrupted]))
    }

    static func isSingleProseResponse(_ chain: [AppChatMessage]) -> Bool {
        chain.count == 1 && chain[0].role == .assistant
            && chain[0].toolCalls.isEmpty && chain[0].toolResults.isEmpty
    }

    static func isInterruptedResponse(_ message: AppChatMessage) -> Bool {
        message.role == .assistant
            && (message.stopReason == StreamInterruptionReason.cancelled.rawValue
                || message.stopReason == StreamInterruptionReason.error.rawValue)
    }

    /// Chooses anchored recovery for interrupted output and the ordinary
    /// retry path for completed prose. An interrupted row without an anchor
    /// fails closed instead of falling through to the ordinary retry.
    static func prepareResponseRetry(
        recoveryAnchor: RecoveryAnchor?,
        messages: [AppChatMessage]
    ) -> ResponseRetryPreparation {
        // An anchor governs only the interrupted prompt. A later user prompt
        // starts a new retry budget and must still permit ordinary Retry.
        if let recoveryAnchor,
            lastPromptAnchorIndex(in: messages).map({ $0 < recoveryAnchor.retainedMessageCount }) ?? true
        {
            switch prepareAnchoredRetry(anchor: recoveryAnchor, messages: messages) {
            case .retry(let plan):
                guard let interrupted = plan.alternates.first,
                    let stagedAnchor = recoveryAnchor.recordingRetryAttempt(
                        plan.attempt,
                        interruptedMessage: interrupted)
                else { return .conflict }
                return .retry(ResponseRetryPlan(
                    transcript: plan.transcript,
                    alternates: plan.alternates,
                    recoveryAttempt: plan.attempt,
                    stagedRecoveryAnchor: stagedAnchor))
            case .conflict:
                return .conflict
            case .exhausted(let limit):
                return .exhausted(limit: limit)
            }
        }

        guard let promptIndex = lastPromptAnchorIndex(in: messages),
            promptIndex + 1 < messages.count,
            isSingleProseResponse(Array(messages[(promptIndex + 1)...]))
        else { return .unavailable }

        var response = messages[promptIndex + 1]
        guard !isInterruptedResponse(response) else { return .conflict }
        response.alternates = []
        return .retry(ResponseRetryPlan(
            transcript: Array(messages[...promptIndex]),
            alternates: [response],
            recoveryAttempt: nil,
            stagedRecoveryAnchor: nil))
    }
}
