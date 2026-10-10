import Foundation
import TurboSpark

/// Offering tools through the engine instead of describing them in the prompt.
///
/// **WHAT CHANGES, AND WHAT DELIBERATELY DOES NOT.** The app used to teach the
/// model its own `<tool_call>` text format in the system prompt (a list of tool
/// names and descriptions, no parameter schemas), read calls back out of the
/// reply with `ToolCallParser`, and replay them as text. In the NATIVE lane the
/// checkpoint's own chat template renders full JSON-schema definitions, the
/// engine parses the checkpoint's own call markup into structured calls, and
/// history replays a call as an assistant `toolCalls` entry plus an id-matched
/// `tool` message. What does not change is everything after a call exists:
/// the same schema validation (`TurnAvailableTools.validate`), permission
/// engine, hooks, risk classification, approval cards and executors. A native
/// call becomes an `AppToolCall` and joins that path at the same point a text
/// call does.
///
/// The lane is per TURN and per CHECKPOINT: it needs the engine to say this
/// install frames tool calls natively (`info.toolCalling.native`), so a
/// checkpoint without that stays on the text lane with its guardrails. A reply
/// that carries no native call still goes through the text gate, which covers a
/// model that answers in the app's own format anyway. Subagents
/// (`SubagentRunner`) have their own loop and stay on the text lane.
extension AppModel {
    /// Whether a turn in `project` offers its tools natively.
    ///
    /// Off for the bounded skill-state prompt, which assembles its own tool
    /// text, and off outside Projects mode where no tools are offered at all.
    func nativeToolLane(project: AppProject?) -> Bool {
        guard nativeToolCallingEnabled,
            let session, session.info.toolCalling.native,
            interactionMode == .projects,
            let project, !project.skillStateEnabled
        else { return false }
        return true
    }

    /// The tools a turn in `project` is offered, captured once.
    ///
    /// Shared by the generation turn and the composer's token estimate, which
    /// used to differ in what they passed (the estimate passed no snapshot and
    /// let the prompt builder re-derive its own); one capture means the meter
    /// prices the tools the turn will actually send.
    func captureTurnTools(project: AppProject?, session: TurboSparkSession) -> TurnAvailableTools {
        let mediaCapability = AppToolMediaCapability(
            supportsImageBearingToolResults: session.info.vision.active,
            maximumPixelCount: session.info.vision.maxPixels)
        return AppToolCatalog.captureTurnAvailableTools(
            for: project,
            globalMcpServers: globalMcpServers,
            contextTokens: maxContextTokens > 0 ? maxContextTokens : nil,
            webToolsEnabled: webSearchEnabled,
            browserAvailability: browserToolAvailability(mediaCapability: mediaCapability))
    }

    /// The functions to offer: the ones the text prompt would have LISTED
    /// (`promptDefinitions`). Deferred MCP tools stay deferred, exactly as in
    /// the text lane's progressive-disclosure contract, and the engine only
    /// reports a call to a function offered here.
    ///
    /// A schema that cannot be re-expressed as JSON is dropped rather than
    /// offered half-described.
    static func nativeToolSpecs(from tools: TurnAvailableTools?) -> [ToolSpec] {
        guard let tools else { return [] }
        let encoder = JSONEncoder()
        return tools.promptDefinitions.compactMap { tool in
            guard !tool.function.name.isEmpty,
                let data = try? encoder.encode(tool.function.parameters),
                let schema = try? JSONDecoder().decode(JSONValue.self, from: data)
            else { return nil }
            return ToolSpec(
                name: tool.function.name,
                description: tool.function.description,
                parameters: schema)
        }
    }

    // MARK: - Budget

    /// The share of the context window the offered tool definitions may take.
    ///
    /// **THE NATIVE LANE IS NOT FREE.** The template renders every offered
    /// tool's full JSON schema; the app's 40-tool set measured about 6,100
    /// tokens against the 650 of the same prompt without it. The text lane
    /// listed names and descriptions only and cost far less. On a large window
    /// that is a fair price for schemas the model can actually follow; on a
    /// small one it would leave no room for the conversation.
    static let nativeToolWindowFraction = 0.4

    /// The longest the definitions may add to a turn's FIRST prefill.
    ///
    /// **SPACE IS NOT THE ONLY COST, AND TIME CAN BE THE WORSE ONE.** On
    /// Bonsai 2 the 6,754-token native prompt took 1,207 s to prefill (about
    /// 5.6 tokens/s) while the same conversation's second turn, with the
    /// prefix cached, took 7.8 s. A checkpoint that slow would make the first
    /// agent turn a twenty-minute wait for definitions. Later turns reuse the
    /// prefix, so this bounds the one-time cost.
    static let nativeToolPrefillBudgetSeconds = 60.0

    /// Why the native lane is or is not used for a set of definitions.
    enum NativeToolBudget: Equatable {
        case allowed
        /// The definitions would take too much of the window.
        case tooLargeForWindow
        /// Prefilling them would take longer than `nativeToolPrefillBudgetSeconds`.
        case tooSlowToPrefill
        /// No prefill speed has been observed for this checkpoint yet, so the
        /// cost in time is unknown. Decided as a no: the text lane works, and
        /// guessing wrong the other way costs minutes.
        case prefillSpeedUnknown
    }

    /// Whether `toolTokens` of definitions fit the share of a window above.
    static func nativeToolsFit(toolTokens: Int, maxContext: UInt32) -> Bool {
        Double(toolTokens) <= Double(maxContext) * nativeToolWindowFraction
    }

    /// The pure decision, so each reason is testable without an engine.
    ///
    /// - Parameter ignoresPrefillSpeed: the user's explicit choice to accept a
    ///   slow first turn (`nativeToolCallingIgnoresSpeed`). It skips only the
    ///   time rule; the window rule always applies, because no amount of
    ///   patience makes definitions that fill the window fit beside the
    ///   conversation.
    static func nativeToolBudget(
        toolTokens: Int, maxContext: UInt32, prefillTokensPerSecond: Double?,
        ignoresPrefillSpeed: Bool = false
    ) -> NativeToolBudget {
        guard nativeToolsFit(toolTokens: toolTokens, maxContext: maxContext) else {
            return .tooLargeForWindow
        }
        if toolTokens == 0 || ignoresPrefillSpeed { return .allowed }
        guard let rate = prefillTokensPerSecond, rate > 0 else { return .prefillSpeedUnknown }
        return Double(toolTokens) / rate <= nativeToolPrefillBudgetSeconds
            ? .allowed : .tooSlowToPrefill
    }

    /// What `specs` add to a prompt, measured by the engine (the checkpoint's
    /// own template decides this, so an estimate from the JSON would be a
    /// guess) and cached for the session. Nil if it cannot be measured.
    func nativeToolTokenCost(_ specs: [ToolSpec], session: TurboSparkSession) async -> Int? {
        guard !specs.isEmpty, let encoded = try? JSONEncoder().encode(specs) else { return 0 }
        var hasher = Hasher()
        hasher.combine(ObjectIdentifier(session))
        hasher.combine(encoded)
        let key = hasher.finalize()
        if let known = nativeToolCostCache[key] { return known }
        let probe = [ChatMessage.user(".")]
        guard let bare = try? await session.countTokens(probe),
            let priced = try? await session.countTokens(probe, tools: specs)
        else { return nil }
        let cost = max(0, priced - bare)
        nativeToolCostCache[key] = cost
        return cost
    }

    /// The budget verdict for `specs` on this session: the definitions' measured
    /// cost against the window, and against this checkpoint's observed prefill
    /// speed. An unmeasurable cost is a no, the safe direction.
    func nativeToolBudget(_ specs: [ToolSpec], session: TurboSparkSession) async -> NativeToolBudget {
        guard let cost = await nativeToolTokenCost(specs, session: session) else {
            return .prefillSpeedUnknown
        }
        return Self.nativeToolBudget(
            toolTokens: cost, maxContext: session.info.maxContext,
            prefillTokensPerSecond: prefillRates.rate(for: session.info.modelPath),
            ignoresPrefillSpeed: nativeToolCallingIgnoresSpeed)
    }

    func nativeToolBudgetAllows(_ specs: [ToolSpec], session: TurboSparkSession) async -> Bool {
        await nativeToolBudget(specs, session: session) == .allowed
    }

    /// Remembers how fast this checkpoint prefilled on the turn that just
    /// finished, for the next turn's budget decision (and the next launch's).
    ///
    /// Only tokens the engine actually prefilled count: a turn that reused its
    /// cached prefix prefilled almost nothing, and dividing the whole prompt by
    /// that time would report an absurdly fast checkpoint. Tiny prefills are
    /// ignored, since their time is mostly fixed overhead.
    func recordPrefillRate(from result: GenerationResult, session: TurboSparkSession) {
        guard let rate = PrefillRateStore.observedRate(
            promptTokens: result.promptTokens, reusedTokens: result.reusedPrefixTokens,
            prefillSeconds: result.prefillSeconds)
        else { return }
        prefillRates.record(rate, for: session.info.modelPath)
    }

    // MARK: - Replay

    /// True when every call on the row came from native tool calling (and
    /// there is at least one). A row mixing the two never occurs: a batch is
    /// one reply, parsed by one lane.
    static func isNativeToolRow(_ message: AppChatMessage) -> Bool {
        !message.toolCalls.isEmpty && message.toolCalls.allSatisfy { $0.nativeCallID != nil }
    }

    /// The engine-side form of a native row's calls, for the assistant message.
    static func engineToolCalls(for message: AppChatMessage) -> [GenerationToolCall] {
        message.toolCalls.compactMap { call in
            guard let id = call.nativeCallID else { return nil }
            return GenerationToolCall(
                id: id, name: call.name, argumentsJSON: call.nativeArgumentsJSON ?? "{}")
        }
    }

    /// The assistant text to replay for a row.
    ///
    /// A native row's reply text holds prose only (the engine took the call
    /// markup out), so in the TEXT lane it needs its calls written back in the
    /// app's text form, or the model would see tool results for calls it never
    /// appears to have made. That happens when the user turns native calling
    /// off, or switches to a checkpoint that cannot use it, mid-chat. Every
    /// other row returns its content untouched.
    static func replayText(for message: AppChatMessage, nativeLane: Bool) -> String {
        guard !nativeLane, isNativeToolRow(message) else { return message.content }
        let invocations = message.toolCalls
            .map(\.rawInvocation)
            .filter { !$0.isEmpty && !message.content.contains($0) }
        guard !invocations.isEmpty else { return message.content }
        return ([message.content].filter { !$0.isEmpty } + invocations).joined(separator: "\n\n")
    }

    /// The prompt messages one stored row contributes.
    ///
    /// **ONE FUNCTION FOR BOTH BUILDERS.** The prompt (`buildAppendOnlyHistory`)
    /// and the composer's token estimate (`buildEstimateParts`) each used to
    /// carry their own copy of this loop, and the native lane would have been a
    /// third and fourth. The legacy branch below is the loop they shared,
    /// moved without change.
    static func historyMessages(
        for message: AppChatMessage,
        nativeLane: Bool,
        mediaCapability: AppToolMediaCapability
    ) -> [ChatMessage] {
        var out: [ChatMessage] = []
        let nativeRow = nativeLane && isNativeToolRow(message)
        let text = replayText(for: message, nativeLane: nativeLane)
        if !text.isEmpty || !message.imagePaths.isEmpty || nativeRow {
            // An unfinished turn is marked as one, so it is not replayed as a
            // completed assistant turn (state#65).
            let content = truncationNote(for: message).map { "\(text)\n\n\($0)" } ?? text
            out.append(
                ChatMessage(
                    role: message.role,
                    content: content,
                    images: message.imagePaths.map {
                        ChatImage.path(
                            EngineImageTranscoder.enginePath(
                                for: AppStorageRoot.resolveStoredPath($0),
                                storedReference: $0))
                    },
                    toolCalls: nativeRow ? engineToolCalls(for: message) : []))
        }
        // A tool result goes back as `.tool`, never `.system`: several
        // templates refuse a mid-history system message (state#32).
        let callsByID = Dictionary(
            message.toolCalls.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        for result in message.toolResults {
            let media = AppToolMediaHistoryAdapter.project(
                result.mediaReferences,
                capability: mediaCapability,
                materialize: { try ManagedAssetStore.shared.materializedURL(for: $0) })
            var resultText = result.modelOutput
            if media.omittedReferenceCount > 0 {
                resultText += "\n[The image result is unavailable to this model.]"
            }
            if nativeRow, let call = callsByID[result.callID], let id = call.nativeCallID {
                // The template wraps a tool result itself, so it is sent bare;
                // wrapping it again in `<tool_response>` would nest the tags.
                out.append(
                    ChatMessage(
                        role: .tool,
                        content: result.isError ? "Error: \(resultText)" : resultText,
                        images: media.images,
                        toolCallId: id,
                        name: call.name))
            } else {
                let tag = result.isError ? "tool_error" : "tool_response"
                out.append(
                    ChatMessage(
                        role: .tool,
                        content: "<\(tag)>\n\(resultText)\n</\(tag)>",
                        images: media.images))
            }
        }
        return out
    }
}

/// Observed prefill speed per checkpoint, remembered across launches.
///
/// A plain per-user preference, not profile data: it describes a model file on
/// this machine, not anything the user wrote. Smoothed so one unusual turn
/// (a thermal dip, a background job) does not flip the lane.
struct PrefillRateStore {
    /// Prefills shorter than this say more about fixed overhead than speed.
    static let minimumTokens = 256
    static let key = "TurboSpark.observedPrefillTokensPerSecond"
    var defaults: UserDefaults = .standard

    /// The speed a turn demonstrated, or nil if it demonstrated none.
    ///
    /// Only tokens the engine actually prefilled count: a turn that reused its
    /// cached prefix prefilled almost nothing, and dividing the whole prompt by
    /// that time would report an absurdly fast checkpoint. Short or sub-second
    /// prefills are ignored, since their time is mostly fixed overhead.
    static func observedRate(promptTokens: Int, reusedTokens: Int, prefillSeconds: Double) -> Double? {
        let fresh = promptTokens - reusedTokens
        guard fresh >= minimumTokens, prefillSeconds >= 1, prefillSeconds.isFinite else { return nil }
        return Double(fresh) / prefillSeconds
    }

    func rate(for modelPath: String) -> Double? {
        let rates = defaults.dictionary(forKey: Self.key) as? [String: Double]
        return rates?[modelPath].flatMap { $0 > 0 ? $0 : nil }
    }

    func record(_ rate: Double, for modelPath: String) {
        guard rate.isFinite, rate > 0 else { return }
        var rates = (defaults.dictionary(forKey: Self.key) as? [String: Double]) ?? [:]
        // Weighted toward the newest observation, so a change in how the
        // machine is running is followed within a couple of turns.
        rates[modelPath] = rates[modelPath].map { $0 * 0.5 + rate * 0.5 } ?? rate
        defaults.set(rates, forKey: Self.key)
    }
}
