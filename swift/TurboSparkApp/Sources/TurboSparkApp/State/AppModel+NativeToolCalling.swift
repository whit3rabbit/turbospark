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
                                for: AppStorageRoot.resolveStoredPath($0)))
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
