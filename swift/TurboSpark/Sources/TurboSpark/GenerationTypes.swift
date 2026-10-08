import Foundation

/// What a finished turn reports.
public struct GenerationResult: Decodable, Sendable, Equatable {
    /// The condition that terminated text generation.
    public enum StopReason: String, Decodable, Sendable {
        /// Reached model end of turn token.
        case endOfTurn
        /// Generated a tool invocation block.
        case toolCalls
        /// Reached end of sequence.
        case eos
        /// Matched a stop string sequence.
        case stopString
        /// Reached maximum token generation budget.
        case maxTokens
        /// The caller pressed Stop. The partial turn in `content` is valid
        /// and the conversation can continue from it.
        case cancelled
        /// A reason this binding does not know, from a newer engine. Decoded
        /// instead of thrown so one new stop reason cannot discard the
        /// finished turn's content and timings.
        case unknown

        public init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = StopReason(rawValue: raw) ?? .unknown
        }
    }

    /// Number of tokens in the prompt prefix.
    public let promptTokens: Int
    /// Number of new tokens generated.
    public let newTokens: Int
    /// How many of `promptTokens` continued from the previous turn's KV
    /// instead of being re-prefilled. Zero on a session's first turn, on one
    /// where the render diverged anywhere, or on a family this cannot help
    /// (recurrent state, a sliding-window ring past its slack) -- never an
    /// error, just a full prefill that turn. Decoded with a default so a
    /// binding built against an older engine still decodes the rest of the
    /// struct.
    public let reusedPrefixTokens: Int
    /// Time spent in the prompt prefill phase in seconds.
    public let prefillSeconds: Double
    /// Time spent in the token decode phase in seconds.
    public let decodeSeconds: Double
    /// Termination reason for this turn.
    public let stopReason: StopReason
    /// Nil when no decoding happened, so a caller cannot plot a rate that
    /// was never measured.
    public let tokensPerSecond: Double?
    /// The reply. THIS is the assistant turn to append to history.
    public let content: String
    /// The model's thinking. Do NOT append it to history: Harmony's own
    /// convention drops prior-turn analysis and Qwen's template drops
    /// prior-turn `<think>` blocks, so feeding it back sends the model
    /// something it was never trained to read.
    public let reasoning: String
    /// The WORST memory pressure seen while this turn decoded: `normal`,
    /// `warn` or `critical`.
    ///
    /// **`normal` when nothing was watching, which is the default.** The
    /// in-loop probe follows the power profile's stepping, and
    /// `performance` (the default) polls nothing -- so on an ordinary
    /// session this is the ABSENCE of a reading rather than a report that
    /// memory was fine. Read `TurboSparkSession.systemTelemetry` for the
    /// machine's current state; this field exists to catch a SPIKE that
    /// happened between two of those polls.
    ///
    /// Decoded with a default so a binding built against an older engine
    /// still decodes the rest of the struct.
    public let peakMemoryPressure: String
    /// Every tool call the model invoked this turn, in emission order. The
    /// same rows arrive on the stream as `.toolCall` events. Empty unless the
    /// engine parsed a call, and decoded with a default so a binding built
    /// against an older engine still decodes the rest of the struct.
    public let toolCalls: [GenerationToolCall]

    private enum CodingKeys: String, CodingKey {
        case promptTokens
        case newTokens
        case reusedPrefixTokens
        case prefillSeconds
        case decodeSeconds
        case stopReason
        case tokensPerSecond
        case content
        case reasoning
        case peakMemoryPressure
        case toolCalls
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        promptTokens = try c.decode(Int.self, forKey: .promptTokens)
        newTokens = try c.decode(Int.self, forKey: .newTokens)
        reusedPrefixTokens = try c.decodeIfPresent(Int.self, forKey: .reusedPrefixTokens) ?? 0
        prefillSeconds = try c.decode(Double.self, forKey: .prefillSeconds)
        decodeSeconds = try c.decode(Double.self, forKey: .decodeSeconds)
        stopReason = try c.decode(StopReason.self, forKey: .stopReason)
        tokensPerSecond = try c.decodeIfPresent(Double.self, forKey: .tokensPerSecond)
        content = try c.decode(String.self, forKey: .content)
        reasoning = try c.decodeIfPresent(String.self, forKey: .reasoning) ?? ""
        peakMemoryPressure =
            try c.decodeIfPresent(String.self, forKey: .peakMemoryPressure) ?? "normal"
        toolCalls = try c.decodeIfPresent([GenerationToolCall].self, forKey: .toolCalls) ?? []
    }
}

/// One streamed event during generation.
public enum GenerationEvent: Sendable, Equatable {
    /// Prefill progress update with completed and total prompt tokens.
    case prefill(done: Int, total: Int)
    /// Incremental generated assistant text content.
    case content(String)
    /// Incremental reasoning or thinking content.
    case reasoning(String)
    /// A parsed tool call, streamed as the model emits it. Fires only when
    /// the caller offered the tool by name, which no `GenerateOptions`
    /// field does yet: this case is the plumbing for that future surface.
    case toolCall(GenerationToolCall)
    /// The model stopped, arriving on the stream itself just before the
    /// turn's call returns. `stopReason` is spelled exactly as
    /// `GenerationResult.stopReason` spells it; `.finished` still follows
    /// with the full result.
    case stopped(stopReason: String, newTokens: Int, promptTokens: Int)
    /// Generation completion event with final result.
    case finished(GenerationResult)
}

/// One parsed tool call invocation.
public struct GenerationToolCall: Sendable, Equatable {
    /// The call's id, generated by the engine's tool-call parser.
    public let id: String
    /// The function the model invoked.
    public let name: String
    /// The arguments as raw JSON text (usually an object), exactly as the
    /// parser recovered them.
    public let argumentsJSON: String

    /// Parses one `TS_EVENT_TOOL` payload: `{"id","name","arguments"}`.
    init?(parsingJSON json: String) {
        guard
            let data = json.data(using: .utf8),
            let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let name = obj["name"] as? String
        else { return nil }
        self.id = obj["id"] as? String ?? ""
        self.name = name
        if let args = obj["arguments"] {
            // `data(withJSONObject:)` raises an uncatchable ObjC exception for a
            // non-container top level (a double-encoded string or null), which
            // would abort the app from inside the C callback. A string is
            // already the raw text; anything else goes through fragments.
            if let text = args as? String {
                self.argumentsJSON = text
            } else {
                guard
                    let argsData = try? JSONSerialization.data(
                        withJSONObject: args, options: [.fragmentsAllowed]),
                    let argsText = String(data: argsData, encoding: .utf8)
                else { return nil }
                self.argumentsJSON = argsText
            }
        } else {
            self.argumentsJSON = ""
        }
    }
}

extension GenerationToolCall: Codable {
    private enum CodingKeys: String, CodingKey { case id, name, arguments }

    /// Builds a call to replay in history, for example the one a turn just
    /// returned (`ChatMessage.assistant(_:toolCalls:)`).
    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }

    /// Decodes one `toolCalls` row: `{"id","name","arguments"}`. `arguments`
    /// is kept as raw JSON text whatever shape the parser recovered (an
    /// object, or text that was already a JSON string).
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? ""
        name = try c.decode(String.self, forKey: .name)
        if let text = try? c.decode(String.self, forKey: .arguments) {
            argumentsJSON = text
        } else if c.contains(.arguments), !(try c.decodeNil(forKey: .arguments)) {
            argumentsJSON = try c.decode(JSONValue.self, forKey: .arguments).jsonString
        } else {
            argumentsJSON = ""
        }
    }

    /// Encodes `arguments` as the JSON text it is. The engine accepts text
    /// here and parses it back into the object the template renders.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(argumentsJSON, forKey: .arguments)
    }
}

/// The decode phase breakdown. Cumulative over every forward pass this
/// session has served, prefill included.
public struct PhaseReport: Decodable, Sendable, Equatable {
    public let calls: UInt64
    public let totalMsPerCall: Double
    public let gpuWaitMs: Double
    public let finalWaitMs: Double
    public let routerMs: Double
    public let expertIoMs: Double
    public let bindMs: Double
    public let pipelineWaitMs: Double
    public let cb1GpuMs: Double
    public let routedCbGpuMs: Double
    public let finalCbGpuMs: Double
    public let expertRequests: UInt64
    public let expertHits: UInt64
    /// Nil before anything has been requested, rather than a 0% rate on no
    /// data.
    public let expertHitRate: Double?
}
