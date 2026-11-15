import Foundation

/// `/chat/completions` with SSE streaming: the common subset only. A field a
/// provider omits (usage, reasoning) degrades to the labeled estimate; there
/// are no per-vendor quirk profiles (ZP-22 Non-Goals).
struct OpenAICompatClient: RemoteFamilyClient {
    let family = RemoteAPIFamily.openAICompatible

    private func headers(key: String) -> [String: String] {
        [
            "authorization": "Bearer \(key)",
            "content-type": "application/json",
            "accept": "text/event-stream",
        ]
    }

    func buildProbeRequest(endpoint: URL, key: String) -> RemoteHTTPRequest {
        var h = headers(key: key)
        h["accept"] = "application/json"
        h["content-type"] = nil
        return RemoteHTTPRequest(url: endpoint, method: "GET", headers: h, body: Data())
    }

    func buildChatRequest(_ request: RemoteChatRequest, endpoint: URL, key: String) throws -> RemoteHTTPRequest {
        var body: [String: Any] = [
            "model": request.model,
            "max_tokens": request.maxTokens,
            "stream": true,
            "stream_options": ["include_usage": true],
        ]
        if let t = request.temperature { body["temperature"] = t }
        if !request.tools.isEmpty {
            body["tools"] = try request.tools.map { spec -> [String: Any] in
                ["type": "function",
                 "function": ["name": spec.name, "description": spec.description,
                              "parameters": try RemoteJSON.toolSchema(spec)] as [String: Any]]
            }
        }
        var messages: [[String: Any]] = []
        if let system = request.system, !system.isEmpty {
            messages.append(["role": "system", "content": system])
        }
        for m in request.messages {
            switch m.role {
            case .user:
                messages.append(["role": "user", "content": m.content])
            case .assistant:
                var entry: [String: Any] = ["role": "assistant", "content": m.content]
                if !m.toolCalls.isEmpty {
                    entry["tool_calls"] = m.toolCalls.map { call -> [String: Any] in
                        ["id": call.id, "type": "function",
                         "function": ["name": call.name, "arguments": call.argumentsJSON]]
                    }
                }
                messages.append(entry)
            case .tool:
                messages.append(["role": "tool", "tool_call_id": m.toolCallID ?? "", "content": m.content])
            }
        }
        body["messages"] = messages
        return RemoteHTTPRequest(
            url: endpoint, method: "POST", headers: headers(key: key), body: try RemoteJSON.data(body))
    }

    func makeDecoder() -> RemoteStreamDecoder { OpenAICompatStreamDecoder() }
}

struct OpenAICompatStreamDecoder: RemoteStreamDecoder {
    private struct PendingCall {
        var id = ""
        var name = ""
        var arguments = ""
    }
    private var calls: [Int: PendingCall] = [:]
    private var promptTokens: Int?
    private var completionTokens: Int?
    private var finishReason: String?
    private var done = false

    mutating func consume(_ event: SSEEvent) throws -> [RemoteStepEvent] {
        let payload = event.data.trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" { return try finalize() }
        guard let json = RemoteJSON.object(payload) else {
            throw RemoteProviderError.badStream("event data is not JSON")
        }
        if let err = json["error"] as? [String: Any] {
            throw RemoteErrorClassifier.classifyStreamError(
                type: (err["type"] as? String) ?? (err["code"] as? String),
                message: err["message"] as? String, key: "")
        }
        var out: [RemoteStepEvent] = []
        if let usage = json["usage"] as? [String: Any] {
            if let p = usage["prompt_tokens"] as? Int { promptTokens = p }
            if let c = usage["completion_tokens"] as? Int { completionTokens = c }
        }
        guard let choice = (json["choices"] as? [[String: Any]])?.first else { return out }
        if let delta = choice["delta"] as? [String: Any] {
            if let text = delta["content"] as? String, !text.isEmpty { out.append(.content(text)) }
            // Two spellings of the same field in the wild.
            let reasoning = (delta["reasoning_content"] as? String) ?? (delta["reasoning"] as? String)
            if let reasoning, !reasoning.isEmpty { out.append(.reasoning(reasoning)) }
            for call in (delta["tool_calls"] as? [[String: Any]]) ?? [] {
                let index = (call["index"] as? Int) ?? 0
                var pending = calls[index] ?? PendingCall()
                if let id = call["id"] as? String, !id.isEmpty { pending.id = id }
                if let fn = call["function"] as? [String: Any] {
                    if let name = fn["name"] as? String, !name.isEmpty { pending.name = name }
                    pending.arguments += (fn["arguments"] as? String) ?? ""
                }
                calls[index] = pending
            }
        }
        if let reason = choice["finish_reason"] as? String { finishReason = reason }
        return out
    }

    mutating func finish() throws -> [RemoteStepEvent] {
        // A server that never sends [DONE] is accepted when it did send a
        // finish reason; otherwise the stream was cut.
        try finalize()
    }

    private mutating func finalize() throws -> [RemoteStepEvent] {
        if done { return [] }
        guard let reason = finishReason else {
            throw RemoteProviderError.badStream("stream ended before a finish reason")
        }
        done = true
        var out: [RemoteStepEvent] = []
        for index in calls.keys.sorted() {
            let call = calls[index]!
            guard !call.name.isEmpty else { throw RemoteProviderError.badStream("tool call without a name") }
            let args = try RemoteJSON.validatedArguments(call.arguments)
            out.append(.toolCall(RemoteToolCall(
                id: call.id.isEmpty ? "call_\(index)" : call.id, name: call.name, argumentsJSON: args)))
        }
        out.append(.usage(promptTokens: promptTokens, outputTokens: completionTokens))
        let mapped: RemoteFinishReason
        switch reason {
        case "tool_calls", "function_call": mapped = .toolCalls
        case "length": mapped = .maxTokens
        case "stop": mapped = calls.isEmpty ? .endOfTurn : .toolCalls
        default: mapped = .other(reason)
        }
        out.append(.finished(mapped))
        return out
    }
}
