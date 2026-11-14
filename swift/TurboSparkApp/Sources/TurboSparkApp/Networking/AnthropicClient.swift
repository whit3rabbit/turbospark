import Foundation

/// `/v1/messages` with SSE streaming.
struct AnthropicClient: RemoteFamilyClient {
    let family = RemoteAPIFamily.anthropic
    static let apiVersion = "2023-06-01"

    private func headers(key: String) -> [String: String] {
        [
            "x-api-key": key,
            "anthropic-version": Self.apiVersion,
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
        ]
        if let system = request.system, !system.isEmpty { body["system"] = system }
        if let t = request.temperature { body["temperature"] = t }
        if !request.tools.isEmpty {
            body["tools"] = try request.tools.map { spec -> [String: Any] in
                ["name": spec.name, "description": spec.description,
                 "input_schema": try RemoteJSON.toolSchema(spec)]
            }
        }
        body["messages"] = Self.messages(request.messages)
        return RemoteHTTPRequest(
            url: endpoint, method: "POST", headers: headers(key: key), body: try RemoteJSON.data(body))
    }

    /// Consecutive tool results fold into one user turn, as the API requires
    /// strict role alternation.
    static func messages(_ messages: [RemoteChatMessage]) -> [[String: Any]] {
        var out: [[String: Any]] = []
        for m in messages {
            switch m.role {
            case .user:
                out.append(["role": "user", "content": m.content])
            case .assistant:
                var blocks: [[String: Any]] = []
                if !m.content.isEmpty { blocks.append(["type": "text", "text": m.content]) }
                for call in m.toolCalls {
                    blocks.append([
                        "type": "tool_use", "id": call.id, "name": call.name,
                        "input": RemoteJSON.object(call.argumentsJSON) ?? [:],
                    ])
                }
                out.append(["role": "assistant", "content": blocks.isEmpty ? [["type": "text", "text": ""]] : blocks])
            case .tool:
                let block: [String: Any] = [
                    "type": "tool_result", "tool_use_id": m.toolCallID ?? "", "content": m.content,
                ]
                if var last = out.last, last["role"] as? String == "user",
                    var content = last["content"] as? [[String: Any]],
                    content.first?["type"] as? String == "tool_result"
                {
                    content.append(block)
                    last["content"] = content
                    out[out.count - 1] = last
                } else {
                    out.append(["role": "user", "content": [block]])
                }
            }
        }
        return out
    }

    func makeDecoder() -> RemoteStreamDecoder { AnthropicStreamDecoder() }
}

struct AnthropicStreamDecoder: RemoteStreamDecoder {
    private struct Block {
        enum Kind { case text, thinking, tool, other }
        var kind: Kind
        var id = ""
        var name = ""
        var json = ""
    }
    private var blocks: [Int: Block] = [:]
    private var inputTokens: Int?
    private var outputTokens: Int?
    private var stopReason: String?
    private var sawStop = false

    mutating func consume(_ event: SSEEvent) throws -> [RemoteStepEvent] {
        guard let json = RemoteJSON.object(event.data) else {
            throw RemoteProviderError.badStream("event data is not JSON")
        }
        let type = (json["type"] as? String) ?? event.event ?? ""
        var out: [RemoteStepEvent] = []
        switch type {
        case "message_start":
            if let usage = (json["message"] as? [String: Any])?["usage"] as? [String: Any] {
                var total = usage["input_tokens"] as? Int
                for extra in ["cache_read_input_tokens", "cache_creation_input_tokens"] {
                    if let n = usage[extra] as? Int { total = (total ?? 0) + n }
                }
                inputTokens = total
                if let o = usage["output_tokens"] as? Int { outputTokens = o }
            }
        case "content_block_start":
            guard let index = json["index"] as? Int, let block = json["content_block"] as? [String: Any]
            else { throw RemoteProviderError.badStream("content_block_start without index") }
            switch block["type"] as? String {
            case "text":
                blocks[index] = Block(kind: .text)
                if let t = block["text"] as? String, !t.isEmpty { out.append(.content(t)) }
            case "thinking":
                blocks[index] = Block(kind: .thinking)
            case "tool_use":
                blocks[index] = Block(
                    kind: .tool, id: block["id"] as? String ?? "", name: block["name"] as? String ?? "")
            default:
                blocks[index] = Block(kind: .other)
            }
        case "content_block_delta":
            guard let index = json["index"] as? Int, let delta = json["delta"] as? [String: Any]
            else { throw RemoteProviderError.badStream("content_block_delta without index") }
            switch delta["type"] as? String {
            case "text_delta":
                if let t = delta["text"] as? String, !t.isEmpty { out.append(.content(t)) }
            case "thinking_delta":
                if let t = delta["thinking"] as? String, !t.isEmpty { out.append(.reasoning(t)) }
            case "input_json_delta":
                blocks[index]?.json += (delta["partial_json"] as? String) ?? ""
            default:
                break
            }
        case "content_block_stop":
            if let index = json["index"] as? Int, let block = blocks.removeValue(forKey: index),
                block.kind == .tool
            {
                guard !block.name.isEmpty else { throw RemoteProviderError.badStream("tool call without a name") }
                let args = try RemoteJSON.validatedArguments(block.json)
                out.append(.toolCall(RemoteToolCall(id: block.id, name: block.name, argumentsJSON: args)))
            }
        case "message_delta":
            if let delta = json["delta"] as? [String: Any], let r = delta["stop_reason"] as? String {
                stopReason = r
            }
            if let usage = json["usage"] as? [String: Any] {
                if let o = usage["output_tokens"] as? Int { outputTokens = o }
                if let i = usage["input_tokens"] as? Int, inputTokens == nil { inputTokens = i }
            }
        case "message_stop":
            if blocks.values.contains(where: { $0.kind == .tool }) {
                throw RemoteProviderError.badStream("stream ended inside a tool call")
            }
            sawStop = true
            out.append(.usage(promptTokens: inputTokens, outputTokens: outputTokens))
            out.append(.finished(Self.reason(stopReason)))
        case "error":
            let err = json["error"] as? [String: Any]
            throw RemoteErrorClassifier.classifyStreamError(
                type: err?["type"] as? String, message: err?["message"] as? String, key: "")
        default:
            break // ping and any event this version does not know
        }
        return out
    }

    mutating func finish() throws -> [RemoteStepEvent] {
        guard sawStop else { throw RemoteProviderError.badStream("stream ended before message_stop") }
        return []
    }

    private static func reason(_ raw: String?) -> RemoteFinishReason {
        switch raw {
        case "end_turn", "stop_sequence", nil: return .endOfTurn
        case "tool_use": return .toolCalls
        case "max_tokens": return .maxTokens
        default: return .other(raw ?? "")
        }
    }
}
