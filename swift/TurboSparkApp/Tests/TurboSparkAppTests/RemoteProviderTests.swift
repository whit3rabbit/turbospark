import Foundation
import Security
import XCTest
@testable import TurboSparkApp
@testable import TurboSpark

// MARK: - Fixtures

/// A scripted transport: no network, ever. Records every request so a test
/// can assert where the key went and that a refused hop was never sent.
final class ScriptedTransport: RemoteHTTPTransport, @unchecked Sendable {
    struct Reply {
        var status = 200
        var headers: [String: String] = [:]
        var chunks: [Data] = []
        /// Thrown after the chunks (a dropped connection).
        var failure: Error?
        /// Never finishes the body (a stalled server).
        var hang = false
        /// Delays the response itself (a server that never answers).
        var delayBeforeResponse: TimeInterval = 0
    }

    private let lock = NSLock()
    private var recorded: [RemoteHTTPRequest] = []
    private let handler: @Sendable (RemoteHTTPRequest, Int) -> Reply

    init(_ handler: @escaping @Sendable (RemoteHTTPRequest, Int) -> Reply) { self.handler = handler }

    convenience init(sse text: String, status: Int = 200) {
        self.init { _, _ in Reply(status: status, chunks: [Data(text.utf8)]) }
    }

    var requests: [RemoteHTTPRequest] {
        lock.lock(); defer { lock.unlock() }
        return recorded
    }

    func send(_ request: RemoteHTTPRequest) async throws -> RemoteHTTPResponse {
        lock.lock()
        recorded.append(request)
        let index = recorded.count - 1
        lock.unlock()
        let reply = handler(request, index)
        if reply.delayBeforeResponse > 0 {
            try await Task.sleep(nanoseconds: UInt64(reply.delayBeforeResponse * 1_000_000_000))
        }
        let body = AsyncThrowingStream<Data, Error> { c in
            for chunk in reply.chunks { c.yield(chunk) }
            if !reply.hang { c.finish(throwing: reply.failure) }
        }
        return RemoteHTTPResponse(status: reply.status, headers: reply.headers, body: body)
    }
}

final class FakeSecretBackend: RemoteSecretBackend, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var items: [String: String] = [:]
    private(set) var readCount = 0
    private(set) var existsCount = 0
    private(set) var writeCount = 0
    var readStatus: OSStatus?

    func exists(account: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        existsCount += 1
        return items[account] != nil
    }
    func read(account: String) -> KeychainReadResult {
        lock.lock(); defer { lock.unlock() }
        readCount += 1
        if let readStatus { return .error(readStatus) }
        return items[account].map { .found($0) } ?? .notFound
    }
    func write(account: String, secret: String) -> OSStatus {
        lock.lock(); defer { lock.unlock() }
        writeCount += 1
        items[account] = secret
        return errSecSuccess
    }
    func delete(account: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        items[account] = nil
        return true
    }
    var touchCount: Int { readCount + existsCount + writeCount }
}

private func collect(_ stream: AsyncThrowingStream<RemoteStepEvent, Error>) async -> (events: [RemoteStepEvent], error: RemoteProviderError?) {
    var events: [RemoteStepEvent] = []
    do {
        for try await e in stream { events.append(e) }
        return (events, nil)
    } catch let error as RemoteProviderError {
        return (events, error)
    } catch {
        return (events, .network("\(error)"))
    }
}

private let secretKey = "sk-ant-api03-TESTKEYTESTKEYTESTKEY1234"
private let base = URL(string: "https://api.example.com")!

private func sse(_ events: [(String?, String)]) -> String {
    events.map { e in (e.0.map { "event: \($0)\n" } ?? "") + "data: \(e.1)\n\n" }.joined()
}

private func chatRequest(tools: [RemoteToolSpec] = []) -> RemoteChatRequest {
    RemoteChatRequest(
        model: "m1", system: "sys",
        messages: [RemoteChatMessage(role: .user, content: "hi")], tools: tools, maxTokens: 64)
}

// MARK: - SSE parser

final class SSEParserTests: XCTestCase {
    func testSplitsAcrossChunksAndHandlesCRLFAndComments() throws {
        var p = SSEParser()
        var out: [SSEEvent] = []
        for part in ["event: a\r\nda", "ta: one\r\n", "data: two\r\n\r\n: comment\n", "data: x\n\n"] {
            out += try p.feed(Data(part.utf8))
        }
        XCTAssertEqual(out, [SSEEvent(event: "a", data: "one\ntwo"), SSEEvent(event: nil, data: "x")])
        XCTAssertFalse(p.hasPartialEvent)
    }

    func testMultibyteScalarSplitAcrossChunksStaysWhole() throws {
        var p = SSEParser()
        let bytes = Array("data: h\u{00E9}llo\n\n".utf8)
        let cut = bytes.firstIndex(of: 0xC3)! + 1
        var out = try p.feed(Data(bytes[..<cut]))
        out += try p.feed(Data(bytes[cut...]))
        XCTAssertEqual(out.first?.data, "h\u{00E9}llo")
    }

    func testPartialEventIsReported() throws {
        var p = SSEParser()
        _ = try p.feed(Data("data: half".utf8))
        XCTAssertTrue(p.hasPartialEvent)
    }

    func testOversizedEventThrows() {
        var p = SSEParser(maxEventBytes: 16)
        XCTAssertThrowsError(try p.feed(Data(String(repeating: "x", count: 40).utf8))) {
            XCTAssertEqual($0 as? SSEParseError, .eventTooLarge)
        }
    }
}

// MARK: - Endpoint policy and redaction

final class RemoteEndpointPolicyTests: XCTestCase {
    func testAcceptsHTTPSAndLoopbackHTTPOnly() throws {
        XCTAssertNoThrow(try RemoteEndpointPolicy.validate(baseURL: "https://api.openai.com/v1"))
        XCTAssertNoThrow(try RemoteEndpointPolicy.validate(baseURL: "http://localhost:8080"))
        XCTAssertNoThrow(try RemoteEndpointPolicy.validate(baseURL: "http://127.0.0.1:1234/v1"))
        XCTAssertNoThrow(try RemoteEndpointPolicy.validate(baseURL: "http://[::1]:1234"))
        for bad in ["http://example.com", "http://192.168.1.5", "ftp://x.com", "https://u:p@x.com",
                    "https://x.com/v1?key=abc", "https://x.com/#f", "", "not a url", "https://"] {
            XCTAssertThrowsError(try RemoteEndpointPolicy.validate(baseURL: bad), bad)
        }
    }

    func testEndpointDoesNotDoubleV1() {
        let a = URL(string: "https://api.anthropic.com")!
        let b = URL(string: "https://api.anthropic.com/v1/")!
        let c = URL(string: "https://x.com/v1")!
        XCTAssertEqual(RemoteEndpointPolicy.endpoint(base: a, family: .anthropic).path, "/v1/messages")
        XCTAssertEqual(RemoteEndpointPolicy.endpoint(base: b, family: .anthropic).path, "/v1/messages")
        XCTAssertEqual(RemoteEndpointPolicy.endpoint(base: c, family: .openAICompatible).path, "/v1/chat/completions")
        XCTAssertEqual(RemoteEndpointPolicy.endpoint(base: c, family: .openAICompatible, probe: true).path, "/v1/models")
    }

    func testRedirectRules() {
        let from = URL(string: "https://api.example.com/v1/x")!
        XCTAssertTrue(RemoteEndpointPolicy.mayFollow(from: from, to: URL(string: "https://api.example.com/v2/x")!))
        XCTAssertFalse(RemoteEndpointPolicy.mayFollow(from: from, to: URL(string: "https://evil.example.net/x")!))
        XCTAssertFalse(RemoteEndpointPolicy.mayFollow(from: from, to: URL(string: "http://api.example.com/x")!))
        XCTAssertFalse(RemoteEndpointPolicy.mayFollow(from: from, to: URL(string: "https://api.example.com:8443/x")!))
    }

    func testRedactorRemovesExactKeyAndKnownShapes() {
        let text = "bad key \(secretKey) and Authorization: Bearer abcdefgh12345678 and sk-abcdefghijklmnop12345"
        let out = RemoteProviderRedactor.redact(text, keys: ["plainsecret"])
        XCTAssertFalse(out.contains(secretKey))
        XCTAssertFalse(out.contains("abcdefgh12345678"))
        XCTAssertFalse(out.contains("sk-abcdefghijklmnop12345"))
        // An exact key of a shape no pattern knows.
        let odd = RemoteProviderRedactor.redact("echo: plainsecret!", keys: ["plainsecret"])
        XCTAssertFalse(odd.contains("plainsecret"))
    }
}

// MARK: - Anthropic

final class AnthropicClientTests: XCTestCase {
    private func run(_ text: String, tools: [RemoteToolSpec] = []) async -> (events: [RemoteStepEvent], error: RemoteProviderError?, transport: ScriptedTransport) {
        let t = ScriptedTransport(sse: text)
        let s = RemoteStreamRunner(transport: t, limits: .standard)
            .stream(client: AnthropicClient(), request: chatRequest(tools: tools), baseURL: base, key: secretKey)
        let r = await collect(s)
        return (r.events, r.error, t)
    }

    func testContentUsageAndFinish() async {
        let text = sse([
            ("message_start", #"{"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":1}}}"#),
            ("ping", #"{"type":"ping"}"#),
            ("future_event", #"{"type":"future_event"}"#),
            ("content_block_start", #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
            ("content_block_delta", #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#),
            ("content_block_delta", #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#),
            ("content_block_stop", #"{"type":"content_block_stop","index":0}"#),
            ("message_delta", #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#),
            ("message_stop", #"{"type":"message_stop"}"#),
        ])
        let r = await run(text)
        XCTAssertNil(r.error)
        XCTAssertEqual(r.events, [
            .content("Hel"), .content("lo"),
            .usage(promptTokens: 11, outputTokens: 7), .finished(.endOfTurn),
        ])
    }

    func testRequestShapeCarriesKeyOnlyInHeader() async throws {
        let r = await run(sse([("message_stop", #"{"type":"message_stop"}"#)]),
                          tools: [RemoteToolSpec(name: "read", description: "d", parametersJSON: #"{"type":"object"}"#)])
        let req = try XCTUnwrap(r.transport.requests.first)
        XCTAssertEqual(req.url.absoluteString, "https://api.example.com/v1/messages")
        XCTAssertEqual(req.headers["x-api-key"], secretKey)
        XCTAssertEqual(req.headers["anthropic-version"], "2023-06-01")
        XCTAssertFalse(req.url.absoluteString.contains(secretKey))
        XCTAssertFalse(String(decoding: req.body, as: UTF8.self).contains(secretKey))
        let body = try XCTUnwrap(RemoteJSON.object(String(decoding: req.body, as: UTF8.self)))
        XCTAssertEqual(body["stream"] as? Bool, true)
        XCTAssertNotNil((body["tools"] as? [[String: Any]])?.first?["input_schema"])
    }

    func testToolCallEmittedOnlyAfterArgumentsParse() async {
        let text = sse([
            ("content_block_start", #"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"read","input":{}}}"#),
            ("content_block_delta", #"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#),
            ("content_block_delta", #"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"a.txt\"}"}}"#),
            ("content_block_stop", #"{"type":"content_block_stop","index":1}"#),
            ("message_delta", #"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}"#),
            ("message_stop", #"{"type":"message_stop"}"#),
        ])
        let r = await run(text)
        XCTAssertNil(r.error)
        XCTAssertEqual(r.events.first, .toolCall(RemoteToolCall(id: "tu_1", name: "read", argumentsJSON: #"{"path":"a.txt"}"#)))
        XCTAssertEqual(r.events.last, .finished(.toolCalls))
    }

    func testTruncatedToolCallRunsNothingAndIsBadStream() async {
        let text = sse([
            ("content_block_start", #"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"read"}}"#),
            ("content_block_delta", #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a"}}"#),
        ])
        let r = await run(text)
        guard case .badStream = r.error else { return XCTFail("expected badStream, got \(String(describing: r.error))") }
        XCTAssertFalse(r.events.contains { if case .toolCall = $0 { return true } else { return false } })
    }

    func testBrokenToolJSONAtBlockStopIsBadStream() async {
        let text = sse([
            ("content_block_start", #"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"x"}}"#),
            ("content_block_delta", #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#),
            ("content_block_stop", #"{"type":"content_block_stop","index":0}"#),
        ])
        let r = await run(text)
        guard case .badStream = r.error else { return XCTFail("\(String(describing: r.error))") }
    }

    func testStreamWithoutMessageStopIsBadStream() async {
        let r = await run(sse([("content_block_delta", #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#)]))
        guard case .badStream = r.error else { return XCTFail("\(String(describing: r.error))") }
    }

    func testInStreamErrorsMapToNamedErrors() async {
        let overloaded = await run(sse([("error", #"{"type":"error","error":{"type":"overloaded_error","message":"x"}}"#)]))
        XCTAssertEqual(overloaded.error, .overloaded)
        let limited = await run(sse([("error", #"{"type":"error","error":{"type":"rate_limit_error","message":"x"}}"#)]))
        XCTAssertEqual(limited.error, .rateLimited)
    }

    func testToolResultsFoldIntoOneUserTurn() {
        let msgs = AnthropicClient.messages([
            RemoteChatMessage(role: .user, content: "go"),
            RemoteChatMessage(role: .assistant, content: "", toolCalls: [
                RemoteToolCall(id: "a", name: "r", argumentsJSON: "{}"), RemoteToolCall(id: "b", name: "r", argumentsJSON: "{}")]),
            RemoteChatMessage(role: .tool, content: "1", toolCallID: "a"),
            RemoteChatMessage(role: .tool, content: "2", toolCallID: "b"),
        ])
        XCTAssertEqual(msgs.count, 3)
        XCTAssertEqual((msgs[2]["content"] as? [[String: Any]])?.count, 2)
    }
}

// MARK: - OpenAI-compatible

final class OpenAICompatClientTests: XCTestCase {
    private func run(_ text: String) async -> (events: [RemoteStepEvent], error: RemoteProviderError?, transport: ScriptedTransport) {
        let t = ScriptedTransport(sse: text)
        let s = RemoteStreamRunner(transport: t, limits: .standard)
            .stream(client: OpenAICompatClient(), request: chatRequest(), baseURL: URL(string: "https://x.example.com/v1")!, key: secretKey)
        let r = await collect(s)
        return (r.events, r.error, t)
    }

    func testContentReasoningUsageDone() async throws {
        let text = sse([
            (nil, #"{"choices":[{"delta":{"reasoning_content":"think"}}]}"#),
            (nil, #"{"choices":[{"delta":{"content":"Hi"}}]}"#),
            (nil, #"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
            (nil, #"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#),
            (nil, "[DONE]"),
        ])
        let r = await run(text)
        XCTAssertNil(r.error)
        XCTAssertEqual(r.events, [.reasoning("think"), .content("Hi"),
                                  .usage(promptTokens: 5, outputTokens: 2), .finished(.endOfTurn)])
        let req = try XCTUnwrap(r.transport.requests.first)
        XCTAssertEqual(req.url.absoluteString, "https://x.example.com/v1/chat/completions")
        XCTAssertEqual(req.headers["authorization"], "Bearer \(secretKey)")
        XCTAssertFalse(String(decoding: req.body, as: UTF8.self).contains(secretKey))
    }

    func testToolCallDeltasAssembleInIndexOrder() async {
        let text = sse([
            (nil, #"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"b","arguments":"{}"}}]}}]}"#),
            (nil, #"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"a","arguments":"{\"p\":"}}]}}]}"#),
            (nil, #"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#),
            (nil, #"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
            (nil, "[DONE]"),
        ])
        let r = await run(text)
        XCTAssertNil(r.error)
        XCTAssertEqual(r.events.first, .toolCall(RemoteToolCall(id: "c1", name: "a", argumentsJSON: #"{"p":1}"#)))
        XCTAssertEqual(r.events[1], .toolCall(RemoteToolCall(id: "c2", name: "b", argumentsJSON: "{}")))
        XCTAssertEqual(r.events.last, .finished(.toolCalls))
    }

    func testTruncatedToolJSONIsBadStreamAndEmitsNoCall() async {
        let text = sse([
            (nil, #"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"a","arguments":"{\"p\":"}}]}}]}"#),
            (nil, #"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
            (nil, "[DONE]"),
        ])
        let r = await run(text)
        guard case .badStream = r.error else { return XCTFail("\(String(describing: r.error))") }
        XCTAssertTrue(r.events.isEmpty)
    }

    func testMissingFinishReasonIsBadStream() async {
        let r = await run(sse([(nil, #"{"choices":[{"delta":{"content":"x"}}]}"#)]))
        guard case .badStream = r.error else { return XCTFail("\(String(describing: r.error))") }
    }

    func testMissingUsageDegradesToNil() async {
        let r = await run(sse([(nil, #"{"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}"#), (nil, "[DONE]")]))
        XCTAssertEqual(r.events.suffix(2), [.usage(promptTokens: nil, outputTokens: nil), .finished(.maxTokens)])
    }
}

// MARK: - Runner: errors, redirects, caps, timeouts

final class RemoteStreamRunnerTests: XCTestCase {
    private func run(
        _ transport: ScriptedTransport, limits: RemoteLimits = .standard,
        client: any RemoteFamilyClient = OpenAICompatClient(), request: RemoteChatRequest? = nil
    ) async -> (events: [RemoteStepEvent], error: RemoteProviderError?) {
        await collect(RemoteStreamRunner(transport: transport, limits: limits)
            .stream(client: client, request: request ?? chatRequest(), baseURL: base, key: secretKey))
    }

    func testHTTPStatusesMapToNamedErrors() async {
        let cases: [(Int, String, RemoteProviderError)] = [
            (401, "{}", .auth), (403, "{}", .auth), (404, "{}", .endpointNotFound),
            (429, "{}", .rateLimited), (529, "{}", .overloaded), (503, "{}", .overloaded),
            (500, "{}", .server(status: 500)),
            (400, #"{"error":{"code":"context_length_exceeded"}}"#, .contextOverflow),
            (400, "This prompt is too long: 300000 tokens", .contextOverflow),
        ]
        for (status, body, expected) in cases {
            let r = await run(ScriptedTransport(sse: body, status: status))
            XCTAssertEqual(r.error, expected, "status \(status)")
        }
        XCTAssertTrue(RemoteProviderError.contextOverflow.offersCompaction)
    }

    func testErrorBodyEchoingTheKeyIsRedacted() async {
        let r = await run(ScriptedTransport(sse: "invalid x-api-key \(secretKey) rejected", status: 400))
        guard case .requestRefused(_, let detail)? = r.error else { return XCTFail("\(String(describing: r.error))") }
        XCTAssertFalse(detail.contains(secretKey))
        XCTAssertFalse(RemoteProviderError.requestRefused(status: 400, detail: detail).diagnostic.contains(secretKey))
    }

    func testCrossHostRedirectIsRefusedAndNeverSent() async {
        let t = ScriptedTransport { req, _ in
            ScriptedTransport.Reply(status: 307, headers: ["location": "https://evil.example.net/steal"])
        }
        let r = await run(t)
        XCTAssertEqual(r.error, .redirectRefused)
        XCTAssertEqual(t.requests.count, 1, "the redirect target must never receive a request")
        XCTAssertFalse(t.requests.contains { $0.url.host == "evil.example.net" })
    }

    func testHTTPDowngradeRedirectIsRefused() async {
        let t = ScriptedTransport { _, _ in
            ScriptedTransport.Reply(status: 302, headers: ["location": "http://api.example.com/chat/completions"])
        }
        let r = await run(t)
        XCTAssertEqual(r.error, .redirectRefused)
        XCTAssertEqual(t.requests.count, 1)
    }

    func testSameHostRedirectIsFollowedWithFreshHeaders() async {
        let done = sse([(nil, #"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#), (nil, "[DONE]")])
        let t = ScriptedTransport { req, i in
            i == 0 ? ScriptedTransport.Reply(status: 308, headers: ["location": "/other/chat/completions"])
                   : ScriptedTransport.Reply(chunks: [Data(done.utf8)])
        }
        let r = await run(t)
        XCTAssertNil(r.error)
        XCTAssertEqual(t.requests.count, 2)
        XCTAssertEqual(t.requests[1].url.path, "/other/chat/completions")
        XCTAssertEqual(t.requests[1].headers["authorization"], "Bearer \(secretKey)")
    }

    func testRedirectLoopIsBounded() async {
        let t = ScriptedTransport { _, _ in
            ScriptedTransport.Reply(status: 302, headers: ["location": "/again"])
        }
        let r = await run(t)
        XCTAssertEqual(r.error, .redirectRefused)
        XCTAssertLessThanOrEqual(t.requests.count, RemoteLimits.standard.maxRedirects + 1)
    }

    func testRequestTooLargeIsRefusedBeforeSending() async {
        var limits = RemoteLimits(); limits.maxRequestBytes = 50
        let t = ScriptedTransport(sse: "")
        let r = await run(t, limits: limits)
        XCTAssertEqual(r.error, .requestTooLarge)
        XCTAssertTrue(t.requests.isEmpty)
    }

    func testResponseTooLarge() async {
        var limits = RemoteLimits(); limits.maxResponseBytes = 64
        let big = "data: " + String(repeating: "a", count: 200) + "\n\n"
        let r = await run(ScriptedTransport(sse: big), limits: limits)
        XCTAssertEqual(r.error, .responseTooLarge)
    }

    func testDroppedConnectionMidStreamIsNetworkErrorAndRunsNoTool() async {
        let partial = sse([(nil, #"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"a","arguments":"{}"}}]}}]}"#)])
        let t = ScriptedTransport { _, _ in
            ScriptedTransport.Reply(chunks: [Data(partial.utf8)], failure: URLError(.networkConnectionLost))
        }
        let r = await run(t)
        guard case .network? = r.error else { return XCTFail("\(String(describing: r.error))") }
        XCTAssertFalse(r.events.contains { if case .toolCall = $0 { return true } else { return false } })
    }

    func testStallAndFirstByteTimeouts() async {
        var limits = RemoteLimits(); limits.firstByteTimeout = 0.15; limits.stallTimeout = 0.15
        let stalled = ScriptedTransport { _, _ in
            ScriptedTransport.Reply(chunks: [Data("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n".utf8)], hang: true)
        }
        let a = await run(stalled, limits: limits)
        XCTAssertEqual(a.error, .stalled)
        XCTAssertEqual(a.events, [.content("a")])

        let silent = ScriptedTransport { _, _ in ScriptedTransport.Reply(delayBeforeResponse: 5) }
        let b = await run(silent, limits: limits)
        XCTAssertEqual(b.error, .firstByteTimeout)
    }

    func testConsumerCancellationEndsAsCancelled() async {
        let t = ScriptedTransport { _, _ in ScriptedTransport.Reply(hang: true) }
        let stream = RemoteStreamRunner(transport: t, limits: .standard)
            .stream(client: OpenAICompatClient(), request: chatRequest(), baseURL: base, key: secretKey)
        let task = Task { await collect(stream) }
        try? await Task.sleep(nanoseconds: 100_000_000)
        task.cancel()
        let r = await task.value
        XCTAssertTrue(r.error == nil || r.error == .cancelled)
    }

    func testProbeUsesFreeModelListAndFallsBackOnlyWhenAbsent() async throws {
        let ok = ScriptedTransport { _, _ in ScriptedTransport.Reply(chunks: [Data("{}".utf8)]) }
        try await RemoteStreamRunner(transport: ok, limits: .standard)
            .probe(client: OpenAICompatClient(), baseURL: base, key: secretKey, model: "m")
        XCTAssertEqual(ok.requests.map(\.method), ["GET"])
        XCTAssertEqual(ok.requests[0].url.path, "/models")

        let done = sse([(nil, #"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#), (nil, "[DONE]")])
        let fallback = ScriptedTransport { req, _ in
            req.method == "GET" ? ScriptedTransport.Reply(status: 404) : ScriptedTransport.Reply(chunks: [Data(done.utf8)])
        }
        try await RemoteStreamRunner(transport: fallback, limits: .standard)
            .probe(client: OpenAICompatClient(), baseURL: base, key: secretKey, model: "m")
        XCTAssertEqual(fallback.requests.map(\.method), ["GET", "POST"])

        let denied = ScriptedTransport { _, _ in ScriptedTransport.Reply(status: 401) }
        do {
            try await RemoteStreamRunner(transport: denied, limits: .standard)
                .probe(client: OpenAICompatClient(), baseURL: base, key: secretKey, model: "m")
            XCTFail("expected auth")
        } catch { XCTAssertEqual(error as? RemoteProviderError, .auth) }
        XCTAssertEqual(denied.requests.count, 1, "an auth failure must not fall back to a paid call")
    }
}

// MARK: - Key store, provider store, service

@MainActor
final class RemoteProviderStoreTests: XCTestCase {
    private var dir: URL!
    private var defaults: UserDefaults!
    private var suite: String!

    override func setUp() async throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("ts_remote_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        suite = "RemoteProviderTests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: dir)
        defaults.removePersistentDomain(forName: suite)
    }

    private func makeStore(_ backend: FakeSecretBackend = FakeSecretBackend()) -> (RemoteProviderStore, FakeSecretBackend, URL) {
        let file = dir.appendingPathComponent("remote_providers.json")
        let keys = RemoteProviderKeyStore(backend: backend, defaults: defaults)
        return (RemoteProviderStore(fileURL: file, keys: keys), backend, file)
    }

    private func model(_ id: String = "m1") -> RemoteProviderModel {
        RemoteProviderModel(id: id, displayName: id, contextWindow: 100_000)
    }

    func testKeychainIsNeverTouchedAtInitOrWithoutOptIn() throws {
        let (store, backend, _) = makeStore()
        XCTAssertEqual(backend.touchCount, 0, "construction must not touch the Keychain")
        let p = try store.add(name: "A", family: .anthropic, baseURL: "https://api.anthropic.com")
        store.setModels([model()], for: p.id)
        _ = store.pickerEntries()
        XCTAssertEqual(backend.touchCount, 0, "building the picker must not touch the Keychain without opt-in")
        XCTAssertThrowsError(try store.keys.keyForSend(provider: p.id)) {
            XCTAssertEqual($0 as? RemoteProviderError, .notConfigured)
        }
        XCTAssertEqual(backend.readCount, 0, "no opt-in means no read")
    }

    func testSessionKeyNeverReachesKeychainOrDisk() throws {
        let (store, backend, file) = makeStore()
        let p = try store.add(name: "A", family: .openAICompatible, baseURL: "https://x.example.com/v1")
        store.setModels([model()], for: p.id)
        store.keys.setSessionKey(secretKey, for: p.id)
        XCTAssertEqual(try store.keys.keyForSend(provider: p.id), secretKey)
        XCTAssertEqual(backend.writeCount, 0)
        XCTAssertFalse(store.keys.isKeychainEnabled(for: p.id))
        let json = try String(contentsOf: file, encoding: .utf8)
        XCTAssertFalse(json.contains(secretKey))
        XCTAssertFalse(json.lowercased().contains("key\""), "the shape file has no key field")
    }

    func testOptInSaveReadsOnceAtFirstSendThenCaches() throws {
        let backend = FakeSecretBackend()
        let (store, _, file) = makeStore(backend)
        let p = try store.add(name: "A", family: .anthropic, baseURL: "https://api.anthropic.com")
        XCTAssertEqual(store.keys.saveToKeychain(secretKey, for: p.id), errSecSuccess)
        XCTAssertTrue(store.keys.isKeychainEnabled(for: p.id))
        XCTAssertFalse(try String(contentsOf: file, encoding: .utf8).contains(secretKey))

        // A fresh process: same backend and defaults, no session memory.
        let relaunched = RemoteProviderKeyStore(backend: backend, defaults: defaults)
        XCTAssertTrue(relaunched.isKeyAvailable(for: p.id))
        XCTAssertEqual(backend.readCount, 0, "availability is an attributes-only probe, never a secret read")
        XCTAssertEqual(try relaunched.keyForSend(provider: p.id), secretKey)
        XCTAssertEqual(try relaunched.keyForSend(provider: p.id), secretKey)
        XCTAssertEqual(backend.readCount, 1, "the secret is read once, at the first send")
    }

    func testDeniedKeychainReadIsANamedErrorAndNeverDeletes() throws {
        let backend = FakeSecretBackend()
        let (store, _, _) = makeStore(backend)
        let p = try store.add(name: "A", family: .anthropic, baseURL: "https://api.anthropic.com")
        store.keys.saveToKeychain(secretKey, for: p.id)
        let relaunched = RemoteProviderKeyStore(backend: backend, defaults: defaults)
        backend.readStatus = errSecAuthFailed
        XCTAssertThrowsError(try relaunched.keyForSend(provider: p.id)) {
            guard case .keyUnavailable? = $0 as? RemoteProviderError else { return XCTFail("\($0)") }
            XCTAssertFalse(($0 as! RemoteProviderError).diagnostic.contains(secretKey))
        }
        XCTAssertEqual(backend.items.count, 1, "a failed read must not delete the stored key")
    }

    func testRemoveForgetsKeychainItemAndOptIn() throws {
        let (store, backend, _) = makeStore()
        let p = try store.add(name: "A", family: .anthropic, baseURL: "https://api.anthropic.com")
        store.keys.saveToKeychain(secretKey, for: p.id)
        store.remove(p.id)
        XCTAssertTrue(backend.items.isEmpty)
        XCTAssertFalse(store.keys.isKeychainEnabled(for: p.id))
        XCTAssertTrue(store.providers.isEmpty)
    }

    func testPickerIsHiddenWithoutProvidersOrKeysAndShowsEnabledModels() throws {
        let (store, _, _) = makeStore()
        XCTAssertTrue(store.pickerEntries().isEmpty, "off by default: no Remote section")
        let p = try store.add(name: "A", family: .openAICompatible, baseURL: "https://x.example.com/v1")
        store.setModels([model("on"), RemoteProviderModel(id: "off", displayName: "off", contextWindow: 1, enabled: false)], for: p.id)
        XCTAssertTrue(store.pickerEntries().isEmpty, "no key, no entry")
        store.keys.setSessionKey(secretKey, for: p.id)
        XCTAssertEqual(store.pickerEntries().map(\.model.id), ["on"])
        try store.update(p.id, enabled: false)
        XCTAssertTrue(store.pickerEntries().isEmpty)
    }

    func testInvalidBaseURLIsRefusedAtSave() {
        let (store, _, _) = makeStore()
        XCTAssertThrowsError(try store.add(name: "A", family: .openAICompatible, baseURL: "http://example.com/v1"))
        XCTAssertTrue(store.providers.isEmpty)
    }

    func testReloadRestoresShapeWithoutKeys() throws {
        let (store, backend, file) = makeStore()
        let p = try store.add(name: "A", family: .anthropic, baseURL: "https://api.anthropic.com")
        store.setModels([model()], for: p.id)
        let again = RemoteProviderStore(fileURL: file, keys: RemoteProviderKeyStore(backend: backend, defaults: defaults))
        XCTAssertEqual(again.providers, store.providers)
        XCTAssertEqual(backend.touchCount, 0)
    }

    func testSendNeedsPerChatConfirmationAndNamesTheHost() throws {
        let (store, _, _) = makeStore()
        let p = try store.add(name: "A", family: .anthropic, baseURL: "https://api.anthropic.com")
        store.setModels([model()], for: p.id)
        let chat = UUID(), other = UUID()
        XCTAssertEqual(store.sendDecision(provider: p.id, model: "m1", chat: chat), .refused(.notConfigured))
        store.keys.setSessionKey(secretKey, for: p.id)
        XCTAssertEqual(store.sendDecision(provider: p.id, model: "m1", chat: chat), .needsConfirmation(host: "api.anthropic.com"))
        store.confirmConsent(provider: p.id, chat: chat)
        XCTAssertEqual(store.sendDecision(provider: p.id, model: "m1", chat: chat), .proceed)
        XCTAssertEqual(store.sendDecision(provider: p.id, model: "m1", chat: other), .needsConfirmation(host: "api.anthropic.com"))
        XCTAssertEqual(store.sendDecision(provider: p.id, model: "nope", chat: chat), .refused(.disabled))
    }

    func testServiceSendsNothingUntilConfirmedAndKeyOnlyToTheConfiguredHost() async throws {
        let (store, _, _) = makeStore()
        let p = try store.add(name: "A", family: .openAICompatible, baseURL: "https://x.example.com/v1")
        store.setModels([model()], for: p.id)
        store.keys.setSessionKey(secretKey, for: p.id)
        let done = sse([(nil, #"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#), (nil, "[DONE]")])
        let t = ScriptedTransport(sse: done)
        let service = RemoteProviderService(store: store, transport: t)
        let chat = UUID()
        XCTAssertThrowsError(try service.stream(providerID: p.id, chat: chat, request: chatRequest()))
        XCTAssertTrue(t.requests.isEmpty, "an unconfirmed chat must send nothing")
        store.confirmConsent(provider: p.id, chat: chat)
        let r = await collect(try service.stream(providerID: p.id, chat: chat, request: chatRequest()))
        XCTAssertNil(r.error)
        XCTAssertEqual(t.requests.count, 1)
        XCTAssertEqual(t.requests[0].url.host, "x.example.com")
    }

    func testTestConnectionReportsNamedFailure() async throws {
        let (store, _, _) = makeStore()
        let p = try store.add(name: "A", family: .anthropic, baseURL: "https://api.anthropic.com")
        store.keys.setSessionKey(secretKey, for: p.id)
        let service = RemoteProviderService(store: store, transport: ScriptedTransport { _, _ in ScriptedTransport.Reply(status: 401) })
        do { try await service.testConnection(providerID: p.id); XCTFail("expected auth") }
        catch { XCTAssertEqual(error as? RemoteProviderError, .auth) }
    }
}

// MARK: - The remote turn, on the real turn loop

/// Drives `executeGenerationTurn` for a chat pointed at a scripted remote
/// model. No local session exists in these tests: that is the point.
@MainActor
final class RemoteTurnTests: XCTestCase {
    private var model: AppModel!
    private var dir: URL!
    private var suite: String!
    private var transport: ScriptedTransport!
    private var providerID: UUID!
    private var chatID: UUID!
    private let key = "sk-ant-api03-TESTKEYTESTKEYTESTKEY1234"

    override func setUp() async throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("ts_remote_turn_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        suite = "RemoteTurnTests.\(UUID().uuidString)"
        model = AppModel()
        model.remoteProviders = RemoteProviderStore(
            fileURL: dir.appendingPathComponent("remote_providers.json"),
            keys: RemoteProviderKeyStore(backend: FakeSecretBackend(), defaults: UserDefaults(suiteName: suite)!))
        let p = try model.remoteProviders.add(name: "Acme", family: .openAICompatible, baseURL: "https://api.acme.example/v1")
        providerID = p.id
        model.remoteProviders.setModels(
            [RemoteProviderModel(id: "acme-1", displayName: "Acme One", contextWindow: 100_000)], for: p.id)
        model.remoteProviders.keys.setSessionKey(key, for: p.id)
        let chat = AppChat(title: "Remote")
        chatID = chat.id
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.interactionMode = .chat
        model.chatToolsEnabled = true
        model.webSearchEnabled = true
        model.mutateTurnMessages(for: chat.id) { $0.append(AppChatMessage(role: .user, content: "hello")) }
    }

    override func tearDown() async throws {
        UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: dir)
        model = nil
    }

    private func select(confirmed: Bool = true) {
        if confirmed { model.remoteProviders.confirmConsent(provider: providerID, chat: chatID) }
        model.requestRemoteModel(providerID: providerID, modelID: "acme-1", chatID: chatID)
    }

    private func settle() async {
        for _ in 0..<400 {
            if !model.generating { return }
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        XCTFail("the remote turn never finished")
    }

    private func proseStream(_ text: String) -> String {
        sse([
            (nil, #"{"choices":[{"delta":{"content":"\#(text)"}}]}"#),
            (nil, #"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
            (nil, #"{"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":5}}"#),
            (nil, "[DONE]"),
        ])
    }

    func testSelectionNeedsConfirmationAndNothingIsSentBeforeIt() {
        select(confirmed: false)
        XCTAssertNil(model.remoteSelections[chatID], "selecting without consent must not switch the chat")
        XCTAssertEqual(model.remoteConfirmation?.host, "api.acme.example")
        model.confirmRemoteSelection()
        XCTAssertEqual(model.remoteSelections[chatID]?.modelID, "acme-1")
        XCTAssertNil(model.remoteConfirmation)
    }

    func testARemoteChatNeedsNoLocalSessionAndCanRun() {
        XCTAssertNil(model.session)
        XCTAssertFalse(model.hasTurnModel)
        select()
        XCTAssertTrue(model.hasTurnModel)
        model.useLocalModel(chatID: chatID)
        XCTAssertFalse(model.hasTurnModel, "local is the default and the way back")
    }

    func testProseTurnStreamsRecordsUsageProviderTaggedAndReleasesTheLane() async throws {
        transport = ScriptedTransport(sse: proseStream("Hi there"))
        model.remoteTransport = transport
        select()
        model.executeGenerationTurn(step: 0, chatID: chatID)
        XCTAssertTrue(model.generating)
        XCTAssertEqual(model.generatingChatID, chatID, "a remote turn claims the same lane")
        await settle()

        let reply = model.chats[0].messages.last
        XCTAssertEqual(reply?.role, .assistant)
        XCTAssertEqual(reply?.content, "Hi there")
        XCTAssertNil(model.generatingChatID)
        XCTAssertNil(model.turnCoordinator.laneHolder)
        XCTAssertEqual(model.chats[0].usage?.byProvider["Acme"]?.promptTokens, 40)
        XCTAssertEqual(model.chats[0].usage?.byProvider["Acme"]?.outputTokens, 5)
        XCTAssertEqual(model.chats[0].usage?.totalPromptTokens, 40)
        XCTAssertEqual(model.remoteUsage[chatID], 45)
        XCTAssertFalse(model.remoteContextReading(chatID: chatID)!.isEstimate)

        let req = try XCTUnwrap(transport.requests.first)
        XCTAssertEqual(req.url.host, "api.acme.example")
        XCTAssertFalse(String(decoding: req.body, as: UTF8.self).contains(key))
        let body = try XCTUnwrap(RemoteJSON.object(String(decoding: req.body, as: UTF8.self)))
        XCTAssertNotNil(body["tools"], "chat tools are offered natively")
    }

    func testBeforeAnyUsageTheRingReadsALabeledEstimate() {
        select()
        let reading = model.remoteContextReading(chatID: chatID)
        XCTAssertEqual(reading?.isEstimate, true)
        XCTAssertEqual(reading?.windowTokens, 100_000)
    }

    func testProviderErrorEndsTheTurnWithALocalizedKeyFreeErrorAndAnInterruptedRow() async {
        transport = ScriptedTransport(sse: "invalid key \(key)", status: 401)
        model.remoteTransport = transport
        select()
        model.executeGenerationTurn(step: 0, chatID: chatID)
        await settle()
        XCTAssertEqual(model.error, model.remoteErrorText(.auth))
        XCTAssertFalse((model.error ?? "").contains(key))
        XCTAssertNil(model.chats[0].usage, "a failed call records no usage")
        XCTAssertNil(model.generatingChatID)
    }

    func testEveryRemoteErrorHasADistinctKeyFreeMessage() {
        let all: [RemoteProviderError] = [
            .notConfigured, .disabled, .invalidBaseURL("x"), .keyUnavailable("denied"), .auth,
            .endpointNotFound, .rateLimited, .overloaded, .contextOverflow, .server(status: 500),
            .requestRefused(status: 400, detail: "d"), .network("n"), .badStream("b"),
            .redirectRefused, .firstByteTimeout, .stalled, .responseTooLarge, .requestTooLarge,
        ]
        let texts = all.map { model.remoteErrorText($0) }
        XCTAssertEqual(Set(texts).count, texts.count)
        XCTAssertTrue(texts.allSatisfy { !$0.contains(key) && !$0.isEmpty })
    }

    func testStopCancelsTheStreamAndKeepsThePartialReply() async {
        transport = ScriptedTransport { _, _ in
            ScriptedTransport.Reply(
                chunks: [Data(#"data: {"choices":[{"delta":{"content":"par"}}]}"#.utf8 + [10, 10])], hang: true)
        }
        model.remoteTransport = transport
        select()
        model.executeGenerationTurn(step: 0, chatID: chatID)
        for _ in 0..<100 where model.outputText.isEmpty { try? await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertEqual(model.outputText, "par")
        model.cancel()
        await settle()
        XCTAssertNil(model.generatingChatID)
        XCTAssertEqual(model.chats[0].messages.last?.content, "par")
    }

    // MARK: Tool ladder equality

    private func toolCallStream() -> String {
        sse([
            (nil, #"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"web_fetch","arguments":"{\"url\":\"https://example.org/a\"}"}}]}}]}"#),
            (nil, #"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
            (nil, "[DONE]"),
        ])
    }

    /// The same call, once as a remote reply and once handed to the shared
    /// tail the way a local reply is, must meet the same approval ladder.
    func testRemoteToolCallMeetsTheSameLadderAsALocalOne() async throws {
        transport = ScriptedTransport(sse: toolCallStream())
        model.remoteTransport = transport
        select()
        model.executeGenerationTurn(step: 0, chatID: chatID)
        await settle()
        let remotePending = try XCTUnwrap(model.pendingToolCall, "a remote tool call must raise the approval card")
        XCTAssertEqual(remotePending.name, "WebFetch", "normalized to the offered tool name, as a local call is")
        XCTAssertEqual(model.pendingToolCallChatID, chatID)
        XCTAssertEqual(
            transport.requests.count, 1,
            "nothing ran: the call is parked behind approval, no tool result was sent")

        // Local-shaped path on a second chat, same call, same shared tail.
        model.clearPendingToolCall()
        let other = AppChat(title: "Local shaped")
        model.chats.append(other)
        model.selectedChatID = other.id
        model.mutateTurnMessages(for: other.id) { $0.append(AppChatMessage(role: .user, content: "hello")) }
        let local = AppModel.makeGenerationResult(
            promptTokens: 1, newTokens: 1, decodeSeconds: 0, finish: .toolCalls, content: "",
            reasoning: "", toolCalls: [RemoteToolCall(id: "c1", name: "web_fetch", argumentsJSON: #"{"url":"https://example.org/a"}"#)])!
        model.generating = true
        model.claimLane(for: other.id)
        let tools = model.captureTurnTools(
            project: nil, mediaCapability: AppToolMediaCapability(supportsImageBearingToolResults: false, maximumPixelCount: 0),
            contextTokens: nil)
        await model.processCompletedTurn(
            completed: CompletedGenerationTurn(
                result: local, content: "", reasoning: "", continuationsUsed: 0,
                outputTokensGenerated: 1, recoveryEvents: []),
            nativeToolCalls: [GenerationToolCall(id: "c1", name: "web_fetch", argumentsJSON: #"{"url":"https://example.org/a"}"#)],
            step: 0, turnChatID: other.id, turnProject: nil, usesSkillState: false,
            turnAvailableTools: tools, turnAllowsToolCalls: true, myEpoch: model.generationEpoch)
        let localPending = try XCTUnwrap(model.pendingToolCall)
        XCTAssertEqual(localPending.name, remotePending.name)
        XCTAssertEqual(localPending.arguments, remotePending.arguments)
        XCTAssertEqual(localPending.category, remotePending.category)
    }

    func testTruncatedToolCallFromTheProviderRunsNothing() async {
        transport = ScriptedTransport(sse: sse([
            (nil, #"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"web_fetch","arguments":"{\"url\":"}}]}}]}"#),
            (nil, #"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#),
            (nil, "[DONE]"),
        ]))
        model.remoteTransport = transport
        select()
        model.executeGenerationTurn(step: 0, chatID: chatID)
        await settle()
        XCTAssertNil(model.pendingToolCall)
        XCTAssertEqual(model.error, model.remoteErrorText(.badStream("")))
    }

    func testCompactionSummaryRunsOnTheSelectedRemoteModel() async throws {
        transport = ScriptedTransport(sse: proseStream("<summary>short</summary>"))
        model.remoteTransport = transport
        select()
        var options = GenerateOptions(); options.maxNewTokens = 100
        let text = try await model.remoteSummary(chatID: chatID, messages: [ChatMessage.user("a long transcript")], options: options)
        XCTAssertTrue(text.contains("short"))
        XCTAssertEqual(transport.requests.first?.url.host, "api.acme.example")
    }

    func testOverflowOffersCompactionNamedError() async {
        transport = ScriptedTransport(sse: #"{"error":{"code":"context_length_exceeded"}}"#, status: 400)
        model.remoteTransport = transport
        select()
        model.executeGenerationTurn(step: 0, chatID: chatID)
        await settle()
        XCTAssertEqual(model.error, model.remoteErrorText(.contextOverflow))
    }

    func testRolloutLogRecordsHostMarkerAndNoKeyWhenEnabled() async throws {
        model.trajectoryLoggingEnabled = true
        defer { model.trajectoryLoggingEnabled = false }
        transport = ScriptedTransport(sse: proseStream("echo \(key) done"))
        model.remoteTransport = transport
        select()
        model.executeGenerationTurn(step: 0, chatID: chatID)
        await settle()
        RolloutStore.shared.flush()
        let log = RolloutStore.shared.readTail(chatID: chatID)
        let entry = try XCTUnwrap(log.records.first { $0.entry.kind == .modelCall }?.entry)
        XCTAssertEqual(entry.remoteHost, "api.acme.example")
        XCTAssertFalse((entry.result ?? "").contains(key))
    }
}
