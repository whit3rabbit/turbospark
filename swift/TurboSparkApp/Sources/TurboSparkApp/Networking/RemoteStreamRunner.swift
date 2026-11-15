import Foundation

/// A family's half of the contract: how to build the request and how to read
/// the stream. Everything else (caps, timeouts, redirects, error mapping,
/// redaction) is shared in `RemoteStreamRunner`.
protocol RemoteFamilyClient: Sendable {
    var family: RemoteAPIFamily { get }
    func buildChatRequest(_ request: RemoteChatRequest, endpoint: URL, key: String) throws -> RemoteHTTPRequest
    func buildProbeRequest(endpoint: URL, key: String) -> RemoteHTTPRequest
    func makeDecoder() -> RemoteStreamDecoder
}

/// Turns SSE events into step events. A decoder owns every partial state, so
/// a stream cut mid tool call never emits that call.
protocol RemoteStreamDecoder {
    mutating func consume(_ event: SSEEvent) throws -> [RemoteStepEvent]
    /// Called once at end of stream. Throws `badStream` when the stream ended
    /// before its terminal event.
    mutating func finish() throws -> [RemoteStepEvent]
}

/// Maps an HTTP failure to a named error. The body is redacted (exact key
/// plus patterns) and clipped before it can land in a case payload.
enum RemoteErrorClassifier {
    private static let overflowPhrases = [
        "context_length_exceeded", "prompt is too long", "maximum context length",
        "context window", "too many tokens", "exceeds the maximum",
    ]

    static func classify(status: Int, body: String, key: String) -> RemoteProviderError {
        let lowered = body.lowercased()
        switch status {
        case 401, 403: return .auth
        case 404: return .endpointNotFound
        case 413: return .contextOverflow
        case 429: return .rateLimited
        case 503, 529: return .overloaded
        case 500...599: return .server(status: status)
        case 400, 422:
            if overflowPhrases.contains(where: { lowered.contains($0) }) { return .contextOverflow }
            fallthrough
        default:
            return .requestRefused(
                status: status,
                detail: RemoteProviderRedactor.redact(body, keys: [key], maxCharacters: 300))
        }
    }

    /// For an `error` object inside a stream (HTTP 200 already sent).
    static func classifyStreamError(type: String?, message: String?, key: String) -> RemoteProviderError {
        let t = (type ?? "").lowercased()
        let m = (message ?? "").lowercased()
        if t.contains("overloaded") { return .overloaded }
        if t.contains("rate_limit") { return .rateLimited }
        if t.contains("authentication") || t.contains("permission") || t == "invalid_api_key" { return .auth }
        if overflowPhrases.contains(where: { m.contains($0) || t.contains($0) }) { return .contextOverflow }
        return .badStream(
            "provider error \(RemoteProviderRedactor.redact(type ?? "unknown", keys: [key], maxCharacters: 80))")
    }
}

/// Wall-clock bookkeeping shared by the consume loop and its watchdog.
private final class StreamActivity: @unchecked Sendable {
    private let lock = NSLock()
    private var last = Date()
    private var gotFirstByte = false
    private var timeout: RemoteProviderError?

    private var done = false
    func finish() { lock.lock(); done = true; lock.unlock() }
    var isDone: Bool { lock.lock(); defer { lock.unlock() }; return done }

    func touch() {
        lock.lock(); last = Date(); gotFirstByte = true; lock.unlock()
    }
    func snapshot() -> (last: Date, first: Bool) {
        lock.lock(); defer { lock.unlock() }
        return (last, gotFirstByte)
    }
    func trip(_ error: RemoteProviderError) {
        lock.lock(); if timeout == nil { timeout = error }; lock.unlock()
    }
    var tripped: RemoteProviderError? {
        lock.lock(); defer { lock.unlock() }
        return timeout
    }
}

/// Runs one remote request end to end.
///
/// Redirect hops are re-checked against `RemoteEndpointPolicy` and the
/// request is rebuilt per hop, so the key header only ever goes to the host
/// the user configured. No automatic retry: every error ends the stream.
struct RemoteStreamRunner: Sendable {
    let transport: any RemoteHTTPTransport
    let limits: RemoteLimits

    /// Streams one chat request.
    func stream(
        client: any RemoteFamilyClient, request: RemoteChatRequest, baseURL: URL, key: String
    ) -> AsyncThrowingStream<RemoteStepEvent, Error> {
        AsyncThrowingStream { continuation in
            let activity = StreamActivity()
            let main = Task {
                defer { activity.finish() }
                do {
                    let endpoint = RemoteEndpointPolicy.endpoint(base: baseURL, family: client.family)
                    let http = try client.buildChatRequest(request, endpoint: endpoint, key: key)
                    guard http.body.count <= limits.maxRequestBytes else {
                        throw RemoteProviderError.requestTooLarge
                    }
                    let response = try await open(http, rebuild: { url in
                        try client.buildChatRequest(request, endpoint: url, key: key)
                    }, key: key)
                    try await consume(
                        response, decoder: client.makeDecoder(), key: key, activity: activity,
                        emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: normalize(error, activity: activity, key: key))
                }
            }
            let watchdog = startWatchdog(activity: activity, owner: main)
            continuation.onTermination = { _ in
                main.cancel()
                watchdog.cancel()
            }
        }
    }

    /// Connection check: a free models-list call, falling back to a one
    /// token completion when the endpoint does not serve the list (404 or
    /// 405). Returns normally on success.
    func probe(
        client: any RemoteFamilyClient, baseURL: URL, key: String, model: String?
    ) async throws {
        let listURL = RemoteEndpointPolicy.endpoint(base: baseURL, family: client.family, probe: true)
        do {
            let response = try await open(
                client.buildProbeRequest(endpoint: listURL, key: key),
                rebuild: { client.buildProbeRequest(endpoint: $0, key: key) }, key: key)
            try await requireSuccess(response, key: key)
            return
        } catch let error as RemoteProviderError {
            // Only "this endpoint has no list" falls through to the paid
            // probe; an auth or network failure is the answer already.
            var noList = error == .endpointNotFound
            if case .requestRefused(405, _) = error { noList = true }
            guard noList, let model, !model.isEmpty else { throw normalizeProbe(error) }
        } catch {
            throw normalize(error, activity: StreamActivity(), key: key)
        }
        var tiny = RemoteChatRequest(
            model: model ?? "", system: nil,
            messages: [RemoteChatMessage(role: .user, content: "ping")], maxTokens: 1)
        tiny.temperature = nil
        for try await _ in stream(client: client, request: tiny, baseURL: baseURL, key: key) {}
    }

    private func normalizeProbe(_ error: RemoteProviderError) -> RemoteProviderError {
        if case .requestRefused(405, _) = error { return .endpointNotFound }
        return error
    }

    // MARK: Internals

    private func startWatchdog(activity: StreamActivity, owner: Task<Void, Never>) -> Task<Void, Never> {
        let first = limits.firstByteTimeout
        let stall = limits.stallTimeout
        let tick = max(0.01, min(0.25, min(first, stall) / 4))
        return Task {
            while !Task.isCancelled, !activity.isDone {
                try? await Task.sleep(nanoseconds: UInt64(tick * 1_000_000_000))
                if Task.isCancelled || activity.isDone { return }
                let snap = activity.snapshot()
                let idle = Date().timeIntervalSince(snap.last)
                if !snap.first, idle > first {
                    activity.trip(.firstByteTimeout)
                    owner.cancel()
                    return
                }
                if snap.first, idle > stall {
                    activity.trip(.stalled)
                    owner.cancel()
                    return
                }
            }
        }
    }

    private func normalize(_ error: Error, activity: StreamActivity, key: String) -> RemoteProviderError {
        if let tripped = activity.tripped { return tripped }
        if let remote = error as? RemoteProviderError { return remote }
        if error is CancellationError || (error as? URLError)?.code == .cancelled { return .cancelled }
        if error is SSEParseError { return .badStream("event too large") }
        let text = RemoteProviderRedactor.redact(error.localizedDescription, keys: [key], maxCharacters: 200)
        return .network(text)
    }

    /// Sends, following same-host redirects under policy.
    private func open(
        _ first: RemoteHTTPRequest, rebuild: (URL) throws -> RemoteHTTPRequest, key: String
    ) async throws -> RemoteHTTPResponse {
        var current = first
        for _ in 0...limits.maxRedirects {
            let response = try await transport.send(current)
            guard (300...399).contains(response.status) else { return response }
            guard let location = response.headers["location"],
                let target = URL(string: location, relativeTo: current.url)?.absoluteURL,
                RemoteEndpointPolicy.mayFollow(from: current.url, to: target)
            else { throw RemoteProviderError.redirectRefused }
            current = try rebuild(target)
        }
        throw RemoteProviderError.redirectRefused
    }

    private func readErrorBody(_ response: RemoteHTTPResponse) async -> String {
        var data = Data()
        do {
            for try await chunk in response.body {
                data.append(chunk)
                if data.count >= limits.maxErrorBodyBytes { break }
            }
        } catch {}
        return String(decoding: data.prefix(limits.maxErrorBodyBytes), as: UTF8.self)
    }

    private func requireSuccess(_ response: RemoteHTTPResponse, key: String) async throws {
        guard (200...299).contains(response.status) else {
            let body = await readErrorBody(response)
            throw RemoteErrorClassifier.classify(status: response.status, body: body, key: key)
        }
    }

    private func consume(
        _ response: RemoteHTTPResponse, decoder: RemoteStreamDecoder, key: String,
        activity: StreamActivity, emit: (RemoteStepEvent) -> Void
    ) async throws {
        try await requireSuccess(response, key: key)
        var decoder = decoder
        var parser = SSEParser()
        var total = 0
        for try await chunk in response.body {
            activity.touch()
            total += chunk.count
            if total > limits.maxResponseBytes { throw RemoteProviderError.responseTooLarge }
            for event in try parser.feed(chunk) {
                for step in try decoder.consume(event) { emit(step) }
            }
        }
        // A cancelled stream ends quietly; surface why before judging it.
        try Task.checkCancellation()
        for step in try decoder.finish() { emit(step) }
    }
}
