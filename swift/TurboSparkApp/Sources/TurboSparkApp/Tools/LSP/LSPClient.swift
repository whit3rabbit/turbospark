import Foundation

/// The byte pipe under an `LSPClient`. The product implementation is a child
/// process (`LSPProcessTransport`); tests use an in-process scripted server so
/// no real language server is ever spawned.
protocol LSPTransport: AnyObject, Sendable {
    /// Begins delivering stdout chunks to `onData` (in order) and calls
    /// `onClose` exactly once when the stream ends or the server exits.
    func start(onData: @escaping @Sendable (Data) -> Void, onClose: @escaping @Sendable () -> Void) throws
    /// Writes one fully framed message. Throws when the pipe is gone.
    func write(_ data: Data) throws
    /// Stops the server. Idempotent; safe to call from any thread.
    func kill()
    /// A capped tail of the server's stderr, for error messages.
    var stderrText: String { get }
}

enum LSPClientError: Error, Equatable, LocalizedError {
    case notRunning
    case timedOut(String)
    case server(code: Int, message: String)
    case protocolViolation(String)

    var errorDescription: String? {
        switch self {
        case .notRunning: return "The language server is not running."
        case .timedOut(let what): return "The language server did not answer \(what) in time."
        case .server(let code, let message): return "The language server returned error \(code): \(message)"
        case .protocolViolation(let detail): return "The language server broke the protocol: \(detail)"
        }
    }
}

struct LSPTimeouts: Sendable {
    var initialize: TimeInterval = 15
    var request: TimeInterval = 8
    var shutdown: TimeInterval = 2
    /// After the first fresh publishDiagnostics, wait this long for a second
    /// one: servers often publish syntax errors first and semantic errors a
    /// moment later. Always capped by the caller's budget.
    var settle: TimeInterval = 0.15
}

/// What a post-edit wait found.
enum LSPDiagnosticsOutcome: Equatable, Sendable {
    case fresh([LSPDiagnostic])
    case timedOut
    case serverGone
}

/// A JSON-RPC client for one language server. It implements the subset OC-10
/// needs: initialize/shutdown, didOpen/didChange/didSave, publishDiagnostics,
/// documentSymbol and workspace/symbol. All state lives on the actor; the
/// transport's reader thread only yields bytes into an ordered stream.
actor LSPClient {
    private struct DocumentState {
        var version = 0
        var diagnostics: [LSPDiagnostic]?
        var publishCount = 0
        var lastPublishedVersion: Int?
    }

    private struct DiagnosticsWaiter {
        let uri: String
        let baseline: Int
        let version: Int
        let continuation: CheckedContinuation<Bool, Never>
        let timeout: Task<Void, Never>
    }

    private let transport: any LSPTransport
    private let rootURL: URL
    private let timeouts: LSPTimeouts
    private var documents: [String: DocumentState] = [:]
    private var pending: [Int: (continuation: CheckedContinuation<Any?, Error>, timeout: Task<Void, Never>)] = [:]
    private var waiters: [UUID: DiagnosticsWaiter] = [:]
    private var canonicalIndex: [String: String] = [:]
    private var nextID = 1
    private var decoder = LSPFrameDecoder()
    private var reader: Task<Void, Never>?
    private var alive = false
    private var closed = false
    private(set) var serverName: String?
    private(set) var failure: String?

    init(transport: any LSPTransport, rootURL: URL, timeouts: LSPTimeouts = LSPTimeouts()) {
        self.transport = transport
        self.rootURL = rootURL
        self.timeouts = timeouts
    }

    var isAlive: Bool { alive && !closed }

    // MARK: Lifecycle

    /// Starts the transport and completes the initialize handshake. A server
    /// that does not answer within the initialize timeout is killed.
    func start() async throws {
        let (stream, continuation) = AsyncStream<Data?>.makeStream(bufferingPolicy: .unbounded)
        do {
            try transport.start(
                onData: { continuation.yield($0) },
                onClose: { continuation.yield(nil); continuation.finish() })
        } catch {
            closed = true
            throw error
        }
        alive = true
        reader = Task { [weak self] in
            for await chunk in stream {
                guard let self else { return }
                if let chunk { await self.receive(chunk) } else { await self.handleClose() }
            }
        }
        let params: [String: Any] = [
            "processId": Int(ProcessInfo.processInfo.processIdentifier),
            "clientInfo": ["name": "TurboSpark"],
            "rootUri": rootURL.absoluteString,
            "workspaceFolders": [["uri": rootURL.absoluteString, "name": rootURL.lastPathComponent]],
            "capabilities": [
                "textDocument": [
                    "synchronization": ["didSave": true, "dynamicRegistration": false],
                    "publishDiagnostics": ["versionSupport": true, "relatedInformation": false],
                    "documentSymbol": ["hierarchicalDocumentSymbolSupport": true],
                ],
                "workspace": ["symbol": ["dynamicRegistration": false], "workspaceFolders": true],
            ],
        ]
        do {
            let result = try await request("initialize", params, timeout: timeouts.initialize)
            if let info = (result as? [String: Any])?["serverInfo"] as? [String: Any] {
                serverName = info["name"] as? String
            }
            try sendObject(["jsonrpc": "2.0", "method": "initialized", "params": [:] as [String: Any]])
        } catch {
            failure = error.localizedDescription
            kill()
            throw error
        }
    }

    /// Polite shutdown (shutdown request, exit notification), then the kill
    /// ladder. Never throws: the server is gone afterwards either way.
    func shutdown() async {
        if alive, !closed {
            _ = try? await request("shutdown", nil, timeout: timeouts.shutdown)
            try? sendObject(["jsonrpc": "2.0", "method": "exit"])
        }
        kill()
    }

    func kill() {
        closed = true
        alive = false
        transport.kill()
        failAll(LSPClientError.notRunning)
        reader?.cancel()
        reader = nil
    }

    // MARK: Documents

    /// Opens the document, or sends a full-text change when it is already
    /// open, then didSave. Returns the baseline a later `awaitDiagnostics`
    /// compares against, taken BEFORE the send so a fast publish is not lost.
    func sync(uri: String, languageID: String, text: String) throws -> (version: Int, baseline: Int) {
        guard isAlive else { throw LSPClientError.notRunning }
        canonicalIndex[Self.canonical(uri)] = uri
        var state = documents[uri] ?? DocumentState()
        let baseline = state.publishCount
        let wasOpen = state.version > 0
        state.version += 1
        documents[uri] = state
        if wasOpen {
            try sendObject([
                "jsonrpc": "2.0", "method": "textDocument/didChange",
                "params": [
                    "textDocument": ["uri": uri, "version": state.version],
                    "contentChanges": [["text": text]],
                ] as [String: Any],
            ])
        } else {
            try sendObject([
                "jsonrpc": "2.0", "method": "textDocument/didOpen",
                "params": [
                    "textDocument": ["uri": uri, "languageId": languageID, "version": state.version, "text": text],
                ] as [String: Any],
            ])
        }
        try sendObject([
            "jsonrpc": "2.0", "method": "textDocument/didSave",
            "params": ["textDocument": ["uri": uri], "text": text] as [String: Any],
        ])
        return (state.version, baseline)
    }

    /// Waits up to `budget` seconds for a publishDiagnostics for `uri` that
    /// postdates the sync (by publish count, and by document version when the
    /// server reports one). The budget is a hard cap: a slow server never
    /// extends the caller's turn.
    func awaitDiagnostics(uri: String, version: Int, baseline: Int, budget: TimeInterval) async -> LSPDiagnosticsOutcome {
        guard isAlive else { return .serverGone }
        let started = ContinuousClock.now
        let arrived = await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
                if isFresh(uri: uri, baseline: baseline, version: version) {
                    continuation.resume(returning: true)
                    return
                }
                let id = UUID()
                let timeout = Task { [weak self] in
                    try? await Task.sleep(nanoseconds: UInt64(max(0, budget) * 1_000_000_000))
                    guard !Task.isCancelled else { return }
                    await self?.expireWaiter(id)
                }
                waiters[id] = DiagnosticsWaiter(
                    uri: uri, baseline: baseline, version: version, continuation: continuation, timeout: timeout)
            }
        } onCancel: {
            Task { await self.cancelAllWaiters() }
        }
        guard arrived else { return isAlive ? .timedOut : .serverGone }
        // A second publish often follows the first (syntax then semantics).
        // The settle wait never outlives the budget.
        let spent = (ContinuousClock.now - started).components
        let spentSeconds = Double(spent.seconds) + Double(spent.attoseconds) / 1e18
        let settle = min(timeouts.settle, max(0, budget - spentSeconds))
        if settle > 0 { try? await Task.sleep(nanoseconds: UInt64(settle * 1_000_000_000)) }
        return .fresh(documents[uri]?.diagnostics ?? [])
    }

    /// The last diagnostics the server published for `uri`, or nil when it
    /// never has (so "clean" and "never checked" stay different answers).
    func knownDiagnostics(uri: String) -> [LSPDiagnostic]? {
        documents[uri]?.diagnostics
    }

    var openDocumentURIs: [String] { documents.filter { $0.value.version > 0 }.map(\.key).sorted() }

    // MARK: Symbols

    func documentSymbols(uri: String) async throws -> [LSPDocumentSymbol] {
        let result = try await request(
            "textDocument/documentSymbol", ["textDocument": ["uri": uri]], timeout: timeouts.request)
        return LSPParse.documentSymbols(result)
    }

    func workspaceSymbols(query: String) async throws -> [LSPWorkspaceSymbol] {
        let result = try await request("workspace/symbol", ["query": query], timeout: timeouts.request)
        return LSPParse.workspaceSymbols(result)
    }

    // MARK: Requests

    private func request(_ method: String, _ params: Any?, timeout: TimeInterval) async throws -> Any? {
        guard isAlive || method == "initialize" else { throw LSPClientError.notRunning }
        let id = nextID
        nextID += 1
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Any?, Error>) in
                let timer = Task { [weak self] in
                    try? await Task.sleep(nanoseconds: UInt64(max(0, timeout) * 1_000_000_000))
                    guard !Task.isCancelled else { return }
                    await self?.expire(id: id, what: method)
                }
                pending[id] = (continuation, timer)
                var message: [String: Any] = ["jsonrpc": "2.0", "id": id, "method": method]
                if let params { message["params"] = params }
                do { try sendObject(message) } catch { settle(id: id, with: .failure(error)) }
            }
        } onCancel: {
            Task { await self.cancel(id: id) }
        }
    }

    private func settle(id: Int, with outcome: Result<Any?, Error>) {
        guard let entry = pending.removeValue(forKey: id) else { return }
        entry.timeout.cancel()
        entry.continuation.resume(with: outcome)
    }

    private func expire(id: Int, what: String) {
        guard pending[id] != nil else { return }
        try? sendObject(["jsonrpc": "2.0", "method": "$/cancelRequest", "params": ["id": id]])
        settle(id: id, with: .failure(LSPClientError.timedOut(what)))
    }

    private func cancel(id: Int) {
        guard pending[id] != nil else { return }
        try? sendObject(["jsonrpc": "2.0", "method": "$/cancelRequest", "params": ["id": id]])
        settle(id: id, with: .failure(CancellationError()))
    }

    private func sendObject(_ object: [String: Any]) throws {
        do {
            try transport.write(try LSPFraming.encode(object: object))
        } catch {
            // A failed write means the pipe is gone: nothing was delivered.
            handleClose()
            throw LSPClientError.notRunning
        }
    }

    // MARK: Receiving

    private func receive(_ chunk: Data) {
        let bodies: [Data]
        do {
            bodies = try decoder.append(chunk)
        } catch {
            failure = "protocol violation (\(error))"
            kill()
            return
        }
        for body in bodies { handle(body) }
    }

    private func handle(_ body: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { return }
        let id = json["id"]
        if let method = json["method"] as? String {
            if let id { answerServerRequest(id: id, method: method, params: json["params"]) }
            else { handleNotification(method: method, params: json["params"]) }
            return
        }
        guard let numericID = LSPParse.int(id) else { return }
        if let error = json["error"] as? [String: Any] {
            settle(id: numericID, with: .failure(LSPClientError.server(
                code: LSPParse.int(error["code"]) ?? 0, message: String((error["message"] as? String ?? "").prefix(300)))))
        } else {
            settle(id: numericID, with: .success(json["result"]))
        }
    }

    /// Requests FROM the server. They are answered so a server waiting on this
    /// client cannot stall: capability registration and progress creation get
    /// an empty success, configuration gets one null per item, anything else
    /// is method-not-found. Nothing the server asks for is performed.
    private func answerServerRequest(id: Any, method: String, params: Any?) {
        var reply: [String: Any] = ["jsonrpc": "2.0", "id": id]
        switch method {
        case "client/registerCapability", "client/unregisterCapability", "window/workDoneProgress/create",
             "workspace/diagnostic/refresh", "workspace/semanticTokens/refresh":
            reply["result"] = NSNull()
        case "workspace/configuration":
            let count = ((params as? [String: Any])?["items"] as? [Any])?.count ?? 0
            reply["result"] = Array(repeating: NSNull(), count: min(count, 64))
        default:
            reply["error"] = ["code": -32601, "message": "Method not found"] as [String: Any]
        }
        try? sendObject(reply)
    }

    private func handleNotification(method: String, params: Any?) {
        guard method == "textDocument/publishDiagnostics", let dict = params as? [String: Any],
              let published = dict["uri"] as? String else { return }
        // Only documents this client opened are tracked: a server publishing
        // for files it found itself must not grow memory without bound.
        // A server may spell a path differently from the way it was opened
        // (/private/var versus /var), so a miss is retried by canonical path.
        let uri = documents[published] != nil ? published : (canonicalIndex[Self.canonical(published)] ?? published)
        guard var state = documents[uri] else { return }
        state.diagnostics = LSPParse.diagnostics(dict["diagnostics"])
        state.publishCount += 1
        state.lastPublishedVersion = LSPParse.int(dict["version"])
        documents[uri] = state
        resumeFreshWaiters(uri: uri)
    }

    private static func canonical(_ uri: String) -> String {
        guard let url = URL(string: uri), url.isFileURL else { return uri }
        return PathContainment.canonical(url).path
    }

    private func isFresh(uri: String, baseline: Int, version: Int) -> Bool {
        guard let state = documents[uri], state.publishCount > baseline else { return false }
        if let published = state.lastPublishedVersion { return published >= version }
        return true
    }

    private func resumeFreshWaiters(uri: String) {
        for (id, waiter) in waiters where waiter.uri == uri
            && isFresh(uri: uri, baseline: waiter.baseline, version: waiter.version) {
            waiters.removeValue(forKey: id)
            waiter.timeout.cancel()
            waiter.continuation.resume(returning: true)
        }
    }

    private func expireWaiter(_ id: UUID) {
        guard let waiter = waiters.removeValue(forKey: id) else { return }
        waiter.continuation.resume(returning: false)
    }

    private func cancelAllWaiters() {
        for (_, waiter) in waiters {
            waiter.timeout.cancel()
            waiter.continuation.resume(returning: false)
        }
        waiters.removeAll()
    }

    private func handleClose() {
        guard !closed else { return }
        let tail = transport.stderrText
        failure = failure ?? (tail.isEmpty ? "the server exited" : "the server exited: \(tail)")
        closed = true
        alive = false
        failAll(LSPClientError.notRunning)
    }

    private func failAll(_ error: Error) {
        let all = pending
        pending.removeAll()
        for (_, entry) in all {
            entry.timeout.cancel()
            entry.continuation.resume(throwing: error)
        }
        cancelAllWaiters()
    }
}
