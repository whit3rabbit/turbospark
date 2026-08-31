import CTurboSpark
import Foundation

/// What port and auth an in-process server starts with. `{}` (every field
/// default) asks the OS to choose a port and starts unauthenticated.
///
/// **UNAUTHENTICATED DOES NOT MEAN PRIVATE TO THIS PROCESS.** The socket is
/// bound on loopback, which keeps it off the network and reachable by EVERY
/// process on this machine -- any local program that can guess or read the
/// port can drive the model. `apiKey` is the only access control this server
/// has; there is no second gate behind it. This comment used to say the
/// server was "reachable only from inside this process", which is not a
/// property a TCP socket can have and is the wrong basis for deciding
/// against setting a key.
public struct ServerOptions: Encodable, Sendable {
    /// 0 (the default) asks the OS for an ephemeral port; read the port
    /// ACTUALLY bound back from `TurboSparkServer.info()`.
    public var port: UInt16
    /// Require this key on every request except `GET /health`, matching
    /// `turbospark-server --api-key`. `nil` (the default) leaves the server
    /// unauthenticated.
    public var apiKey: String?

    public init(port: UInt16 = 0, apiKey: String? = nil) {
        self.port = port
        self.apiKey = apiKey
    }
}

/// What `TurboSparkServer.info()` reports.
public struct ServerInfo: Decodable, Sendable, Equatable {
    /// The port ACTUALLY bound, never the one requested: `ServerOptions.port
    /// == 0` asks the OS to choose one, so this is the only place that
    /// number is knowable.
    public let port: UInt16
    /// The IP ACTUALLY bound, from the same `local_addr` read the port comes
    /// from. Build a URL out of this rather than restating `127.0.0.1`: the
    /// literal is what this library binds TODAY, and a caller spelling it by
    /// hand has no way to notice the day that changes.
    public let host: String
    /// The FIRST attached model, or `""` when none is.
    ///
    /// **Show `models` instead.** This field exists for a reader written when
    /// a server could serve only one, and on a two-model server it is half
    /// the truth. It is kept rather than removed because removing it would
    /// break that reader silently, which is the failure it is here to avoid.
    public let modelId: String
    /// Every attached model, in attachment order -- the same ids and the same
    /// order `GET /v1/models` reports, because both read one registry.
    public let models: [String]
    public let authEnabled: Bool
    /// Seconds since the server started, from a monotonic clock.
    public let uptimeSeconds: UInt64

    /// Spelled out because a hand-written `init(from:)` suppresses the
    /// synthesized one.
    private enum CodingKeys: String, CodingKey {
        case port, host, modelId, models, authEnabled, uptimeSeconds
    }

    /// Decoded tolerantly for the two fields added after this struct
    /// shipped, so a binding built against a newer engine and run against an
    /// older archive still decodes rather than throwing and losing the whole
    /// struct (`swift/CLAUDE.md` Gotcha 13's rule, applied at the ABI rather
    /// than at the on-disk store).
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        port = try c.decode(UInt16.self, forKey: .port)
        host = try c.decode(String.self, forKey: .host)
        modelId = try c.decode(String.self, forKey: .modelId)
        models = try c.decodeIfPresent([String].self, forKey: .models)
            ?? (modelId.isEmpty ? [] : [modelId])
        authEnabled = try c.decode(Bool.self, forKey: .authEnabled)
        uptimeSeconds = try c.decodeIfPresent(UInt64.self, forKey: .uptimeSeconds) ?? 0
    }

    /// `http://host:port`, the address a client should actually call.
    ///
    /// The host is bracketed when it contains a `:`, which is how an IPv6
    /// literal has to appear in a URL. The engine binds IPv4 today, so that
    /// branch is unreachable and costs one line -- the point is that this
    /// helper stays correct without one, rather than becoming the next thing
    /// that is true only by coincidence.
    public var baseURL: URL? {
        let literal = host.contains(":") ? "[\(host)]" : host
        return URL(string: "http://\(literal):\(port)")
    }
}

/// Something a running server did, drained through `TurboSparkServer.poll`.
///
/// **`requested` AND `served` DIFFER WHENEVER THE SINGLE-MODEL FALLBACK
/// FIRED**, which is the common case rather than an edge one: a client
/// sending its own default model name (Claude Code sends
/// `claude-sonnet-4-6`) is served by whatever single model is attached. A
/// console that showed one of them would hide the fact somebody debugging
/// routing is looking for.
public enum ServerEvent: Decodable, Sendable, Equatable {
    case requestStarted(id: UInt64, atMs: UInt64, method: String, path: String)
    case requestRouted(id: UInt64, requested: String?, served: String, stream: Bool)
    case generated(
        id: UInt64, model: String, promptTokens: UInt32, newTokens: UInt32,
        prefillSeconds: Double, decodeSeconds: Double, stopReason: String)
    case requestFinished(id: UInt64, status: UInt16, durationMs: UInt32)
    case modelAttached(atMs: UInt64, model: String)
    case modelDetached(atMs: UInt64, model: String)
    /// A `kind` this binding does not know, carried rather than thrown.
    ///
    /// A newer engine adding an event kind must not make a whole poll fail
    /// to decode and take the KNOWN events down with it -- the console would
    /// go blank rather than showing one unrecognized row.
    case unknown(kind: String)

    /// The request these events belong to, when they belong to one.
    /// `modelAttached` and `modelDetached` are properties of the server
    /// rather than of any request, and carry none.
    public var requestID: UInt64? {
        switch self {
        case let .requestStarted(id, _, _, _): return id
        case let .requestRouted(id, _, _, _): return id
        case let .generated(id, _, _, _, _, _, _): return id
        case let .requestFinished(id, _, _): return id
        case .modelAttached, .modelDetached, .unknown: return nil
        }
    }

    private enum CodingKeys: String, CodingKey {
        case kind, id, atMs, method, path, requested, served, stream
        case model, promptTokens, newTokens, prefillSeconds, decodeSeconds, stopReason
        case status, durationMs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try c.decode(String.self, forKey: .kind)
        switch kind {
        case "requestStarted":
            self = .requestStarted(
                id: try c.decode(UInt64.self, forKey: .id),
                atMs: try c.decode(UInt64.self, forKey: .atMs),
                method: try c.decode(String.self, forKey: .method),
                path: try c.decode(String.self, forKey: .path))
        case "requestRouted":
            self = .requestRouted(
                id: try c.decode(UInt64.self, forKey: .id),
                requested: try c.decodeIfPresent(String.self, forKey: .requested),
                served: try c.decode(String.self, forKey: .served),
                stream: try c.decode(Bool.self, forKey: .stream))
        case "generated":
            self = .generated(
                id: try c.decode(UInt64.self, forKey: .id),
                model: try c.decode(String.self, forKey: .model),
                promptTokens: try c.decode(UInt32.self, forKey: .promptTokens),
                newTokens: try c.decode(UInt32.self, forKey: .newTokens),
                prefillSeconds: try c.decode(Double.self, forKey: .prefillSeconds),
                decodeSeconds: try c.decode(Double.self, forKey: .decodeSeconds),
                stopReason: try c.decode(String.self, forKey: .stopReason))
        case "requestFinished":
            self = .requestFinished(
                id: try c.decode(UInt64.self, forKey: .id),
                status: try c.decode(UInt16.self, forKey: .status),
                durationMs: try c.decode(UInt32.self, forKey: .durationMs))
        case "modelAttached":
            self = .modelAttached(
                atMs: try c.decode(UInt64.self, forKey: .atMs),
                model: try c.decode(String.self, forKey: .model))
        case "modelDetached":
            self = .modelDetached(
                atMs: try c.decode(UInt64.self, forKey: .atMs),
                model: try c.decode(String.self, forKey: .model))
        default:
            self = .unknown(kind: kind)
        }
    }
}

/// One drain: what happened, and what was lost.
public struct ServerEventBatch: Decodable, Sendable, Equatable {
    public let events: [ServerEvent]
    /// Events the engine's ring discarded since the PREVIOUS poll, oldest
    /// first. Zero for any host polling on a timer.
    ///
    /// **SHOW IT.** A console that quietly drops rows reads as "nothing
    /// happened in that window", which is the same thing it reads as when
    /// the server genuinely was idle -- and those are the two states someone
    /// watching it is trying to tell apart.
    public let dropped: UInt64

    /// Public so a consumer can build one, which is what lets a host's own
    /// aggregation be tested against synthetic traffic rather than only
    /// against a real bound socket. The synthesized memberwise init is
    /// internal to this module and would not cross the package boundary.
    public init(events: [ServerEvent], dropped: UInt64) {
        self.events = events
        self.dropped = dropped
    }
}

/// A background in-process HTTP server, started from `TurboSparkSession
/// .startServer(options:)` or `TurboSparkServer.start(options:)`.
///
/// **THE MODEL SET IS MUTABLE WHILE IT RUNS.** A server can start with
/// nothing attached and gain models through `attach(_:)`, which is what a
/// GUI does: the socket binds before its user has chosen anything to load.
/// Attaching and detaching take effect immediately, with no rebind and no
/// interruption to a request already in flight on another model.
///
/// **DETACHING IS WHAT RELEASES A MODEL.** Every attached entry holds the
/// engine alive independently of the `TurboSparkSession` it came from, so
/// letting that session go does NOT free the weights, the KV cache or the
/// compiled pipelines -- `detach(modelId:)` or `stop()` does. A host that
/// clears a session without one of those has a multi-gigabyte mapping
/// resident with nothing in its own UI still showing the model as loaded.
///
/// **A CLASS WITH A LOCK RATHER THAN AN `actor`**, for the same reason
/// `TurboSparkSession` is: `stop()` has to be callable from any thread
/// without suspending behind whatever request the server is mid-way through
/// serving, and it has to be safe to call more than once (from a caller AND
/// from `deinit`) without double-freeing the underlying C handle -- the C
/// layer itself has no way to tell a second `ts_server_stop` on the same
/// pointer apart from the first, so this type is what keeps that invariant.
public final class TurboSparkServer: @unchecked Sendable {
    private struct Handle: @unchecked Sendable { let raw: OpaquePointer }
    private let handle: Handle
    private let lock = NSLock()
    private var stopped = false

    init(raw: OpaquePointer) {
        self.handle = Handle(raw: raw)
    }

    /// Starts a server with NO model attached.
    ///
    /// The socket binds and `GET /health` answers (reporting
    /// `"state": "empty"`); every generation route returns 503 until
    /// `attach(_:)` adds one. That is the order a GUI wants -- the address
    /// exists, and can be shown and copied, before the user has decided what
    /// to load.
    public static func start(options: ServerOptions = ServerOptions()) throws -> TurboSparkServer {
        let optionsJSON = try JSONEncoder().encode(options)
        let json = String(decoding: optionsJSON, as: UTF8.self)
        var out: OpaquePointer?
        let status = json.withCString { opts in
            ts_server_start(nil, opts, &out)
        }
        guard status == 0, let out else {
            throw TurboSparkError.fromLastError(status)
        }
        return TurboSparkServer(raw: out)
    }

    /// Adds an open session's model, returning the id clients address it by
    /// (the install directory's own name).
    ///
    /// Throws if a model with that id is already attached. That refusal is
    /// deliberate rather than a limitation: the id is what a request's
    /// `model` field names and what `detach(modelId:)` keys on, so a
    /// silently renamed second copy would be reachable under a name the
    /// caller never learned.
    ///
    /// From here the server holds the engine alive on its own; see the
    /// type's own note on what that obliges a host to do.
    @discardableResult
    public func attach(_ session: TurboSparkSession) throws -> String {
        lock.lock()
        defer { lock.unlock() }
        try checkRunning()
        return try session.withRawHandle { sessionHandle in
            try takeString { ts_server_attach_session(handle.raw, sessionHandle, $0) }
        }
    }

    /// Removes a model by id, releasing this server's reference to its
    /// engine. Throws if nothing was attached under that id.
    ///
    /// **THIS IS WHAT ACTUALLY FREES THE MODEL.** Releasing the
    /// `TurboSparkSession` does not: the server holds its own reference.
    public func detach(modelId: String) throws {
        lock.lock()
        defer { lock.unlock() }
        try checkRunning()
        try modelId.withCString { id in
            try check(ts_server_detach_model(handle.raw, id))
        }
    }

    /// Takes up to `max` buffered events. Each is returned exactly once, so
    /// a caller polling on a timer appends what it gets.
    ///
    /// `max: 0` means no bound, which is what a final drain before shutdown
    /// wants. Anything over the bound stays queued rather than being
    /// discarded, so a burst arrives late and never silently short.
    ///
    /// Returns an empty batch rather than throwing once the server has been
    /// stopped: a poll timer and a Stop button race by nature, and a host
    /// should not have to catch an error for the ordinary case of one
    /// arriving a tick late.
    public func poll(max: UInt32 = 256) -> ServerEventBatch {
        lock.lock()
        defer { lock.unlock() }
        guard !stopped else { return ServerEventBatch(events: [], dropped: 0) }
        guard
            let json = try? takeString({ ts_server_poll_events_json(handle.raw, max, $0) }),
            let batch = try? decode(ServerEventBatch.self, from: json)
        else {
            return ServerEventBatch(events: [], dropped: 0)
        }
        return batch
    }

    /// Callers hold `lock` already.
    private func checkRunning() throws {
        guard !stopped else {
            throw TurboSparkError(
                code: .invalidArgument,
                message: "this server has already been stopped"
            )
        }
    }

    deinit {
        stopLocked()
    }

    /// Signals the server to stop and blocks until its background thread has
    /// actually exited. Idempotent and safe from any thread; a second call
    /// (or a call after `deinit` would have run anyway) is a no-op.
    public func stop() {
        stopLocked()
    }

    private func stopLocked() {
        lock.lock()
        defer { lock.unlock() }
        guard !stopped else { return }
        stopped = true
        ts_server_stop(handle.raw)
    }

    /// `{ port, modelId, authEnabled }`, read fresh on every call.
    ///
    /// **The lock is held ACROSS the C call, not just across the `stopped`
    /// check.** `ts_server_stop` frees the handle, so releasing the lock
    /// between reading the flag and using the pointer leaves a window in
    /// which another thread's `stop()` (or a `deinit` on any thread) frees
    /// what this call is about to dereference. That window is exactly what
    /// this type exists to close, and checking-then-unlocking reads as
    /// careful while reintroducing it. The call is a JSON snapshot of three
    /// fields, so holding the lock costs a caller nothing.
    public func info() throws -> ServerInfo {
        lock.lock()
        defer { lock.unlock() }
        try checkRunning()
        return try decode(
            ServerInfo.self,
            from: try takeString { ts_server_info_json(handle.raw, $0) }
        )
    }
}
