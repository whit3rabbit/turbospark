import CTurboSpark
import Foundation

/// What port and auth an in-process server starts with. `{}` (every field
/// default) asks the OS to choose a port and starts unauthenticated --
/// appropriate for a server bound to loopback and reachable only from
/// inside this process.
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
    public let modelId: String
    public let authEnabled: Bool
}

/// A background in-process HTTP server, started from `TurboSparkSession
/// .startServer(options:)` and sharing that session's engine.
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
    public func info() throws -> ServerInfo {
        lock.lock()
        let alreadyStopped = stopped
        lock.unlock()
        guard !alreadyStopped else {
            throw TurboSparkError(
                code: .invalidArgument,
                message: "this server has already been stopped"
            )
        }
        return try decode(
            ServerInfo.self,
            from: try takeString { ts_server_info_json(handle.raw, $0) }
        )
    }
}
