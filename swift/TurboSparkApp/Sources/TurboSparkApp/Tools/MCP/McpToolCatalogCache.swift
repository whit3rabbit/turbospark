import Foundation

/// Process-wide cache of MCP `tools/list` results, read synchronously by
/// catalog and prompt assembly.
///
/// Why a cache at all: discovery spawns the server process and pays a full
/// JSON-RPC handshake (`McpClientEngine.discoverTools`), which takes
/// seconds. A turn must not pay that inline, so `AppToolCatalog` reads
/// whatever is cached and background refreshes keep it warm. A stale entry
/// is safe by construction: a tool advertised but since removed from the
/// server fails the call server-side with an honest "unknown tool" error
/// the model can see and report.
///
/// Entries are keyed by normalized server name -- the same identity key
/// `AppToolRegistry.executeMcpCall` and `AppToolPermissionEngine` resolve
/// by -- and invalidated by transport fingerprint, so editing a server's
/// command or URL re-discovers on the next refresh. There is no TTL on
/// purpose: refreshes are event-driven (project selection, server edits,
/// explicit refresh), never on a timer that spawns processes while the
/// user is reading.
public final class McpToolCatalogCache: @unchecked Sendable {
    public static let shared = McpToolCatalogCache()

    private struct Entry {
        var tools: [McpDiscoveredTool]
        var transport: McpTransportSpec
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private struct Request {
        let id: UUID
        let transport: McpTransportSpec
    }
    private var inFlight: [String: Request] = [:]
    /// The newest discovery failure per server, kept beside (not instead of)
    /// `entries` so a failed refresh still re-queues exactly as before and an
    /// older tools list stays advertised. Read only by the settings status.
    private struct Failure {
        var message: String
        var transport: McpTransportSpec
    }
    private var failures: [String: Failure] = [:]

    /// Posted (object: the cache) after any change the status UI reads:
    /// discovery start, success, failure, removal. Lets the pane redraw on
    /// events instead of polling. May arrive on any thread.
    public static let didChangeNotification = Notification.Name("McpToolCatalogCacheDidChange")
    public typealias Discovery = @Sendable (McpServerConfig, URL?) async throws -> [McpDiscoveredTool]
    private let discover: Discovery

    public init(discover: @escaping Discovery = { config, directory in
        try await McpClientEngine.shared.discoverTools(for: config, workingDirectory: directory)
    }) {
        self.discover = discover
    }

    /// The cached tools for `serverName`, or nil when never discovered.
    public func tools(forServerName serverName: String) -> [McpDiscoveredTool]? {
        lock.lock(); defer { lock.unlock() }
        return entries[McpServerConfig.normalizedName(serverName)]?.tools
    }

    /// Whether `config` has no cached tools yet or its transport changed
    /// since they were cached.
    public func isStale(for config: McpServerConfig) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard let entry = entries[McpServerConfig.normalizedName(config.name)] else { return true }
        return entry.transport != config.transport
    }

    public func setTools(_ tools: [McpDiscoveredTool], for config: McpServerConfig) {
        lock.lock()
        let key = McpServerConfig.normalizedName(config.name)
        inFlight.removeValue(forKey: key)
        entries[key] = Entry(tools: tools, transport: config.transport)
        failures.removeValue(forKey: key)
        lock.unlock()
        postChange()
    }

    /// Records a failed explicit connection test so the status reads Failed
    /// until the next success. Leaves cached tools in place: advertising is
    /// unchanged by a test, exactly as before.
    public func recordFailure(_ message: String, for config: McpServerConfig) {
        lock.lock()
        failures[McpServerConfig.normalizedName(config.name)] =
            Failure(message: message, transport: config.transport)
        lock.unlock()
        postChange()
    }

    /// The newest failure message for `config`, or nil when the last finished
    /// attempt succeeded or the failure was for a different transport.
    public func lastFailure(for config: McpServerConfig) -> String? {
        lock.lock(); defer { lock.unlock() }
        guard let failure = failures[McpServerConfig.normalizedName(config.name)],
            failure.transport == config.transport
        else { return nil }
        return failure.message
    }

    /// Whether a background discovery for `config`'s current transport is
    /// in flight.
    public func isDiscovering(_ config: McpServerConfig) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return inFlight[McpServerConfig.normalizedName(config.name)]?.transport == config.transport
    }

    /// The cached tool count for `config`, or nil when none is cached for
    /// its current transport.
    public func cachedToolCount(for config: McpServerConfig) -> Int? {
        lock.lock(); defer { lock.unlock() }
        guard isStaleLocked(for: config) == false else { return nil }
        return entries[McpServerConfig.normalizedName(config.name)]?.tools.count
    }

    private func postChange() {
        NotificationCenter.default.post(name: Self.didChangeNotification, object: self)
    }

    public func removeServer(named serverName: String) {
        lock.lock()
        let key = McpServerConfig.normalizedName(serverName)
        entries.removeValue(forKey: key)
        inFlight.removeValue(forKey: key)
        failures.removeValue(forKey: key)
        // Removal, rename and disable all land here: the persistent session
        // for that server must not keep running under a config that is gone.
        Task { await McpClientEngine.shared.invalidateSessions(serverName: serverName) }
        lock.unlock()
        postChange()
    }

    public func removeAll() {
        lock.lock()
        entries.removeAll()
        inFlight.removeAll()
        failures.removeAll()
        lock.unlock()
        postChange()
    }

    /// Kicks off background discovery for every ENABLED server in `servers`
    /// whose cache entry is missing or stale, and prunes entries for names
    /// no longer present. Returns the server names queued.
    ///
    /// Repeated calls share the current request. Reconfiguration invalidates
    /// its identity so a late completion cannot publish or clear newer work.
    @discardableResult
    public func refreshEnabled(servers: [McpServerConfig], workingDirectory: URL?) -> [String] {
        refreshTasks(servers: servers, workingDirectory: workingDirectory).map(\.name)
    }

    // Returning handles lets tests await publication, not just the discovery
    // callback. Production callers retain the fire-and-forget interface.
    func refreshTasks(servers: [McpServerConfig], workingDirectory: URL?)
        -> [(name: String, task: Task<Void, Never>)] {
        let enabled = servers.filter { $0.isEnabled }
        var queued: [(config: McpServerConfig, id: UUID)] = []
        lock.lock()
        let liveKeys = Set(enabled.map { McpServerConfig.normalizedName($0.name) })
        entries = entries.filter { liveKeys.contains($0.key) }
        inFlight = inFlight.filter { liveKeys.contains($0.key) }
        failures = failures.filter { liveKeys.contains($0.key) }
        for config in enabled {
            let key = McpServerConfig.normalizedName(config.name)
            if let request = inFlight[key] {
                if request.transport == config.transport { continue }
                inFlight.removeValue(forKey: key)
            }
            guard isStaleLocked(for: config) else { continue }
            let id = UUID()
            inFlight[key] = Request(id: id, transport: config.transport)
            queued.append((config, id))
        }
        lock.unlock()
        // Pruning or queueing can both change what the status shows.
        postChange()

        return queued.map { config, id in
            let discover = self.discover
            let task = Task { [weak self] in
                let outcome: Result<[McpDiscoveredTool], Error>
                do {
                    outcome = .success(try await discover(config, workingDirectory))
                } catch {
                    outcome = .failure(error)
                }
                self?.complete(config: config, id: id, outcome: outcome)
            }
            return (config.name, task)
        }
    }

    /// `isStale` for a caller already holding `lock`.
    private func isStaleLocked(for config: McpServerConfig) -> Bool {
        guard let entry = entries[McpServerConfig.normalizedName(config.name)] else { return true }
        return entry.transport != config.transport
    }

    private func complete(
        config: McpServerConfig, id: UUID, outcome: Result<[McpDiscoveredTool], Error>
    ) {
        lock.lock()
        let key = McpServerConfig.normalizedName(config.name)
        // Reset, removal, or replacement revokes both publication and cleanup.
        guard inFlight[key]?.id == id else {
            lock.unlock()
            return
        }
        inFlight.removeValue(forKey: key)
        switch outcome {
        case .success(let tools):
            entries[key] = Entry(tools: tools, transport: config.transport)
            failures.removeValue(forKey: key)
        case .failure(let error):
            // Entries are untouched, as with the old `try?`: a failure keeps
            // the server stale so the next refresh retries it.
            failures[key] = Failure(message: error.localizedDescription, transport: config.transport)
        }
        lock.unlock()
        postChange()
    }
}
