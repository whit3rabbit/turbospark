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
    private var inFlight: Set<String> = []

    public init() {}

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
        lock.lock(); defer { lock.unlock() }
        entries[McpServerConfig.normalizedName(config.name)] = Entry(tools: tools, transport: config.transport)
    }

    public func removeServer(named serverName: String) {
        lock.lock(); defer { lock.unlock() }
        let key = McpServerConfig.normalizedName(serverName)
        entries.removeValue(forKey: key)
        inFlight.remove(key)
    }

    public func removeAll() {
        lock.lock(); defer { lock.unlock() }
        entries.removeAll()
        inFlight.removeAll()
    }

    /// Kicks off background discovery for every ENABLED server in `servers`
    /// whose cache entry is missing or stale, and prunes entries for names
    /// no longer present. Returns the server names queued.
    ///
    /// Safe to call repeatedly: an in-flight server is skipped, a fresh
    /// entry is skipped, and each name runs at most one discovery at a
    /// time. Failures leave any previous entry in place and simply clear
    /// the in-flight mark, so the next event retries.
    @discardableResult
    public func refreshEnabled(servers: [McpServerConfig], workingDirectory: URL?) -> [String] {
        let enabled = servers.filter { $0.isEnabled }
        var queued: [String] = []

        lock.lock()
        for config in enabled {
            let key = McpServerConfig.normalizedName(config.name)
            guard !inFlight.contains(key), isStaleLocked(for: config) else { continue }
            inFlight.insert(key)
            queued.append(config.name)
        }
        // Drop cache rows for servers that no longer exist or were disabled,
        // so a disabled server's tools stop being advertised immediately.
        let liveKeys = Set(enabled.map { McpServerConfig.normalizedName($0.name) })
        for key in entries.keys where !liveKeys.contains(key) {
            entries.removeValue(forKey: key)
        }
        lock.unlock()

        for config in queued.compactMap({ name in enabled.first(where: { $0.name == name }) }) {
            let snapshot = config
            Task { [weak self] in
                defer { self?.clearInFlight(named: snapshot.name) }
                guard let tools = try? await McpClientEngine.shared.discoverTools(
                    for: snapshot, workingDirectory: workingDirectory)
                else { return }
                self?.setTools(tools, for: snapshot)
            }
        }
        return queued
    }

    /// `isStale` for a caller already holding `lock`.
    private func isStaleLocked(for config: McpServerConfig) -> Bool {
        guard let entry = entries[McpServerConfig.normalizedName(config.name)] else { return true }
        return entry.transport != config.transport
    }

    private func clearInFlight(named serverName: String) {
        lock.lock(); defer { lock.unlock() }
        inFlight.remove(McpServerConfig.normalizedName(serverName))
    }
}
