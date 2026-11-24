import Foundation

/// What the MCP Servers pane says about one server's connection.
///
/// Derived only from events the app already handles: the enable switch, an
/// explicit connection test, and the catalog cache's background `tools/list`
/// discovery. Nothing here opens a connection or polls. "Connected" means the
/// last handshake and `tools/list` succeeded for the current transport, not
/// that a process is alive right now: stdio sessions exit when idle and
/// respawn on the next call, so a live-process flag would flap.
enum McpServerRuntimeStatus: Equatable {
    case disabled
    case connecting
    case connected(toolCount: Int)
    case failed(reason: String)
    /// Enabled, but no discovery has finished for the current transport.
    case notConnected

    /// Precedence: the switch wins (a disabled server is never advertised),
    /// then work in flight, then the latest finished outcome. A success
    /// clears the recorded failure in the cache, so a failure that is still
    /// present is the newest result.
    static func resolve(
        isEnabled: Bool,
        isTesting: Bool,
        isDiscovering: Bool,
        cachedToolCount: Int?,
        lastFailure: String?
    ) -> McpServerRuntimeStatus {
        guard isEnabled else { return .disabled }
        if isTesting || isDiscovering { return .connecting }
        if let lastFailure {
            let reason = lastFailure.trimmingCharacters(in: .whitespacesAndNewlines)
            return .failed(reason: reason)
        }
        if let cachedToolCount { return .connected(toolCount: cachedToolCount) }
        return .notConnected
    }
}
