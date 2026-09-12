import Foundation

public struct McpTransportSpec: Equatable, Sendable { let command: String }
public struct McpDiscoveredTool: Sendable { let name: String }
public struct McpServerConfig: Sendable {
    let name: String
    let isEnabled: Bool
    let transport: McpTransportSpec
    static func normalizedName(_ name: String) -> String { name.lowercased() }
}
public actor McpClientEngine {
    public static let shared = McpClientEngine()
    private var pending: CheckedContinuation<[McpDiscoveredTool], Never>?
    public func discoverTools(for config: McpServerConfig, workingDirectory: URL?) async throws -> [McpDiscoveredTool] {
        await withCheckedContinuation { pending = $0 }
    }
    func hasStarted() -> Bool { pending != nil }
    func complete() { pending?.resume(returning: [McpDiscoveredTool(name: "stale")]); pending = nil }
}

@main struct Audit {
    static func main() async {
        let cache = McpToolCatalogCache()
        let server = McpServerConfig(name: "fixture", isEnabled: true, transport: .init(command: "fixture"))
        let tasks = cache.refreshTasks(servers: [server], workingDirectory: nil)
        precondition(tasks.map(\.name) == ["fixture"])
        while !(await McpClientEngine.shared.hasStarted()) { await Task.yield() }
        cache.removeAll()
        precondition(cache.tools(forServerName: "fixture") == nil)
        await McpClientEngine.shared.complete()
        await tasks[0].task.value
        precondition(cache.tools(forServerName: "fixture") == nil)
        print("PASS: in-flight discovery cannot repopulate cache after removeAll")
    }
}
