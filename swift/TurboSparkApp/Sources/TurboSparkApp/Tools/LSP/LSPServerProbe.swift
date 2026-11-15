import Foundation

/// OC-10: the Settings "Test" action. It starts the configured server in a
/// fresh EMPTY temporary folder (never a user project), completes the
/// initialize handshake, and shuts it down. The same launch policy applies as
/// for real use, so a row that tests fine launches the same way later.
enum LSPServerProbe {
    static func test(
        _ config: LSPServerConfig,
        transportFactory: LanguageServerManager.TransportFactory = LSPProductionTransport.make
    ) async -> Result<String, Error> {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-lsp-probe-\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: folder, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        } catch { return .failure(error) }
        defer { try? FileManager.default.removeItem(at: folder) }
        let root = PathContainment.canonical(folder)
        do {
            let transport = try await transportFactory(config, root)
            let client = LSPClient(transport: transport, rootURL: root)
            // The initialize timeout (15 s) bounds a hung server.
            try await client.start()
            let name = await client.serverName ?? config.name
            await client.shutdown()
            return .success(name)
        } catch {
            return .failure(error)
        }
    }
}
