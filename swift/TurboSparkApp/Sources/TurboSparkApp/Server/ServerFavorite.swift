import Foundation

/// Server recipes belong to the existing per-user settings store. No keys
/// or captured traffic are serialized with a favorite.
public struct ServerFavorite: Codable, Identifiable, Equatable, Sendable {
    public var id = UUID()
    public var name: String
    public var host: String
    public var port: UInt16
    public var modelPaths: [String]
    public var contextTokens: Int
}

extension AppModel {
    func saveServerFavorite(name: String) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        let ids = Set(serverInfo?.models ?? [])
        let paths = installed.filter { ids.contains(Self.servedModelID(for: $0)) }.map(\.path)
        serverFavorites.append(ServerFavorite(name: name, host: serverHost,
            port: serverPinnedPort, modelPaths: paths, contextTokens: maxContextTokens))
        persistSettingsDebounced()
    }

    func loadServerFavorite(_ favorite: ServerFavorite) {
        guard server == nil, !serverBusy else { return }
        serverHost = favorite.host
        serverPinnedPort = favorite.port
        serverPortIsValid = true
        maxContextTokens = min(max(0, favorite.contextTokens), Int(UInt32.max))
        persistSettingsDebounced()
        serverBusy = true
        Task {
            await performServerStart(attachChat: false)
            guard let started = server else { return }
            for path in favorite.modelPaths {
                guard server === started else { return }
                if serverInfo?.models.contains((path as NSString).lastPathComponent) == true { continue }
                guard let candidate = installed.first(where: { $0.path == path }) else {
                    showToast("Model is no longer installed: \(path)", style: .warning)
                    continue
                }
                await attachServerModelAndWait(candidate)
            }
        }
    }
}
