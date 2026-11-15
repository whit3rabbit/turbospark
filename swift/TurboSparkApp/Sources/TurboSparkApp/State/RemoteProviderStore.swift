import Foundation

/// Persisted shape of the provider list. No secrets: `RemoteProvider` has no
/// key field, and `RemoteProviderKeyStore` owns every key.
struct RemoteProviderArchive: Codable, Equatable {
    var providers: [RemoteProvider] = []
}

/// Per-chat consent that content may leave the Mac.
///
/// Remote use is explicit and off by default: the first send to a provider
/// in a chat, and a switch from local to remote mid-chat, wait for the
/// person's confirmation naming the provider host. The ledger is in memory
/// and per chat, so a relaunch asks again.
struct RemoteConsentLedger: Equatable {
    private var confirmed: Set<String> = []

    private func token(_ provider: UUID, _ chat: UUID) -> String { "\(provider)|\(chat)" }

    func isConfirmed(provider: UUID, chat: UUID) -> Bool {
        confirmed.contains(token(provider, chat))
    }

    mutating func confirm(provider: UUID, chat: UUID) {
        confirmed.insert(token(provider, chat))
    }

    mutating func revoke(provider: UUID) {
        confirmed = confirmed.filter { !$0.hasPrefix("\(provider)|") }
    }
}

/// What the turn loop must do before a remote send.
enum RemoteSendDecision: Equatable {
    case proceed
    /// Show the notice ("prompts and tool results leave this machine") naming
    /// `host`, and send only after the person confirms.
    case needsConfirmation(host: String)
    case refused(RemoteProviderError)
}

/// The provider and model configuration. Local stays the default
/// everywhere: with no provider there is no Remote section and nothing here
/// runs.
@MainActor
final class RemoteProviderStore: ObservableObject {
    static let fileName = "remote_providers.json"

    @Published private(set) var providers: [RemoteProvider]
    @Published private(set) var consent = RemoteConsentLedger()
    let keys: RemoteProviderKeyStore
    private let fileURL: URL?

    /// Loads the shape file only. The Keychain is not consulted.
    init(fileURL: URL? = AppStorageRoot.file(RemoteProviderStore.fileName),
         keys: RemoteProviderKeyStore = RemoteProviderKeyStore()) {
        self.fileURL = fileURL
        self.keys = keys
        if let fileURL,
            let archive = AppJSONStore.load(RemoteProviderArchive.self, from: fileURL, label: "Remote providers")
        {
            providers = archive.providers
        } else {
            providers = []
        }
    }

    private func persist() {
        guard let fileURL else { return }
        AppJSONStore.save(RemoteProviderArchive(providers: providers), to: fileURL, label: "Remote providers")
    }

    func provider(id: UUID) -> RemoteProvider? { providers.first { $0.id == id } }

    /// Adds a provider after validating its base URL. A provider is created
    /// disabled-by-key: until a key is held it is not usable.
    @discardableResult
    func add(name: String, family: RemoteAPIFamily, baseURL: String) throws -> RemoteProvider {
        _ = try RemoteEndpointPolicy.validate(baseURL: baseURL)
        let provider = RemoteProvider(
            id: UUID(), name: name.trimmingCharacters(in: .whitespacesAndNewlines),
            family: family, baseURL: baseURL.trimmingCharacters(in: .whitespacesAndNewlines))
        providers.append(provider)
        persist()
        return provider
    }

    /// The family is fixed at creation, so it is not editable here.
    func update(_ id: UUID, name: String? = nil, baseURL: String? = nil, enabled: Bool? = nil) throws {
        guard let i = providers.firstIndex(where: { $0.id == id }) else { return }
        if let baseURL {
            _ = try RemoteEndpointPolicy.validate(baseURL: baseURL)
            providers[i].baseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let name { providers[i].name = name }
        if let enabled { providers[i].enabled = enabled }
        persist()
    }

    func setModels(_ models: [RemoteProviderModel], for id: UUID) {
        guard let i = providers.firstIndex(where: { $0.id == id }) else { return }
        providers[i].models = models
        persist()
    }

    /// Removes the provider, its Keychain item and its consent.
    func remove(_ id: UUID) {
        providers.removeAll { $0.id == id }
        keys.forget(for: id)
        consent.revoke(provider: id)
        persist()
    }

    /// Enabled models of usable providers, for the composer's Remote section.
    /// Hidden entirely when there are none. Asks the key store by attributes
    /// only, so building the picker never prompts.
    func pickerEntries() -> [(provider: RemoteProvider, model: RemoteProviderModel)] {
        providers.filter { $0.enabled && keys.isKeyAvailable(for: $0.id) }
            .flatMap { p in p.models.filter(\.enabled).map { (p, $0) } }
    }

    func confirmConsent(provider: UUID, chat: UUID) {
        consent.confirm(provider: provider, chat: chat)
    }

    /// The gate every remote send passes first. Pure over the store's state.
    func sendDecision(provider id: UUID, model modelID: String, chat: UUID) -> RemoteSendDecision {
        guard let provider = provider(id: id) else { return .refused(.notConfigured) }
        guard provider.enabled, provider.models.contains(where: { $0.id == modelID && $0.enabled }) else {
            return .refused(.disabled)
        }
        let url: URL
        do { url = try RemoteEndpointPolicy.validate(baseURL: provider.baseURL) } catch let error as RemoteProviderError {
            return .refused(error)
        } catch { return .refused(.invalidBaseURL("unreadable")) }
        guard keys.isKeyAvailable(for: id) else { return .refused(.notConfigured) }
        guard consent.isConfirmed(provider: id, chat: chat) else {
            return .needsConfirmation(host: url.host ?? provider.baseURL)
        }
        return .proceed
    }
}
