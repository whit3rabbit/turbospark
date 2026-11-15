import Foundation

/// The one entry point a turn takes to reach a remote provider. It applies
/// every gate in order and only then hands the key to a family client:
///
/// 1. the send decision (provider and model enabled, base URL valid, key
///    available, this chat confirmed that content leaves the Mac);
/// 2. the key read (Keychain only here, only for an opted-in credential);
/// 3. the shared runner (TLS, redirect, size and time caps).
///
/// A refusal at any step ends the turn with a named `RemoteProviderError`
/// and sends nothing.
@MainActor
struct RemoteProviderService {
    let store: RemoteProviderStore
    var transport: any RemoteHTTPTransport = URLSessionRemoteTransport()
    var limits: RemoteLimits = .standard

    static func client(for family: RemoteAPIFamily) -> any RemoteFamilyClient {
        switch family {
        case .anthropic: return AnthropicClient()
        case .openAICompatible: return OpenAICompatClient()
        }
    }

    func stream(
        providerID: UUID, chat: UUID, request: RemoteChatRequest
    ) throws -> AsyncThrowingStream<RemoteStepEvent, Error> {
        switch store.sendDecision(provider: providerID, model: request.model, chat: chat) {
        case .refused(let error): throw error
        case .needsConfirmation: throw RemoteProviderError.notConfigured
        case .proceed: break
        }
        guard let provider = store.provider(id: providerID) else { throw RemoteProviderError.notConfigured }
        let base = try RemoteEndpointPolicy.validate(baseURL: provider.baseURL)
        let key = try store.keys.keyForSend(provider: providerID)
        return RemoteStreamRunner(transport: transport, limits: limits).stream(
            client: Self.client(for: provider.family), request: request, baseURL: base, key: key)
    }

    /// The Test connection button. Needs no chat consent: it sends no
    /// conversation, only the key to the host the user configured, on an
    /// explicit click.
    func testConnection(providerID: UUID) async throws {
        guard let provider = store.provider(id: providerID) else { throw RemoteProviderError.notConfigured }
        let base = try RemoteEndpointPolicy.validate(baseURL: provider.baseURL)
        let key = try store.keys.keyForSend(provider: providerID)
        try await RemoteStreamRunner(transport: transport, limits: limits).probe(
            client: Self.client(for: provider.family), baseURL: base, key: key,
            model: provider.models.first(where: \.enabled)?.id)
    }
}
