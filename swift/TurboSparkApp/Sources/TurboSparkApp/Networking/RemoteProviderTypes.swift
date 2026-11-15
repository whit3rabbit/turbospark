import Foundation

/// The wire protocol a remote provider speaks. Persisted as the raw value.
/// Fixed when the provider is created: changing family means adding a new
/// provider (ZP-22).
enum RemoteAPIFamily: String, Codable, CaseIterable, Sendable {
    case anthropic
    case openAICompatible = "openai-compatible"
}

/// One model a provider serves, as the user entered it.
struct RemoteProviderModel: Codable, Equatable, Identifiable, Sendable {
    /// The id the API expects in the request body.
    var id: String
    var displayName: String
    /// Tokens. Drives the pre-send estimate and the compaction trigger for a
    /// remote chat, never a local `countTokens`.
    var contextWindow: Int
    var supportsVision: Bool = false
    var supportsReasoning: Bool = false
    var enabled: Bool = true
}

/// A named upstream endpoint. **Holds no secret**: the key lives in
/// `RemoteProviderKeyStore`, and this shape is what is persisted.
struct RemoteProvider: Codable, Equatable, Identifiable, Sendable {
    var id: UUID
    var name: String
    var family: RemoteAPIFamily
    var baseURL: String
    var models: [RemoteProviderModel] = []
    var enabled: Bool = true
}

/// The named failures a remote turn can end in. Mirrors the local engine's
/// error rows: each ends the turn, none retries on its own, and the user
/// resends.
///
/// **NO CASE CARRIES A KEY, AND NO DESCRIPTION IS BUILT FROM RAW TEXT.**
/// Provider error bodies and transport messages pass through
/// `RemoteProviderRedactor` before they reach a case's payload.
enum RemoteProviderError: Error, Equatable, Sendable {
    case notConfigured
    case disabled
    case invalidBaseURL(String)
    /// The Keychain read failed (locked, denied). Nothing was sent.
    case keyUnavailable(String)
    case auth
    case endpointNotFound
    case rateLimited
    case overloaded
    case contextOverflow
    case server(status: Int)
    case requestRefused(status: Int, detail: String)
    case network(String)
    case badStream(String)
    case redirectRefused
    case firstByteTimeout
    case stalled
    case responseTooLarge
    case requestTooLarge
    case cancelled

    /// Plain-English diagnostic. The UI increment maps each case to a
    /// localized string; this is for logs and tests.
    var diagnostic: String {
        switch self {
        case .notConfigured: return "remote provider has no key configured"
        case .disabled: return "remote provider or model is disabled"
        case .invalidBaseURL(let why): return "invalid base URL: \(why)"
        case .keyUnavailable(let why): return "key unavailable: \(why)"
        case .auth: return "authentication failed (check the API key)"
        case .endpointNotFound: return "endpoint not found (check the base URL)"
        case .rateLimited: return "rate limited by the provider"
        case .overloaded: return "provider overloaded"
        case .contextOverflow: return "context window exceeded"
        case .server(let status): return "provider error (HTTP \(status))"
        case .requestRefused(let status, let detail): return "request refused (HTTP \(status)): \(detail)"
        case .network(let why): return "network error: \(why)"
        case .badStream(let why): return "malformed stream: \(why)"
        case .redirectRefused: return "redirect refused"
        case .firstByteTimeout: return "no response from the provider"
        case .stalled: return "the stream stalled"
        case .responseTooLarge: return "response too large"
        case .requestTooLarge: return "request too large"
        case .cancelled: return "cancelled"
        }
    }

    /// Whether the transcript row should offer compaction.
    var offersCompaction: Bool { self == .contextOverflow }
}

// MARK: - Step events

enum RemoteFinishReason: Equatable, Sendable {
    case endOfTurn
    case toolCalls
    case maxTokens
    case other(String)
}

/// A complete tool call. Only emitted once its argument JSON parsed, so a
/// stream cut mid-call yields `badStream` and nothing here.
struct RemoteToolCall: Equatable, Sendable {
    var id: String
    var name: String
    /// A JSON object, as text.
    var argumentsJSON: String
}

/// What a family client hands the turn loop: content, reasoning, tool calls,
/// usage, finish. Engine-neutral on purpose: the local path's
/// `GenerationEvent` maps onto the same shape (see the ZP-22 draft, seam
/// record).
enum RemoteStepEvent: Equatable, Sendable {
    case content(String)
    case reasoning(String)
    case toolCall(RemoteToolCall)
    /// API-reported counts. Either may be nil when the provider omits it; the
    /// ring then falls back to the labeled estimate.
    case usage(promptTokens: Int?, outputTokens: Int?)
    case finished(RemoteFinishReason)
}

// MARK: - Request

struct RemoteToolSpec: Equatable, Sendable {
    var name: String
    var description: String
    /// A JSON Schema object, as text.
    var parametersJSON: String
}

struct RemoteChatMessage: Equatable, Sendable {
    enum Role: String, Sendable { case user, assistant, tool }
    var role: Role
    var content: String
    /// Assistant turns that called tools.
    var toolCalls: [RemoteToolCall] = []
    /// Tool-result rows name the call they answer.
    var toolCallID: String?
}

struct RemoteChatRequest: Equatable, Sendable {
    var model: String
    var system: String?
    var messages: [RemoteChatMessage]
    var tools: [RemoteToolSpec] = []
    var maxTokens: Int
    var temperature: Double?
}

/// Caps and clocks. Values are fixed at implementation and stated in
/// `swift/docs/SWIFT_REMOTE_PROVIDERS.md`; tests shrink them.
struct RemoteLimits: Sendable {
    var firstByteTimeout: TimeInterval = 60
    var stallTimeout: TimeInterval = 90
    /// Request body ceiling. A history that serializes larger is refused
    /// before anything is sent.
    var maxRequestBytes = 8 << 20
    /// Response ceiling for one turn's stream.
    var maxResponseBytes = 32 << 20
    /// How much of an error body is read, then redacted and clipped.
    var maxErrorBodyBytes = 16 << 10
    var maxRedirects = 3

    static let standard = RemoteLimits()
}
