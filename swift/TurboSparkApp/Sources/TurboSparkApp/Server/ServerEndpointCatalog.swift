import Foundation

/// The routes this server actually serves, and how to call them.
///
/// **ONLY IMPLEMENTED ROUTES APPEAR HERE.** Routes are kept in sync with
/// `turbospark_server::build_router_with_options`. A user reads the list as a
/// capability statement and would build against it.
///
/// Pure and free of SwiftUI so the list and the snippets can be tested
/// against the router's own route table.
public enum ServerAPIFamily: String, CaseIterable, Identifiable, Sendable {
    case openAI
    case anthropic
    case ollama
    case turbospark

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .openAI: return "OpenAI"
        case .anthropic: return "Anthropic"
        case .ollama: return "Ollama"
        case .turbospark: return "Service"
        }
    }

    public var blurb: String {
        switch self {
        case .openAI:
            return "The default for most SDKs and editor plugins. Point the base URL here."
        case .anthropic:
            return "Native, so Claude Code and other Anthropic clients need no proxy in between."
        case .ollama:
            return "For tools that only speak Ollama. Streams NDJSON, not SSE."
        case .turbospark:
            return "Liveness and discovery."
        }
    }
}

public struct ServerEndpoint: Identifiable, Equatable, Sendable {
    public let method: String
    public let path: String
    public let family: ServerAPIFamily
    public let summary: String
    /// True when the route streams. Worth showing because the FRAMING
    /// differs by family: SSE everywhere except Ollama, which sends NDJSON.
    public let streams: Bool

    public var id: String { "\(method) \(path)" }
}

public enum ServerEndpointCatalog {
    /// Every route `turbospark_server::build_router_with_options` registers.
    ///
    /// **KEPT IN STEP BY A TEST, NOT BY DISCIPLINE.** `ServerEndpointCatalogTests`
    /// asserts this list against a literal copy of the router's own routes,
    /// so adding a route without a row here (or leaving a row for a route
    /// that was removed) fails rather than quietly misinforming a user.
    public static let all: [ServerEndpoint] = [
        ServerEndpoint(
            method: "GET", path: "/health", family: .turbospark,
            summary: "Liveness, the attached model ids, and the engine version.",
            streams: false),
        ServerEndpoint(
            method: "GET", path: "/v1/models", family: .openAI,
            summary: "Every attached model. This is what a client's model picker reads.",
            streams: false),
        ServerEndpoint(
            method: "GET", path: "/v1/models/{id}", family: .openAI,
            summary: "One model by id.", streams: false),
        ServerEndpoint(
            method: "POST", path: "/v1/chat/completions", family: .openAI,
            summary: "Chat completions, with tools and reasoning.", streams: true),
        ServerEndpoint(
            method: "POST", path: "/v1/completions", family: .openAI,
            summary: "Legacy raw prompt, with no chat template applied.", streams: true),
        ServerEndpoint(
            method: "POST", path: "/v1/responses", family: .openAI,
            summary: "The Responses API, item-shaped.", streams: true),
        ServerEndpoint(
            method: "POST", path: "/v1/embeddings", family: .openAI,
            summary: "Vector embeddings for text strings.", streams: false),
        ServerEndpoint(
            method: "POST", path: "/v1/messages", family: .anthropic,
            summary: "Messages, with tools, thinking and images.", streams: true),
        ServerEndpoint(
            method: "POST", path: "/v1/messages/count_tokens", family: .anthropic,
            summary: "What a real call on this request would prefill. Generates nothing.",
            streams: false),
        ServerEndpoint(
            method: "GET", path: "/api/tags", family: .ollama,
            summary: "The model list, in Ollama's shape.", streams: false),
        ServerEndpoint(
            method: "GET", path: "/api/version", family: .ollama,
            summary: "This engine's version, not an Ollama one.", streams: false),
        ServerEndpoint(
            method: "POST", path: "/api/show", family: .ollama,
            summary: "Details for one model.", streams: false),
        ServerEndpoint(
            method: "POST", path: "/api/chat", family: .ollama,
            summary: "Chat. Streams by default, unlike every OpenAI route here.",
            streams: true),
        ServerEndpoint(
            method: "POST", path: "/api/generate", family: .ollama,
            summary: "Raw prompt, through the model's own template.", streams: true),
        ServerEndpoint(
            method: "POST", path: "/api/embeddings", family: .ollama,
            summary: "Vector embedding for one prompt.", streams: false),
        ServerEndpoint(
            method: "POST", path: "/api/embed", family: .ollama,
            summary: "Batch vector embeddings.", streams: false),
    ]

    public static func endpoints(for family: ServerAPIFamily) -> [ServerEndpoint] {
        all.filter { $0.family == family }
    }
}

/// A ready-to-paste way to reach a running server from a particular tool.
public struct ServerConnectSnippet: Identifiable, Equatable, Sendable {
    public let id: String
    public let title: String
    /// What the snippet is for, in one line.
    public let note: String
    public let language: String
    public let body: String
}

public enum ServerConnectRecipes {
    /// Snippets for a server at `baseURL`, serving `modelID`.
    ///
    /// **THE LIVE PORT AND KEY ARE SUBSTITUTED IN.** A snippet with a
    /// placeholder in it is a snippet the reader has to edit, which is the
    /// step people get wrong -- and the port is OS-assigned here, so it is
    /// not something they could know without reading it off this pane.
    ///
    /// `apiKey` nil means the server is unauthenticated, and the snippets
    /// say so with a dummy value rather than omitting the header: most
    /// clients require one to be set even when it is never checked, and
    /// leaving it out produces a confusing client-side failure.
    ///
    /// **EVERY INTERPOLATED VALUE IS ESCAPED FOR THE CONTEXT IT LANDS IN.**
    /// The key is whatever the user typed into the Advanced pane and the
    /// model id is an install directory's name, and both go straight into
    /// commands the reader pastes into a terminal: an unescaped quote ends
    /// the argument, and a `$`, backtick or `;` runs part of the value as
    /// shell. `TurboSparkAgent.launchCommand` had this fixed for the menu
    /// bar's agent exports and this surface was missed (same class, one
    /// pane over). Bash assignments and URLs take the double-quoted form,
    /// curl's header and JSON body take the single-quoted form, and the
    /// Python and JSON literals take JSON escaping, which is also a valid
    /// Python string literal.
    public static func snippets(baseURL: String, modelID: String, apiKey: String?)
        -> [ServerConnectSnippet]
    {
        let key = apiKey ?? "unused"
        let model = modelID.isEmpty ? "<load a model first>" : modelID
        let curlBody = """
        {"model":"\(Self.jsonEscaped(model))","stream":true,
             "messages":[{"role":"user","content":"hello"}]}
        """
        let ollamaBody = """
        {"model":"\(Self.jsonEscaped(model))",
             "messages":[{"role":"user","content":"hello"}]}
        """
        return [
            ServerConnectSnippet(
                id: "claude-code",
                title: "Claude Code",
                note: "Anthropic-native, so nothing sits in between.",
                language: "bash",
                body: """
                    ANTHROPIC_BASE_URL=\(Self.shellDoubleQuoted(baseURL)) \\
                    ANTHROPIC_API_KEY=\(Self.shellDoubleQuoted(key)) \\
                    CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=true \\
                      claude
                    """),
            ServerConnectSnippet(
                id: "openai-python",
                title: "OpenAI SDK",
                note: "Any client that takes a base URL: Cursor, Continue, LangChain.",
                language: "python",
                body: """
                    from openai import OpenAI

                    client = OpenAI(base_url=\(Self.pythonString(baseURL + "/v1")), api_key=\(Self.pythonString(key)))
                    reply = client.chat.completions.create(
                        model=\(Self.pythonString(model)),
                        messages=[{"role": "user", "content": "hello"}],
                    )
                    print(reply.choices[0].message.content)
                    """),
            ServerConnectSnippet(
                id: "curl",
                title: "curl",
                note: "Streams as it decodes.",
                language: "bash",
                body: """
                    curl -sN \(Self.shellSingleQuoted("\(baseURL)/v1/chat/completions")) \\
                      -H 'content-type: application/json' \\
                      -H \(Self.shellSingleQuoted("authorization: Bearer \(key)")) \\
                      -d \(Self.shellSingleQuoted(curlBody))
                    """),
            ServerConnectSnippet(
                id: "ollama",
                title: "Ollama clients",
                note: "Set the host and the tool finds the models by itself.",
                language: "bash",
                body: """
                    OLLAMA_HOST=\(Self.shellDoubleQuoted(baseURL)) ollama list
                    curl -s \(Self.shellSingleQuoted("\(baseURL)/api/chat")) \\
                      -d \(Self.shellSingleQuoted(ollamaBody))
                    """),
        ]
    }

    /// Escapes for a POSIX double-quoted string: backslash first, then the
    /// three characters a double-quoted shell context still treats
    /// specially -- the closing quote, parameter/command substitution, and
    /// command substitution's other spelling. Same set as
    /// `TurboSparkAgent.launchCommand`'s helper, restated here because that
    /// one is private to the TurboSpark module.
    private static func shellDoubleQuoted(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "$", with: "\\$")
            .replacingOccurrences(of: "`", with: "\\`")
        return "\"\(escaped)\""
    }

    /// Escapes for a POSIX single-quoted string. Nothing is special inside
    /// single quotes except the closing quote itself, which cannot be
    /// escaped -- the spelling is to close, escape a bare quote, and reopen.
    private static func shellSingleQuoted(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    /// JSON string escaping without the surrounding quotes, with `/` left
    /// alone: a JSON encoder would write `\/`, which Python keeps as a
    /// literal backslash and which turns a pasted URL into a broken one.
    private static func jsonEscaped(_ value: String) -> String {
        // The encoder is invoked on the string rather than hand-written so
        // control characters and unusual scalars follow JSON's own table
        // rather than a subset somebody remembers.
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes
        guard
            let data = try? encoder.encode([value]),
            data.first == UInt8(ascii: "["), data.last == UInt8(ascii: "]")
        else { return value }
        let inner = data.dropFirst().dropLast()
        guard
            inner.first == UInt8(ascii: "\""), inner.last == UInt8(ascii: "\"")
        else { return value }
        return String(decoding: inner.dropFirst().dropLast(), as: UTF8.self)
    }

    /// A Python double-quoted literal. JSON string escaping is valid Python
    /// for the characters that matter (`"`, `\`, newlines), so one table
    /// serves both languages.
    private static func pythonString(_ value: String) -> String {
        "\"\(jsonEscaped(value))\""
    }
}
