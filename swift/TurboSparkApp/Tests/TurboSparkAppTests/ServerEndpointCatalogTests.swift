import XCTest

@testable import TurboSparkApp

/// The endpoint list and the connect snippets.
///
/// **THE POINT OF THE FIRST TEST IS THAT THE LIST CANNOT SILENTLY ROT.** A
/// user reads the endpoint list as a capability statement, so a row for a
/// route that does not exist sends them off to build against nothing, and a
/// missing row hides a route they could have used. Neither shows up in a
/// screenshot. The literal below is a copy of
/// `turbospark_server::build_router_with_options`'s own route table, so a
/// route added or removed there without touching the catalog fails here.
final class ServerEndpointCatalogTests: XCTestCase {

    /// Every route the router registers, spelled exactly as it does.
    private let routerRoutes: Set<String> = [
        "GET /health",
        "POST /v1/chat/completions",
        "POST /v1/completions",
        "POST /v1/responses",
        "POST /v1/embeddings",
        "POST /v1/messages",
        "POST /v1/messages/count_tokens",
        "GET /v1/models",
        "GET /v1/models/{id}",
        "GET /api/tags",
        "GET /api/version",
        "POST /api/show",
        "POST /api/chat",
        "POST /api/generate",
        "POST /api/embeddings",
        "POST /api/embed",
    ]

    func testTheCatalogMatchesTheRoutersOwnRoutes() {
        let catalog = Set(ServerEndpointCatalog.all.map(\.id))
        XCTAssertEqual(
            catalog, routerRoutes,
            "the endpoint list and the router disagree: "
                + "\(catalog.symmetricDifference(routerRoutes).sorted())")
    }

    /// Every advertised endpoint is part of the router's registered routes.
    func testEveryAdvertisedEndpointMatchesRouter() {
        for endpoint in ServerEndpointCatalog.all {
            XCTAssertTrue(
                routerRoutes.contains(endpoint.id),
                "advertised route \(endpoint.id) is not registered in the router")
        }
    }

    func testEveryFamilyHasAtLeastOneRoute() {
        for family in ServerAPIFamily.allCases {
            XCTAssertFalse(
                ServerEndpointCatalog.endpoints(for: family).isEmpty,
                "\(family.title) is offered as a filter and matches nothing")
        }
    }

    /// The Ollama chat route streams, and its framing differs from every
    /// other streaming route here. The pane draws that badge off this flag.
    func testTheStreamingRoutesAreMarkedAsSuch() {
        let streaming = Set(ServerEndpointCatalog.all.filter(\.streams).map(\.id))
        XCTAssertTrue(streaming.contains("POST /v1/chat/completions"))
        XCTAssertTrue(streaming.contains("POST /api/chat"))
        XCTAssertFalse(
            streaming.contains("POST /v1/messages/count_tokens"),
            "count_tokens generates nothing, so it cannot stream")
        XCTAssertFalse(streaming.contains("GET /health"))
    }

    // MARK: - Snippets

    /// **THE LIVE PORT IS SUBSTITUTED IN, WHICH IS THE WHOLE POINT.** The
    /// port is OS-assigned, so a snippet carrying a placeholder is one the
    /// reader cannot complete from anything they already know.
    func testEverySnippetCarriesTheLiveAddress() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:54321", modelID: "gemma4.gturbo", apiKey: "sk-live")
        XCTAssertFalse(snippets.isEmpty)
        for snippet in snippets {
            XCTAssertTrue(
                snippet.body.contains("127.0.0.1:54321"),
                "\(snippet.title) does not carry the address")
            XCTAssertFalse(
                snippet.body.contains("<port>"),
                "\(snippet.title) still has a placeholder in it")
        }
    }

    func testTheKeyIsSubstitutedWhenOneIsSet() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "m", apiKey: "sk-live")
        let claude = snippets.first { $0.id == "claude-code" }
        XCTAssertEqual(
            claude?.body.contains("sk-live"), true,
            "a set key belongs in the snippet, or the reader pastes a call that 401s")
    }

    /// With no key the snippets send a dummy rather than dropping the
    /// header: most clients require SOME value even when it is never
    /// checked, and omitting it produces a confusing client-side failure
    /// instead of a working call.
    func testAnUnauthenticatedServerStillGetsAPlaceholderKey() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "m", apiKey: nil)
        let claude = snippets.first { $0.id == "claude-code" }
        XCTAssertEqual(claude?.body.contains("ANTHROPIC_API_KEY=\"unused\""), true)
    }

    /// With nothing loaded the model field says what to do rather than
    /// leaving an empty string that would paste as a broken request.
    func testAnEmptyModelIdIsNamedRatherThanLeftBlank() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "", apiKey: nil)
        let python = snippets.first { $0.id == "openai-python" }
        XCTAssertEqual(python?.body.contains("<load a model first>"), true)
    }

    // MARK: - Snippet escaping

    /// **A HAND-TYPED KEY CANNOT BREAK OUT OF THE SNIPPET IT IS PASTED
    /// INTO.** Same class as `TurboSparkAgent.launchCommand`'s escaping
    /// test: these commands go straight into a terminal, and an unescaped
    /// quote ends the assignment while a `$` or backtick runs part of the
    /// key as a substitution. Bash assignments take double-quote escaping;
    /// inside curl's single quotes the same characters are literal, which
    /// is why the raw key survives there unchanged.
    func testSnippetsEscapeShellMetacharactersInTheKey() {
        let raw = "a\\b\"c$d`e;rm"
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "m", apiKey: raw)
        let claude = snippets.first { $0.id == "claude-code" }?.body ?? ""
        XCTAssertTrue(
            claude.contains("ANTHROPIC_API_KEY=\"a\\\\b\\\"c\\$d\\`e;rm\""),
            "the bash assignment must escape every double-quote metacharacter: \(claude)")
        XCTAssertFalse(
            claude.contains("ANTHROPIC_API_KEY=\"a\\b"),
            "an unescaped backslash or quote leaked into the assignment")
        let curl = snippets.first { $0.id == "curl" }?.body ?? ""
        XCTAssertTrue(
            curl.contains("'authorization: Bearer a\\b\"c$d`e;rm'"),
            "single quotes make these same characters literal: \(curl)")
    }

    /// A single quote ends curl's single-quoted arguments, and a model id
    /// is an install directory's name, which a user can legitimately give a
    /// quote. Both spell out of it the POSIX way, and a `"` in the model id
    /// is JSON-escaped inside the body.
    func testAQuoteInTheKeyOrModelIdCannotEndACurlArgument() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "mo'del\"m", apiKey: "sk'x")
        let curl = snippets.first { $0.id == "curl" }?.body ?? ""
        XCTAssertTrue(
            curl.contains("Bearer sk'\\''x"),
            "the header must close, escape and reopen around the quote: \(curl)")
        XCTAssertTrue(
            curl.contains("\"model\":\"mo'\\''del\\\"m\""),
            "the body's shell quoting and its JSON escaping must compose: \(curl)")
    }

    /// The Python snippet builds a source file, so the key and model go
    /// through string-literal escaping there: a `"` ends the argument and
    /// anything after it is Python, not a key.
    func testSnippetsEscapePythonStringLiterals() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "mo\"del", apiKey: "a\\b\"c")
        let python = snippets.first { $0.id == "openai-python" }?.body ?? ""
        XCTAssertTrue(
            python.contains("api_key=\"a\\\\b\\\"c\""),
            "the key must arrive as one Python string literal: \(python)")
        XCTAssertTrue(
            python.contains("model=\"mo\\\"del\""),
            "the model id must arrive as one Python string literal: \(python)")
    }

    /// `/` stays `/`: JSON's default escaping writes `\/`, which Python
    /// keeps as a literal backslash, so a pasted base URL would point at a
    /// path that does not exist. The URL must survive verbatim.
    func testURLsKeepTheirSlashesInThePythonSnippet() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:54321", modelID: "m", apiKey: nil)
        let python = snippets.first { $0.id == "openai-python" }?.body ?? ""
        XCTAssertTrue(
            python.contains("base_url=\"http://127.0.0.1:54321/v1\""),
            "the base URL must survive without JSON slash escaping: \(python)")
    }
}
