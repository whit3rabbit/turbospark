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
        "POST /v1/messages",
        "POST /v1/messages/count_tokens",
        "GET /v1/models",
        "GET /v1/models/{id}",
        "GET /api/tags",
        "GET /api/version",
        "POST /api/show",
        "POST /api/chat",
        "POST /api/generate",
    ]

    func testTheCatalogMatchesTheRoutersOwnRoutes() {
        let catalog = Set(ServerEndpointCatalog.all.map(\.id))
        XCTAssertEqual(
            catalog, routerRoutes,
            "the endpoint list and the router disagree: "
                + "\(catalog.symmetricDifference(routerRoutes).sorted())")
    }

    /// **NO EMBEDDINGS ROW, AND THE ABSENCE IS DELIBERATE.** This engine has
    /// no embedding path at all, so listing one would be a capability claim
    /// a reader could act on. Stated as a test so a later session adding it
    /// out of symmetry has to come past this line.
    func testNoEndpointIsAdvertisedThatTheEngineCannotServe() {
        XCTAssertFalse(
            ServerEndpointCatalog.all.contains { $0.path.contains("embeddings") },
            "this engine has no embedding path; a listed route that 404s is worse than none")
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
    /// header: most clients require SOME value even when nothing checks it,
    /// and omitting it produces a confusing client-side failure instead of a
    /// working call.
    func testAnUnauthenticatedServerStillGetsAPlaceholderKey() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "m", apiKey: nil)
        let claude = snippets.first { $0.id == "claude-code" }
        XCTAssertEqual(claude?.body.contains("ANTHROPIC_API_KEY=unused"), true)
    }

    /// With nothing loaded the model field says what to do rather than
    /// leaving an empty string that would paste as a broken request.
    func testAnEmptyModelIdIsNamedRatherThanLeftBlank() {
        let snippets = ServerConnectRecipes.snippets(
            baseURL: "http://127.0.0.1:1", modelID: "", apiKey: nil)
        let python = snippets.first { $0.id == "openai-python" }
        XCTAssertEqual(python?.body.contains("<load a model first>"), true)
    }
}
