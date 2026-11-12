import XCTest
@testable import TurboSparkApp

/// DNS rebinding: the destination validator resolves a host once and
/// URLSession resolves it again to connect. The fixture is on loopback and a
/// permissive injected validator stands in for "DNS said public at validation
/// time", so the only thing that can stop the fetch is the check of the
/// address the socket really connected to.
final class WebRebindingTests: NetworkPolicyPinnedTestCase {
    func testConnectionToAPrivateAddressIsRefusedEvenWhenValidationPassed() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let url = server.url("/ok")
        do {
            _ = try await WebFetchExecutor.fetch(
                url: url.absoluteString,
                destinationValidator: { _ in },
                connectedAddressValidator: HttpRequestExecutor.refuseConnectedPrivateAddress)
            XCTFail("a connection that landed on loopback must be refused")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("private"), error.localizedDescription)
            XCTAssertTrue(error.localizedDescription.contains("rebinding"), error.localizedDescription)
        }
    }

    /// Control: without the connected-address check the same fetch succeeds,
    /// so the refusal above is caused by that check and not by the fixture.
    func testControlFetchSucceedsWithoutTheConnectedAddressCheck() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let body = try await WebFetchExecutor.fetch(
            url: server.url("/ok").absoluteString, destinationValidator: { _ in })
        XCTAssertFalse(body.isEmpty)
    }

    func testTheCheckSeesTheRealPeerAddress() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        var seen: [String] = []
        let config = URLSessionConfiguration.ephemeral
        _ = try await HttpRequestExecutor.performRequest(
            URLRequest(url: server.url("/ok")),
            configuration: config, customSession: nil,
            validate: { _ in },
            validateConnectedAddress: { seen.append($0) })
        XCTAssertFalse(seen.isEmpty, "the metrics delegate must report at least one connection")
        XCTAssertTrue(seen.allSatisfy { AppToolSandbox.isPrivateOrMetadataHost($0) }, "\(seen)")
    }

    func testEmptyPeerAddressFailsClosed() {
        XCTAssertThrowsError(try HttpRequestExecutor.refuseConnectedPrivateAddress(""))
        XCTAssertNoThrow(try HttpRequestExecutor.refuseConnectedPrivateAddress("93.184.216.34"))
        XCTAssertNoThrow(try HttpRequestExecutor.refuseConnectedPrivateAddress("2606:4700::1111"))
    }

    func testAdditionalNonPublicRangesAreClassifiedPrivate() {
        for host in ["198.18.0.1", "198.19.255.255", "192.0.0.8", "224.0.0.251", "239.255.255.250",
                     "240.0.0.1", "255.255.255.255", "ff02::fb", "ff05::1"] {
            XCTAssertTrue(AppToolSandbox.isPrivateOrMetadataHost(host), "\(host) must be non-public")
        }
        for host in ["198.17.255.255", "198.20.0.1", "192.0.2.1", "223.255.255.255", "93.184.216.34",
                     "2606:4700::1111"] {
            XCTAssertFalse(AppToolSandbox.isPrivateOrMetadataHost(host), "\(host) must stay public")
        }
    }

    // MARK: - Web search path (issue S1)

    /// SearXNG takes a user-supplied base URL. Under the strict policy a
    /// loopback instance is refused before any request reaches the fixture.
    func testSearXNGLoopbackBaseUrlIsRefusedBeforeSending() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let base = server.url("/").absoluteString
        do {
            _ = try await WebSearchExecutor.searchSearXNG(query: "swift", baseUrl: base)
            XCTFail("a loopback SearXNG URL must be refused under the strict policy")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("denied"), error.localizedDescription)
        }
        XCTAssertEqual(server.requestCount(for: "/search"), 0, "nothing may reach the fixture")
    }

    func testSearXNGMetadataBaseUrlIsRefused() async {
        do {
            _ = try await WebSearchExecutor.searchSearXNG(
                query: "swift", baseUrl: "http://169.254.169.254/latest")
            XCTFail("a metadata SearXNG URL must be refused")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("denied"), error.localizedDescription)
        }
    }

    /// A public name that resolves to loopback stands in for DNS rebinding:
    /// the pre-flight check passes (injected) and the connected-address
    /// check must discard the response on the search path too.
    func testSearchFetchRefusesAConnectionToAPrivateAddress() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        do {
            _ = try await WebSearchExecutor.fetchLimited(
                request: URLRequest(url: server.url("/search")),
                session: nil,
                validate: { _ in })
            XCTFail("a search connection that landed on loopback must be refused")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("rebinding"), error.localizedDescription)
        }
    }

    func testSearchFetchValidatesInjectedSessionRequestsBeforeSending() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        do {
            _ = try await WebSearchExecutor.fetchLimited(
                request: URLRequest(url: server.url("/search")), session: URLSession.shared)
            XCTFail("an injected session must not bypass destination validation")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("denied"), error.localizedDescription)
        }
        XCTAssertEqual(server.requestCount(for: "/search"), 0)
    }
}
