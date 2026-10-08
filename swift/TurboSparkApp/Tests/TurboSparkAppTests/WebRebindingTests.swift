import XCTest
@testable import TurboSparkApp

/// DNS rebinding: the destination validator resolves a host once and
/// URLSession resolves it again to connect. The fixture is on loopback and a
/// permissive injected validator stands in for "DNS said public at validation
/// time", so the only thing that can stop the fetch is the check of the
/// address the socket really connected to.
final class WebRebindingTests: XCTestCase {
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
}
