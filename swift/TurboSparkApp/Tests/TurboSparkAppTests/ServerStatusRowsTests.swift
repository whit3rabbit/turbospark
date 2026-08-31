import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The settings panel's two server rows, and the key resolution that decides
/// what the second one says.
///
/// **THESE ROWS WERE UNTESTABLE UNTIL THEY WERE A VALUE.** Both facts used to
/// be spelled inline in `AppSettingsView`, reachable only by launching the app
/// and looking -- which is exactly how the Address row came to interpolate a
/// `127.0.0.1` literal beside a read-back port, and how `authEnabled` came to
/// be decoded and rendered nowhere at all.
///
/// Every `ServerInfo` here is DECODED from the JSON `ts_server_info_json`
/// actually emits rather than built by hand, so these cases also pin the wire
/// spelling: a `host` renamed on the Rust side fails to decode here rather
/// than arriving as a silent default.
final class ServerStatusRowsTests: XCTestCase {
    private func info(
        host: String = "127.0.0.1",
        port: UInt16 = 8080,
        authEnabled: Bool = false
    ) throws -> ServerInfo {
        let json = """
            {
                "port": \(port),
                "host": "\(host)",
                "modelId": "gemma4",
                "authEnabled": \(authEnabled)
            }
            """
        return try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))
    }

    // MARK: - Address

    func testAddressIsBuiltFromTheReportedHostAndPort() throws {
        let rows = ServerStatusRows(info: try info(host: "127.0.0.1", port: 63063))
        XCTAssertEqual(rows.address, "http://127.0.0.1:63063")
    }

    /// **THE CASE THAT DISCRIMINATES.** Measured rather than asserted:
    /// replacing the address with a hardcoded `"http://127.0.0.1:\(port)"`
    /// reddens this case and `testAnIPv6HostIsBracketed`, and leaves
    /// `testAddressIsBuiltFromTheReportedHostAndPort` GREEN -- that one uses
    /// the address the engine happens to bind, so it cannot tell a reading
    /// from a literal. A file of only-loopback cases would have proved
    /// nothing about the bug this row was written to fix.
    ///
    /// It is the app-side twin of `c_surface.rs`'s
    /// `the_reported_host_is_the_address_actually_bound`: that test proves the
    /// FFI reports the bound host, this one proves the UI reads what it
    /// reports.
    func testAddressFollowsAHostThatIsNotLoopback() throws {
        let rows = ServerStatusRows(info: try info(host: "100.64.1.7", port: 8080))
        XCTAssertEqual(
            rows.address, "http://100.64.1.7:8080",
            "the row must render the host the server REPORTED, not the one this engine happens to bind"
        )
    }

    /// The bracket rule an IPv6 literal needs in a URL. Unreachable today (the
    /// engine binds IPv4) and asserted anyway, so the helper is correct
    /// without that coincidence rather than because of it.
    func testAnIPv6HostIsBracketed() throws {
        let rows = ServerStatusRows(info: try info(host: "::1", port: 8080))
        XCTAssertEqual(rows.address, "http://[::1]:8080")
    }

    // MARK: - Auth

    func testAuthRowReportsNoneWhenTheServerStartedWithoutAKey() throws {
        let rows = ServerStatusRows(info: try info(authEnabled: false))
        XCTAssertEqual(rows.authLabel, "none")
        XCTAssertTrue(rows.authIsWarning, "an unauthenticated server is the state a user must notice")
    }

    func testAuthRowReportsRequiredWhenTheServerStartedWithAKey() throws {
        let rows = ServerStatusRows(info: try info(authEnabled: true))
        XCTAssertEqual(rows.authLabel, "API key required")
        XCTAssertFalse(rows.authIsWarning)
    }

    // MARK: - What the typed field resolves to

    func testAKeyIsTrimmedRatherThanTakenLiterally() {
        XCTAssertEqual(AppModel.serverAPIKey(from: "  sk-test  "), "sk-test")
        XCTAssertEqual(AppModel.serverAPIKey(from: "sk-test"), "sk-test")
    }

    func testAnEmptyOrBlankFieldMeansNoAuth() {
        XCTAssertNil(AppModel.serverAPIKey(from: ""))
        XCTAssertNil(AppModel.serverAPIKey(from: "   "))
        XCTAssertNil(AppModel.serverAPIKey(from: "\n\t "))
    }

    /// **THE WHOLE CHAIN THE AUTH ROW EXISTS FOR, in one case.** A field
    /// holding nothing but spaces resolves to no key, which starts an
    /// unauthenticated server, which reports `authEnabled: false`, which the
    /// panel must say out loud. Before the row existed this ran end to end and
    /// looked identical to a key that took.
    func testAWhitespaceOnlyKeyEndsUpRenderedAsNoAuth() throws {
        let resolved = AppModel.serverAPIKey(from: "   ")
        XCTAssertNil(resolved, "spaces are not a key")

        // What the engine then reports, given no key (pinned against the Rust
        // side by c_surface.rs's server_options_accept_an_api_key case).
        let rows = ServerStatusRows(info: try info(authEnabled: resolved != nil))
        XCTAssertEqual(rows.authLabel, "none")
        XCTAssertTrue(rows.authIsWarning)
    }
}
