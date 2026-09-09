import XCTest
import TurboSpark
@testable import TurboSparkApp

final class ServerDaemonLaunchTests: XCTestCase {
    /// The key and the port the user configured must reach the daemon: it is
    /// a separate process that inherits nothing, and without the key it
    /// starts with NO access control at all.
    func testEverythingConfiguredIsCarried() {
        let args = ServerDaemonLaunch.args(
            port: 9090,
            apiKey: "sk-abc",
            guardrails: .off)
        XCTAssertEqual(args, ["--port", "9090", "--guardrails", "off", "--api-key", "sk-abc"])
    }

    /// Port 0 is "automatic" for the in-app server and is not passed: the
    /// daemon records the handed-in port in its meta file, so an OS-assigned
    /// one would surface as 0. Its own default applies instead.
    func testAutomaticPortPassesNoPortFlag() {
        let args = ServerDaemonLaunch.args(port: 0, apiKey: nil, guardrails: .on)
        XCTAssertEqual(args, ["--guardrails", "on"])
    }

    /// No key means no flag -- the daemon then runs unauthenticated, which is
    /// what an empty key field means everywhere else in this app.
    func testNoKeyPassesNoKeyFlag() {
        let args = ServerDaemonLaunch.args(port: 8080, apiKey: nil, guardrails: .on)
        XCTAssertEqual(args, ["--port", "8080", "--guardrails", "on"])
        XCTAssertFalse(args.contains("--api-key"))
    }

    /// The guardrails flag is always carried: `.select` resolves to `.on` in
    /// `AppModel.serverGuardrails`, so the daemon's served path agrees with
    /// what the Advanced pane reports for the in-app server.
    func testGuardrailsIsAlwaysCarried() {
        let on = ServerDaemonLaunch.args(port: 0, apiKey: nil, guardrails: .on)
        let off = ServerDaemonLaunch.args(port: 0, apiKey: nil, guardrails: .off)
        XCTAssertEqual(on, ["--guardrails", "on"])
        XCTAssertEqual(off, ["--guardrails", "off"])
    }
}
