import TurboSpark
import XCTest

@testable import TurboSparkApp

/// F2 (state#28): a stop pressed while a start was still binding used to
/// vanish.
///
/// `AppModel.server` is published only after the awaited bind, so
/// `stopServer()` found nil and returned -- leaving a server LISTENING that
/// the UI showed as stopped, with no control left that could reach it.
///
/// **THIS NEEDS A REAL BIND AND GETS ONE.** `ts_server_start` takes a nullable
/// session (`crates/ffi/CLAUDE.md` Gotcha 14), so a server starts with no
/// model at all -- no install, no Metal, milliseconds. The ordering is
/// deterministic rather than racy: `startServer()` sets `serverBusy`
/// synchronously and then spawns a `Task`, whose body cannot run until the
/// synchronous caller yields, so a `stopServer()` on the next line always
/// arrives mid-flight.
@MainActor
final class ServerLifecycleTests: XCTestCase {
    /// A port the OS just handed out and released. Racy in principle, fine in
    /// a test, and far better than a hardcoded number that collides with
    /// whatever else the developer happens to be running.
    private func borrowFreePort() throws -> UInt16 {
        let probe = try TurboSparkServer.start(options: ServerOptions(port: 0))
        let port = try probe.info().port
        probe.stop()
        return port
    }

    private func waitUntil(_ condition: () -> Bool, timeout: TimeInterval = 5) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    // MARK: - state#72: a stop during an ATTACH must not orphan the session

    func testAnAttachDoesNotCompleteAgainstAServerThatIsGoneOrReplaced() throws {
        // `attachModelToServer` captures `server` as a local, awaits a model
        // open that runs for tens of seconds on a real install, and then
        // writes into `serverAttachedSessions`. `stopServer()` in between
        // clears both -- and every remover starts with `guard let server`, so
        // the resurrected entry is unreachable and the weights stay resident
        // for the life of the process.
        //
        // The call site itself needs a real install to reach (the guard sits
        // after `TurboSparkSession(modelPath:)`), so what is asserted here is
        // the decision it makes.
        let first = try TurboSparkServer.start(options: ServerOptions(port: 0))
        defer { first.stop() }

        XCTAssertFalse(
            AppModel.attachMayComplete(captured: first, current: nil),
            "The server was stopped while the model was loading: nothing may be attached to it.")
        XCTAssertTrue(
            AppModel.attachMayComplete(captured: first, current: first),
            "The ordinary case must still attach, or the guard is a blanket refusal.")

        let second = try TurboSparkServer.start(options: ServerOptions(port: 0))
        defer { second.stop() }
        XCTAssertFalse(
            AppModel.attachMayComplete(captured: first, current: second),
            "Stop-then-Start leaves a DIFFERENT server running, and a model opened for the first "
                + "one must not be served by it.")
    }

    func testAStopDuringAStartDoesNotLeaveAServerListening() async throws {
        let appModel = AppModel()
        let port = try borrowFreePort()
        appModel.serverPinnedPort = port

        appModel.startServer()
        XCTAssertTrue(appModel.serverBusy, "Precondition: the bind has not run yet.")
        appModel.stopServer()

        try await waitUntil { !appModel.serverBusy }
        XCTAssertNil(appModel.server, "The UI must show no server.")

        // **THE DISCRIMINATING ASSERTION.** "The UI shows nil" was already
        // true with the bug -- that IS the bug. What separates the two states
        // is whether the port is still held: binding it again succeeds only
        // if the aborted start really stopped what it started.
        let rebind = try TurboSparkServer.start(options: ServerOptions(port: port))
        defer { rebind.stop() }
        XCTAssertEqual(
            try rebind.info().port, port,
            "The port must be free. A server left listening here is unreachable by any control.")
    }

    func testAnOrdinaryStartAndStopStillWorks() async throws {
        // The guard above must not be a blanket refusal: a start nobody
        // cancelled has to publish its server as it always did.
        let appModel = AppModel()
        appModel.startServer()
        try await waitUntil { !appModel.serverBusy }

        XCTAssertNotNil(appModel.server, "A start nobody cancelled must publish its server.")
        XCTAssertNotNil(appModel.serverInfo)

        appModel.stopServer()
        XCTAssertNil(appModel.server)
    }

    func testStoppingWhenNothingIsRunningIsHarmless() {
        let appModel = AppModel()
        appModel.stopServer()
        XCTAssertNil(appModel.server)
        XCTAssertFalse(appModel.serverBusy)
    }
}
