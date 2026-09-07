import XCTest

@testable import TurboSparkApp

/// The ThermalForge fan bridge: `thermalforge status` decoding against the
/// JSON captured on real hardware, the held/ramp predicates, the quit
/// decision, PATH resolution, and the pin/restore flow driven through a
/// fixture script standing in for the real binary.
@MainActor
final class FanControllerTests: XCTestCase {
    // MARK: - Fixtures

    /// Verbatim `thermalforge status` output from Mac16,5 (two fans, one
    /// under the machine's own curve), captured 2026-09-06.
    private static let realStatusJSON = """
    {
      "fans" : [
        {
          "actual_rpm" : 1876,
          "index" : 0,
          "max_rpm" : 5777,
          "min_rpm" : 1350,
          "mode" : "auto",
          "target_rpm" : 1920
        },
        {
          "actual_rpm" : 2059,
          "index" : 1,
          "max_rpm" : 5777,
          "min_rpm" : 1350,
          "mode" : "auto",
          "target_rpm" : 2073
        }
      ],
      "temperatures" : {
        "TAOL" : 26.1,
        "TB0T" : 37,
        "TCDX" : 81.9
      }
    }
    """

    private func decode(_ json: String) throws -> FanStatus {
        try JSONDecoder().decode(FanStatus.self, from: Data(json.utf8))
    }

    private func makeFan(
        index: Int = 0, actual: Int, target: Int, mode: String = "auto"
    ) -> FanStatus.Fan {
        FanStatus.Fan(
            index: index, actualRPM: actual, targetRPM: target,
            minRPM: 1350, maxRPM: 5777, mode: mode)
    }

    /// Creates a scratch directory with an executable shell script standing
    /// in for `thermalforge`. `failsMax` makes `max` exit 1, the shape of a
    /// daemon that is down. `max` flips the status fixture to a pinned
    /// (manual, at-target) reading; `auto` flips it back and drops
    /// `restored` as a marker the sync quit path can be asserted on.
    private func makeFixtureScript(failsMax: Bool = false) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("FanControllerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let script = dir.appendingPathComponent("thermalforge")
        let body = """
        #!/bin/sh
        case "$1" in
          status)
            if [ -f "\(dir.path)/pinned" ]; then
              echo '{"fans":[{"actual_rpm":5774,"index":0,"max_rpm":5777,"min_rpm":1350,"mode":"manual","target_rpm":5777}]}'
            else
              echo '{"fans":[{"actual_rpm":1876,"index":0,"max_rpm":5777,"min_rpm":1350,"mode":"auto","target_rpm":1920}]}'
            fi
            ;;
          max)
        \(failsMax ? "    exit 1" : "    : > \"\(dir.path)/pinned\"")
            ;;
          auto)
            rm -f "\(dir.path)/pinned"
            : > "\(dir.path)/restored"
            ;;
          *)
            exit 1
            ;;
        esac
        """
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
    }

    // MARK: - Decoding

    func test_realStatusFixtureDecodes() throws {
        let status = try decode(Self.realStatusJSON)
        XCTAssertEqual(status.fans.count, 2)
        XCTAssertEqual(status.fans[0].actualRPM, 1876)
        XCTAssertEqual(status.fans[0].targetRPM, 1920)
        XCTAssertEqual(status.fans[0].mode, "auto")
        XCTAssertEqual(status.fans[1].maxRPM, 5777)
        XCTAssertEqual(status.temperatures["TCDX"], 81.9)
        // An integral JSON temperature decodes as Double.
        XCTAssertEqual(status.temperatures["TB0T"], 37.0)
    }

    func test_statusWithoutTemperaturesStillDecodes() throws {
        let status = try decode(
            #"{"fans":[{"actual_rpm":100,"index":0,"max_rpm":6000,"min_rpm":1000,"mode":"auto","target_rpm":1100}]}"#)
        XCTAssertEqual(status.fans.count, 1)
        XCTAssertTrue(status.temperatures.isEmpty)
        XCTAssertNil(status.hottestSensor)
    }

    // MARK: - Held and ramp predicates

    func test_anyFanHeldTracksMode() {
        XCTAssertFalse(
            FanStatus(fans: [makeFan(actual: 1876, target: 1920, mode: "auto")]).anyFanHeld)
        XCTAssertTrue(
            FanStatus(fans: [makeFan(actual: 5774, target: 5777, mode: "manual")]).anyFanHeld)
        // One held fan among held and free: a hold anywhere means held.
        XCTAssertTrue(
            FanStatus(fans: [
                makeFan(index: 0, actual: 1876, target: 1920, mode: "auto"),
                makeFan(index: 1, actual: 5774, target: 5777, mode: "manual"),
            ]).anyFanHeld)
    }

    func test_rampPredicateAtTheNinetyFivePercentBoundary() {
        // Exactly 95% of target: holding. One below: still ramping.
        XCTAssertTrue(makeFan(actual: 95, target: 100).isNearTarget)
        XCTAssertFalse(makeFan(actual: 94, target: 100).isNearTarget)
        // The captured pinned state settles within 1% of target.
        XCTAssertTrue(makeFan(actual: 5774, target: 5777, mode: "manual").isNearTarget)
        // In auto the target can sit below actual; harmless, since only a
        // post-pin check consumes the predicate.
        XCTAssertTrue(makeFan(actual: 1380, target: 1350).isNearTarget)
    }

    func test_allFansNearTargetRequiresEveryFan() {
        let bothRamped = FanStatus(fans: [
            makeFan(index: 0, actual: 5774, target: 5777),
            makeFan(index: 1, actual: 5718, target: 5777),
        ])
        XCTAssertTrue(bothRamped.allFansNearTarget)
        let oneLagging = FanStatus(fans: [
            makeFan(index: 0, actual: 5774, target: 5777),
            makeFan(index: 1, actual: 2000, target: 5777),
        ])
        XCTAssertFalse(oneLagging.allFansNearTarget)
        // No fans at all is not "at target"; it is nothing measured.
        XCTAssertFalse(FanStatus(fans: []).allFansNearTarget)
    }

    func test_hottestSensorPicksTheMaximum() throws {
        let status = try decode(Self.realStatusJSON)
        XCTAssertEqual(status.hottestSensor, FanStatus.SensorReading(name: "TCDX", celsius: 81.9))
    }

    // MARK: - Quit decision

    func test_quitDecisionRestoresByDefaultAndHonorsTheToggle() {
        // Held + default: restore. That is the whole safety net.
        XCTAssertTrue(FanController.shouldRestoreOnQuit(heldAnyFan: true, keepPinnedOnQuit: false))
        // Held + keep toggle: the user owns the machine state.
        XCTAssertFalse(FanController.shouldRestoreOnQuit(heldAnyFan: true, keepPinnedOnQuit: true))
        // Never held: nothing to restore, either way.
        XCTAssertFalse(FanController.shouldRestoreOnQuit(heldAnyFan: false, keepPinnedOnQuit: false))
    }

    // MARK: - Binary discovery

    func test_locateExecutablePrefersTheFirstPATHDirectory() throws {
        let dirA = try makeExecutableFixture()
        let dirB = try makeExecutableFixture()
        defer { cleanup(URL(fileURLWithPath: dirA.path)); cleanup(URL(fileURLWithPath: dirB.path)) }

        let first = FanController.locateExecutable(pathEnvironment: "\(dirA.path):\(dirB.path)")
        XCTAssertEqual(first?.path, dirA.appendingPathComponent("thermalforge").path)

        let second = FanController.locateExecutable(pathEnvironment: dirB.path)
        XCTAssertEqual(second?.path, dirB.appendingPathComponent("thermalforge").path)
    }

    func test_locateExecutableRejectsMissingAndNonExecutable() throws {
        // knownPaths: [] throughout: the defaults point at this machine's
        // real install, which would satisfy every lookup below.
        XCTAssertNil(FanController.locateExecutable(
            pathEnvironment: "/nonexistent-fanforge-dir", knownPaths: []))
        XCTAssertNil(FanController.locateExecutable(pathEnvironment: "", knownPaths: []))
        XCTAssertNil(FanController.locateExecutable(pathEnvironment: "::", knownPaths: []))
        // A non-executable file does not count (isExecutableFile, not exists).
        let dir = try makeNonExecutableFixture()
        defer { cleanup(URL(fileURLWithPath: dir.path)) }
        XCTAssertNil(FanController.locateExecutable(pathEnvironment: dir.path, knownPaths: []))
    }

    func test_locateExecutableFallsBackToKnownInstallPaths() throws {
        // A GUI launch inherits /usr/bin:/bin, where a Homebrew install
        // never appears; the known-location fallback is what keeps the
        // feature visible for exactly those users.
        let dir = try makeExecutableFixture()
        defer { cleanup(URL(fileURLWithPath: dir.path)) }
        let known = dir.appendingPathComponent("thermalforge").path
        let found = FanController.locateExecutable(
            pathEnvironment: "/usr/bin:/bin", knownPaths: [known])
        XCTAssertEqual(found?.path, known)
        // PATH still wins when it resolves something.
        let dirB = try makeExecutableFixture()
        defer { cleanup(URL(fileURLWithPath: dirB.path)) }
        let alsoFound = FanController.locateExecutable(
            pathEnvironment: dirB.path, knownPaths: [known])
        XCTAssertEqual(alsoFound?.path, dirB.appendingPathComponent("thermalforge").path)
    }

    func test_controllerIsUnavailableForAMissingBinary() {
        let controller = FanController(
            executablePath: "/nonexistent-fanforge-dir/thermalforge", pollInterval: nil)
        XCTAssertFalse(controller.isAvailable)
    }

    func test_missingBinaryDoesNotStartPollingTimer() {
        let controller = FanController(
            executablePath: "/nonexistent-fanforge-dir/thermalforge", pollInterval: 1.0)
        XCTAssertFalse(controller.isAvailable)
        XCTAssertFalse(controller.isPolling)
    }

    func test_pollingTimerStartsWhenAvailableAndStopsOnRequest() throws {
        let dir = try makeExecutableFixture()
        defer { cleanup(URL(fileURLWithPath: dir.path)) }
        let binary = dir.appendingPathComponent("thermalforge").path
        let controller = FanController(executablePath: binary, pollInterval: 10.0)
        XCTAssertTrue(controller.isAvailable)
        XCTAssertTrue(controller.isPolling)
        controller.stopPolling()
        XCTAssertFalse(controller.isPolling)
    }

    // MARK: - End to end through the fixture script

    func test_refreshStatusDecodesFixtureOutput() async throws {
        let script = try makeFixtureScript()
        defer { cleanup(script) }
        let controller = FanController(executablePath: script.path, pollInterval: nil)
        await controller.refreshStatus()
        XCTAssertNil(controller.statusError)
        XCTAssertEqual(controller.status?.fans.first?.mode, "auto")
        XCTAssertEqual(controller.status?.fans.first?.actualRPM, 1876)
    }

    func test_pinMaxVerifiesTheRampAndRestoresIt() async throws {
        let script = try makeFixtureScript()
        defer { cleanup(script) }
        let controller = FanController(executablePath: script.path, pollInterval: nil)
        await controller.pinMax()
        XCTAssertNil(controller.lastError)
        XCTAssertEqual(controller.status?.fans.first?.mode, "manual")
        XCTAssertTrue(controller.status?.allFansNearTarget ?? false)
        XCTAssertFalse(controller.isBusy)

        await controller.restoreAuto()
        XCTAssertNil(controller.lastError)
        XCTAssertEqual(controller.status?.fans.first?.mode, "auto")
    }

    func test_pinMaxReportsADaemonFailure() async throws {
        let script = try makeFixtureScript(failsMax: true)
        defer { cleanup(script) }
        let controller = FanController(executablePath: script.path, pollInterval: nil)
        await controller.pinMax()
        // The hint names the daemon, because a bare exit code is not
        // actionable and the usual cause is the socket being down.
        XCTAssertTrue(controller.lastError?.contains("daemon") ?? false)
        XCTAssertFalse(controller.isBusy)
    }

    func test_restoreOnQuitIfNeededRestoresObservedHoldsOnly() async throws {
        let script = try makeFixtureScript()
        let dir = script.deletingLastPathComponent()
        defer { cleanup(script) }
        let marker = dir.appendingPathComponent("restored")
        let controller = FanController(executablePath: script.path, pollInterval: nil)

        // Nothing held: the quit path must not touch the machine.
        await controller.refreshStatus()
        controller.restoreOnQuitIfNeeded()
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))

        // A hold appears, but the user asked to keep it: still untouched.
        try? "".write(to: dir.appendingPathComponent("pinned"), atomically: true, encoding: .utf8)
        await controller.refreshStatus()
        controller.keepFansPinnedOnQuit = true
        controller.restoreOnQuitIfNeeded()
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))

        // Default: the observed hold is released, synchronously.
        controller.keepFansPinnedOnQuit = false
        controller.restoreOnQuitIfNeeded()
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
        await controller.refreshStatus()
        XCTAssertEqual(controller.status?.fans.first?.mode, "auto")
    }

    // MARK: - Fixture helpers

    private func makeExecutableFixture() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("FanControllerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let binary = dir.appendingPathComponent("thermalforge")
        try "#!/bin/sh".write(to: binary, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: binary.path)
        return dir
    }

    private func makeNonExecutableFixture() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("FanControllerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "#!/bin/sh".write(
            to: dir.appendingPathComponent("thermalforge"), atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o644], ofItemAtPath: dir.appendingPathComponent("thermalforge").path)
        return dir
    }
}
