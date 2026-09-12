import AppKit
import Foundation

/// Decoded `thermalforge status` output (github.com/ProducerGuy/ThermalForge).
///
/// Schema captured on Mac16,5 2026-09-06; `temperatures` is defaulted rather
/// than required because its key set is whatever SMC keys the machine exposes
/// and a future version may omit the map entirely.
struct FanStatus: Codable, Equatable {
    struct Fan: Codable, Equatable, Identifiable {
        var index: Int
        var actualRPM: Int
        var targetRPM: Int
        var minRPM: Int
        var maxRPM: Int
        /// "auto" under the machine's own curve, "manual" while a hold is
        /// set (`thermalforge max` / `set`). Anything that is not "auto"
        /// counts as held, so a future profile mode stays honest.
        var mode: String

        var id: Int { index }
        var isHeld: Bool { mode != "auto" }
        /// Fans settle NEAR the commanded target, never exactly on it, so
        /// the ramp check is a fraction rather than an equality. In "auto"
        /// the target can sit slightly below actual, which also satisfies
        /// this; the predicate is only consumed after a `max` pin.
        var isNearTarget: Bool {
            Double(actualRPM) >= FanStatus.rampFraction * Double(targetRPM)
        }

        enum CodingKeys: String, CodingKey {
            case index
            case actualRPM = "actual_rpm"
            case targetRPM = "target_rpm"
            case minRPM = "min_rpm"
            case maxRPM = "max_rpm"
            case mode
        }
    }

    struct SensorReading: Equatable {
        var name: String
        var celsius: Double
    }

    /// Fraction of the commanded target the tach must reach for a pin to
    /// count as holding. Measured ramp on Mac16,5 lands within 1% of target.
    static let rampFraction = 0.95

    var fans: [Fan]
    var temperatures: [String: Double]

    var anyFanHeld: Bool { fans.contains(where: \.isHeld) }
    var allFansNearTarget: Bool { !fans.isEmpty && fans.allSatisfy(\.isNearTarget) }
    var hottestSensor: SensorReading? {
        temperatures.max { $0.value < $1.value }.map { SensorReading(name: $0.key, celsius: $0.value) }
    }

    init(fans: [Fan], temperatures: [String: Double] = [:]) {
        self.fans = fans
        self.temperatures = temperatures
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.fans = try container.decode([Fan].self, forKey: .fans)
        self.temperatures = try container.decodeIfPresent([String: Double].self, forKey: .temperatures) ?? [:]
    }

    enum CodingKeys: String, CodingKey {
        case fans
        case temperatures
    }
}

/// App-lifetime bridge to the `thermalforge` CLI for status-bar fan control.
///
/// **A HOLD SET THROUGH THIS CLI SURVIVES THIS APP'S DEATH.** `thermalforge
/// max` talks to a root LaunchDaemon over `/tmp/thermalforge.sock`, and the
/// watchdog inside ThermalForge covers only its own menu bar app, so once
/// this process pins the fans nothing restores them but an explicit
/// `thermalforge auto`. That is why the quit restore exists
/// (`restoreOnQuitIfNeeded`, called from the app delegate) and why its
/// toggle defaults to restoring.
///
/// Control commands need the daemon; `status` does not (it reads the SMC
/// directly), so the RPM readout keeps working when the daemon is down and
/// only the buttons fail. The daemon answers "Failed to connect to daemon
/// socket" for a short window after a restart, so every control command is
/// retried once after a short delay.
@MainActor
final class FanController: ObservableObject {
    // AppModel loads this singleton even without a fan view. Tests must not
    // discover or control the host hardware; explicit fixture instances still work.
    static let shared = FanController(
        executablePath: AppStorageRoot.isRunningTests ? "" : nil,
        pollInterval: AppStorageRoot.isRunningTests ? nil : defaultPollInterval)

    @Published private(set) var isAvailable: Bool
    @Published private(set) var status: FanStatus?
    @Published private(set) var isBusy = false
    /// From `max` / `auto` failing, or the post-pin ramp check timing out.
    @Published private(set) var lastError: String?
    /// From the 5 s status poll, kept apart from `lastError` so a poll
    /// failure cannot clear a control error and vice versa.
    @Published private(set) var statusError: String?
    /// Live copy of the persisted setting, written by `AppModel` at load and
    /// by the popover toggle. The settings write is debounced, so the quit
    /// path must not read it back off disk.
    @Published var keepFansPinnedOnQuit = false

    private let executableURL: URL?
    private var pollTimer: Timer?
    private let pollInterval: TimeInterval?

    nonisolated static let defaultPollInterval: TimeInterval = 5

    /// `executablePath` nil means locate `thermalforge` on `PATH`. Tests pass
    /// an explicit path (real or nonexistent) and `pollInterval: nil`.
    init(executablePath: String? = nil, pollInterval: TimeInterval? = FanController.defaultPollInterval) {
        let resolvedPath =
            executablePath
            ?? Self.locateExecutable(
                pathEnvironment: ProcessInfo.processInfo.environment["PATH"] ?? ""
            )?.path
        self.executableURL = resolvedPath.map(URL.init(fileURLWithPath:))
        self.isAvailable = {
            guard let resolvedPath else { return false }
            return FileManager.default.isExecutableFile(atPath: resolvedPath)
        }()
        self.pollInterval = pollInterval
        startPolling()
    }

    /// Resolves `thermalforge`: PATH directories first, mirroring how a
    /// shell would resolve the bare name `power.sh` invokes, then the
    /// standard install locations. The fallback exists because a GUI app
    /// launched from Finder or the Dock inherits a minimal PATH
    /// (/usr/bin:/bin:...) where neither Homebrew prefix ever appears, so
    /// PATH alone would hide the feature from exactly the users who have
    /// the binary installed.
    nonisolated static let knownInstallPaths = [
        "/usr/local/bin/thermalforge",
        "/opt/homebrew/bin/thermalforge",
    ]

    nonisolated static func locateExecutable(
        pathEnvironment: String,
        knownPaths: [String] = FanController.knownInstallPaths,
        fileManager: FileManager = .default
    ) -> URL? {
        var candidates: [String] = []
        for dir in pathEnvironment.split(separator: ":", omittingEmptySubsequences: true) {
            candidates.append(
                URL(fileURLWithPath: String(dir)).appendingPathComponent("thermalforge").path)
        }
        candidates.append(contentsOf: knownPaths)
        for path in candidates where fileManager.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        return nil
    }

    var isPolling: Bool { pollTimer != nil }

    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func startPolling() {
        guard isAvailable, let interval = pollInterval else { return }
        Task { [weak self] in await self?.refreshStatus() }
        pollTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in await self?.refreshStatus() }
        }
    }

    // MARK: - Actions

    /// Pins every fan to maximum. Verifies the ramp by polling `status`:
    /// an exit-0 `max` is not proof the daemon is holding.
    func pinMax() async {
        guard isAvailable, !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        guard await runControl(["max"]) != nil else {
            lastError = Self.daemonHint(command: "max")
            return
        }
        lastError = nil
        for _ in 0..<10 {
            await refreshStatus()
            if status?.allFansNearTarget == true { return }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        lastError = "Fans did not reach target RPM within 10 s; the pin may not be holding."
    }

    /// Returns the fans to the machine's own curve.
    func restoreAuto() async {
        guard isAvailable, !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        guard await runControl(["auto"]) != nil else {
            lastError = Self.daemonHint(command: "auto")
            return
        }
        lastError = nil
        await refreshStatus()
    }

    /// Runs from `applicationWillTerminate`, synchronously on purpose: the
    /// terminate path cannot await, so this spawns a bare `Process` with a
    /// bounded wait. Only a hold we actually OBSERVED in a status poll is
    /// restored, and only when the user has not asked to keep it.
    func restoreOnQuitIfNeeded() {
        guard let executableURL else { return }
        // `status` comes from an async poll that may not have landed yet --
        // quitting shortly after launch, before the first `refreshStatus()`
        // completes, sees `status == nil` here. Reading that as "nothing
        // held" is the failure this function exists to prevent: fans pinned
        // by a PREVIOUS crashed or force-quit session then stay pinned
        // indefinitely, because every subsequent quick launch-and-quit skips
        // the restore the same way. `applicationWillTerminate` cannot await,
        // so when nothing has been observed yet this falls back to a
        // synchronous probe (mirroring `runSync`'s bounded-wait `Process`
        // use below) rather than assuming; the common case (status already
        // polled at least once) costs no extra process spawn.
        let heldAnyFan = status?.anyFanHeld ?? statusSync(from: executableURL)?.anyFanHeld ?? false
        guard Self.shouldRestoreOnQuit(heldAnyFan: heldAnyFan, keepPinnedOnQuit: keepFansPinnedOnQuit)
        else { return }
        runSync(["auto"], from: executableURL)
    }

    /// Synchronous `thermalforge status`, for `restoreOnQuitIfNeeded`'s one
    /// caller only -- the terminate path cannot await `refreshStatus()`, and
    /// treating "never polled" as "not held" is exactly the mistake this
    /// exists to avoid. Bounded the same 2 s `runSync` waits below.
    private func statusSync(from executableURL: URL) -> FanStatus? {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = ["status"]
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        var collected = Data()
        let deadline = Date().addingTimeInterval(2.0)
        while process.isRunning && Date() < deadline {
            collected.append(outputPipe.fileHandleForReading.availableData)
            usleep(50_000)
        }
        if process.isRunning {
            process.terminate()
            return nil
        }
        collected.append(outputPipe.fileHandleForReading.readDataToEndOfFile())
        guard process.terminationStatus == 0 else { return nil }
        return try? JSONDecoder().decode(FanStatus.self, from: collected)
    }

    /// The quit decision, factored out pure for the test suite.
    nonisolated static func shouldRestoreOnQuit(heldAnyFan: Bool, keepPinnedOnQuit: Bool) -> Bool {
        heldAnyFan && !keepPinnedOnQuit
    }

    private static func daemonHint(command: String) -> String {
        "thermalforge \(command) failed. Is the ThermalForge daemon running?"
            + " Install it once with: sudo thermalforge install"
    }

    // MARK: - Plumbing

    func refreshStatus() async {
        guard let executableURL else { return }
        guard
            let output = try? await ProcessExecutor.run(
                executableURL: executableURL, arguments: ["status"],
                timeoutSeconds: 5, outputCapBytes: 256_000),
            output.exitCode == 0
        else {
            statusError = "thermalforge status failed."
            return
        }
        guard let decoded = try? JSONDecoder().decode(FanStatus.self, from: Data(output.stdout.utf8)) else {
            statusError = "thermalforge status returned unreadable output."
            return
        }
        self.status = decoded
        self.statusError = nil
    }

    /// One control command, retried once: the daemon refuses connections for
    /// a short window after a restart (observed 2026-09-06).
    private func runControl(_ arguments: [String]) async -> ProcessExecutor.Output? {
        guard let executableURL else { return nil }
        for attempt in 0...1 {
            if let output = try? await ProcessExecutor.run(
                executableURL: executableURL, arguments: arguments,
                timeoutSeconds: 10, outputCapBytes: 64_000),
                output.exitCode == 0
            {
                return output
            }
            if attempt == 0 {
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
        return nil
    }

    private func runSync(_ arguments: [String], from executableURL: URL) {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        do { try process.run() } catch { return }
        let deadline = Date().addingTimeInterval(2.0)
        while process.isRunning && Date() < deadline {
            usleep(50_000)
        }
        if process.isRunning { process.terminate() }
    }
}
