import Foundation
import Sparkle

/// Owns the Sparkle updater for the whole app: one controller, created once,
/// shared by the app-menu item and the General settings pane.
///
/// WHY A GUARD INSTEAD OF ALWAYS STARTING: `swift run` builds are bare
/// executables -- no `TurboSpark.app` wrapper, so no Info.plist, no SUFeedURL,
/// and a different preferences domain than the installed app (swift/CLAUDE.md
/// Gotcha 12). Starting Sparkle there produces console errors and, on the
/// second run, update prompts against a bundle that cannot be sanely updated.
/// Sparkle only starts when the executable actually lives inside a .app.
///
/// THE FEED IS THE GITHUB RELEASE ITSELF. The DMG we already attach to every
/// release is a supported Sparkle 2 update archive, and the feed is the
/// `appcast.xml` release asset reached through the evergreen
/// `releases/latest/download/` URL -- no feed host beyond GitHub Releases.
/// Trust comes from the EdDSA signature (`SUPublicEDKey` in the bundle's
/// Info.plist), not from Apple code-signing identity: the app is ad-hoc
/// signed, so the signature comparison that needs a Team ID cannot be used.
@MainActor
final class SparkleUpdateController: NSObject, ObservableObject {
    static let shared = SparkleUpdateController()

    /// Mirrors Sparkle's automatic-check preference for the settings toggle.
    /// Sparkle owns persistence; the observation keeps this value current when
    /// Sparkle changes it outside the SwiftUI binding.
    @Published var automaticallyChecksForUpdates: Bool {
        didSet {
            guard let updater else { return }
            if updater.automaticallyChecksForUpdates != automaticallyChecksForUpdates {
                updater.automaticallyChecksForUpdates = automaticallyChecksForUpdates
            }
        }
    }

    /// KVO-backed because scheduled checks start without going through this
    /// wrapper. Both manual update buttons use this value.
    @Published private(set) var canCheckForUpdates = false

    private var controller: SPUStandardUpdaterController?
    private var updater: SPUUpdater?
    private var silentDriver: AutoInstallUserDriver?
    private var automaticChecksObservation: NSKeyValueObservation?
    private var canCheckObservation: NSKeyValueObservation?

    /// Debug/harness override: point the feed at any appcast (used by the
    /// local two-version update smoke against a localhost server). Deliberate
    /// and documented rather than DEBUG-only -- the smoke must run against a
    /// release build, and EdDSA still enforces feed trust, so an attacker
    /// controlled override can only serve signed updates.
    static var feedURLOverride: String? {
        ProcessInfo.processInfo.environment["TURBOSPARK_UPDATE_FEED_URL"]
    }

    /// Harness mode: with the feed override set AND this flag, run the
    /// updater behind `AutoInstallUserDriver` so the two-version smoke can
    /// prove download -> verify -> install -> RELAUNCH unattended (the
    /// standard driver needs a human at its relaunch dialog). Normal runs
    /// never set both variables; nothing else changes.
    static var silentUpdateMode: Bool {
        ProcessInfo.processInfo.environment["TURBOSPARK_UPDATE_SILENT"] == "1"
            && feedURLOverride != nil
    }

    /// True only for a real bundle: the guard that keeps `swift run` builds
    /// away from Sparkle. Also the switch the settings pane uses to disable
    /// controls that would be dead weight in a dev run.
    static var isBundledApp: Bool {
        Bundle.main.bundleURL.pathExtension == "app"
    }

    /// The stamped release version, straight from the bundle the same way
    /// ChatParitySheets and the profile exporters read it.
    static var displayVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString")
            as? String ?? "unknown"
    }

    override private init() {
        automaticallyChecksForUpdates = false
        super.init()
    }

    /// Idempotent. Starts Sparkle only in a real bundle; every call site
    /// can call it unconditionally.
    func start() {
        guard Self.isBundledApp, updater == nil else {
            if !Self.isBundledApp {
                NSLog(
                    "TurboSpark updates: not a bundled app; update checker disabled"
                )
            }
            return
        }

        // The standard controller schedules the next automatic check when it
        // starts. It fires immediately only when the saved interval is due.
        // Sparkle must be created on the main thread; the whole class is
        // MainActor for that reason.
        if Self.silentUpdateMode {
            let driver = AutoInstallUserDriver()
            silentDriver = driver
            let u = SPUUpdater(
                hostBundle: Bundle.main,
                applicationBundle: Bundle.main,
                userDriver: driver,
                delegate: self)
            // Disable the scheduler only while starting. Otherwise a recent
            // SULastCheckTime can delay the smoke, while an overdue check can
            // race the forced check below. Restore the user's preference once
            // the manual session owns the updater.
            let automaticChecks = u.automaticallyChecksForUpdates
            u.automaticallyChecksForUpdates = false
            do {
                try u.start()
            } catch {
                u.automaticallyChecksForUpdates = automaticChecks
                NSLog("TurboSpark updates: failed to start silent updater: %@", error.localizedDescription)
                return
            }
            attach(to: u)
            NSLog(
                "TurboSpark updates: Sparkle started in silent harness mode (version %@, feed %@)",
                Self.displayVersion,
                u.feedURL?.absoluteString ?? "nil")
            u.checkForUpdates()
            u.automaticallyChecksForUpdates = automaticChecks
            return
        }

        let vc = SPUStandardUpdaterController(
            startingUpdater: true,
            updaterDelegate: self,
            userDriverDelegate: nil)
        controller = vc
        attach(to: vc.updater)
        NSLog(
            "TurboSpark updates: Sparkle started (version %@, feed %@)",
            Self.displayVersion,
            vc.updater.feedURL?.absoluteString ?? "nil")
    }

    func checkForUpdates() {
        guard let updater, updater.canCheckForUpdates else {
            NSLog("TurboSpark updates: manual check requested while updater is unavailable")
            return
        }
        updater.checkForUpdates()
    }

    private func attach(to updater: SPUUpdater) {
        self.updater = updater
        automaticallyChecksForUpdates = updater.automaticallyChecksForUpdates
        canCheckForUpdates = updater.canCheckForUpdates

        automaticChecksObservation = updater.observe(
            \.automaticallyChecksForUpdates, options: [.new]
        ) { [weak self] _, change in
            let value = change.newValue ?? false
            MainActor.assumeIsolated { [weak self] in
                self?.automaticallyChecksForUpdates = value
            }
        }
        canCheckObservation = updater.observe(
            \.canCheckForUpdates, options: [.new]
        ) { [weak self] _, change in
            let value = change.newValue ?? false
            MainActor.assumeIsolated { [weak self] in
                self?.canCheckForUpdates = value
            }
        }
    }
}

extension SparkleUpdateController: SPUUpdaterDelegate {
    func feedURLString(for updater: SPUUpdater) -> String? {
        Self.feedURLOverride
    }

    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        NSLog(
            "TurboSpark updates: found version %@ (current %@)",
            item.displayVersionString,
            Self.displayVersion)
    }

    func updaterDidNotFindUpdate(_ updater: SPUUpdater) {
        NSLog(
            "TurboSpark updates: no update found (current %@)",
            Self.displayVersion)
    }

    func updater(_ updater: SPUUpdater, didAbortWithError error: Error) {
        NSLog("TurboSpark updates: aborted: \(error.localizedDescription)")
    }

    func updater(_ updater: SPUUpdater, willInstallUpdate item: SUAppcastItem) {
        NSLog(
            "TurboSpark updates: installing version %@", item.displayVersionString)
    }

    func updater(
        _ updater: SPUUpdater,
        userDidMake choice: SPUUserUpdateChoice,
        forUpdate update: SUAppcastItem,
        state: SPUUserUpdateState
    ) {
        let message = "delegate saw user choice \(choice.rawValue) for \(update.displayVersionString)"
        if Self.silentUpdateMode {
            AutoInstallUserDriver.harnessLog(message)
        } else {
            NSLog("TurboSpark updates: %@", message)
        }
    }
}

/// Harness-only user driver (see `SparkleUpdateController.silentUpdateMode`):
/// answers every Sparkle prompt with "yes" and logs the funnel so the
/// two-version smoke can assert download -> verify -> install -> relaunch.
/// Logs go straight to a file next to the rest of the smoke artifacts --
/// stderr from the installer-side callbacks has proven unreliable to
/// capture. Never used in normal runs.
@MainActor
final class AutoInstallUserDriver: NSObject, SPUUserDriver {
    private static let harnessLogURL = URL(
        fileURLWithPath: "/tmp/turbospark-update-smoke.log")

    override init() {
        do {
            try Data().write(to: Self.harnessLogURL, options: .atomic)
        } catch {
            NSLog(
                "TurboSpark updates[smoke]: failed to reset log: %@",
                error.localizedDescription)
        }
        super.init()
    }

    static func harnessLog(_ message: String) {
        NSLog("TurboSpark updates[smoke]: %@", message)
        let line = "\(Date()) \(message)\n"
        do {
            try appendHarnessLog(line, to: harnessLogURL)
        } catch {
            NSLog(
                "TurboSpark updates[smoke]: failed to append log: %@",
                error.localizedDescription)
        }
    }

    static func appendHarnessLog(_ line: String, to url: URL) throws {
        let data = Data(line.utf8)
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            _ = try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } else {
            try data.write(to: url, options: .atomic)
        }
    }

    private func log(_ message: String) {
        Self.harnessLog("\(message) [thread=\(Thread.current)]")
    }

    // Sparkle's reply blocks dispatch their work async to the main queue;
    // replying from the callback synchronously starves that handoff in
    // practice, so match the standard driver and reply on a later turn.
    private func replyLater(_ reply: @escaping (SPUUserUpdateChoice) -> Void, _ choice: SPUUserUpdateChoice) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            reply(choice)
        }
    }

    func show(
        _ request: SPUUpdatePermissionRequest,
        reply: @escaping (SUUpdatePermissionResponse) -> Void
    ) {
        log("permission request -> allow")
        reply(SUUpdatePermissionResponse(automaticUpdateChecks: true, sendSystemProfile: false))
    }

    func showReady(toInstallAndRelaunch reply: @escaping (SPUUserUpdateChoice) -> Void) {
        log("ready -> install and relaunch")
        replyLater(reply, .install)
    }

    func showUserInitiatedUpdateCheck(cancellation: @escaping () -> Void) {
        log("user-initiated check")
    }

    func showUpdateFound(
        with appcastItem: SUAppcastItem,
        state: SPUUserUpdateState,
        reply: @escaping (SPUUserUpdateChoice) -> Void
    ) {
        log("update found \(appcastItem.displayVersionString) -> install")
        replyLater(reply, .install)
    }

    func showUpdateReleaseNotes(with downloadData: SPUDownloadData) {}

    func showUpdateReleaseNotesFailedToDownloadWithError(_ error: Error) {}

    func showUpdateNotFoundWithError(_ error: Error) async {
        log("no update found: \(error.localizedDescription)")
    }

    func showUpdaterError(_ error: Error) async {
        log("updater error: \(error.localizedDescription)")
    }

    func showDownloadInitiated(cancellation: @escaping () -> Void) {
        log("download started")
    }

    func showDownloadDidReceiveExpectedContentLength(_ expectedContentLength: UInt64) {}

    func showDownloadDidReceiveData(ofLength length: UInt64) {}

    func showDownloadDidStartExtractingUpdate() {
        log("extracting")
    }

    func showExtractionReceivedProgress(_ progress: Double) {}

    func showInstallingUpdate(
        withApplicationTerminated applicationTerminated: Bool,
        retryTerminatingApplication: @escaping () -> Void
    ) {
        log("installing (appTerminated=\(applicationTerminated))")
    }

    func showUpdateInstalledAndRelaunched(_ relaunched: Bool) async {
        log("installed and relaunched=\(relaunched)")
    }

    func dismissUpdateInstallation() {
        log("installation dismissed")
    }
}
