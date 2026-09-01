import Foundation

/// The one directory every on-disk store lives under.
///
/// **THIS EXISTS BECAUSE THE TEST SUITE DESTROYED A USER'S CHATS.**
/// Seven call sites each spelled
/// `applicationSupportDirectory + "TurboSpark"` for themselves, with no test
/// seam anywhere, so an `AppModel()` built in a test read and WROTE the real
/// `~/Library/Application Support/TurboSpark/`. `AppModel.chats` persists on
/// mutation and the archive is written WHOLE with `.atomic`, so a test that
/// set up a one-chat fixture replaced the user's entire history with it --
/// silently, on every `swift test`, and the fixture reappeared each run after
/// being deleted in the UI. Found 2026-08-31 with `HookDecisionRoutingTests`'s
/// "T10 chat" sitting alone in a real archive.
///
/// Two properties make that unrepeatable. The redirect is **automatic**, so a
/// new test cannot forget to opt in -- the failure mode is silent data loss,
/// which is exactly the kind nobody discovers by writing a test for it. And it
/// keys on the TEST HOST rather than on a flag any one test sets, because the
/// stores are reached through `.shared` singletons and static enums whose
/// first touch can happen before any test body runs.
public enum AppStorageRoot {
    /// Set this to redirect the stores anywhere, for a sandbox or a fixture.
    public static let overrideEnvironmentKey = "TURBOSPARK_STATE_DIR"

    /// Whether this process is a test runner rather than the app.
    ///
    /// `XCTestConfigurationFilePath` is set by `xctest` for an XCTest bundle;
    /// the `XCTestCase` class lookup covers a runner that does not set it, and
    /// costs one Objective-C lookup once.
    public static var isRunningTests: Bool {
        let environment = ProcessInfo.processInfo.environment
        if environment["XCTestConfigurationFilePath"] != nil { return true }
        if environment["XCTestBundlePath"] != nil { return true }
        if environment["XCTestSessionIdentifier"] != nil { return true }
        return NSClassFromString("XCTestCase") != nil
    }

    /// The store root: the user's Application Support directory in the app, a
    /// throwaway directory under a test host, or whatever the override names.
    public static let directory: URL = {
        let fileManager = FileManager.default

        if let override = ProcessInfo.processInfo.environment[overrideEnvironmentKey],
           !override.isEmpty {
            let url = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
            try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        if isRunningTests {
            // Per process, so parallel suites cannot collide, and under the
            // temp directory so nothing survives to be mistaken for real data.
            let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
                .appendingPathComponent(
                    "TurboSparkTests-\(ProcessInfo.processInfo.processIdentifier)",
                    isDirectory: true)
            try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        let appSupport = fileManager
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let url = appSupport.appendingPathComponent("TurboSpark", isDirectory: true)
        try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()

    /// A subdirectory of the store root, created on demand.
    public static func subdirectory(_ name: String) -> URL {
        let url = directory.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A file directly under the store root.
    public static func file(_ name: String) -> URL {
        directory.appendingPathComponent(name)
    }
}
