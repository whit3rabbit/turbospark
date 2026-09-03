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

/// Shared read/write behaviour for the app's JSON stores.
///
/// Every store spelled its own `try? data.write(...)` and its own decode
/// fallback, and the two halves failed differently and badly:
///
/// **A SWALLOWED WRITE LOSES THE SESSION WITH NO SIGN** (state#24). `save()` was
/// `try? encoder.encode(...)` then `try? data.write(...)`, so a full disk, a
/// permissions problem or a directory that `AppStorageRoot` failed to create
/// (it uses `try?` too) discarded the whole archive silently on quit.
/// `load()` was made to report to stderr for exactly this reason years after
/// the fact; `save()` is the mirror and had never been.
///
/// **A SWALLOWED DECODE BECOMES REAL DATA LOSS ON THE NEXT WRITE.** Reporting
/// a decode failure and returning the empty default is right as far as it
/// goes -- but the file is still on disk, and the next mutation overwrites it
/// whole and atomically (`swift/CLAUDE.md` Gotcha 13). By the time anyone
/// reads the stderr line the original is gone. The file is QUARANTINED
/// instead: renamed with a timestamp, so a fresh one is written beside it and
/// the user's data is recoverable by hand.
public enum AppJSONStore {
    /// The last write failure, for a caller that wants to surface it.
    ///
    /// Not thrown: `save()` is called from `didSet`-shaped paths that cannot
    /// propagate, and the point is that the failure stops being INVISIBLE.
    public private(set) nonisolated(unsafe) static var lastWriteError: String?

    /// Clears the recorded write failure once it has been shown.
    public static func clearLastWriteError() {
        lastWriteError = nil
    }

    /// Decodes `url`, quarantining it if it is present but unreadable.
    ///
    /// Returns nil when the file is absent (a first run, which is not an
    /// error) or when it was quarantined; the caller supplies its own empty
    /// default, as before.
    public static func load<T: Decodable>(
        _ type: T.Type, from url: URL, label: String,
        decoder: JSONDecoder = JSONDecoder()
    ) -> T? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        do {
            return try decoder.decode(type, from: data)
        } catch {
            let quarantined = quarantine(url)
            FileHandle.standardError.write(
                ("TurboSpark: \(label) failed to decode, starting empty: \(error)\n"
                    + (quarantined.map { "TurboSpark: the unreadable file was kept at \($0.path)\n" }
                        ?? "TurboSpark: the unreadable file could NOT be preserved\n"))
                    .data(using: .utf8)!)
            return nil
        }
    }

    /// Writes `value` atomically, recording rather than swallowing a failure.
    @discardableResult
    public static func save<T: Encodable>(
        _ value: T, to url: URL, label: String,
        encoder: JSONEncoder = JSONEncoder()
    ) -> Bool {
        do {
            let data = try encoder.encode(value)
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            let message = "\(label) could not be saved: \(error.localizedDescription)"
            lastWriteError = message
            FileHandle.standardError.write("TurboSpark: \(message)\n".data(using: .utf8)!)
            return false
        }
    }

    /// Moves an unreadable file aside, returning where it went.
    private static func quarantine(_ url: URL) -> URL? {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let destination = url.deletingPathExtension()
            .appendingPathExtension("corrupt-\(stamp)")
            .appendingPathExtension(url.pathExtension)
        do {
            try FileManager.default.moveItem(at: url, to: destination)
            return destination
        } catch {
            return nil
        }
    }
}
