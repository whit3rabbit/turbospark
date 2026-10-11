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
private var testStorageDirectoryToClean: URL?

private func registerTestStorageCleanup(_ url: URL) {
    testStorageDirectoryToClean = url
    atexit {
        if let url = testStorageDirectoryToClean {
            try? FileManager.default.removeItem(at: url)
        }
    }
}

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

    /// Where the base root lives, and whether a user profile may redirect it.
    /// `profilesApply` is false exactly when the root was pinned by the
    /// override environment or a test host: profiles are a real-app concept,
    /// and a redirected run or a test must always see the root it was given,
    /// never someone's profile folder.
    private static let resolvedBase: (url: URL, profilesApply: Bool) = {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment

        if let override = environment[overrideEnvironmentKey],
           !override.isEmpty {
            let url = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
            try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            return (url, false)
        }

        if isRunningTests {
            // Per process, so parallel suites cannot collide, and per LAUNCH:
            // process ids get reused, and a run handed a previous run's
            // directory reads that run's leftover stores as its own. The pid
            // stays in the name so a leaked directory still traces to the
            // process that made it. Under the temp directory, so nothing
            // survives to be mistaken for real data.
            let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
                .appendingPathComponent(
                    "TurboSparkTests-\(ProcessInfo.processInfo.processIdentifier)-\(UUID().uuidString)",
                    isDirectory: true)
            try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
            registerTestStorageCleanup(url)
            return (url, false)
        }

        let appSupport = fileManager
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let url = appSupport.appendingPathComponent("TurboSpark", isDirectory: true)
        try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        return (url, true)
    }()

    /// The machine-level store root, shared by every profile: the Default
    /// user's stores live here directly, other profiles in `profiles/<id>/`
    /// under it, and the profile registry itself sits at its top level.
    public static var machineRoot: URL { resolvedBase.url }

    /// The store root for THIS run: the machine root for the Default user,
    /// that user's `profiles/<id>/` folder for anyone else. Evaluated once,
    /// like before profiles existed -- which is also why switching profiles
    /// is a relaunch rather than a re-pointing.
    public static let directory: URL = {
        guard resolvedBase.profilesApply, let store = UserProfileStore.storeDirectory() else {
            return resolvedBase.url
        }
        return store
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

    /// Resolves a path stored in chat history. New generated artifacts use a
    /// relative path so a profile's archive never hardcodes its root; older
    /// absolute paths and remote URLs retain their existing meaning.
    public static func resolveStoredPath(_ path: String) -> String {
        if ManagedAssetStore.assetID(from: path) != nil {
            do {
                return try ManagedAssetStore.shared.materializedURL(for: path).path
            } catch {
                // The empty path is kept for callers that cannot take an
                // optional, but the cause must name the asset or every later
                // turn fails with an unattributable `image : ...` error.
                AppJSONStore.recordReadFailure(
                    label: "Managed file \(String(path.suffix(12)))", error: error)
                return ""
            }
        }
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") || URL(string: expanded)?.scheme != nil {
            return expanded
        }
        return directory.appendingPathComponent(expanded).standardizedFileURL.path
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
///
/// A `label` is a catalog key naming the store ("Chat archive"). The latched
/// message reaches a toast, so it resolves in the picker's language; the
/// stderr line resolves the same key in English, because a log line is not UI.
public enum AppJSONStore {
    /// The last write failure, for a caller that wants to surface it.
    ///
    /// Not thrown: `save()` is called from `didSet`-shaped paths that cannot
    /// propagate, and the point is that the failure stops being INVISIBLE.
    public private(set) static var lastWriteError: String? {
        get { latchLock.withLock { latchWriteError } }
        set { latchLock.withLock { latchWriteError = newValue } }
    }

    /// The error latches are written on the chat writer queue and read on the
    /// main thread; an unsynchronized String is a torn read at best.
    private static let latchLock = NSLock()
    private nonisolated(unsafe) static var latchWriteError: String?
    private nonisolated(unsafe) static var latchReadError: String?
    private nonisolated(unsafe) static var latchReadFailureCount = 0

    /// Clears the recorded write failure once it has been shown.
    public static func clearLastWriteError() {
        lastWriteError = nil
    }

    /// The last read failure, for a caller that wants to surface it.
    ///
    /// Same reason `lastWriteError` exists: `load()` is called from `init()`
    /// and cannot propagate, and the point is that the failure stops being
    /// invisible.
    public private(set) static var lastReadError: String? {
        get { latchLock.withLock { latchReadError } }
        set { latchLock.withLock { latchReadError = newValue } }
    }

    /// Monotonic count of read failures, so a caller can tell "this load
    /// failed" from "this load found nothing" without relying on the message.
    public private(set) static var readFailureCount: Int {
        get { latchLock.withLock { latchReadFailureCount } }
        set { latchLock.withLock { latchReadFailureCount = newValue } }
    }

    /// Clears the recorded read failure once it has been shown.
    public static func clearLastReadError() {
        lastReadError = nil
    }

    /// Decodes `url`, quarantining it if it is present but unreadable.
    ///
    /// Returns nil when the file is absent (a first run, which is not an
    /// error) or when it was quarantined; the caller supplies its own empty
    /// default, as before.
    ///
    /// **ABSENT AND UNREADABLE ARE DIFFERENT THINGS, AND `try?` MADE THEM ONE**
    /// (state#42). A permissions failure, an I/O error or a file the OS
    /// refuses to map came back as nil, the caller substituted its empty
    /// default, and the next mutation overwrote an INTACT file whole and
    /// atomically -- the exact data loss the decode arm below was written to
    /// prevent, reached by the one path that skipped it. A file that EXISTS
    /// and cannot be read is quarantined and reported like a corrupt one.
    public static func load<T: Decodable>(
        _ type: T.Type, from url: URL, label: String.LocalizationValue,
        decoder: JSONDecoder = JSONDecoder()
    ) -> T? {
        if let key = ProfileRepository.protectedRecordKey(for: url) {
            do {
                return try ProfileRepository.shared.load(type, key: key, legacyURL: url)
            } catch {
                recordReadFailure(label: label, error: error)
                return nil
            }
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            guard FileManager.default.fileExists(atPath: url.path) else {
                // A genuine first run. Not an error, and nothing to preserve.
                return nil
            }
            report(label: label, url: url, error: error, failure: .unreadable)
            return nil
        }
        do {
            return try decoder.decode(type, from: data)
        } catch {
            report(label: label, url: url, error: error, failure: .undecodable)
            return nil
        }
    }

    private enum ReadFailure {
        case unreadable
        case undecodable

        var logVerb: String {
            switch self {
            case .unreadable: return "could not be read"
            case .undecodable: return "failed to decode"
            }
        }
    }

    /// Quarantines an unreadable file and records why, on stderr and on
    /// `lastReadError`.
    private static func report(
        label: String.LocalizationValue, url: URL, error: Error, failure: ReadFailure
    ) {
        readFailureCount += 1
        let quarantined = quarantine(url)
        // Whole-sentence keys per failure, so no translation has to agree
        // with an English verb spliced into it.
        let name = AppStrings.localized(label)
        let detail = error.localizedDescription
        let summary: String
        switch failure {
        case .unreadable:
            summary = AppStrings.localized("\(name) could not be read, starting empty: \(detail).")
        case .undecodable:
            summary = AppStrings.localized("\(name) failed to decode, starting empty: \(detail).")
        }
        let kept =
            quarantined.map { AppStrings.localized("The unreadable file was kept at \($0.path).") }
            ?? AppStrings.localized("The unreadable file could not be preserved.")
        lastReadError = "\(summary) \(kept)"
        let keptLog =
            quarantined.map { "the unreadable file was kept at \($0.path)" }
            ?? "the unreadable file could NOT be preserved"
        FileHandle.standardError.write(
            ("TurboSpark: \(englishName(label)) \(failure.logVerb), starting empty: \(error)\n"
                + "TurboSpark: \(keptLog)\n").data(using: .utf8)!)
    }

    /// `label` for the stderr line, which stays English.
    private static func englishName(_ label: String.LocalizationValue) -> String {
        AppStrings.localized(label, language: .english)
    }

    /// Writes `value` atomically, recording rather than swallowing a failure.
    @discardableResult
    public static func save<T: Encodable>(
        _ value: T, to url: URL, label: String.LocalizationValue,
        encoder: JSONEncoder = JSONEncoder()
    ) -> Bool {
        if let key = ProfileRepository.protectedRecordKey(for: url) {
            do {
                try ProfileRepository.shared.save(value, key: key)
                return true
            } catch {
                recordWriteFailure(label: label, error: error)
                return false
            }
        }
        do {
            let data = try encoder.encode(value)
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            recordWriteFailure(label: label, error: error)
            return false
        }
    }

    static func recordReadFailure(label: String.LocalizationValue, error: Error) {
        let name = AppStrings.localized(label)
        let detail = error.localizedDescription
        let message = AppStrings.localized("\(name) could not be read: \(detail)")
        // One critical section so a concurrent reader never sees the count
        // bumped without its message.
        latchLock.withLock {
            latchReadFailureCount += 1
            latchReadError = message
        }
        FileHandle.standardError.write(
            "TurboSpark: \(englishName(label)) could not be read: \(detail)\n".data(using: .utf8)!)
    }

    static func recordWriteFailure(label: String.LocalizationValue, error: Error) {
        let name = AppStrings.localized(label)
        let detail = error.localizedDescription
        lastWriteError = AppStrings.localized("\(name) could not be saved: \(detail)")
        FileHandle.standardError.write(
            "TurboSpark: \(englishName(label)) could not be saved: \(detail)\n".data(using: .utf8)!)
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
