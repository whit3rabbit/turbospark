import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Tasks 6.4 and 6.5: the strings the Installed Models list adds, in all 21
/// languages.
///
/// `LocalizationParityTests` holds every catalog key to full translation and
/// matching specifiers, but its source scan cannot see an interpolated literal,
/// and a missing key there renders the English source silently. Each key the
/// new code uses is named here, and the sentences with names in them are
/// rendered through every language's compiled bundle.
final class InstalledModelStringsTests: XCTestCase {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// Keys this change added to the catalog.
    static let newKeys: [String] = [
        "Missing",
        "Unknown library",
        "About %@ to load",
        "Connect %@ to load",
        "The files for this model are missing.",
        "Remove from list",
        "Remove %@ from the list?",
        "TurboSpark stops listing this model. Its files are not deleted.",
        "This permanently deletes the model files (%@) from the drive of the library %@.",
        "A model is installing or moving, so '%@' was left as it is. Try again when it finishes.",
        "'%@' is no longer in the model list.",
        "'%@' could not be changed because of a file system error.",
        "'%@' is in a read-only library, so it was left as it is.",
        "TurboSpark could not change '%@'.",
        "TurboSpark could not change '%@': %@",
        "Removed '%@' from the list.",
        "By Location",
        "Available now",
        "Hide models whose drive is not connected or whose files are missing",
        "No Models Available Now",
        "Connect a drive, or turn off Available now to see every model.",
    ]

    /// Keys the new code reuses from the existing catalog.
    static let reusedKeys: [String] = [
        "Not connected", "This Mac", "Custom Folder", "Delete Model", "Cancel", "Group By",
        "%@ is on %@, which is not connected. Connect the drive and try again.",
        "Delete %@?",
        "This will permanently remove the model files (%@) from disk at:\n%@",
        "Deleted model '%@'",
        "Stop Humanize before deleting its model.",
    ]

    private func loadStrings() throws -> [String: [String: Any]] {
        let url = packageRoot.appendingPathComponent("Localization/Localizable.xcstrings")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let root = try XCTUnwrap(object as? [String: Any])
        return try XCTUnwrap(root["strings"] as? [String: [String: Any]])
    }

    private func value(_ localization: Any?) -> String? {
        let node = localization as? [String: Any]
        return (node?["stringUnit"] as? [String: Any])?["value"] as? String
    }

    private func specifiers(_ text: String) -> [String] {
        let regex = try! NSRegularExpression(pattern: #"%(\d+\$)?(lld|ld|lf|d|@|f)"#)
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).map { String(text[Range($0.range, in: text)!]) }.sorted()
    }

    func testEveryNewKeyIsInTheCatalogInAllLanguages() throws {
        let strings = try loadStrings()
        let languages = AppLanguage.allCases.filter { $0 != .system }.map(\.rawValue)
        XCTAssertEqual(languages.count, 21)
        var problems: [String] = []
        for key in Self.newKeys + Self.reusedKeys {
            guard let localizations = strings[key]?["localizations"] as? [String: Any] else {
                problems.append("missing key: \(key)")
                continue
            }
            for language in languages where (value(localizations[language]) ?? "").isEmpty {
                problems.append("\(language): empty or missing: \(key)")
            }
        }
        XCTAssertEqual(problems, [])
    }

    func testTheNewKeysKeepTheirSpecifiersInEveryLanguage() throws {
        let strings = try loadStrings()
        var problems: [String] = []
        for key in Self.newKeys {
            guard let localizations = strings[key]?["localizations"] as? [String: Any] else { continue }
            let expected = specifiers(key)
            for (language, node) in localizations {
                if let text = value(node), specifiers(text) != expected {
                    problems.append("\(language): \(key.prefix(50)) -> \(text.prefix(60))")
                }
            }
        }
        XCTAssertEqual(problems.sorted(), [])
    }

    func testTheEnglishValueIsTheKey() throws {
        let strings = try loadStrings()
        for key in Self.newKeys {
            let english = value((strings[key]?["localizations"] as? [String: Any])?["en"])
            XCTAssertEqual(english, key, key)
        }
    }

    func testNoTranslationOfANewKeyIsTheEnglishSentenceInALanguageThatDoesNotShareIt() throws {
        // A copy of the English text passes the parity gate. Languages that
        // legitimately keep a short English word are listed by hand.
        let strings = try loadStrings()
        let sentences = Self.newKeys.filter { $0.count > 25 }
        let nonLatin = ["ar", "he", "hi", "ja", "ko", "ru", "uk", "zh-Hans", "zh-Hant"]
        var problems: [String] = []
        for key in sentences {
            let localizations = strings[key]?["localizations"] as? [String: Any] ?? [:]
            for language in nonLatin where value(localizations[language]) == key {
                problems.append("\(language): \(key)")
            }
        }
        XCTAssertEqual(problems, [])
    }

    // MARK: Rendered through every compiled bundle

    private var languages: [AppLanguage] { AppLanguage.allCases.filter { $0 != .system } }

    func testTheDeleteSentencesRenderInEveryLanguageWithTheirNames() {
        let alias = "Qwen Max"
        let errors: [(String, TurboSparkError)] = [
            ("busy", TurboSparkError(code: .busy, message: "busy: x")),
            ("unknown", TurboSparkError(code: .invalidArgument, message: "unknown_install: x")),
            ("io", TurboSparkError(code: .invalidArgument, message: "io: x")),
            ("readOnly", TurboSparkError(code: .invalidArgument, message: "library_read_only: x")),
            ("generic", TurboSparkError(code: .invalidArgument, message: "use_install_id: x")),
            // No code of ours: the engine's own reason is kept, escaped.
            ("uncoded", TurboSparkError(code: .invalidArgument, message: "the disk said no")),
        ]
        for language in languages {
            for (label, error) in errors {
                let text = ModelDeleteText.failure(error, alias: alias, libraryName: "T7", language: language)
                XCTAssertTrue(text.contains(alias), "\(language.rawValue) \(label): \(text)")
                XCTAssertFalse(text.contains("%@"), "\(language.rawValue) \(label): \(text)")
                if language != .english {
                    let english = ModelDeleteText.failure(error, alias: alias, libraryName: "T7", language: .english)
                    XCTAssertNotEqual(text, english, "\(language.rawValue) \(label) is English")
                }
            }
            let away = ModelDeleteText.notConnected(alias: alias, libraryName: "T7", language: language)
            XCTAssertTrue(away.contains(alias) && away.contains("T7"), "\(language.rawValue): \(away)")
            let removed = ModelDeleteText.removedFromList(alias: alias, language: language)
            XCTAssertTrue(removed.contains(alias), "\(language.rawValue): \(removed)")
            XCTAssertFalse(removed.contains("%@"))
            if language != .english {
                XCTAssertNotEqual(removed, ModelDeleteText.removedFromList(alias: alias, language: .english))
            }
        }
    }

    func testTheRowSentencesRenderInEveryLanguage() {
        let lib = LibraryFixtures.external("t7-id", name: "T7")
        let away = LibraryFixtures.external("away-id", name: "Archive", available: false)
        let connection = DriveConnection(kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "u", isLocal: true)
        for language in languages {
            let offline = InstalledModelRowModel(
                model: LibraryFixtures.row("qwen", library: away, availability: .notConnected), library: away,
                connection: nil, measuredMBps: nil, isMoE: false, physicalMemoryBytes: 64 << 30, language: language)
            let missing = InstalledModelRowModel(
                model: LibraryFixtures.row("qwen", library: lib, availability: .missing), library: lib,
                connection: connection, measuredMBps: nil, isMoE: false, physicalMemoryBytes: 64 << 30,
                language: language)
            let live = InstalledModelRowModel(
                model: InstalledModel(
                    alias: "qwen", repo: "x/y", path: "/p", family: "qwen", installBytes: 7_000_000_000,
                    library: lib.id, libraryName: "T7", installId: "t7-id:text/qwen", locationKind: .library),
                library: lib, connection: connection, measuredMBps: nil, isMoE: false,
                physicalMemoryBytes: 64 << 30, language: language)

            XCTAssertEqual(offline.loadBlockedReason?.contains("Archive"), true, language.rawValue)
            XCTAssertEqual(offline.loadBlockedReason?.contains("%@"), false, language.rawValue)
            XCTAssertFalse(missing.loadBlockedReason?.isEmpty ?? true, language.rawValue)
            XCTAssertEqual(live.estimateText?.contains("%@"), false, language.rawValue)
            XCTAssertFalse(live.estimateText?.isEmpty ?? true, language.rawValue)
            // The number follows the language's own digits (Arabic may use
            // Eastern Arabic ones), so the check is against the same formatter.
            let tenSeconds = InstalledModelRowModel.durationText(seconds: 10, language: language)
            XCTAssertEqual(
                live.estimateText?.contains(tenSeconds), true,
                "\(language.rawValue): \(live.estimateText ?? "") should hold \(tenSeconds)")
            if language != .english {
                XCTAssertNotEqual(offline.statusText, "Not connected", language.rawValue)
                XCTAssertNotEqual(missing.statusText, "Missing", language.rawValue)
                XCTAssertNotEqual(live.estimateText, "About 10 sec to load", language.rawValue)
                XCTAssertNotEqual(offline.loadBlockedReason, "Connect Archive to load", language.rawValue)
                XCTAssertNotEqual(missing.loadBlockedReason, "The files for this model are missing.", language.rawValue)
            }
        }
    }

    func testTheConfirmationsRenderInEveryLanguageAndNameTheLibrary() {
        let library = LibraryFixtures.row(
            "qwen", library: LibraryFixtures.external("t7-id", name: "T7"))
        let thisMac = LibraryFixtures.row("qwen")
        for language in languages {
            let files = ModelDeleteDialog(model: library, libraryName: "T7", language: language)
            XCTAssertEqual(files.kind, .libraryFiles)
            XCTAssertTrue(files.message.contains("T7"), "\(language.rawValue): \(files.message)")
            XCTAssertTrue(files.title.contains("qwen"), "\(language.rawValue): \(files.title)")
            XCTAssertFalse(files.message.contains("%@") || files.title.contains("%@"))

            let list = ModelDeleteDialog(
                model: LibraryFixtures.row("qwen", library: LibraryFixtures.external("t7-id", name: "T7"), availability: .notConnected),
                libraryName: "T7", language: language)
            XCTAssertEqual(list.kind, .listOnly)
            XCTAssertTrue(list.title.contains("qwen"), "\(language.rawValue): \(list.title)")

            let mac = ModelDeleteDialog(model: thisMac, libraryName: nil, language: language)
            XCTAssertEqual(mac.kind, .thisMacFiles)
            XCTAssertTrue(mac.message.contains(thisMac.path), "\(language.rawValue)")

            if language != .english {
                let english = ModelDeleteDialog(model: library, libraryName: "T7", language: .english)
                XCTAssertNotEqual(files.message, english.message, language.rawValue)
                XCTAssertNotEqual(list.confirmTitle, "Remove from list", language.rawValue)
            }
        }
    }

    func testTheLibraryDeleteConfirmationNamesTheLibraryAndSaysTheFilesAreDeleted() {
        let row = InstalledModel(
            alias: "qwen", repo: "x/y", path: "/Volumes/T7/TurboSpark/text/qwen.gturbo", family: "qwen",
            installBytes: 2_000_000_000, library: "t7-id", libraryName: "T7",
            installId: "t7-id:text/qwen", availability: .available, locationKind: .library)
        let dialog = ModelDeleteDialog(model: row, libraryName: "T7", language: .english)

        XCTAssertEqual(dialog.kind, .libraryFiles)
        XCTAssertEqual(dialog.title, "Delete qwen?")
        XCTAssertEqual(
            dialog.message,
            "This permanently deletes the model files (\(MetricFormat.storage(2_000_000_000))) from the drive of the library T7.")
        XCTAssertEqual(dialog.confirmTitle, "Delete Model")
        // A drive-supplied name is escaped, never shown raw.
        let hostile = ModelDeleteDialog(model: row, libraryName: "T7\u{202E}gpj", language: .english)
        XCTAssertFalse(hostile.message.contains("\u{202E}"))
    }

    func testTheGroupAndFilterWordsRenderInEveryCompiledBundle() {
        for language in languages where language != .english {
            let bundle = AppStrings.bundle(for: language)
            for key in [
                "By Location", "Available now", "Missing", "Remove from list", "No Models Available Now",
                "Unknown library",
            ] {
                let value = bundle.localizedString(forKey: key, value: "MISSING", table: nil)
                XCTAssertNotEqual(value, "MISSING", "\(language.rawValue): \(key) did not compile")
                XCTAssertNotEqual(value, key, "\(language.rawValue): \(key) is English")
            }
        }
    }
}
