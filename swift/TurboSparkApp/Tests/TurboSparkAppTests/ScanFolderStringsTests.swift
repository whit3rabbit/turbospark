import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.12: the words of an additional scan folder that is not reachable, in
/// all 21 languages.
///
/// `LocalizationParityTests` holds every catalog key to full translation and
/// matching specifiers, but its source scan skips interpolated literals and a
/// key missing there renders its English source silently. So the new key and
/// every key the row reuses are named here, the sentences are rendered through
/// every language's compiled bundle, and a scan resolves EVERY literal the
/// touched files pass to `Text` or `AppStrings.localized` against the catalog.
final class ScanFolderStringsTests: XCTestCase {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var sourcesRoot: URL { packageRoot.appendingPathComponent("Sources/TurboSparkApp") }

    /// Keys this change added to the catalog.
    static let newKeys: [String] = ["Connect %@ to scan this folder."]

    /// Keys the row reuses. "Folder not found" and "Not connected" already
    /// exist; the second is the word every other surface uses for the same
    /// condition, so the row says it the same way.
    static let reusedKeys: [String] = [
        "Not connected", "Folder not found", "Models: %lld", "Connect the drive to use this.",
        "Reveal in Finder", "Remove this folder from scan list",
    ]

    /// The files this change touched. A literal in any of them must be a key.
    static let sourceFiles = [
        "Components/CustomModelFoldersSectionView.swift",
        "Components/ScanFolderRowView.swift",
        "State/ScanFolderRowModel.swift",
    ]

    private var languages: [String] { AppLanguage.allCases.filter { $0 != .system }.map(\.rawValue) }

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

    // MARK: - The catalog

    func testEveryKeyTheRowUsesIsInTheCatalogInAllLanguages() throws {
        let strings = try loadStrings()
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

    func testTheNewKeyKeepsItsSpecifiersInEveryLanguage() throws {
        let strings = try loadStrings()
        var problems: [String] = []
        for key in Self.newKeys {
            let localizations = try XCTUnwrap(strings[key]?["localizations"] as? [String: Any], key)
            let expected = specifiers(key)
            XCTAssertEqual(expected, ["%@"])
            for (language, node) in localizations {
                if let text = value(node), specifiers(text) != expected {
                    problems.append("\(language): \(key) -> \(text)")
                }
            }
        }
        XCTAssertEqual(problems.sorted(), [])
    }

    func testTheEnglishValueIsTheKey() throws {
        let strings = try loadStrings()
        for key in Self.newKeys {
            XCTAssertEqual(value((strings[key]?["localizations"] as? [String: Any])?["en"]), key, key)
        }
    }

    func testNoTranslationInANonLatinLanguageIsTheEnglishSentence() throws {
        let strings = try loadStrings()
        let nonLatin = ["ar", "he", "hi", "ja", "ko", "ru", "uk", "zh-Hans", "zh-Hant"]
        var problems: [String] = []
        for key in Self.newKeys {
            let localizations = strings[key]?["localizations"] as? [String: Any] ?? [:]
            for language in nonLatin where value(localizations[language]) == key {
                problems.append("\(language): \(key)")
            }
        }
        XCTAssertEqual(problems, [])
    }

    // MARK: - Rendered through every compiled bundle

    func testTheNewKeyCompilesIntoEveryLanguagesBundle() {
        for language in AppLanguage.allCases where language != .system && language != .english {
            let bundle = AppStrings.bundle(for: language)
            for key in Self.newKeys {
                let text = bundle.localizedString(forKey: key, value: "MISSING", table: nil)
                XCTAssertNotEqual(text, "MISSING", "\(language.rawValue): \(key) did not compile")
                XCTAssertNotEqual(text, key, "\(language.rawValue): \(key) is English")
            }
        }
    }

    func testEveryRowSentenceRendersInEveryLanguage() {
        let away = ScanFolderReading(status: .notConnected(volumeName: "Samsung T7"), modelCount: 0)
        let absent = ScanFolderReading(status: .notFound, modelCount: 0)
        let live = ScanFolderReading(status: .connected, modelCount: 12)
        for language in AppLanguage.allCases where language != .system {
            let offline = ScanFolderRowModel(path: "/Volumes/Samsung T7/models", reading: away, language: language)
            let gone = ScanFolderRowModel(path: "/Volumes/Samsung T7/models", reading: absent, language: language)
            let counted = ScanFolderRowModel(path: "/Volumes/Samsung T7/models", reading: live, language: language)

            for text in [offline.statusText, offline.hintText, offline.reveal.disabledReason, gone.statusText,
                gone.reveal.disabledReason, counted.countText, offline.accessibilityLabel]
            {
                let shown = text ?? ""
                XCTAssertFalse(shown.isEmpty, language.rawValue)
                XCTAssertFalse(shown.contains("%@") || shown.contains("%lld"), "\(language.rawValue): \(shown)")
            }
            XCTAssertTrue(offline.hintText?.contains("Samsung T7") == true, "\(language.rawValue): \(offline.hintText ?? "")")
            XCTAssertTrue(offline.accessibilityLabel.contains("Samsung T7"), language.rawValue)
            // The count is a number in the language's own digits, so only the
            // absence of a placeholder is checked outside English.
            if language == .english {
                XCTAssertEqual(counted.countText, "Models: 12")
            } else {
                XCTAssertNotEqual(offline.statusText, "Not connected", language.rawValue)
                XCTAssertNotEqual(gone.statusText, "Folder not found", language.rawValue)
                XCTAssertNotEqual(offline.reveal.disabledReason, "Connect the drive to use this.", language.rawValue)
                XCTAssertNotEqual(counted.countText, "Models: 12", language.rawValue)
            }
        }
    }

    func testTheStatusWordIsTheOneTheOtherSurfacesUse() {
        // The same condition reads the same on the installed model row and
        // here: one key, one translation per language.
        for language in AppLanguage.allCases where language != .system {
            let row = ScanFolderRowModel(
                path: "/Volumes/T7/m", reading: ScanFolderReading(status: .notConnected(volumeName: "T7"), modelCount: 0),
                language: language)
            let model = InstalledModelRowModel(
                model: LibraryFixtures.row(
                    "qwen", library: LibraryFixtures.external("away", name: "Away", available: false),
                    availability: .notConnected),
                library: nil, connection: nil, measuredMBps: nil, language: language)
            XCTAssertEqual(row.statusText, model.statusText, language.rawValue)
            XCTAssertEqual(row.statusText, AppStrings.localized("Not connected", language: language), language.rawValue)
        }
    }

    // MARK: - Every literal the touched files pass is a key

    /// The text of the literal that opens `source` at `start` (just after the
    /// opening quote), with escapes resolved and every interpolation turned into
    /// the specifier its type gives it: `%lld` for a name that ends in `count`,
    /// `%@` for anything else (every other interpolation in these files is a
    /// String). Nil for an unterminated literal.
    private func catalogKey(of source: [Character], from start: Int) -> String? {
        var out = ""
        var index = start
        while index < source.count {
            let character = source[index]
            if character == "\"" { return out }
            guard character == "\\", index + 1 < source.count else {
                out.append(character)
                index += 1
                continue
            }
            let next = source[index + 1]
            if next == "(" {
                var depth = 1
                var cursor = index + 2
                var inner = ""
                while cursor < source.count, depth > 0 {
                    if source[cursor] == "(" { depth += 1 }
                    if source[cursor] == ")" { depth -= 1 }
                    if depth > 0 { inner.append(source[cursor]) }
                    cursor += 1
                }
                out += inner.lowercased().hasSuffix("count") ? "%lld" : "%@"
                index = cursor
            } else if next == "u", index + 2 < source.count, source[index + 2] == "{",
                let close = source[(index + 3)...].firstIndex(of: "}"),
                let value = UInt32(String(source[(index + 3)..<close]), radix: 16),
                let scalar = Unicode.Scalar(value)
            {
                out.append(Character(scalar))
                index = close + 1
            } else {
                out.append(next == "n" ? "\n" : next)
                index += 2
            }
        }
        return nil
    }

    func testEveryLiteralTheTouchedFilesPassToTextOrLocalizedIsACatalogKey() throws {
        let strings = try loadStrings()
        var checked = 0
        var problems: [String] = []
        for file in Self.sourceFiles {
            let code = MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: sourcesRoot.appendingPathComponent(file), encoding: .utf8))
            let characters = Array(code)
            for trigger in ["Text(\"", "AppStrings.localized(\""] {
                var search = code.startIndex..<code.endIndex
                while let found = code.range(of: trigger, range: search) {
                    search = found.upperBound..<code.endIndex
                    let offset = code.distance(from: code.startIndex, to: found.upperBound)
                    guard let key = catalogKey(of: characters, from: offset) else {
                        problems.append("\(file): unterminated literal after \(trigger)")
                        continue
                    }
                    if trigger == "Text(\"" {
                        let tail = String(code[found.upperBound...].prefix(400))
                        let closing = tail.firstIndex(of: ")").map { String(tail[..<$0]) } ?? tail
                        if !closing.contains("bundle: .module") {
                            problems.append("\(file): Text literal without bundle: .module -> \(key.prefix(50))")
                        }
                    }
                    checked += 1
                    if strings[key] == nil { problems.append("\(file): no catalog key -> \(key)") }
                }
            }
        }
        XCTAssertGreaterThanOrEqual(checked, 12, "the scan found almost nothing: is it looking at the right files?")
        XCTAssertEqual(problems, [])
    }

    func testTheTouchedFilesNeverPassABareLiteralToAKeyOnlyInitializer() throws {
        let triggers = [
            ".help(\"", ".accessibilityLabel(\"", ".accessibilityHint(\"", "Button(\"", "Label(\"",
            "ProgressView(\"",
        ]
        for file in Self.sourceFiles {
            let code = MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: sourcesRoot.appendingPathComponent(file), encoding: .utf8))
            for trigger in triggers {
                // A call's own name, not the tail of a longer one:
                // `AppIconButton("trash", ...)` takes an SF Symbol name.
                let pattern = (trigger.hasPrefix(".") ? "" : "(?<![A-Za-z0-9_])") + NSRegularExpression.escapedPattern(for: trigger)
                let regex = try NSRegularExpression(pattern: pattern)
                let found = regex.firstMatch(in: code, range: NSRange(code.startIndex..., in: code))
                XCTAssertNil(found, "\(file) passes a bare literal to \(trigger)")
            }
        }
    }
}
