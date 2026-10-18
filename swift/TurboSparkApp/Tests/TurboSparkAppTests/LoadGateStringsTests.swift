import Foundation
import XCTest

@testable import TurboSparkApp

/// Tasks 6.7 and 6.8: the strings the load gate adds, in all 21 languages.
///
/// `LocalizationParityTests` holds every catalog key to full translation and
/// matching specifiers, but its source scan skips interpolated literals, and a key
/// missing there renders its English source silently. So each key the gate uses is
/// named here, the sentences with names and durations in them are rendered through
/// every compiled bundle, and a scan resolves EVERY literal the new files pass to
/// `Text` or `AppStrings.localized`, interpolated ones included, against the
/// catalog.
final class LoadGateStringsTests: XCTestCase {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var sourcesRoot: URL { packageRoot.appendingPathComponent("Sources/TurboSparkApp") }

    /// Keys this change added to the catalog.
    static let newKeys: [String] = [
        "Load anyway",
        "This drive's connection is slow, so loading will take about %@.",
        "This drive's connection is slow, so loading will take a long time.",
        "This model likely streams from the drive, so generation speed will be limited by the drive.",
        "Load %@ anyway?",
    ]

    /// Keys the gate reuses from the existing catalog.
    static let reusedKeys: [String] = [
        "Connect %@ to load", "The files for this model are missing.", "Unknown library", "Cancel",
        "Start Chat", "Remove from list", "Not connected", "Missing",
    ]

    /// The files the gate added or rewrote. A literal in any of them must be a key.
    static let newSourceFiles = [
        "State/LoadGate.swift", "State/InstalledModelActionPlan.swift", "State/AppModel+LoadGate.swift",
        "State/LoadableModels.swift", "Components/LoadAnywayConfirmation.swift",
        "Installation/InstalledModelActionBarView.swift",
    ]

    private var languages: [AppLanguage] { AppLanguage.allCases.filter { $0 != .system } }

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

    func testEveryKeyTheGateUsesIsInTheCatalogInAllLanguages() throws {
        let strings = try loadStrings()
        XCTAssertEqual(languages.count, 21)
        var problems: [String] = []
        for key in Self.newKeys + Self.reusedKeys {
            guard let localizations = strings[key]?["localizations"] as? [String: Any] else {
                problems.append("missing key: \(key)")
                continue
            }
            for language in languages where (value(localizations[language.rawValue]) ?? "").isEmpty {
                problems.append("\(language.rawValue): empty or missing: \(key)")
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
            XCTAssertEqual(value((strings[key]?["localizations"] as? [String: Any])?["en"]), key, key)
        }
    }

    func testNoTranslationInANonLatinLanguageIsTheEnglishSentence() throws {
        let strings = try loadStrings()
        let nonLatin = ["ar", "he", "hi", "ja", "ko", "ru", "uk", "zh-Hans", "zh-Hant"]
        var problems: [String] = []
        for key in Self.newKeys {
            let localizations = strings[key]?["localizations"] as? [String: Any] ?? [:]
            for language in nonLatin where value(localizations[language]) == key { problems.append("\(language): \(key)") }
        }
        XCTAssertEqual(problems, [])
    }

    func testTheNewKeysCompileIntoEveryLanguagesBundle() {
        for language in languages where language != .english {
            let bundle = AppStrings.bundle(for: language)
            for key in Self.newKeys {
                let text = bundle.localizedString(forKey: key, value: "MISSING", table: nil)
                XCTAssertNotEqual(text, "MISSING", "\(language.rawValue): \(key) did not compile")
                XCTAssertNotEqual(text, key, "\(language.rawValue): \(key) is English")
            }
        }
    }

    // MARK: Rendered through every compiled bundle

    func testEverySentenceRendersInEveryLanguageWithItsNames() {
        for language in languages {
            let duration = InstalledModelRowModel.durationText(seconds: 175, language: language)
            let dense = LoadGateText.text(for: .slowDrive(estimateSeconds: 175), language: language)
            let noEstimate = LoadGateText.text(for: .slowDrive(estimateSeconds: nil), language: language)
            let streaming = LoadGateText.text(for: .streamsFromDrive, language: language)
            let title = LoadGateText.confirmationTitle(alias: "Qwen Max", language: language)
            let action = LoadGateText.loadAnywayTitle(language: language)
            let connect = LoadGateText.statement(for: .notConnected(libraryName: "T7"), language: language)!
            let missing = LoadGateText.statement(for: .missing, language: language)!

            XCTAssertTrue(dense.contains(duration), "\(language.rawValue): \(dense) should hold \(duration)")
            XCTAssertTrue(title.contains("Qwen Max"), "\(language.rawValue): \(title)")
            XCTAssertTrue(connect.contains("T7"), "\(language.rawValue): \(connect)")
            for text in [dense, noEstimate, streaming, title, action, connect, missing] {
                XCTAssertFalse(text.contains("%@") || text.contains("%lld"), "\(language.rawValue): \(text)")
            }
            XCTAssertEqual(
                Set([dense, noEstimate, streaming]).count, 3, "\(language.rawValue): the wordings must differ")
            if language != .english {
                let english = LoadGateText.text(for: .slowDrive(estimateSeconds: 175), language: .english)
                XCTAssertNotEqual(dense, english, language.rawValue)
                XCTAssertNotEqual(
                    streaming, LoadGateText.text(for: .streamsFromDrive, language: .english), language.rawValue)
                XCTAssertNotEqual(action, "Load anyway", language.rawValue)
                XCTAssertNotEqual(noEstimate, LoadGateText.text(for: .slowDrive(estimateSeconds: nil), language: .english))
            }
        }
    }

    // MARK: - Every literal the new files pass is a key

    /// The text of the literal that opens `source` at `start` (just after the
    /// opening quote), with escapes resolved and every interpolation turned into
    /// the specifier its type gives it: `%lld` for a name that ends in `count`,
    /// `%@` for anything else (every other interpolation in these files is a
    /// String). Nil for an unterminated literal.
    private func catalogKey(of source: [Character], from start: Int) -> (key: String, end: Int)? {
        var out = ""
        var index = start
        while index < source.count {
            let character = source[index]
            if character == "\"" { return (out, index) }
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

    func testEveryLiteralTheNewFilesPassToTextOrLocalizedIsACatalogKey() throws {
        let strings = try loadStrings()
        var checked = 0
        var problems: [String] = []
        for file in Self.newSourceFiles {
            let code = MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: sourcesRoot.appendingPathComponent(file), encoding: .utf8))
            let characters = Array(code)
            for trigger in ["Text(\"", "AppStrings.localized(\""] {
                var search = code.startIndex..<code.endIndex
                while let found = code.range(of: trigger, range: search) {
                    search = found.upperBound..<code.endIndex
                    let offset = code.distance(from: code.startIndex, to: found.upperBound)
                    guard let (key, end) = catalogKey(of: characters, from: offset) else {
                        problems.append("\(file): unterminated literal after \(trigger)")
                        continue
                    }
                    // `Text("x", bundle: .module)` only: a bare literal renders English.
                    // The call's tail is read from the literal's CLOSING quote: an
                    // interpolation inside the literal has parentheses of its own.
                    if trigger == "Text(\"" {
                        let tail = String(characters[(end + 1)...].prefix(400))
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
        XCTAssertGreaterThanOrEqual(checked, 10, "the scan found almost nothing: is it looking at the right files?")
        XCTAssertEqual(problems, [])
    }

    func testTheNewFilesNeverPassABareLiteralToAKeyOnlyInitializer() throws {
        let triggers = [
            ".help(\"", ".accessibilityLabel(\"", ".accessibilityHint(\"", "Button(\"", "Label(\"",
            "ProgressView(\"", ".confirmationDialog(\"", ".alert(\"",
        ]
        for file in Self.newSourceFiles {
            let code = MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: sourcesRoot.appendingPathComponent(file), encoding: .utf8))
            for trigger in triggers {
                XCTAssertFalse(code.contains(trigger), "\(file) passes a bare literal to \(trigger)")
            }
        }
    }
}
