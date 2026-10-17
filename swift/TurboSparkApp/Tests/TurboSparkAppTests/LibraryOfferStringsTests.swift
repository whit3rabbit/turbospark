import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.6: the strings the add-library offer adds, in all 21 languages.
///
/// `LocalizationParityTests` holds every catalog key to full translation and
/// matching specifiers, but its source scan skips interpolated literals, and a
/// key missing there renders its English source silently. So each key the new
/// code uses is named here, the sentences with names in them are rendered
/// through every compiled bundle (`LibraryOfferBannerModelTests`), and a scan
/// resolves EVERY literal the new files pass to `Text` or `AppStrings.localized`,
/// interpolated ones included, against the catalog.
final class LibraryOfferStringsTests: XCTestCase {
    /// `swift/TurboSparkApp`.
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var sourcesRoot: URL { packageRoot.appendingPathComponent("Sources/TurboSparkApp") }

    /// Keys this change added to the catalog.
    static let newPlainKeys: [String] = [
        "Add the library %@?",
        "Not now",
        "A newer version of TurboSpark made this library. You can use its models, but you cannot change it.",
        "Adding the library\u{2026}",
        "TurboSpark found the library %@. Add it from Installed Models or Models & Storage.",
        "TurboSpark will not offer this library again. You can add it later in Models & Storage settings.",
    ]

    static let newPluralKey = "%lld more libraries found"

    /// Keys the offer reuses from the existing catalog.
    static let reusedKeys: [String] = [
        "Add Library", "Not available while a model is installing or moving.", "Unknown library",
        "Added the library %@.", "Installed Models", "Models & Storage",
    ]

    /// The files the offer added. A literal in any of them must be a key.
    static let newSourceFiles = [
        "Components/LibraryOfferBanner.swift",
        "State/LibraryOfferBannerModel.swift",
        "State/LibraryOfferController.swift",
        "State/LibraryOfferAnnouncer.swift",
    ]

    private var languages: [String] { AppLanguage.allCases.filter { $0 != .system }.map(\.rawValue) }

    private func loadStrings() throws -> [String: [String: Any]] {
        let url = packageRoot.appendingPathComponent("Localization/Localizable.xcstrings")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let root = try XCTUnwrap(object as? [String: Any])
        return try XCTUnwrap(root["strings"] as? [String: [String: Any]])
    }

    private func units(_ localization: Any?) -> [String] {
        guard let node = localization as? [String: Any] else { return [] }
        if let unit = node["stringUnit"] as? [String: Any], let value = unit["value"] as? String {
            return [value]
        }
        guard let plural = (node["variations"] as? [String: Any])?["plural"] as? [String: Any] else { return [] }
        return plural.values.compactMap { (($0 as? [String: Any])?["stringUnit"] as? [String: Any])?["value"] as? String }
    }

    private func plain(_ strings: [String: [String: Any]], _ key: String, _ language: String) -> String? {
        let localizations = strings[key]?["localizations"] as? [String: Any]
        return units(localizations?[language]).first
    }

    private func specifiers(_ text: String) -> [String] {
        let regex = try! NSRegularExpression(pattern: #"%(\d+\$)?(lld|ld|lf|d|@|f)"#)
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).map { String(text[Range($0.range, in: text)!]) }.sorted()
    }

    // MARK: - The catalog

    func testEveryKeyTheOfferUsesIsInTheCatalogInAllLanguages() throws {
        let strings = try loadStrings()
        XCTAssertEqual(languages.count, 21)
        var problems: [String] = []
        for key in Self.newPlainKeys + [Self.newPluralKey] + Self.reusedKeys {
            guard let localizations = strings[key]?["localizations"] as? [String: Any] else {
                problems.append("missing key: \(key)")
                continue
            }
            for language in languages {
                let values = units(localizations[language])
                if values.isEmpty || values.contains(where: { $0.isEmpty }) {
                    problems.append("\(language): empty or missing: \(key)")
                }
            }
        }
        XCTAssertEqual(problems, [])
    }

    func testThePlainKeysKeepTheirSpecifiersInEveryLanguage() throws {
        let strings = try loadStrings()
        var problems: [String] = []
        for key in Self.newPlainKeys {
            guard let localizations = strings[key]?["localizations"] as? [String: Any] else { continue }
            let expected = specifiers(key)
            for (language, node) in localizations {
                for value in units(node) where specifiers(value) != expected {
                    problems.append("\(language): \(key.prefix(50)) -> \(value.prefix(60))")
                }
            }
        }
        XCTAssertEqual(problems.sorted(), [])
    }

    func testTheEnglishValueIsTheKey() throws {
        let strings = try loadStrings()
        for key in Self.newPlainKeys {
            XCTAssertEqual(plain(strings, key, "en"), key, key)
        }
    }

    func testThePluralKeyCarriesTheSameCategoriesAsTheModelCountInEveryLanguage() throws {
        let strings = try loadStrings()
        let mine = try XCTUnwrap(strings[Self.newPluralKey]?["localizations"] as? [String: Any])
        let reference = try XCTUnwrap(strings["%lld model(s)"]?["localizations"] as? [String: Any])
        func categories(_ node: Any?) -> [String: String] {
            let plural = ((node as? [String: Any])?["variations"] as? [String: Any])?["plural"] as? [String: Any]
            return (plural ?? [:]).compactMapValues {
                (($0 as? [String: Any])?["stringUnit"] as? [String: Any])?["value"] as? String
            }
        }
        for language in languages {
            let ours = categories(mine[language])
            XCTAssertFalse(ours.isEmpty, "\(language) has no plural variations")
            XCTAssertEqual(Set(ours.keys), Set(categories(reference[language]).keys), "\(language) categories")
            XCTAssertNotNil(ours["other"], "\(language) needs the `other` category")
            XCTAssertTrue(ours["other"]?.contains("%lld") ?? false, "\(language) other: \(ours["other"] ?? "")")
            for (category, value) in ours {
                // A translation may leave the number out (one, two, zero spell
                // it) but must never invent a specifier.
                XCTAssertTrue(
                    specifiers(value).allSatisfy { $0 == "%lld" } && specifiers(value).count <= 1,
                    "\(language) \(category): \(value)")
            }
        }
        let english = categories(mine["en"])
        XCTAssertEqual(english["one"], "%lld more library found")
        XCTAssertEqual(english["other"], "%lld more libraries found")
    }

    func testNoTranslationInANonLatinLanguageIsTheEnglishSentence() throws {
        let strings = try loadStrings()
        let nonLatin = ["ar", "he", "hi", "ja", "ko", "ru", "uk", "zh-Hans", "zh-Hant"]
        var problems: [String] = []
        for key in Self.newPlainKeys {
            for language in nonLatin where plain(strings, key, language) == key {
                problems.append("\(language): \(key)")
            }
        }
        XCTAssertEqual(problems, [])
    }

    func testTheSentencesNameThePlacesTheWayEachLanguageNamesThem() throws {
        // The toast and the Not now tooltip point at two places by name. A
        // sentence that spells them differently from the sidebar and the
        // settings tab sends the user looking for something that is not there.
        let strings = try loadStrings()
        let toast = Self.newPlainKeys[4]
        let help = Self.newPlainKeys[5]
        var problems: [String] = []
        for language in languages where language != "en" {
            let installed = try XCTUnwrap(plain(strings, "Installed Models", language))
            let storage = try XCTUnwrap(plain(strings, "Models & Storage", language))
            let toastText = try XCTUnwrap(plain(strings, toast, language))
            let helpText = try XCTUnwrap(plain(strings, help, language))
            if !toastText.contains(installed) { problems.append("\(language) toast lacks \(installed)") }
            if !toastText.contains(storage) { problems.append("\(language) toast lacks \(storage)") }
            if !helpText.contains(storage) { problems.append("\(language) help lacks \(storage)") }
        }
        XCTAssertEqual(problems, [])
    }

    func testTheNewKeysCompileIntoEveryLanguagesBundle() {
        for language in AppLanguage.allCases where language != .system && language != .english {
            let bundle = AppStrings.bundle(for: language)
            for key in ["Not now", "Adding the library\u{2026}", Self.newPlainKeys[2]] {
                let value = bundle.localizedString(forKey: key, value: "MISSING", table: nil)
                XCTAssertNotEqual(value, "MISSING", "\(language.rawValue): \(key) did not compile")
                XCTAssertNotEqual(value, key, "\(language.rawValue): \(key) is English")
            }
        }
    }

    // MARK: - Every literal the new files pass is a key

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
                    guard let key = catalogKey(of: characters, from: offset) else {
                        problems.append("\(file): unterminated literal after \(trigger)")
                        continue
                    }
                    // `Text("x", bundle: .module)` only: a bare literal renders
                    // English. The tail after the closing quote is checked.
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
        XCTAssertGreaterThanOrEqual(checked, 6, "the scan found almost nothing: is it looking at the right files?")
        XCTAssertEqual(problems, [])
    }

    func testTheNewFilesNeverPassABareLiteralToAKeyOnlyInitializer() throws {
        let triggers = [
            ".help(\"", ".accessibilityLabel(\"", ".accessibilityHint(\"", "Button(\"", "Label(\"",
            "ProgressView(\"",
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
