import Foundation
import XCTest

@testable import TurboSparkApp

/// Tasks 6.9 and 6.10: the strings the install destination picker, the free-space
/// gate, the install failures and the offline install statements add, in all 21
/// languages.
///
/// `LocalizationParityTests` holds every catalog key to full translation and
/// matching specifiers, but its source scan skips interpolated literals, and a key
/// missing there renders its English source silently. So each key is named here,
/// each sentence with a name in it is rendered through every compiled bundle (in
/// `InstallFailureTextTests` and `InstallDestinationGateTests`), and a scan
/// resolves EVERY literal the new files pass to `Text` or `AppStrings.localized`,
/// interpolated ones included, against the catalog.
final class InstallDestinationStringsTests: XCTestCase {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var sourcesRoot: URL { packageRoot.appendingPathComponent("Sources/TurboSparkApp") }

    /// Keys this change added to the catalog.
    static let newKeys: [String] = [
        "Install to",
        "%@ is the default library but is not connected, so this goes to This Mac.",
        "Choose the library this model is installed into.",
        "%@ is not connected. Connect it or choose another library.",
        "This needs up to %@ temporarily, but %@ has only %@ free. Verified download ranges are kept until the install passes verification, then removed.",
        "%@ was disconnected during the install. Nothing was added. Connect it and try again.",
        "%@ is read-only, so nothing was installed there. Choose another library.",
        "That library is no longer in the list, so nothing was installed. Choose another library.",
        "A model with this name is already installed on %@. Move it from there instead.",
        "A model with this name is already installed in another place. Move it from there instead.",
        "Installed on %@, which is not connected. Connect the drive to use it.",
    ]

    /// Keys the change reuses from the existing catalog.
    static let reusedKeys: [String] = [
        "This Mac", "%@ free", "Unknown library", "Not connected",
        "A model is installing or moving. Try again when it finishes.", "Install anyway?",
    ]

    /// The files this change added. A literal in any of them must be a key.
    static let newSourceFiles = [
        "State/InstallDestinationModel.swift", "State/InstallFailureText.swift",
        "State/AppModel+InstallDestination.swift", "State/HubInstallState.swift",
        "Components/InstallDestinationPicker.swift", "Audio/AudioOfflineInstalls.swift",
        "Installation/ModelInstallGate.swift",
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

    func testEveryKeyIsInTheCatalogInAllLanguages() throws {
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

    /// Plain specifiers are filled in the order the call site passes its
    /// arguments, so a translation may not reorder them (the multiset check
    /// cannot see an order). The way to keep the order is to write each
    /// translation around the English order, and the way to stay in it is to use
    /// no positional specifier at all, which this pins.
    func testNoNewKeyUsesAPositionalSpecifier() throws {
        let strings = try loadStrings()
        for key in Self.newKeys {
            let localizations = strings[key]?["localizations"] as? [String: Any] ?? [:]
            for (language, node) in localizations {
                XCTAssertFalse(
                    (value(node) ?? "").contains("$@"), "\(language): \(key.prefix(40)) uses a positional specifier")
            }
        }
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
                XCTAssertNotEqual(text, "MISSING", "\(language.rawValue): \(key.prefix(40)) did not compile")
                XCTAssertNotEqual(text, key, "\(language.rawValue): \(key.prefix(40)) is English")
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
        XCTAssertGreaterThanOrEqual(checked, 8, "the scan found almost nothing: is it looking at the right files?")
        XCTAssertEqual(problems, [])
    }

    func testTheNewFilesNeverPassABareLiteralToAKeyOnlyInitializer() throws {
        let triggers = [
            ".help(\"", ".accessibilityLabel(\"", ".accessibilityHint(\"", "Button(\"", "Label(\"",
            "ProgressView(\"", ".confirmationDialog(\"", ".alert(\"", "Picker(\"", "Menu(\"",
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
