import Foundation
import XCTest

@testable import TurboSparkApp

/// Tasks 6.1 and 6.2: the strings the libraries section adds, in all 21
/// languages.
///
/// `LocalizationParityTests` holds every catalog key to full translation and
/// matching specifiers, but its source scan cannot see an interpolated literal
/// (`Text("... \(name)", bundle: .module)`), and a missing key there renders
/// the English source silently. This list is the other half: each key the
/// section uses is named here, so deleting one, or forgetting one, fails here.
final class ModelLibraryStringsTests: XCTestCase {
    /// `swift/TurboSparkApp`.
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static let plainKeys: [String] = [
        "Model libraries",
        "Keep models on drives connected to this Mac. Your settings and model list stay on this Mac when a drive is unplugged.",
        "Add Library\u{2026}",
        "Add Library",
        "Choose a drive or a folder on a drive to use as a model library",
        "Added the library %@.",
        "Removed the library %@.",
        "%@ is now the default library.",
        "This Mac",
        "%@ free",
        "Connection: %@",
        "Internal drive",
        "Connected",
        "Not connected",
        "Not connected, last seen %@",
        "Slow connection. Use a USB-C cable rated for 10 Gb/s or faster.",
        "Measured read speed: %@",
        "Your earlier model store converts to a library when its drive is connected.",
        "Default library for new downloads",
        "Set as default",
        "Test speed",
        "Testing speed\u{2026}",
        "Stop waiting for this drive",
        "Not available while a model is installing or moving.",
        "Connect the drive to use this.",
        "Remove the library %@?",
        "TurboSpark stops listing this library and its models. The files on the drive are not changed.",
        "TurboSpark stops listing this library and its models, and new downloads go to This Mac. The files on the drive are not changed.",
        "Remove Library",
        "Stop waiting for this drive?",
        "TurboSpark forgets the earlier model store. Its models stay on the drive and are not changed. To use them later, add the folder as a library.",
        "Some models from your earlier model store could not be copied to this Mac yet. TurboSpark tries again each time it starts and when the drive connects.",
        "TurboSpark has not finished converting your earlier model store. It tries again each time it starts and when the drive connects.",
        "TurboSpark could not stop waiting for this drive. Try again.",
        "This Mac does not need a speed test.",
        "This library has no large model file to read yet. Install a model on it, then test again.",
        "The speed test could not read this drive: %@",
        "This Mac already covers that disk. Choose a folder on a different drive.",
        "Libraries must be on a directly connected drive, not on a network share.",
        "That drive's file system cannot store a single file of 4 GiB or more. Use a drive formatted as APFS or exFAT.",
        "That drive is read-only. A library needs a drive TurboSpark can write to.",
        "That location is not a folder. Choose a folder or a drive.",
        "That folder overlaps the library %@, which already covers it.",
        "That folder overlaps a library that is already listed.",
        "That folder is inside the TurboSpark home folder, which This Mac already covers.",
        "TurboSpark could not identify that drive. Choose a drive that is connected directly to this Mac.",
        "The library information on that drive could not be read or written.",
        "TurboSpark could not use that location because of a file system error.",
        "A model is installing or moving. Try again when it finishes.",
        "That library is no longer in the list.",
        "This Mac cannot be removed.",
        "That drive is not connected. Connect it and try again.",
        "TurboSpark could not change the library list.",
        // The two labels earlier tasks left without a catalog entry
        // (tasks 5.6 and 5.10): shown only when the file cannot be read or
        // saved.
        "Library offer dismissals",
        "Moved store conversion",
    ]

    static let pluralKeys: [String] = ["%lld model(s)"]

    /// Keys the section reuses from the existing catalog.
    static let reusedKeys: [String] = [
        "Default", "Unknown", "Reveal in Finder", "Remove", "Cancel", "Dismiss",
    ]

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

    private func specifiers(_ text: String) -> [String] {
        let regex = try! NSRegularExpression(pattern: #"%(\d+\$)?(lld|ld|lf|d|@|f)"#)
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).map { String(text[Range($0.range, in: text)!]) }.sorted()
    }

    func testEveryKeyTheSectionUsesIsInTheCatalogInAllLanguages() throws {
        let strings = try loadStrings()
        let languages = AppLanguage.allCases.filter { $0 != .system }.map(\.rawValue)
        XCTAssertEqual(languages.count, 21)
        var problems: [String] = []
        for key in Self.plainKeys + Self.pluralKeys + Self.reusedKeys {
            guard let entry = strings[key], let localizations = entry["localizations"] as? [String: Any] else {
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

    func testPlainKeysKeepTheirSpecifiersInEveryLanguage() throws {
        let strings = try loadStrings()
        var problems: [String] = []
        for key in Self.plainKeys {
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
        for key in Self.plainKeys {
            let english = units((strings[key]?["localizations"] as? [String: Any])?["en"])
            XCTAssertEqual(english, [key], key)
        }
    }

    func testThePluralKeyCarriesPluralVariationsInEveryLanguage() throws {
        let strings = try loadStrings()
        let localizations = try XCTUnwrap(strings["%lld model(s)"]?["localizations"] as? [String: Any])
        for language in AppLanguage.allCases.filter({ $0 != .system }) {
            let node = try XCTUnwrap(localizations[language.rawValue] as? [String: Any], language.rawValue)
            let plural = try XCTUnwrap(
                (node["variations"] as? [String: Any])?["plural"] as? [String: Any],
                "\(language.rawValue) has no plural variations")
            XCTAssertNotNil(plural["other"], "\(language.rawValue) needs the `other` category")
        }
        // The categories each language needs (CLDR).
        func categories(_ language: String) -> Set<String> {
            let node = localizations[language] as? [String: Any]
            let plural = (node?["variations"] as? [String: Any])?["plural"] as? [String: Any]
            return Set(plural?.keys.map { $0 } ?? [])
        }
        XCTAssertEqual(categories("en"), ["one", "other"])
        XCTAssertEqual(categories("ru"), ["one", "few", "many", "other"])
        XCTAssertEqual(categories("uk"), ["one", "few", "many", "other"])
        XCTAssertEqual(categories("pl"), ["one", "few", "many", "other"])
        XCTAssertEqual(categories("ar"), ["zero", "one", "two", "few", "many", "other"])
        XCTAssertEqual(categories("he"), ["one", "two", "other"])
        XCTAssertEqual(categories("ja"), ["other"])
    }

    func testTheSectionsOwnSentencesRenderInEveryCompiledBundle() {
        // A key that exists in the catalog but did not compile resolves to its
        // English source in every language. Each language's bundle is asked.
        for language in AppLanguage.allCases where language != .system && language != .english {
            let bundle = AppStrings.bundle(for: language)
            for key in ["Connected", "Not connected", "Set as default", "Test speed", "Remove Library"] {
                let value = bundle.localizedString(forKey: key, value: "MISSING", table: nil)
                XCTAssertNotEqual(value, "MISSING", "\(language.rawValue): \(key) did not compile")
                XCTAssertNotEqual(value, key, "\(language.rawValue): \(key) is English")
            }
        }
    }
}
