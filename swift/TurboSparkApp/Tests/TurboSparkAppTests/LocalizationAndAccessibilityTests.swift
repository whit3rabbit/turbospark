import AppKit
import SwiftUI
@testable import TurboSparkApp
import XCTest

@MainActor
final class LocalizationAndAccessibilityTests: XCTestCase {
    func testAppLanguageEnumProperties() {
        XCTAssertEqual(AppLanguage.allCases.count, 14) // system + 13 languages
        XCTAssertTrue(AppLanguage.arabic.isRTL)
        XCTAssertEqual(AppLanguage.arabic.layoutDirection, .rightToLeft)
        XCTAssertFalse(AppLanguage.english.isRTL)
        XCTAssertEqual(AppLanguage.english.layoutDirection, .leftToRight)
        XCTAssertFalse(AppLanguage.spanish.isRTL)
        XCTAssertFalse(AppLanguage.japanese.isRTL)
        XCTAssertFalse(AppLanguage.simplifiedChinese.isRTL)

        XCTAssertEqual(AppLanguage.resolve("es"), .spanish)
        XCTAssertEqual(AppLanguage.resolve("fr"), .french)
        XCTAssertEqual(AppLanguage.resolve("unknown"), .system)
    }

    func testLocalizableXCStringsValidityAndLanguageCoverage() throws {
        // Read Localizable.xcstrings directly from Resources folder or Bundle
        let bundle = Bundle.module
        let xcstringsURL = bundle.url(forResource: "Localizable", withExtension: "xcstrings")
            ?? URL(fileURLWithPath: "Sources/TurboSparkApp/Resources/Localizable.xcstrings")

        guard FileManager.default.fileExists(atPath: xcstringsURL.path) else {
            XCTFail("Localizable.xcstrings not found at path: \(xcstringsURL.path)")
            return
        }

        let data = try Data(contentsOf: xcstringsURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(json)
        XCTAssertEqual(json?["sourceLanguage"] as? String, "en")

        guard let strings = json?["strings"] as? [String: [String: Any]] else {
            XCTFail("strings dictionary missing in Localizable.xcstrings")
            return
        }

        XCTAssertGreaterThan(strings.count, 50, "Expected comprehensive localization dictionary with > 50 strings")

        let expectedLanguages = [
            "en", "es", "fr", "de", "it", "pt-BR", "ru", "ja", "ko", "zh-Hans", "zh-Hant", "ar", "hi"
        ]

        for (key, entry) in strings {
            guard let localizations = entry["localizations"] as? [String: Any] else {
                XCTFail("Key '\(key)' is missing localizations map")
                continue
            }

            for lang in expectedLanguages {
                guard let langEntry = localizations[lang] as? [String: Any] else {
                    XCTFail("Key '\(key)' is missing localization for language '\(lang)'")
                    continue
                }

                if let variations = langEntry["variations"] as? [String: Any] {
                    XCTAssertNotNil(variations["plural"], "Key '\(key)' has variations for '\(lang)' but no plural rule")
                } else if let stringUnit = langEntry["stringUnit"] as? [String: Any] {
                    let val = stringUnit["value"] as? String
                    XCTAssertNotNil(val, "Key '\(key)' has nil value for language '\(lang)'")
                    XCTAssertFalse(val?.isEmpty ?? true, "Key '\(key)' has empty value for language '\(lang)'")
                } else {
                    XCTFail("Key '\(key)' has unrecognized localization entry for language '\(lang)'")
                }
            }
        }
    }

    func testResponseMarkdownRendererLinkAndTooltipAttributes() {
        let renderer = ResponseMarkdownRenderer()
        let markdown = "Check out [TurboSpark Documentation](https://github.com/turbospark/turbospark) for details."
        let result = renderer.render(markdown)

        XCTAssertFalse(result.usedFallback)
        let attrString = result.attributedString
        XCTAssertGreaterThan(attrString.length, 0)

        var foundLink = false
        var foundTooltip = false

        attrString.enumerateAttributes(in: NSRange(location: 0, length: attrString.length)) { attrs, range, _ in
            if let linkURL = attrs[.link] as? URL {
                foundLink = true
                XCTAssertEqual(linkURL.absoluteString, "https://github.com/turbospark/turbospark")
            }
            if let tooltip = attrs[.toolTip] as? String {
                foundTooltip = true
                XCTAssertEqual(tooltip, "https://github.com/turbospark/turbospark")
            }
        }

        XCTAssertTrue(foundLink, "Rendered attributed string should contain NSAttributedString.Key.link")
        XCTAssertTrue(foundTooltip, "Rendered attributed string should contain NSAttributedString.Key.toolTip for hover")
    }

    func testLanguageDetector() {
        XCTAssertEqual(LanguageDetector.detectTextLanguage("This is a simple english sentence."), .english)
        XCTAssertEqual(LanguageDetector.detectTextLanguage("Esta es una frase en español."), .spanish)
        XCTAssertTrue(LanguageDetector.isRTL(languageCode: "ar"))
        XCTAssertFalse(LanguageDetector.isRTL(languageCode: "en"))
    }
}
