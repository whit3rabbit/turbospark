import AppKit
import SwiftUI
@testable import TurboSparkApp
import XCTest

@MainActor
final class LocalizationAndAccessibilityTests: XCTestCase {
    func testAppLanguageEnumProperties() {
        XCTAssertEqual(AppLanguage.allCases.count, 22) // system + 21 languages
        XCTAssertTrue(AppLanguage.arabic.isRTL)
        XCTAssertEqual(AppLanguage.arabic.layoutDirection, .rightToLeft)
        XCTAssertTrue(AppLanguage.hebrew.isRTL)
        XCTAssertEqual(AppLanguage.hebrew.layoutDirection, .rightToLeft)
        XCTAssertFalse(AppLanguage.english.isRTL)
        XCTAssertEqual(AppLanguage.english.layoutDirection, .leftToRight)
        XCTAssertFalse(AppLanguage.spanish.isRTL)
        XCTAssertFalse(AppLanguage.japanese.isRTL)
        XCTAssertFalse(AppLanguage.simplifiedChinese.isRTL)
        XCTAssertFalse(AppLanguage.dutch.isRTL)

        XCTAssertEqual(AppLanguage.resolve("es"), .spanish)
        XCTAssertEqual(AppLanguage.resolve("fr"), .french)
        XCTAssertEqual(AppLanguage.resolve("nl"), .dutch)
        XCTAssertEqual(AppLanguage.resolve("pl"), .polish)
        XCTAssertEqual(AppLanguage.resolve("he"), .hebrew)
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
            "en", "es", "fr", "de", "it", "pt-BR", "ru", "ja", "ko", "zh-Hans", "zh-Hant", "ar", "hi",
            "nl", "pl", "tr", "uk", "sv", "vi", "id", "he"
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
        XCTAssertEqual(LanguageDetector.detectTextLanguage("Esta es una frase en espa\u{00F1}ol."), .spanish)
        XCTAssertTrue(LanguageDetector.isRTL(languageCode: "ar"))
        XCTAssertFalse(LanguageDetector.isRTL(languageCode: "en"))
    }

    func testAccessibilityAndTooltipCatalogKeysCoverage() throws {
        let bundle = Bundle.module
        let xcstringsURL = bundle.url(forResource: "Localizable", withExtension: "xcstrings")
            ?? URL(fileURLWithPath: "Sources/TurboSparkApp/Resources/Localizable.xcstrings")

        guard FileManager.default.fileExists(atPath: xcstringsURL.path) else {
            XCTFail("Localizable.xcstrings not found at path: \(xcstringsURL.path)")
            return
        }

        let data = try Data(contentsOf: xcstringsURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let strings = json["strings"] as? [String: [String: Any]] else {
            XCTFail("Failed to parse strings in Localizable.xcstrings")
            return
        }

        let requiredKeys = [
            "Personal", "Engine & Coding", "Installed Models", "Discover Models",
            "Model Settings", "Installed", "Rescan", "Add Folder\u{2026}", "Discover Hub \u{2192}",
            "Show Favorites Only", "Filter favorites", "Clear search",
            "Filter installed models...", "Filter by architecture", "Filter by drafter",
            "Filter by source", "Sort and group models", "Sort By", "Group By",
            "Architecture", "Drafter", "Source", "Rescan storage", "Add folder",
            "Discover models in hub", "Active model", "Manage installed models\u{2026}",
            "Discover new models\u{2026}", "Open model folder\u{2026}", "Reload model",
            "No models installed yet", "No model installed", "Select a model to load",
            "Not loaded", "Load", "Load model", "Eject model", "Reasoning Effort",
            "Fits Memory", "Streams Fast", "Resident", "Streams", "MoE Streaming",
            "Custom Tags", "Organization & Notes", "Tag name", "Add Tag",
            "Remove tag %@", "Copy model path", "Attach %@ to server",
            "Copy code snippet", "Copy code snippet to clipboard", "Start Server",
            "Stop Server", "Restart Server", "Server Running", "Server Stopped",
            "Running", "Stopped", "Local Endpoint", "API Keys", "Filter",
            "Preview %@", "Remove %@", "Install %@", "Switch to %@",
            "Chat actions for %@", "Project actions for %@", "More prompt examples",
            "Clear filters", "Dismiss error", "Dismiss notification", "Close preview",
            "Attach files", "Send", "Stop", "Generating response", "Thinking",
            "Thought process", "Approve", "Deny", "Always Allow", "Run", "View",
            "Settings", "Share message", "Read message out loud", "Stop reading out loud",
            "Copy message text", "Copy message"
        ]

        for key in requiredKeys {
            XCTAssertNotNil(strings[key], "Required accessibility/tooltip key '\(key)' is missing from Localizable.xcstrings")
        }
    }
}

