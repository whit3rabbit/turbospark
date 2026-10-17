import Foundation
import XCTest

@testable import TurboSparkApp

/// The four sentences the eject coordinator shows (tasks 5.11 and 5.12), in
/// English and in every language the app offers.
///
/// The parity gate skips interpolated literals, so a key with a `%@` in it is
/// checked here instead: each sentence is rendered through the compiled
/// bundle of every language, as the toast renders it.
final class EjectNoticeTextTests: XCTestCase {
    private var languages: [AppLanguage] {
        AppLanguage.allCases.filter { $0 != .system }
    }

    func testTheSentencesReadAsIntendedInEnglish() {
        XCTAssertEqual(
            EjectNoticeText.message(for: .ejected(models: ["qwen"]), language: .english),
            "Unloaded qwen and ejected the drive.")
        XCTAssertEqual(
            EjectNoticeText.message(for: .ejectFailed(models: ["qwen"], reason: "Resource busy"), language: .english),
            "Unloaded qwen, but the drive could not be ejected: Resource busy")
        XCTAssertEqual(
            EjectNoticeText.message(for: .removedWithoutEject(models: ["qwen", "gemma"]), language: .english),
            "Unloaded qwen and gemma because its drive was disconnected. Eject a drive before disconnecting it.")
        XCTAssertEqual(
            EjectNoticeText.message(for: .unloadBlocked(models: ["qwen"]), language: .english),
            "Could not unload qwen because it is still in use, so the drive was not ejected.")
    }

    func testEverySentenceIsTranslatedIntoEveryLanguage() {
        let notices: [EjectNotice] = [
            .ejected(models: ["qwen"]),
            .ejectFailed(models: ["qwen"], reason: "REASON"),
            .removedWithoutEject(models: ["qwen"]),
            .unloadBlocked(models: ["qwen"]),
        ]
        XCTAssertEqual(languages.count, 21)
        var untranslated: [String] = []
        for notice in notices {
            let english = EjectNoticeText.message(for: notice, language: .english)
            for language in languages where language != .english {
                let text = EjectNoticeText.message(for: notice, language: language)
                if text == english { untranslated.append("\(language.rawValue): \(english)") }
                XCTAssertTrue(text.contains("qwen"), "\(language.rawValue): the model name is missing in \(text)")
                XCTAssertFalse(text.contains("%@"), "\(language.rawValue): a placeholder was left unfilled")
                if case .ejectFailed = notice {
                    XCTAssertTrue(text.contains("REASON"), "\(language.rawValue): the reason is missing")
                    // The model comes first in every translation (plain `%@`
                    // specifiers cannot be reordered).
                    let nameAt = text.range(of: "qwen")?.lowerBound
                    let reasonAt = text.range(of: "REASON")?.lowerBound
                    if let nameAt, let reasonAt {
                        XCTAssertLessThan(nameAt, reasonAt, "\(language.rawValue): arguments in English order")
                    }
                }
            }
        }
        XCTAssertEqual(untranslated, [], "these fell back to English")
    }

    func testSeveralModelsAreJoinedInTheLanguagesOwnListStyle() {
        let english = EjectNoticeText.nameList(["a", "b", "c"], language: .english)
        XCTAssertTrue(english.contains("a") && english.contains("b") && english.contains("c"))
        XCTAssertEqual(english, "a, b, and c", "the system list style, not a hand-rolled comma join")
        XCTAssertEqual(EjectNoticeText.nameList(["only"], language: .english), "only")
        XCTAssertEqual(EjectNoticeText.nameList(["a", "b"], language: .german), "a und b")
    }

    func testAnUnboundedListOfNamesIsCapped() {
        let names = (1...200).map { "model-number-\($0)" }
        let list = EjectNoticeText.nameList(names, language: .english)
        XCTAssertLessThanOrEqual(list.count, EjectNoticeText.nameListLimit)
        XCTAssertTrue(list.hasSuffix("..."))
        XCTAssertTrue(list.hasPrefix("model-number-1,"))
    }

    func testANameFromADriveCannotBreakOutOfTheSentence() {
        // A folder name with a newline and a right-to-left override.
        let hostile = "evil\n\u{202E}gnp.exe"
        let text = EjectNoticeText.message(for: .ejected(models: [hostile]), language: .english)
        XCTAssertFalse(text.contains("\n"))
        XCTAssertFalse(text.unicodeScalars.contains("\u{202E}"))
        XCTAssertTrue(text.contains("\\n"))
        XCTAssertTrue(text.contains("\\u{202e}"))

        let blocked = EjectNoticeText.message(for: .unloadBlocked(models: [hostile]), language: .english)
        XCTAssertFalse(blocked.contains("\n"))
        XCTAssertFalse(blocked.unicodeScalars.contains("\u{202E}"))
        XCTAssertTrue(blocked.contains("\\n"))
    }

    func testTheBlockedSentenceCapsAListOfNamesLikeTheOthers() {
        let names = (1...200).map { "model-number-\($0)" }
        let text = EjectNoticeText.message(for: .unloadBlocked(models: names), language: .english)
        XCTAssertLessThan(text.count, EjectNoticeText.nameListLimit + 100)
        XCTAssertTrue(text.contains("model-number-1,"))
        XCTAssertFalse(text.contains("model-number-200"))
    }

    func testTheSystemsReasonIsEscapedAndCapped() {
        let reason = "busy\r\n" + String(repeating: "x", count: 500)
        let text = EjectNoticeText.message(for: .ejectFailed(models: ["q"], reason: reason), language: .english)
        XCTAssertFalse(text.contains("\n"))
        XCTAssertFalse(text.contains("\r"))
        XCTAssertLessThan(text.count, 300)
    }

    func testEachNoticeHasItsOwnStyle() {
        XCTAssertEqual(EjectNoticeText.style(for: .ejected(models: [])), .success)
        XCTAssertEqual(EjectNoticeText.style(for: .ejectFailed(models: [], reason: "")), .error)
        XCTAssertEqual(EjectNoticeText.style(for: .removedWithoutEject(models: [])), .warning)
        XCTAssertEqual(EjectNoticeText.style(for: .unloadBlocked(models: [])), .error)
    }
}
