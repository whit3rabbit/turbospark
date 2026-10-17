import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.2 (requirements 1.4, 1.5, 1.7, 1.10 and the rest of the add and
/// remove refusals): one localized sentence per engine code, in every language.
///
/// The parity gate skips interpolated literals, so a sentence with a `%@` in it
/// is checked here instead: each one is rendered through the compiled bundle of
/// every language, as the section renders it.
final class LibraryActionTextTests: XCTestCase {
    private var languages: [AppLanguage] {
        AppLanguage.allCases.filter { $0 != .system }
    }

    private func fault(_ code: String, _ sentence: String = "x", status: TurboSparkError.Code = .invalidArgument)
        -> TurboSparkError
    {
        TurboSparkError(code: status, message: "\(code): \(sentence)")
    }

    private func english(_ error: Error) -> String {
        LibraryActionText.sentence(for: error, language: .english)
    }

    // MARK: - One sentence per code

    func testEveryLocationRefusalHasItsOwnSentence() {
        let expected: [(String, String)] = [
            ("same_disk_as_home", "This Mac already covers that disk. Choose a folder on a different drive."),
            ("network", "Libraries must be on a directly connected drive, not on a network share."),
            (
                "small_file_limit",
                "That drive's file system cannot store a single file of 4 GiB or more. Use a drive formatted as APFS or exFAT."
            ),
            ("read_only", "That drive is read-only. A library needs a drive TurboSpark can write to."),
            ("not_a_directory", "That location is not a folder. Choose a folder or a drive."),
            ("inside_home", "That folder is inside the TurboSpark home folder, which This Mac already covers."),
            (
                "unknown_volume",
                "TurboSpark could not identify that drive. Choose a drive that is connected directly to this Mac."
            ),
        ]
        for (code, sentence) in expected {
            XCTAssertEqual(english(fault(code, "/V/X is a place")), sentence, code)
        }
        XCTAssertEqual(LibraryRefusal.allCases.count, 8, "a new refusal needs a sentence here")
    }

    func testEveryRefusalMapsToADifferentSentence() {
        var seen: [String: String] = [:]
        for refusal in LibraryRefusal.allCases {
            let sentence = english(fault(refusal.rawValue, "/V/X overlaps the library Fast, which already covers it"))
            XCTAssertNil(seen[sentence], "\(refusal.rawValue) and \(seen[sentence] ?? "") share a sentence")
            seen[sentence] = refusal.rawValue
        }
        XCTAssertEqual(seen.count, 8)
    }

    func testTheSameDiskAndNetworkSentencesSayWhatToDo() {
        XCTAssertTrue(english(fault("same_disk_as_home")).contains("This Mac already covers"))
        XCTAssertTrue(english(fault("network")).contains("directly connected drive"))
    }

    func testTheSmallFileLimitNamesTheLimit() {
        for language in languages {
            let text = LibraryActionText.sentence(for: fault("small_file_limit"), language: language)
            XCTAssertTrue(text.contains("4 GiB"), "\(language.rawValue): \(text)")
        }
    }

    func testLibraryErrorsHaveTheirOwnSentences() {
        XCTAssertEqual(english(fault("unknown_library")), "That library is no longer in the list.")
        XCTAssertEqual(english(fault("internal_library")), "This Mac cannot be removed.")
        XCTAssertEqual(english(fault("not_connected")), "That drive is not connected. Connect it and try again.")
        XCTAssertEqual(english(fault("marker")), "The library information on that drive could not be read or written.")
        XCTAssertEqual(english(fault("io")), "TurboSpark could not use that location because of a file system error.")
    }

    func testBusyIsTheLibraryBusyOnlyWithItsPrefix() {
        let busy = fault("busy", "an install is running", status: .busy)
        XCTAssertEqual(english(busy), "A model is installing or moving. Try again when it finishes.")
        // The audio device being busy shares the status and has no prefix.
        let audio = TurboSparkError(code: .busy, message: "audio device busy")
        XCTAssertEqual(english(audio), "TurboSpark could not change the library list.")
        XCTAssertNotEqual(english(audio), english(busy))
    }

    func testUnknownAndForeignErrorsGetTheGenericSentence() {
        let generic = "TurboSpark could not change the library list."
        XCTAssertEqual(english(fault("something_new")), generic)
        XCTAssertEqual(english(fault("home_only")), generic)
        XCTAssertEqual(english(TurboSparkError(code: .unknown, message: "boom")), generic)
        XCTAssertEqual(english(CancellationError()), generic)
        XCTAssertEqual(english(FakeServiceError.scripted("x")), generic)
    }

    // MARK: - Naming the library that already covers it

    func testNestedNamesTheCoveringLibraryFromTheEnginesSentence() {
        let error = fault("nested", "/Volumes/Fast/stuff overlaps the library Fast, which already covers it")
        XCTAssertEqual(english(error), "That folder overlaps the library Fast, which already covers it.")
    }

    func testNestedKeepsANameThatContainsTheLeadWords() {
        let error = fault("nested", "/V/x overlaps the library My overlaps the library Two, which already covers it")
        XCTAssertTrue(english(error).contains("My overlaps the library Two"), english(error))
    }

    func testNestedWithoutAReadableNameFallsBack() {
        XCTAssertEqual(
            english(fault("nested", "something else entirely")),
            "That folder overlaps a library that is already listed.")
        XCTAssertEqual(
            english(fault("nested", "")), "That folder overlaps a library that is already listed.")
    }

    func testTheCoveringNameIsEscapedAndCut() {
        let hostile = "evil\n\u{202E}gnp.exe"
        let text = english(fault("nested", "/V/x overlaps the library \(hostile), which already covers it"))
        XCTAssertFalse(text.contains("\n"))
        XCTAssertFalse(text.unicodeScalars.contains("\u{202E}"))
        XCTAssertTrue(text.contains("evil\\n\\u{202e}gnp.exe"), text)

        let long = String(repeating: "n", count: 400)
        let cut = english(fault("nested", "/V/x overlaps the library \(long), which already covers it"))
        XCTAssertLessThan(cut.count, 220)
    }

    func testTheCoveringNameParserHandlesEveryShape() {
        XCTAssertEqual(
            LibraryActionText.coveringLibraryName(in: "/V/x overlaps the library Fast, which already covers it"),
            "Fast")
        XCTAssertEqual(
            LibraryActionText.coveringLibraryName(in: "/V/x overlaps the library Fast"), "Fast",
            "a sentence cut by the engine's cap has no tail")
        XCTAssertNil(LibraryActionText.coveringLibraryName(in: "/V/x overlaps the library , which already covers it"))
        XCTAssertNil(LibraryActionText.coveringLibraryName(in: "no marker here"))
        XCTAssertNil(LibraryActionText.coveringLibraryName(in: ""))
    }

    // MARK: - Every language

    func testEverySentenceIsTranslatedIntoEveryLanguage() {
        let errors: [(String, TurboSparkError)] = [
            ("same_disk_as_home", fault("same_disk_as_home")),
            ("network", fault("network")),
            ("small_file_limit", fault("small_file_limit")),
            ("read_only", fault("read_only")),
            ("not_a_directory", fault("not_a_directory")),
            ("nested", fault("nested", "/V/x overlaps the library COVERING, which already covers it")),
            ("nested fallback", fault("nested", "nothing")),
            ("inside_home", fault("inside_home")),
            ("unknown_volume", fault("unknown_volume")),
            ("marker", fault("marker")),
            ("io", fault("io")),
            ("busy", fault("busy", status: .busy)),
            ("unknown_library", fault("unknown_library")),
            ("internal_library", fault("internal_library")),
            ("not_connected", fault("not_connected")),
            ("generic", fault("something_new")),
        ]
        XCTAssertEqual(languages.count, 21)
        var untranslated: [String] = []
        for (name, error) in errors {
            let source = LibraryActionText.sentence(for: error, language: .english)
            for language in languages where language != .english {
                let text = LibraryActionText.sentence(for: error, language: language)
                if text == source { untranslated.append("\(language.rawValue): \(name)") }
                XCTAssertFalse(text.contains("%@"), "\(language.rawValue): \(name): a placeholder was left unfilled")
                XCTAssertFalse(text.isEmpty)
                if name == "nested" {
                    XCTAssertTrue(text.contains("COVERING"), "\(language.rawValue): the library name is missing in \(text)")
                }
            }
        }
        XCTAssertEqual(untranslated, [], "these fell back to English")
    }

    func testTheOtherSectionSentencesAreTranslatedIntoEveryLanguage() {
        for language in languages {
            let added = LibraryActionText.added(name: "NAME", language: language)
            XCTAssertTrue(added.contains("NAME"), "\(language.rawValue): \(added)")
            XCTAssertFalse(added.contains("%@"))
            for (name, text) in [
                ("removed", LibraryActionText.removed(name: "NAME", language: language)),
                ("default", LibraryActionText.nowDefault(name: "NAME", language: language)),
            ] {
                XCTAssertTrue(text.contains("NAME"), "\(language.rawValue): \(name): \(text)")
                XCTAssertFalse(text.contains("%@"), "\(language.rawValue): \(name)")
                if language != .english {
                    XCTAssertNotEqual(
                        text,
                        name == "removed"
                            ? LibraryActionText.removed(name: "NAME", language: .english)
                            : LibraryActionText.nowDefault(name: "NAME", language: .english),
                        "\(language.rawValue): \(name) fell back to English")
                }
            }
            let unreadable = LibraryActionText.speedTestSentence(
                for: DriveSpeedTestError.unreadable("REASON"), language: language)
            XCTAssertTrue(unreadable.contains("REASON"), "\(language.rawValue): \(unreadable)")
            for (name, error) in [
                ("this mac", DriveSpeedTestError.thisMac), ("no large file", DriveSpeedTestError.noLargeFile),
            ] {
                let text = LibraryActionText.speedTestSentence(for: error, language: language)
                XCTAssertFalse(text.isEmpty, "\(language.rawValue): \(name)")
                if language != .english {
                    XCTAssertNotEqual(
                        text, LibraryActionText.speedTestSentence(for: error, language: .english),
                        "\(language.rawValue): \(name) fell back to English")
                }
            }
        }
    }

    // MARK: - Speed test sentences

    func testSpeedTestRefusalsHaveSentences() {
        func text(_ error: Error) -> String {
            LibraryActionText.speedTestSentence(for: error, language: .english)
        }
        XCTAssertEqual(text(DriveSpeedTestError.thisMac), "This Mac does not need a speed test.")
        XCTAssertEqual(
            text(DriveSpeedTestError.noLargeFile),
            "This library has no large model file to read yet. Install a model on it, then test again.")
        XCTAssertEqual(
            text(DriveSpeedTestError.unreadable("Input/output error")),
            "The speed test could not read this drive: Input/output error")
        // An error of another kind gives its own (escaped) description.
        let other = text(FakeServiceError.scripted("x"))
        XCTAssertTrue(other.hasPrefix("The speed test could not read this drive: "), other)
        XCTAssertTrue(other.contains("scripted"), other)
    }

    func testTheUnreadableReasonIsEscapedAndCut() {
        let reason = "bad\n" + String(repeating: "x", count: 600)
        let text = LibraryActionText.speedTestSentence(for: DriveSpeedTestError.unreadable(reason), language: .english)
        XCTAssertFalse(text.contains("\n"))
        XCTAssertLessThan(text.count, 300)
    }

    func testTheOutcomeSentencesEscapeTheName() {
        XCTAssertEqual(LibraryActionText.added(name: "a\nb", language: .english), "Added the library a\\nb.")
        XCTAssertEqual(LibraryActionText.removed(name: "a\nb", language: .english), "Removed the library a\\nb.")
        XCTAssertEqual(
            LibraryActionText.nowDefault(name: "a\nb", language: .english), "a\\nb is now the default library.")
    }
}
