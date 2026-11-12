import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.9 (requirement 2.4): one localized sentence per install error code.
///
/// The engine answers a refused or failed install as `"<code>: <sentence>"`. A
/// host localizes by the code and never shows the engine's English, so each code
/// here is pinned to its sentence, and the names that go into a sentence come
/// from a drive and are escaped.
final class InstallFailureTextTests: XCTestCase {
    private func error(_ code: String, _ sentence: String = "the engine's own English") -> TurboSparkError {
        TurboSparkError(code: .invalidArgument, message: "\(code): \(sentence)")
    }

    private func sentence(
        _ code: String, name: String? = "T7", place: String? = nil, language: AppLanguage = .english
    ) -> String? {
        InstallFailureText.sentence(
            for: error(code), alias: "qwen", libraryName: name, place: place, language: language)
    }

    // MARK: One sentence per code

    func testADriveDisconnectedMidInstallSaysNothingWasAdded() {
        XCTAssertEqual(
            sentence("library_disconnected"),
            "T7 was disconnected during the install. Nothing was added. Connect it and try again.")
    }

    func testALibraryThatIsNotConnected() {
        XCTAssertEqual(
            sentence("not_connected"), "T7 is not connected. Connect it or choose another library.")
    }

    func testAReadOnlyLibrary() {
        XCTAssertEqual(
            sentence("library_read_only"),
            "T7 is read-only, so nothing was installed there. Choose another library.")
    }

    func testALibraryThatIsNoLongerListed() {
        XCTAssertEqual(
            sentence("unknown_library"),
            "That library is no longer in the list, so nothing was installed. Choose another library.")
    }

    func testAnAliasInstalledElsewhereNamesWhereWhenItIsKnown() {
        XCTAssertEqual(
            sentence("alias_elsewhere", place: "This Mac"),
            "A model with this name is already installed on This Mac. Move it from there instead.")
        XCTAssertEqual(
            sentence("alias_elsewhere", place: "Spare"),
            "A model with this name is already installed on Spare. Move it from there instead.")
    }

    func testAnAliasInstalledElsewhereWithoutAKnownPlaceStillReadsRight() {
        XCTAssertEqual(
            sentence("alias_elsewhere", place: nil),
            "A model with this name is already installed in another place. Move it from there instead.")
    }

    func testBusyUsesTheSentenceTheLibraryActionsUse() {
        XCTAssertEqual(
            sentence("busy"), "A model is installing or moving. Try again when it finishes.")
        XCTAssertEqual(sentence("busy"), LibraryActionText.busy(language: .english))
    }

    func testNoTwoCodesShareASentence() {
        let codes = [
            "library_disconnected", "not_connected", "library_read_only", "unknown_library",
            "alias_elsewhere", "busy",
        ]
        let sentences = codes.compactMap { sentence($0, place: "This Mac") }
        XCTAssertEqual(sentences.count, codes.count)
        XCTAssertEqual(Set(sentences).count, codes.count)
    }

    func testTheEnginesEnglishIsNeverShown() {
        for code in ["library_disconnected", "not_connected", "library_read_only", "unknown_library",
                     "alias_elsewhere", "busy"]
        {
            let text = sentence(code, place: "This Mac")
            XCTAssertFalse(text?.contains("the engine's own English") ?? true, code)
            XCTAssertFalse(text?.contains(code) ?? true, "the code is not wording: \(code)")
        }
    }

    // MARK: What is not handled here

    func testAnErrorWithNoInstallCodeIsLeftToTheOrdinaryFailurePath() {
        XCTAssertNil(
            InstallFailureText.sentence(
                for: FakeServiceError.scripted("network down"), alias: "qwen", libraryName: "T7", place: nil))
        XCTAssertNil(
            InstallFailureText.sentence(
                for: TurboSparkError(code: .invalidArgument, message: "install cancelled"), alias: "qwen",
                libraryName: "T7", place: nil))
    }

    func testCodesThatAreNotInstallFaultsAreLeftToTheOrdinaryFailurePath() {
        // A location refusal's spelling is also an ordinary word, and an engine
        // message that happens to start "network: " is not a library fault.
        for code in ["network", "io", "marker", "same_disk_as_home", "some_future_code"] {
            XCTAssertNil(sentence(code), code)
        }
    }

    // MARK: Names come from a drive

    func testALibraryNameIsEscapedAndCut() throws {
        let hostile = "T7\u{202E}gnp.exe\n" + String(repeating: "z", count: 400)
        let text = try XCTUnwrap(sentence("library_disconnected", name: hostile))
        XCTAssertFalse(text.contains("\u{202E}"))
        XCTAssertFalse(text.contains("\n"))
        XCTAssertTrue(text.contains("\\u{202e}"), text)
        XCTAssertLessThan(text.count, 260)
    }

    func testAMissingNameReadsAsAnUnknownLibrary() throws {
        let unknown = AppStrings.localized("Unknown library", language: .english)
        XCTAssertTrue(try XCTUnwrap(sentence("library_disconnected", name: nil)).contains(unknown))
        XCTAssertTrue(try XCTUnwrap(sentence("library_disconnected", name: "")).contains(unknown))
    }

    // MARK: Where a model is installed already

    func testThePlaceIsReadFromTheRowThatHoldsTheAlias() {
        let thisMac = LibraryFixtures.row("qwen")
        let drive = LibraryFixtures.row(
            "qwen", library: LibraryFixtures.external("t7-id", name: "T7"), availability: .notConnected)
        XCTAssertEqual(InstallFailureText.place(of: thisMac, language: .english), "This Mac")
        XCTAssertEqual(InstallFailureText.place(of: drive, language: .english), "T7")
        XCTAssertEqual(
            InstallFailureText.place(of: thisMac, language: .german),
            AppStrings.localized("This Mac", language: .german))
        XCTAssertNil(InstallFailureText.place(of: nil, language: .english))
    }

    func testACustomFolderHasNoNameableLibrary() {
        let custom = InstalledModel(
            alias: "qwen", repo: "o/qwen", path: "/Users/x/Models/qwen.gturbo", family: "qwen",
            library: nil, libraryName: "custom location", installId: "internal:text/qwen",
            availability: .available, locationKind: .custom)
        XCTAssertNil(InstallFailureText.place(of: custom, language: .english))
    }

    // MARK: Every language

    func testEverySentenceRendersInEveryLanguageWithItsName() {
        let codes = [
            "library_disconnected", "not_connected", "library_read_only", "alias_elsewhere",
        ]
        for language in AppLanguage.allCases where language != .system {
            for code in codes {
                let text = sentence(code, name: "T7", place: "T7", language: language)
                XCTAssertNotNil(text, "\(language.rawValue) \(code)")
                XCTAssertTrue(text?.contains("T7") ?? false, "\(language.rawValue) \(code): \(text ?? "")")
                XCTAssertFalse(text?.contains("%@") ?? true, "\(language.rawValue) \(code): \(text ?? "")")
                if language != .english {
                    XCTAssertNotEqual(text, sentence(code, name: "T7", place: "T7", language: .english),
                        "\(language.rawValue) \(code) is still English")
                }
            }
            for code in ["unknown_library", "busy"] {
                let text = sentence(code, language: language)
                XCTAssertNotNil(text, "\(language.rawValue) \(code)")
                if language != .english {
                    XCTAssertNotEqual(text, sentence(code, language: .english), "\(language.rawValue) \(code)")
                }
            }
            let unplaced = sentence("alias_elsewhere", place: nil, language: language)
            XCTAssertNotNil(unplaced)
            XCTAssertFalse(unplaced?.contains("%@") ?? true)
        }
    }

    func testTheOfflineInstallStatementRendersInEveryLanguage() {
        for language in AppLanguage.allCases where language != .system {
            let text = InstallFailureText.offlineInstallStatement(libraryName: "T7", language: language)
            XCTAssertTrue(text.contains("T7"), "\(language.rawValue): \(text)")
            XCTAssertFalse(text.contains("%@"), "\(language.rawValue): \(text)")
            if language != .english {
                XCTAssertNotEqual(
                    text, InstallFailureText.offlineInstallStatement(libraryName: "T7", language: .english),
                    language.rawValue)
            }
        }
        XCTAssertEqual(
            InstallFailureText.offlineInstallStatement(libraryName: "T7", language: .english),
            "Installed on T7, which is not connected. Connect the drive to use it.")
    }
}
