import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.6 (requirement 4.3): the words of the add-library offer banner and of
/// the one toast that announces it, pinned without a view.
///
/// The banner draws `LibraryOfferBannerModel` and nothing else, so what a
/// sighted user reads and what VoiceOver reads come from the same strings.
final class LibraryOfferBannerModelTests: XCTestCase {
    private func offer(
        _ name: String = "Spare", models: Int = 4, readOnly: Bool = false, id: String = "spare-id",
        root: String = "/Volumes/Spare"
    ) -> LibraryOffer {
        LibraryOffer(volumeRoot: root, markerId: id, name: name, modelCount: models, readOnly: readOnly)
    }

    private func banner(
        _ offer: LibraryOffer, queued: Int = 0, language: AppLanguage = .english
    ) -> LibraryOfferBannerModel {
        LibraryOfferBannerModel(offer: offer, queuedCount: queued, language: language)
    }

    private var languages: [AppLanguage] { AppLanguage.allCases.filter { $0 != .system } }

    // MARK: - Title and count

    func testTheTitleNamesTheLibrary() {
        let model = banner(offer("Spare"))
        XCTAssertEqual(model.title, "Add the library Spare?")
        XCTAssertEqual(model.markerId, "spare-id")
    }

    func testTheCountReusesTheModelPluralForZeroOneAndMany() {
        XCTAssertEqual(banner(offer(models: 0)).countText, "0 models")
        XCTAssertEqual(banner(offer(models: 1)).countText, "1 model")
        XCTAssertEqual(banner(offer(models: 2)).countText, "2 models")
        XCTAssertEqual(banner(offer(models: 12)).countText, "12 models")
    }

    func testANegativeCountFromAMarkerNeverReadsNegative() {
        XCTAssertEqual(banner(offer(models: -3)).countText, "0 models")
    }

    // MARK: - Read-only and queued lines

    func testAWritableLibraryHasNoReadOnlyNote() {
        XCTAssertNil(banner(offer(readOnly: false)).readOnlyText)
    }

    func testAReadOnlyLibrarySaysSoInOneSentence() {
        XCTAssertEqual(
            banner(offer(readOnly: true)).readOnlyText,
            "A newer version of TurboSpark made this library. You can use its models, but you cannot change it.")
    }

    func testTheQueuedLineIsAbsentForNoneAndPluralForTheRest() {
        XCTAssertNil(banner(offer(), queued: 0).queuedText)
        XCTAssertEqual(banner(offer(), queued: 1).queuedText, "1 more library found")
        XCTAssertEqual(banner(offer(), queued: 2).queuedText, "2 more libraries found")
        XCTAssertEqual(banner(offer(), queued: 11).queuedText, "11 more libraries found")
    }

    func testANegativeQueuedCountIsNoLine() {
        XCTAssertNil(banner(offer(), queued: -1).queuedText)
    }

    // MARK: - A drive-supplied name is escaped, cut, and never blank

    func testABidiOverrideInTheNameIsEscapedNotShownRaw() {
        let model = banner(offer("Spare\u{202E}gpj.exe"))
        XCTAssertFalse(model.title.contains("\u{202E}"))
        XCTAssertTrue(model.title.contains("\\u{202e}"), model.title)
        XCTAssertFalse(model.accessibilityLabel.contains("\u{202E}"))
    }

    func testControlCharactersInTheNameAreEscapedNotShownRaw() {
        let model = banner(offer("Line one\nLine two\u{0007}"))
        XCTAssertFalse(model.title.contains("\n"))
        XCTAssertFalse(model.title.contains("\u{0007}"))
        XCTAssertTrue(model.title.contains("\\n"), model.title)
    }

    func testALongNameIsCutToTheLimitWithAnEllipsisMarker() {
        let long = String(repeating: "Archive ", count: 60)
        let model = banner(offer(long))
        let name = String(model.title.dropFirst("Add the library ".count).dropLast())
        XCTAssertLessThanOrEqual(name.count, LibraryOfferBannerModel.nameLimit)
        XCTAssertTrue(name.hasSuffix("..."), name)
        XCTAssertLessThan(model.title.count, 140)
    }

    func testABlankNameFallsBackToTheVolumeFolderThenToUnknown() {
        XCTAssertEqual(banner(offer("  \n ", root: "/Volumes/T7 2")).title, "Add the library T7 2?")
        XCTAssertEqual(banner(offer("", root: "/")).title, "Add the library Unknown library?")
    }

    func testThePlaceholderThisMacIsNotSpecialInAnOfferName() {
        // A marker may call its library anything, including the internal
        // library's name. The offer shows what it was told, never a guess.
        XCTAssertEqual(banner(offer("This Mac")).title, "Add the library This Mac?")
    }

    // MARK: - One combined accessibility label

    func testTheAccessibilityLabelReadsTitleCountNoteThenQueued() {
        let model = banner(offer("Spare", models: 4, readOnly: true), queued: 2)
        XCTAssertEqual(
            model.accessibilityLabel,
            "Add the library Spare? 4 models. A newer version of TurboSpark made this library. You can use its models, but you cannot change it. 2 more libraries found"
        )
    }

    func testTheAccessibilityLabelOfAPlainOfferIsTitleAndCount() {
        XCTAssertEqual(banner(offer("Spare", models: 1)).accessibilityLabel, "Add the library Spare? 1 model")
    }

    // MARK: - Why the buttons are disabled

    func testTheDisabledReasonSaysWhyInTheLanguageShown() {
        XCTAssertNil(LibraryOfferBannerModel.disabledReason(adding: false, busy: false, language: .english))
        XCTAssertEqual(
            LibraryOfferBannerModel.disabledReason(adding: false, busy: true, language: .english),
            "Not available while a model is installing or moving.")
        XCTAssertEqual(
            LibraryOfferBannerModel.disabledReason(adding: true, busy: false, language: .english),
            "Adding the library\u{2026}")
        // Both at once: the add in flight is what the user just did.
        XCTAssertEqual(
            LibraryOfferBannerModel.disabledReason(adding: true, busy: true, language: .english),
            "Adding the library\u{2026}")
    }

    // MARK: - The toast sentence

    func testTheAnnouncementNamesTheLibraryAndBothPlacesToAddIt() {
        XCTAssertEqual(
            LibraryOfferBannerModel.announcement(for: offer("Spare"), language: .english),
            "TurboSpark found the library Spare. Add it from Installed Models or Models & Storage.")
    }

    func testTheAnnouncementEscapesAndCutsTheName() {
        let text = LibraryOfferBannerModel.announcement(
            for: offer(String(repeating: "X", count: 200) + "\u{202E}"), language: .english)
        XCTAssertFalse(text.contains("\u{202E}"))
        XCTAssertLessThan(text.count, 140, "the toast shows two lines: a long name must not push the rest out")
        XCTAssertTrue(text.contains("..."))
        XCTAssertTrue(text.hasSuffix("Add it from Installed Models or Models & Storage."), text)
    }

    func testTheAnnouncementEscapesABidiOverride() {
        let text = LibraryOfferBannerModel.announcement(for: offer("Spare\u{202E}gpj"), language: .english)
        XCTAssertFalse(text.contains("\u{202E}"))
        XCTAssertTrue(text.contains("\\u{202e}"), text)
    }

    // MARK: - Every language

    func testEveryLanguageRendersEverySentenceWithItsNameAndNoPlaceholder() {
        for language in languages {
            let model = banner(offer("Spare", models: 4, readOnly: true), queued: 2, language: language)
            let announcement = LibraryOfferBannerModel.announcement(for: offer("Spare"), language: language)
            let adding = LibraryOfferBannerModel.disabledReason(adding: true, busy: false, language: language) ?? ""
            let busy = LibraryOfferBannerModel.disabledReason(adding: false, busy: true, language: language) ?? ""
            let notNow = LibraryOfferBannerModel.notNowHelp(language: language)
            for (label, text) in [
                ("title", model.title), ("count", model.countText), ("readOnly", model.readOnlyText ?? ""),
                ("queued", model.queuedText ?? ""), ("announcement", announcement), ("adding", adding),
                ("busy", busy), ("notNowHelp", notNow), ("accessibility", model.accessibilityLabel),
            ] {
                XCTAssertFalse(text.isEmpty, "\(language.rawValue) \(label) is empty")
                XCTAssertFalse(text.contains("%@") || text.contains("%lld"), "\(language.rawValue) \(label): \(text)")
            }
            XCTAssertTrue(model.title.contains("Spare"), "\(language.rawValue): \(model.title)")
            XCTAssertTrue(announcement.contains("Spare"), "\(language.rawValue): \(announcement)")
            if language != .english {
                let english = banner(offer("Spare", models: 4, readOnly: true), queued: 2, language: .english)
                XCTAssertNotEqual(model.title, english.title, "\(language.rawValue) title is English")
                XCTAssertNotEqual(model.readOnlyText, english.readOnlyText, "\(language.rawValue) note is English")
                XCTAssertNotEqual(model.queuedText, english.queuedText, "\(language.rawValue) queued is English")
                XCTAssertNotEqual(
                    announcement, LibraryOfferBannerModel.announcement(for: offer("Spare"), language: .english),
                    "\(language.rawValue) toast is English")
                XCTAssertNotEqual(adding, "Adding the library\u{2026}", "\(language.rawValue) is English")
                XCTAssertNotEqual(notNow, LibraryOfferBannerModel.notNowHelp(language: .english), language.rawValue)
            }
        }
    }

    func testTheCountFollowsEachLanguagesOwnPluralForms() {
        // Russian one, few and many; Arabic zero, one, two; Japanese has one form.
        XCTAssertNotEqual(
            banner(offer(models: 1), language: .russian).countText,
            banner(offer(models: 5), language: .russian).countText)
        XCTAssertNotEqual(
            banner(offer(models: 2), language: .russian).countText,
            banner(offer(models: 5), language: .russian).countText)
        XCTAssertNotEqual(
            banner(offer(models: 0), language: .arabic).countText,
            banner(offer(models: 3), language: .arabic).countText)
        XCTAssertNotEqual(
            banner(offer(models: 2), language: .arabic).countText,
            banner(offer(models: 1), language: .arabic).countText)
        XCTAssertNotEqual(
            banner(offer(), queued: 1, language: .russian).queuedText,
            banner(offer(), queued: 5, language: .russian).queuedText)
        XCTAssertNotEqual(
            banner(offer(), queued: 1, language: .arabic).queuedText,
            banner(offer(), queued: 2, language: .arabic).queuedText)
    }

    // MARK: - From the state

    @MainActor
    func testNoBannerWithoutAPendingOffer() {
        let state = LibraryState(
            service: FakeModelLibraryService(), probe: FakeDriveProbe(),
            dismissalsURL: FileManager.default.temporaryDirectory.appendingPathComponent("none-\(UUID()).json"))
        XCTAssertNil(LibraryOfferBannerModel.make(from: state, language: .english))
    }
}
