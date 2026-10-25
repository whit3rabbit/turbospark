import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.11 (requirements 7.3, 7.4 and 7.5): the words of a move, in every
/// language the app offers.
///
/// **ONE SENTENCE PER ENGINE CODE, AND EACH ENDS IN "THE ORIGINAL WAS KEPT".**
/// The engine answers `"<code>: <sentence>"` and a host localizes by the code,
/// never by the sentence. A failed move always leaves the original where it was,
/// so every failure sentence ends with that reassurance, built from one key so a
/// test can hold every sentence in every language to it.
final class ModelMoveTextTests: XCTestCase {
    private let gb: UInt64 = 1_000_000_000
    private var languages: [AppLanguage] { AppLanguage.allCases.filter { $0 != .system } }

    private func context(
        required: UInt64? = 4_000_000_000, available: UInt64? = 3_000_000_000, away: String? = nil
    ) -> MoveFailureContext {
        MoveFailureContext(
            alias: "qwen", sourceName: "This Mac", destinationName: "T7", requiredBytes: required,
            availableBytes: available, awayDriveName: away)
    }

    private func failure(
        _ code: String, language: AppLanguage = .english, context: MoveFailureContext? = nil
    ) -> MoveFailureText {
        ModelMoveText.failure(
            for: LibraryFixtures.moveError(code), context: context ?? self.context(), language: language)
    }

    /// Every code a move can answer (4.4 wire contract), and the one the app maps
    /// to the cancel line.
    static let failureCodes = [
        "not_connected", "library_read_only", "home_only", "same_library", "unknown_install",
        "unknown_library", "destination_taken", "insufficient_space", "verify_failed",
        "free_space_unknown", "io", "busy",
    ]

    // MARK: Failures

    func testEveryFailureCodeHasOneDistinctSentenceThatNeverShowsTheEnginesText() {
        var seen: [String: String] = [:]
        for code in Self.failureCodes {
            let result = failure(code)
            XCTAssertEqual(result.kind, .failed, code)
            XCTAssertFalse(result.text.isEmpty, code)
            XCTAssertFalse(result.text.contains("engine sentence"), "\(code): the engine's own text is never shown")
            XCTAssertFalse(result.text.contains(code), "\(code): a wire code is not user text")
            if let earlier = seen[result.text] { XCTFail("\(code) reads like \(earlier)") }
            seen[result.text] = code
        }
        XCTAssertEqual(seen.count, Self.failureCodes.count)
    }

    func testEveryFailureEndsWithTheOriginalWasKept() {
        for language in languages {
            let kept = ModelMoveText.originalKept(language: language)
            XCTAssertFalse(kept.isEmpty)
            for code in Self.failureCodes {
                let text = failure(code, language: language).text
                XCTAssertTrue(
                    text.hasSuffix(kept), "\(language.rawValue) \(code): does not end with the reassurance -> \(text)")
                XCTAssertGreaterThan(text.count, kept.count, "\(language.rawValue) \(code): the reason is missing")
            }
            // A code this build does not know, and an error that is not the engine's.
            let unknown = ModelMoveText.failure(
                for: LibraryFixtures.moveError("from_the_future"), context: context(), language: language)
            XCTAssertTrue(unknown.text.hasSuffix(kept), language.rawValue)
            let other = ModelMoveText.failure(for: CocoaError(.fileReadUnknown), context: context(), language: language)
            XCTAssertTrue(other.text.hasSuffix(kept), language.rawValue)
            XCTAssertEqual(other.kind, .failed)
        }
    }

    func testTheReassuranceIsNotRepeatedByTheReasons() {
        let kept = ModelMoveText.originalKept(language: .english)
        for code in Self.failureCodes {
            let text = failure(code).text
            XCTAssertEqual(text.components(separatedBy: kept).count, 2, "\(code): said once -> \(text)")
        }
    }

    func testACancelIsAnInfoLineNotAFailure() {
        let result = failure("cancelled")
        XCTAssertEqual(result.kind, .cancelled)
        XCTAssertEqual(result.text, "Move cancelled. The original was not changed.")
        for language in languages {
            XCTAssertEqual(
                failure("cancelled", language: language).text,
                AppStrings.localized("Move cancelled. The original was not changed.", language: language))
        }
    }

    func testADriveLostMidMoveIsNamedWhenWeKnowWhichOne() {
        let named = failure("not_connected", context: context(away: "T7")).text
        XCTAssertTrue(named.contains("T7"), named)
        XCTAssertEqual(named, "T7 was disconnected during the move. The original was kept.")
        let unnamed = failure("not_connected", context: context(away: nil)).text
        XCTAssertEqual(unnamed, "A drive was disconnected during the move. The original was kept.")
    }

    func testInsufficientSpaceUsesTheFiguresTheAppComputed() {
        let text = failure("insufficient_space", context: context(required: 4 * gb, available: 3 * gb)).text
        XCTAssertTrue(text.contains("T7"), text)
        XCTAssertTrue(text.contains(MetricFormat.storage(3 * gb)), text)
        XCTAssertTrue(text.contains(MetricFormat.storage(4 * gb)), text)
        let engineOnly = failure("insufficient_space", context: context(required: nil, available: nil)).text
        XCTAssertFalse(engineOnly.contains(MetricFormat.storage(0)), "a figure we do not have is not invented")
        XCTAssertTrue(engineOnly.hasSuffix(ModelMoveText.originalKept(language: .english)))
    }

    func testInsufficientSpaceQuotesFiguresOnlyWhenTheyAgreeWithTheSentence() {
        // "has only 3 GB free, but this model needs 2 GB" disproves itself. The recorded
        // size can be the smaller one (a listing that records less than is on disk), and
        // then the sentence is the plain one.
        let plain = "T7 does not have enough free space for this model. The original was kept."
        let smaller = failure("insufficient_space", context: context(required: 2 * gb, available: 3 * gb)).text
        XCTAssertEqual(smaller, plain)
        let equal = failure("insufficient_space", context: context(required: 3 * gb, available: 3 * gb)).text
        XCTAssertEqual(equal, plain, "needing exactly what is free is not 'only' that much free")
        let nothing = failure("insufficient_space", context: context(required: nil, available: nil)).text
        XCTAssertEqual(nothing, plain)
        let noFree = failure("insufficient_space", context: context(required: 2 * gb, available: nil)).text
        XCTAssertEqual(noFree, plain)

        let quoted = failure("insufficient_space", context: context(required: 4 * gb, available: 3 * gb)).text
        XCTAssertEqual(
            quoted,
            "T7 has only \(MetricFormat.storage(3 * gb)) free, but this model needs \(MetricFormat.storage(4 * gb)). The original was kept.")
    }

    func testTheSpaceSentencesExistInEveryLanguageAndDiffer() {
        for language in languages {
            let plain = failure(
                "insufficient_space", language: language, context: context(required: nil, available: nil)).text
            let quoted = failure("insufficient_space", language: language).text
            XCTAssertNotEqual(plain, quoted, language.rawValue)
            XCTAssertTrue(plain.hasSuffix(ModelMoveText.originalKept(language: language)), language.rawValue)
            XCTAssertTrue(plain.contains("T7"), "\(language.rawValue): the library is named -> \(plain)")
        }
    }

    func testAReadOnlyLibraryNamesBothLibrariesAndClaimsNeither() {
        // The engine answers `library_read_only` for a read-only source as well as a
        // read-only destination, and the app has no flag to say which.
        let text = failure("library_read_only").text
        XCTAssertEqual(text, "This Mac or T7 is read-only, so the model cannot be moved. The original was kept.")
        XCTAssertFalse(text.contains("there"), "no claim that the destination is the one: \(text)")
        let reversed = failure(
            "library_read_only",
            context: MoveFailureContext(
                alias: "qwen", sourceName: "T7", destinationName: "This Mac", requiredBytes: nil, availableBytes: nil,
                awayDriveName: nil)
        ).text
        XCTAssertEqual(reversed, "T7 or This Mac is read-only, so the model cannot be moved. The original was kept.")
        for language in languages {
            let localized = failure("library_read_only", language: language).text
            XCTAssertTrue(localized.contains("T7"), "\(language.rawValue): \(localized)")
            XCTAssertTrue(localized.hasSuffix(ModelMoveText.originalKept(language: language)), language.rawValue)
        }
    }

    func testAModelBeingMovedHasItsOwnSentenceInEveryLanguage() {
        XCTAssertEqual(
            ModelMoveText.modelBeingMoved(language: .english),
            "This model is being moved. Try again when the move finishes.")
        XCTAssertNotEqual(
            ModelMoveText.modelBeingMoved(language: .english), LibraryActionText.busy(language: .english),
            "the busy sentence is about any install or move; this one is about THIS model")
        for language in languages where language != .english {
            let text = ModelMoveText.modelBeingMoved(language: language)
            XCTAssertNotEqual(text, ModelMoveText.modelBeingMoved(language: .english), language.rawValue)
            XCTAssertFalse(text.isEmpty)
        }
    }

    func testNamesFromADriveAreEscapedInEveryFailure() {
        let hostile = MoveFailureContext(
            alias: "a\u{202E}b", sourceName: "S\u{202E}rc", destinationName: "D\u{202E}st\nline",
            requiredBytes: 4 * gb, availableBytes: 3 * gb, awayDriveName: "Aw\u{202E}ay")
        for code in Self.failureCodes {
            let text = ModelMoveText.failure(
                for: LibraryFixtures.moveError(code), context: hostile, language: .english).text
            XCTAssertFalse(text.contains("\u{202E}"), "\(code): a direction override reaches the screen")
            XCTAssertFalse(text.contains("\n"), "\(code): a line break from a drive name")
        }
    }

    func testABusyCodeIsReadFromTheLibraryFaultNotFromTheStatus() {
        // The audio device's busy status has no library prefix and is not the
        // library's `busy`: it falls to the generic sentence.
        let device = TurboSparkError(code: .busy, message: "the audio device is busy")
        let text = ModelMoveText.failure(for: device, context: context(), language: .english).text
        XCTAssertNotEqual(text, failure("busy").text)
        XCTAssertTrue(text.hasSuffix(ModelMoveText.originalKept(language: .english)))
    }

    // MARK: Successes

    private func success(
        _ kind: String, stale: String? = nil, path: String? = nil, language: AppLanguage = .english
    ) throws -> MoveSuccessText {
        let outcome = try LibraryFixtures.moveOutcome(kind, installId: "t7-id:text/qwen", staleMarker: stale, path: path)
        return ModelMoveText.success(outcome, alias: "qwen", destinationName: "T7", language: language)
    }

    func testAMovedModelSaysWhereItWentAndNothingMore() throws {
        let result = try success("moved")
        XCTAssertEqual(result.headline, "Moved qwen to T7.")
        XCTAssertNil(result.detail)
    }

    func testAStaleMarkerAndAFailedCleanupStillSayTheModelMoved() throws {
        let stale = try success("source_marker_stale", stale: "Old Drive")
        XCTAssertEqual(stale.headline, "Moved qwen to T7.")
        XCTAssertTrue(stale.detail?.contains("Old Drive") ?? false, stale.detail ?? "")
        XCTAssertTrue(stale.detail?.contains("works from its new place") ?? false)

        let unnamed = try success("source_marker_stale", stale: nil)
        XCTAssertNotNil(unnamed.detail)
        XCTAssertNotEqual(unnamed.detail, stale.detail)

        let cleanup = try success("source_cleanup_failed", stale: "Old Drive", path: "/Volumes/Old/TurboSpark/text/qwen.gturbo")
        XCTAssertEqual(cleanup.headline, "Moved qwen to T7.")
        XCTAssertTrue(cleanup.detail?.contains("/Volumes/Old/TurboSpark/text/qwen.gturbo") ?? false, cleanup.detail ?? "")
        XCTAssertTrue(cleanup.detail?.contains("works from its new place") ?? false)
        XCTAssertNotEqual(cleanup.detail, stale.detail)
    }

    func testAnOutcomeThisBuildDoesNotKnowIsStillASuccess() throws {
        let result = try success("quarantined")
        XCTAssertEqual(result.headline, "Moved qwen to T7.")
        XCTAssertNil(result.detail)
    }

    func testTheLeftoverPathIsEscapedAndNeverATerminalCommand() throws {
        let hostile = try success("source_cleanup_failed", path: "/Volumes/X/a\u{202E}b\nrm -rf")
        let detail = try XCTUnwrap(hostile.detail)
        XCTAssertFalse(detail.contains("\u{202E}"))
        XCTAssertFalse(detail.contains("\n"))
    }

    // MARK: Refusals

    /// Every refusal that has a sentence of its own. An install and a move share
    /// the one busy sentence (see `testAnInstallOrAMoveIsRefusedWithTheExistingBusySentence`).
    private var everyRefusal: [MoveRefusal] {
        [
            .moveRunning, .loadedInChat, .loadedInImages, .attachedToServer, .usedByHumanize,
            .residentInAudio, .turnRunning, .modelOpening, .noDestination, .sameLibrary,
            .destinationNotConnected(name: "T7"), .destinationNotConnected(name: nil), .sizeUnavailable,
            .freeSpaceUnknown(name: "T7"),
            .insufficientSpace(required: 4_000_000_000, available: 3_000_000_000, name: "T7"),
        ]
    }

    func testEveryRefusalIsOneDistinctSentence() {
        var seen: [String: MoveRefusal] = [:]
        for refusal in everyRefusal {
            let text = ModelMoveText.refusal(refusal, language: .english)
            XCTAssertFalse(text.isEmpty, "\(refusal)")
            if let earlier = seen[text] { XCTFail("\(refusal) reads like \(earlier)") }
            seen[text] = refusal
        }
    }

    func testTheRefusalsThatNameADriveNameIt() {
        let away = ModelMoveText.refusal(.destinationNotConnected(name: "T7"), language: .english)
        XCTAssertEqual(away, ModelInstallGate.notConnectedSentence(name: "T7", language: .english))
        let unlisted = ModelMoveText.refusal(.destinationNotConnected(name: nil), language: .english)
        XCTAssertTrue(unlisted.contains(AppStrings.localized("Unknown library", language: .english)), unlisted)
        XCTAssertTrue(ModelMoveText.refusal(.freeSpaceUnknown(name: "T7"), language: .english).contains("T7"))
        let space = ModelMoveText.refusal(
            .insufficientSpace(required: 4 * gb, available: 3 * gb, name: "T7"), language: .english)
        for part in ["T7", MetricFormat.storage(3 * gb), MetricFormat.storage(4 * gb)] {
            XCTAssertTrue(space.contains(part), "\(part) -> \(space)")
        }
    }

    func testAnInstallOrAMoveIsRefusedWithTheExistingBusySentence() {
        let busy = LibraryActionText.busy(language: .english)
        XCTAssertEqual(ModelMoveText.refusal(.moveRunning, language: .english), busy)
        XCTAssertEqual(ModelMoveText.refusal(.installRunning, language: .english), busy)
        XCTAssertEqual(failure("busy").text, busy + " " + ModelMoveText.originalKept(language: .english))
    }

    // MARK: Every language

    func testEverySentenceRendersInEveryLanguageAndIsTranslated() throws {
        let nonLatin: Set<AppLanguage> = [.arabic, .hebrew, .hindi, .japanese, .korean, .russian, .ukrainian, .simplifiedChinese, .traditionalChinese]
        for language in languages {
            var sentences: [String] = []
            sentences += everyRefusal.map { ModelMoveText.refusal($0, language: language) }
            sentences += Self.failureCodes.map { failure($0, language: language).text }
            sentences.append(failure("cancelled", language: language).text)
            sentences.append(failure("from_the_future", language: language).text)
            let moved = try success("moved", language: language)
            let stale = try success("source_marker_stale", stale: "Old", language: language)
            let unnamedStale = try success("source_marker_stale", language: language)
            let cleanup = try success("source_cleanup_failed", path: "/p/q", language: language)
            sentences += [moved.headline, stale.detail ?? "", unnamedStale.detail ?? "", cleanup.detail ?? ""]
            for sentence in sentences {
                XCTAssertFalse(sentence.isEmpty, language.rawValue)
                XCTAssertFalse(sentence.contains("%@"), "\(language.rawValue): an unfilled specifier -> \(sentence)")
                XCTAssertFalse(sentence.contains("%lld"), "\(language.rawValue): an unfilled specifier -> \(sentence)")
            }
            if nonLatin.contains(language) {
                let english = [
                    failure("io", language: .english).text, ModelMoveText.originalKept(language: .english),
                ]
                XCTAssertFalse(english.contains(failure("io", language: language).text), language.rawValue)
                XCTAssertNotEqual(ModelMoveText.originalKept(language: language), english[1], language.rawValue)
            }
            XCTAssertEqual(Set(sentences).count, sentences.count, "\(language.rawValue): two sentences collapsed into one")
        }
    }
}
