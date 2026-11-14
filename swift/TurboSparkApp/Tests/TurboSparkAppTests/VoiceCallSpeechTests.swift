import XCTest
@testable import TurboSparkApp

/// The detector and the sentence planner: both pure, both held against
/// synthetic input.
final class VoiceCallSpeechDetectorTests: XCTestCase {
    private func block(_ seconds: Double, _ amplitude: Float) -> [Float] {
        (0..<Int(seconds * 16_000)).map { $0 % 2 == 0 ? amplitude : -amplitude }
    }

    func testOnsetNeedsSustainedSpeechNotABlip() {
        var detector = VoiceCallEnergyDetector()
        XCTAssertEqual(detector.process(block(0.08, 0.2), onsetScale: 1), [])
        XCTAssertEqual(detector.process(block(0.2, 0), onsetScale: 1), [])
        var fresh = VoiceCallEnergyDetector()
        XCTAssertEqual(fresh.process(block(0.2, 0.2), onsetScale: 1), [.speechStarted])
    }

    func testSilenceWindowEndsTheUtteranceOnlyAfterItElapses() {
        var detector = VoiceCallEnergyDetector()
        XCTAssertEqual(detector.process(block(0.4, 0.2), onsetScale: 1), [.speechStarted])
        XCTAssertEqual(detector.process(block(0.6, 0), onsetScale: 1), [], "0.6 s is inside the 0.7 s window")
        XCTAssertEqual(detector.process(block(0.2, 0), onsetScale: 1), [.speechEnded])
    }

    func testSoftSpeechInsideAnUtteranceIsNotAGap() {
        var detector = VoiceCallEnergyDetector()
        _ = detector.process(block(0.4, 0.2), onsetScale: 1)
        // Above half the onset level: still speech, so the window never runs out.
        XCTAssertEqual(detector.process(block(2, 0.015), onsetScale: 1), [])
    }

    func testOnsetScaleRaisesTheBarThatSpeakerBleedMustClear() {
        var bleed = VoiceCallEnergyDetector()
        XCTAssertEqual(bleed.process(block(0.5, 0.04), onsetScale: VoiceCallTuning.bargeInScale), [])
        var voice = VoiceCallEnergyDetector()
        XCTAssertEqual(voice.process(block(0.5, 0.2), onsetScale: VoiceCallTuning.bargeInScale), [.speechStarted])
        var quietRoom = VoiceCallEnergyDetector()
        XCTAssertEqual(quietRoom.process(block(0.5, 0.04), onsetScale: 1), [.speechStarted])
    }

    func testResetForgetsAnOpenUtterance() {
        var detector = VoiceCallEnergyDetector()
        _ = detector.process(block(0.4, 0.2), onsetScale: 1)
        detector.reset()
        XCTAssertEqual(detector.process(block(1, 0), onsetScale: 1), [])
    }

    func testNonFiniteSamplesAreSilence() {
        var detector = VoiceCallEnergyDetector()
        XCTAssertEqual(detector.process([Float](repeating: .nan, count: 16_000), onsetScale: 1), [])
    }
}

final class VoiceCallSpeechPlannerTests: XCTestCase {
    private func planner() -> VoiceCallSpeechPlanner { VoiceCallSpeechPlanner(placeholders: CallHarnessPlaceholders.value) }

    func testSentencesAreEmittedAsTheyComplete() {
        var p = planner()
        XCTAssertEqual(p.ingest("First sentence. Seco", final: false), ["First sentence."])
        XCTAssertEqual(p.ingest("First sentence. Second one! Thi", final: false), ["Second one!"])
        XCTAssertEqual(p.ingest("First sentence. Second one! Third", final: true), ["Third"])
    }

    func testATrailingPeriodWaitsForTheNextCharacter() {
        var p = planner()
        // "3." might still become "3.5".
        XCTAssertEqual(p.ingest("Use version 3.", final: false), [])
        XCTAssertEqual(p.ingest("Use version 3.5 now.", final: false), [])
        XCTAssertEqual(p.ingest("Use version 3.5 now.", final: true), ["Use version 3.5 now."])
    }

    func testNumberedListMarkersAndAbbreviationsAreNotSentenceEnds() {
        var p = planner()
        let out = p.ingest("1. Install it\n2. Run it, e.g. now.\n", final: true)
        XCTAssertEqual(out, ["1. Install it", "2. Run it, e.g. now."])
    }

    func testFencedCodeIsSkippedWithOneSpokenPlaceholder() {
        var p = planner()
        let text = "Try this:\n```swift\nlet x = 1\nprint(x)\n```\nThat prints one.\n"
        XCTAssertEqual(p.ingest(text, final: true), ["Try this:", "Code block skipped.", "That prints one."])
    }

    func testAnOpenFenceHoldsUntilItClosesAndNothingInsideIsSpoken() {
        var p = planner()
        XCTAssertEqual(p.ingest("Look:\n```\nsecret code", final: false), ["Look:"])
        XCTAssertEqual(p.ingest("Look:\n```\nsecret code\n```\nDone.\n", final: false), ["Code block skipped.", "Done."])
    }

    func testUnclosedFenceIsSkippedWhenTheReplyEnds() {
        var p = planner()
        XCTAssertEqual(p.ingest("Look:\n```\nnever closed", final: true), ["Look:", "Code block skipped."])
    }

    func testPathsAndLinksBecomePlaceholdersInsideTheSentence() {
        var p = planner()
        let out = p.ingest("Edit /Users/me/app/main.swift and src/lib.rs, see https://example.com/docs. Use and/or freely.", final: true)
        XCTAssertEqual(out, ["Edit a file path and a file path, see a link.", "Use and/or freely."])
    }

    func testMarkdownScaffoldingAndTablesAreNotRead() {
        var p = planner()
        let out = p.ingest("## Heading\n- **bold** item with `code`\n| a | b |\n", final: true)
        XCTAssertEqual(out, ["Heading", "bold item with code"])
    }

    func testARewrittenBufferStartsOver() {
        var p = planner()
        XCTAssertEqual(p.ingest("Step one done.\n", final: false), ["Step one done."])
        // The agent loop cleared the live buffer and a new step began.
        XCTAssertEqual(p.ingest("Next step now.\n", final: false), ["Next step now."])
    }

    func testNothingIsSpokenTwice() {
        var p = planner()
        XCTAssertEqual(p.ingest("Hello there. ", final: false), ["Hello there."])
        XCTAssertEqual(p.ingest("Hello there. ", final: true), [])
    }
}

enum CallHarnessPlaceholders {
    static let value = VoiceCallSpeechPlanner.Placeholders(
        code: "Code block skipped.", path: "a file path", link: "a link")
}
