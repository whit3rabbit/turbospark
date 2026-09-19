import Foundation
import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Tests for the large-paste policy: split detection, context-fit
/// truncation, and the AppModel conversion end to end.
final class PromptPastePolicyTests: XCTestCase {
    private static let threshold = PromptPastePolicy.attachmentThreshold

    override func setUp() {
        super.setUp()
        // Each case starts from "no chats": `AppModel()` loads the archive
        // the shared (test-redirected, per `AppStorageRoot`) storage root
        // holds, and an earlier case's converted draft would otherwise walk
        // in through `selectedChat` (the same setup GhostChatTests uses).
        try? FileManager.default.removeItem(at: AppStorageRoot.file("chats_archive.json"))
    }

    private func repeated(_ text: String, _ count: Int) -> String {
        String(repeating: text, count: count)
    }

    // MARK: - splitPaste: no conversion

    func testBelowThresholdChangeStaysInline() {
        let previous = "hello "
        let current = previous + repeated("x", 999)
        XCTAssertNil(PromptPastePolicy.splitPaste(previous: previous, current: current))
    }

    func testDeletionAndIdenticalTextNeverConvert() {
        let long = repeated("y", Self.threshold * 2)
        XCTAssertNil(PromptPastePolicy.splitPaste(previous: long, current: "a"))
        XCTAssertNil(PromptPastePolicy.splitPaste(previous: long, current: long))
    }

    func testSingleKeystrokeCannotConvert() {
        let previous = repeated("a", Self.threshold * 2)
        XCTAssertNil(PromptPastePolicy.splitPaste(previous: previous, current: previous + "b"))
    }

    // MARK: - splitPaste: conversion

    func testPasteAtThresholdConverts() {
        let pasted = repeated("x", Self.threshold)
        let split = PromptPastePolicy.splitPaste(previous: "", current: pasted)
        XCTAssertNotNil(split)
        XCTAssertEqual(split?.pasted, pasted)
        XCTAssertEqual(split?.draft, "")
    }

    func testPasteInMiddleKeepsSurroundings() {
        let head = "first line\n"
        let tail = "\nlast line"
        let pasted = repeated("data ", 1_000)  // 5,000 chars
        let split = PromptPastePolicy.splitPaste(
            previous: head + tail, current: head + pasted + tail)
        XCTAssertNotNil(split)
        XCTAssertEqual(split?.pasted, pasted)
        XCTAssertEqual(split?.draft, head + tail)
    }

    func testPasteOverSelectionReplacesSelection() {
        let before = "keep "
        let selection = repeated("x", 4_800)  // selected text
        let after = " end"
        let pasted = repeated("y", 4_400)
        let split = PromptPastePolicy.splitPaste(
            previous: before + selection + after,
            current: before + pasted + after)
        XCTAssertNotNil(split)
        XCTAssertEqual(split?.pasted, pasted)
        XCTAssertEqual(split?.draft, before + after)
    }

    func testMultibytePasteSplitsOnCharacterBoundaries() {
        let head = "你好，世界 "
        let tail = " 🌍 fin"
        let pasted = repeated("🚀データ", 2_000)  // 8,000 characters, all multibyte
        let split = PromptPastePolicy.splitPaste(
            previous: head + tail, current: head + pasted + tail)
        XCTAssertNotNil(split)
        XCTAssertEqual(split?.pasted, pasted)
        XCTAssertEqual(split?.draft, head + tail)
        XCTAssertEqual(split?.pasted.count, 8_000)
    }

    func testPasteOverLongerSelectionStillConverts() {
        // Replacing a long selection with a shorter paste SHRINKS the
        // draft; the insert is what decides, not the delta.
        let selection = repeated("x", 12_000)  // selected text
        let pasted = repeated("y", 4_400)  // pasted over it
        let split = PromptPastePolicy.splitPaste(
            previous: "keep " + selection + " end",
            current: "keep " + pasted + " end")
        XCTAssertNotNil(split)
        XCTAssertEqual(split?.pasted, pasted)
        XCTAssertEqual(split?.draft, "keep  end")
    }

    // MARK: - fit

    func testFitUnderBudgetIsUntouched() {
        let text = repeated("a", 100)
        XCTAssertEqual(PromptPastePolicy.fit(text, freeTokens: 1_000), .fits)
    }

    func testFitAtExactBudgetIsUntouched() {
        let text = repeated("a", 4_000)  // 1,000 tokens * 4 chars
        XCTAssertEqual(PromptPastePolicy.fit(text, freeTokens: 1_000), .fits)
    }

    func testFitOverBudgetTruncatesHeadToCharacterBudget() {
        let text = repeated("a", 5_000)
        guard case .truncated(let kept) = PromptPastePolicy.fit(text, freeTokens: 1_000)
        else { return XCTFail("expected truncation") }
        XCTAssertEqual(kept.count, 4_000)
        XCTAssertEqual(kept, String(text.prefix(4_000)))
    }

    func testFitTruncationIsCharacterSafeForMultibyteText() {
        let text = repeated("🚀", 5_000)
        guard case .truncated(let kept) = PromptPastePolicy.fit(text, freeTokens: 1_000)
        else { return XCTFail("expected truncation") }
        // 4,000 CHARACTERS, not 4,000 bytes (which would be 1,000 emoji):
        // the budget counts characters, so the cut must land on one.
        XCTAssertEqual(kept.count, 4_000)
        XCTAssertTrue(kept.allSatisfy { $0 == "🚀" })
    }

    func testFitWithNoFreeTokensRefuses() {
        XCTAssertEqual(PromptPastePolicy.fit("anything", freeTokens: 0), .noRoom)
        XCTAssertEqual(PromptPastePolicy.fit("anything", freeTokens: -5), .noRoom)
    }

    // MARK: - AppModel end to end

    @MainActor
    func testLargePasteConvertsIntoAttachmentChip() {
        let model = AppModel()
        model.writePromptTextDirectly("hello ")
        let pasted = repeated("data ", 1_000)  // 5,000 chars

        model.processLargePaste(previous: "hello ", current: "hello " + pasted)

        XCTAssertEqual(model.promptText, "hello ")
        XCTAssertEqual(model.promptAttachments.count, 1)
        let chip = model.promptAttachments.first
        XCTAssertEqual(chip?.fileName, AppModel.pasteAttachmentFileName)
        XCTAssertEqual(chip?.extractedText, pasted)
        XCTAssertEqual(chip?.characterCount, pasted.count)
        XCTAssertFalse(chip?.wasTruncatedDuringExtraction ?? true)
    }

    @MainActor
    func testSmallChangeDoesNotConvert() {
        let model = AppModel()
        // The editor's binding has already applied the new text by the
        // time onChange fires; a non-paste change must leave it alone.
        model.writePromptTextDirectly("typing more")
        model.processLargePaste(previous: "typing ", current: "typing more")
        XCTAssertEqual(model.promptAttachments.count, 0)
        XCTAssertEqual(model.promptText, "typing more")
    }

    @MainActor
    func testProgrammaticWriteIsSuppressedFromConversion() {
        let model = AppModel()
        // A history recall or queued restore legitimately writes thousands
        // of characters; it must stay inline.
        let recalled = repeated("prompt ", 1_000)  // 7,000 chars
        model.writePromptTextDirectly(recalled)
        XCTAssertEqual(model.promptText, recalled)
        XCTAssertEqual(model.promptAttachments.count, 0)
    }

    @MainActor
    func testUndoRepasteKeepsASingleChip() {
        let model = AppModel()
        let pasted = repeated("x", Self.threshold + 10)
        model.processLargePaste(previous: "", current: pasted)
        XCTAssertEqual(model.promptAttachments.count, 1)

        // Cmd+Z restores the paste through the editor, which reads as a
        // fresh paste of the same text; the chip must not duplicate.
        model.processLargePaste(previous: model.promptText, current: pasted)
        XCTAssertEqual(model.promptAttachments.count, 1)
        XCTAssertEqual(model.promptText, "")
    }

    @MainActor
    func testPasteOverFreeWindowIsTruncatedAndFlagged() {
        let model = AppModel()
        // No session loaded: the resolved window is the 4,096 fallback and
        // maxNewTokens defaults to 2,048, so the free budget is 2,048
        // tokens = 8,192 characters.
        let pasted = repeated("a", 20_000)
        model.processLargePaste(previous: "", current: pasted)

        XCTAssertEqual(model.promptAttachments.count, 1)
        let chip = model.promptAttachments.first
        XCTAssertEqual(chip?.characterCount, 8_192)
        XCTAssertEqual(chip?.extractedText, String(pasted.prefix(8_192)))
        XCTAssertEqual(chip?.wasTruncatedDuringExtraction, true)
        XCTAssertEqual(model.promptText, "")
    }

    @MainActor
    func testPasteWithNoFreeTokensIsRefusedAndStaysInline() {
        let model = AppModel()
        model.maxNewTokens = 4_096  // == the no-session window: zero room
        let pasted = repeated("a", Self.threshold + 100)
        // The editor's binding already holds the paste when onChange fires.
        model.writePromptTextDirectly(pasted)
        model.processLargePaste(previous: "", current: pasted)

        XCTAssertEqual(model.promptAttachments.count, 0)
        XCTAssertEqual(model.promptText, pasted, "the paste stays inline when nothing fits")
    }
}
