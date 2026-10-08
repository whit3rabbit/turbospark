import XCTest
@testable import TurboSparkApp

/// Pins the pure helpers that were pulled out of per-streamed-token view
/// bodies. These assert behavior and call counts, not timings.
@MainActor
final class PerTokenHotPathTests: XCTestCase {
    private func chat(_ title: String, ageDays: Double, pinned: Bool = false, now: Date) -> AppChat {
        AppChat(title: title, updatedAt: now.addingTimeInterval(-ageDays * 86_400), isPinned: pinned)
    }

    // MARK: ChatSidebarLayout

    func testLayoutSplitsPinnedThenBucketsKeepingSortOrder() {
        let now = Date()
        let chats = AppChat.sortedForSidebar([
            chat("old", ageDays: 30, now: now),
            chat("today-a", ageDays: 0, now: now),
            chat("pin", ageDays: 5, pinned: true, now: now),
            chat("today-b", ageDays: 0.0001, now: now),
        ])
        let layout = ChatSidebarLayout.make(sortedHistory: chats, searchText: "  ", now: now)
        XCTAssertFalse(layout.isSearching)
        XCTAssertEqual(layout.pinned.map(\.title), ["pin"])
        XCTAssertEqual(layout.groups.map(\.bucket), [.today, .older], "empty buckets are dropped, order follows allCases")
        XCTAssertEqual(layout.groups[0].chats.map(\.title), ["today-a", "today-b"], "input order is kept inside a bucket")
        XCTAssertFalse(layout.isEmpty)
    }

    func testLayoutSearchIsFlatAndSuppressesGrouping() {
        let now = Date()
        let chats = [chat("Alpha plan", ageDays: 0, pinned: true, now: now), chat("beta", ageDays: 1, now: now)]
        let layout = ChatSidebarLayout.make(sortedHistory: chats, searchText: "ALPHA", now: now)
        XCTAssertTrue(layout.isSearching)
        XCTAssertEqual(layout.matches.map(\.title), ["Alpha plan"])
        XCTAssertTrue(layout.pinned.isEmpty)
        XCTAssertTrue(layout.groups.isEmpty)
        XCTAssertTrue(ChatSidebarLayout.make(sortedHistory: chats, searchText: "zzz", now: now).isEmpty)
        XCTAssertTrue(ChatSidebarLayout.make(sortedHistory: [], searchText: "", now: now).isEmpty)
    }

    // MARK: Diff parsing cache

    func testDiffBlockCacheParsesOncePerDistinctDiff() {
        let diff = "@@ -1,2 +1,2 @@\n-a\n+b\n c\n"
        let cache = DiffBlockCache()
        let first = cache.blocks(for: diff)
        for _ in 0..<50 { XCTAssertEqual(cache.blocks(for: diff).count, first.count) }
        XCTAssertEqual(cache.parseCount, 1, "re-rendering with the same diff must not reparse")
        _ = cache.blocks(for: diff + "+more\n")
        XCTAssertEqual(cache.parseCount, 2, "a changed diff must reparse")
    }

    func testParseDiffTracksLineNumbersAcrossHunks() {
        let blocks = WorktreeDiffView.parseDiff("@@ -10,2 +20,2 @@ x = -5\n-old\n+new\n")
        let lines = blocks.flatMap { block -> [ParsedDiffLine] in
            if case .lines(_, let l) = block { return l }
            return []
        }
        XCTAssertEqual(lines.first(where: { $0.kind == .deletion })?.oldLineNumber, 10)
        XCTAssertEqual(lines.first(where: { $0.kind == .addition })?.newLineNumber, 20)
    }

    // MARK: Status bar sampling

    func testSampleGateAdmitsAtMostOncePerInterval() {
        var gate = SampleRateGate()
        let t0 = Date()
        XCTAssertTrue(gate.admit(now: t0))
        XCTAssertFalse(gate.admit(now: t0.addingTimeInterval(0.1)))
        XCTAssertFalse(gate.admit(now: t0.addingTimeInterval(0.49)))
        XCTAssertTrue(gate.admit(now: t0.addingTimeInterval(0.5)))
        // 80 tok/s for 2 s is 160 triggers; about 4 samples get through.
        var gate2 = SampleRateGate()
        let admitted = (0..<160).filter { gate2.admit(now: t0.addingTimeInterval(Double($0) / 80)) }
        XCTAssertEqual(admitted.count, 4)
        var forced = SampleRateGate()
        XCTAssertTrue(forced.admit(now: t0))
        XCTAssertTrue(forced.admit(now: t0, minInterval: 0), "the timer path is never throttled")
    }

    // MARK: Count fast paths

    func testPreviewGateStillCountsGraphemesNotBytes() {
        let threshold = MessageContentPreview.collapseThreshold
        // One Character, many UTF-8 bytes: bytes exceed the threshold, Characters do not.
        let family = String(repeating: "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", count: threshold / 2)
        XCTAssertGreaterThan(family.utf8.count, threshold)
        XCTAssertLessThanOrEqual(family.count, threshold)
        XCTAssertNil(MessageContentPreview.make(family), "the byte fast path must not collapse a short message")
        XCTAssertNotNil(MessageContentPreview.make(String(repeating: "a", count: threshold + 1)))
    }

    func testSubagentStreamedTextStaysBoundedAndKeepsTheTail() {
        let state = SubagentRunState(id: "x", mode: .foreground, chatID: nil)
        for i in 0..<5_000 { state.apply(.content("chunk\(i);")) }
        XCTAssertLessThanOrEqual(state.streamedText.count, SubagentRunState.streamedTextLimit)
        XCTAssertTrue(state.streamedText.hasSuffix("chunk4999;"))
    }
}
