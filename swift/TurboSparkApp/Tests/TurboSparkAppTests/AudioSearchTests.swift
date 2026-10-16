import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

@MainActor
final class AudioSearchTests: XCTestCase {
    private var root: URL!
    private var library: AudioLibraryStore!
    private var repository: ProfileRepository!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { [root] in root! }, profileIDProvider: { "search" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        repository = ProfileRepository(store: vault)
        library = AudioLibraryStore(repository: repository)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func take(_ title: String, _ segments: [(Double, Double, String)], status: AudioLibraryItem.Status = .completed) -> AudioLibraryItem {
        var item = AudioLibraryItem(title: title, kind: .recording)
        item.status = status
        item.addTranscript(segments.map { AudioLibrarySegment(start: $0.0, end: $0.1, text: $0.2) }, source: .final)
        return item
    }

    func testSavedTranscriptsAreSearchableByWordWithTimes() throws {
        XCTAssertNotNil(try repository.audioIndex(), "the optional index schema opened")
        try library.save(take("Standup", [(0, 4, "We shipped the lighthouse release."), (12.5, 16, "Budget review is next week.")]))
        let hits = library.lexicalSearch("budget")
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits.first?.start, 12.5)
        XCTAssertTrue(hits.first?.snippet.contains("Budget") == true)
        XCTAssertEqual(library.lexicalSearch("lightho").count, 1, "the last term matches as a prefix")
        XCTAssertEqual(library.lexicalSearch("standup").first?.start, nil, "a title match has no time")
    }

    func testAnEditReplacesOldWordsAndDeleteRemovesAll() throws {
        var item = take("Call", [(0, 2, "alpha bravo")])
        try library.save(item)
        XCTAssertEqual(library.lexicalSearch("alpha").count, 1)
        item.addTranscript([AudioLibrarySegment(start: 0, end: 2, text: "charlie delta")], source: .user)
        try library.save(item)
        XCTAssertTrue(library.lexicalSearch("alpha").isEmpty, "the edited transcript replaces the old words")
        XCTAssertEqual(library.lexicalSearch("charlie").count, 1)
        try library.delete(item.id)
        XCTAssertTrue(library.lexicalSearch("charlie").isEmpty)
        XCTAssertTrue(library.lexicalSearch("call").isEmpty)
    }

    func testATakeStillRecordingIsNotIndexed() throws {
        try library.save(take("Live", [(0, 2, "zulu")], status: .recording))
        XCTAssertTrue(library.lexicalSearch("zulu").isEmpty)
    }

    func testTypedQueriesCannotInjectSearchSyntax() throws {
        try library.save(take("A", [(0, 1, "near or and not")]))
        for hostile in [#""a" OR b* NEAR("#, "title:x", "-not", "(((", "\"", "*", "a AND"] {
            _ = library.lexicalSearch(hostile)
        }
        XCTAssertEqual(ProfileDatabase.audioMatchExpression(#"budget" OR x*"#), #""budget" "or" "x"*"#)
        XCTAssertNil(ProfileDatabase.audioMatchExpression("  ( * ) "))
    }

    func testVectorsRoundTripAndKeysRecordEvenWithoutWords() throws {
        let index = try XCTUnwrap(repository.audioIndex())
        let id = UUID()
        let rows = [AudioVectorRow(itemID: id, index: 0, start: 0, end: 10, text: nil, vector: [0.6, 0.8])]
        try index.replaceAudioVectors(itemID: id, kind: .audio, model: "m@1", key: "m@1|abc", rows: rows)
        XCTAssertEqual(try index.audioVectorKey(itemID: id, kind: .audio), "m@1|abc", "a sound-only item still records its key")
        XCTAssertEqual(try index.loadAudioVectors(kind: .audio, model: "m@1"), rows)
        XCTAssertTrue(try index.loadAudioVectors(kind: .audio, model: "other").isEmpty)
        XCTAssertNil(ProfileDatabase.vector(Data(count: 7), dimensions: 2), "a blob of the wrong size is not trusted")
        XCTAssertEqual(ProfileDatabase.vector(ProfileDatabase.vectorBlob([1, -2.5]), dimensions: 2), [1, -2.5])
    }

    func testFusionRanksMultiLaneMatchesFirstAndCapsHits() {
        let a = UUID(), b = UUID(), c = UUID()
        let words = [AudioLexicalHit(itemID: a, start: 1, end: 2, snippet: "x"), AudioLexicalHit(itemID: b, start: 3, end: 4, snippet: "y")]
        let meaning = [AudioSemanticHit(itemID: b, start: 5, text: "z", score: 0.9), AudioSemanticHit(itemID: c, start: 0, text: "w", score: 0.8)]
        let fused = AudioSearchFusion.fuse(words: words, meaning: meaning, sound: [])
        XCTAssertEqual(fused.map(\.itemID), [b, a, c], "two lanes outrank one; among single-lane items the words lane leads")
        XCTAssertEqual(fused.first?.lanes, [.words, .meaning])
        let many = (0..<6).map { AudioLexicalHit(itemID: a, start: Double($0), end: nil, snippet: "\($0)") }
        XCTAssertEqual(AudioSearchFusion.fuse(words: many, meaning: [], sound: []).first?.hits.count, AudioSearchFusion.maxHitsPerLane)
    }

    func testChunksStayWithinTheTokenBudgetAndKeepTimes() {
        let words = { (text: String) in text.split(separator: " ").count }
        let segments = [
            AudioSearchProjection.Segment(start: 0, end: 2, text: "one two three"),
            AudioSearchProjection.Segment(start: 2, end: 4, text: "four five"),
            AudioSearchProjection.Segment(start: 4, end: 9, text: (0..<11).map { "w\($0)" }.joined(separator: " ")),
        ]
        let chunks = AudioSearchChunker.chunks(segments, budget: 5, count: words)
        XCTAssertTrue(chunks.allSatisfy { words($0.text) <= 5 }, "\(chunks)")
        XCTAssertEqual(chunks.first?.text, "one two three four five")
        XCTAssertEqual(chunks.first?.end, 4)
        XCTAssertEqual(chunks.map { words($0.text) }.reduce(0, +), 16, "no word is dropped when an overlong segment is split")
    }

    func testSemanticRankAppliesTheFloorAndSorts() {
        let id = UUID()
        let rows = [
            AudioVectorRow(itemID: id, index: 0, start: 0, end: 1, text: "a", vector: [1, 0]),
            AudioVectorRow(itemID: id, index: 1, start: 1, end: 2, text: "b", vector: [0.6, 0.8]),
            AudioVectorRow(itemID: id, index: 2, start: 2, end: 3, text: "c", vector: [0, 1]),
            AudioVectorRow(itemID: id, index: 3, start: 3, end: 4, text: "d", vector: [1, 0, 0]),
        ]
        let hits = AudioSemanticSearch.rank([1, 0], rows, floor: 0.5, limit: 10)
        XCTAssertEqual(hits.map(\.text), ["a", "b"], "below the floor or of another size is skipped")
        XCTAssertEqual(AudioSemanticSearch.normalized([3, 4]), [0.6, 0.8])
    }

    func testSubstringFallbackFindsTextTheIndexCannotSegment() {
        let item = take("会議", [(5, 6, "予算の話")])
        let hits = AudioLibrarySearch.substringHits([item], "予算")
        XCTAssertEqual(hits.first?.start, 5)
    }
}
