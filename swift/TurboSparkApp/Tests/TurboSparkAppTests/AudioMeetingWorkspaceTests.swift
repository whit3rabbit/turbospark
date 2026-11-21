import Foundation
import XCTest
@testable import TurboSparkApp

/// The meeting surface: the user's own notes, notes generation prompts, the
/// meeting list filter and Ask's retrieval.
final class AudioMeetingWorkspaceTests: XCTestCase {
    // MARK: User notes

    func testRecordSavedBeforeUserNotesDecodesWithNone() throws {
        let item = AudioLibraryItem(title: "Old", kind: .recording)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(item)) as? [String: Any])
        json.removeValue(forKey: "userNotes")
        let decoded = try JSONDecoder().decode(AudioLibraryItem.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(decoded.userNotes)
        XCTAssertFalse(decoded.hasUserNotes)
        XCTAssertNoThrow(try decoded.validate())
    }

    func testOversizedUserNotesFailValidation() {
        var item = AudioLibraryItem(title: "Huge", kind: .recording)
        item.userNotes = String(repeating: "a", count: AudioLibraryItem.maxUserNotesBytes + 1)
        XCTAssertThrowsError(try item.validate())
        item.userNotes = "  \n"
        XCTAssertFalse(item.hasUserNotes)
    }

    @MainActor private func makeController() throws -> (AudioWorkspaceController, AudioLibraryStore, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "meeting-notes" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        controller.liveTranscription = false
        return (controller, library, root)
    }

    /// Notes typed while a capture keeps saving the take must survive the
    /// capture's writes, and the capture's writes must survive the notes.
    @MainActor func testUserNotesAndCaptureWritesDoNotOverwriteEachOther() throws {
        let (controller, library, root) = try makeController()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        var item = AudioLibraryItem(title: "Standup", kind: .recording)
        item.status = .recording
        try controller.save(item)
        controller.stageUserNotes("- ship Friday", for: item.id)
        // A capture write lands before the debounce fires.
        controller.update(item.id) { $0.markers.append(AudioRecordingMarker(time: 4, title: "Marker 1")) }
        XCTAssertEqual(controller.userNotes(for: item), "- ship Friday", "staged text shows before it is saved")
        controller.flushUserNotes(for: item.id)
        let saved = try XCTUnwrap(library.item(item.id))
        XCTAssertEqual(saved.userNotes, "- ship Friday")
        XCTAssertEqual(saved.markers.count, 1)
        XCTAssertTrue(controller.pendingUserNotes.isEmpty)
        controller.update(item.id) { $0.markers.append(AudioRecordingMarker(time: 9, title: "Marker 2")) }
        XCTAssertEqual(try library.item(item.id)?.userNotes, "- ship Friday")
    }

    @MainActor func testShutdownSavesStagedUserNotes() throws {
        let (controller, library, root) = try makeController()
        defer { try? FileManager.default.removeItem(at: root) }
        let item = AudioLibraryItem(title: "Review", kind: .recording)
        try controller.save(item)
        controller.stageUserNotes("decide on vendor", for: item.id)
        controller.shutdown()
        XCTAssertEqual(try library.item(item.id)?.userNotes, "decide on vendor")
    }

    @MainActor func testClearingNotesStoresNone() throws {
        let (controller, library, root) = try makeController()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        var item = AudioLibraryItem(title: "Sync", kind: .recording)
        item.userNotes = "x"
        try controller.save(item)
        XCTAssertTrue(controller.saveUserNotes("", for: item.id))
        XCTAssertNil(try library.item(item.id)?.userNotes)
    }

    // MARK: Meeting list filter

    func testMeetingListMatchesEveryWordAcrossFields() {
        var a = AudioLibraryItem(title: "Roadmap review", kind: .recording)
        a.userNotes = "Budget for Q3"
        var b = AudioLibraryItem(title: "Caf\u{e9} sync", kind: .recording)
        b.transcripts = [AudioTranscriptRevision(source: .final, segments: [AudioLibrarySegment(start: 0, end: 1, text: "Launch moves to June")])]
        let items = [a, b]
        XCTAssertEqual(SpeechToTextLibrary.matching(items, query: "").count, 2)
        XCTAssertEqual(SpeechToTextLibrary.matching(items, query: "roadmap budget").map(\.id), [a.id])
        XCTAssertEqual(SpeechToTextLibrary.matching(items, query: "cafe JUNE").map(\.id), [b.id])
        XCTAssertTrue(SpeechToTextLibrary.matching(items, query: "roadmap june").isEmpty)
    }

    // MARK: Notes prompts

    func testShortMeetingGetsFinalFormatAndUserNotesFirst() {
        let prompt = AudioMeetingNotesPrompt.single(transcript: "Alex: we ship Friday", userNotes: "  ship date?  ")
        XCTAssertTrue(prompt.system.contains("### Action Items"))
        XCTAssertTrue(prompt.system.contains("note-taker's own notes"))
        XCTAssertTrue(prompt.user.hasPrefix("Note-taker's notes:\nship date?\n\nTranscript:\nAlex: we ship Friday"))
        let bare = AudioMeetingNotesPrompt.single(transcript: "t", userNotes: "   ")
        XCTAssertFalse(bare.system.contains("note-taker"))
        XCTAssertEqual(bare.user, "Transcript:\nt")
    }

    func testMergeLevelsReduceToOne() {
        XCTAssertEqual(AudioMeetingNotesPrompt.mergeLevels(partCount: 1), [])
        XCTAssertEqual(AudioMeetingNotesPrompt.mergeLevels(partCount: 6), [1])
        XCTAssertEqual(AudioMeetingNotesPrompt.mergeLevels(partCount: 7), [2, 1])
        XCTAssertEqual(AudioMeetingNotesPrompt.mergeLevels(partCount: 37), [7, 2, 1])
    }

    func testOnlyTheFinalMergeCarriesUserNotes() {
        let partial = AudioMeetingNotesPrompt.merge(parts: ["a", "b"], userNotes: "mine", final: false)
        XCTAssertFalse(partial.user.contains("mine"))
        XCTAssertFalse(partial.system.contains("### Summary"))
        let final = AudioMeetingNotesPrompt.merge(parts: ["a", "b"], userNotes: "mine", final: true)
        XCTAssertTrue(final.user.hasPrefix("Note-taker's notes:\nmine"))
        XCTAssertTrue(final.user.contains("Part 2:\nb"))
        XCTAssertTrue(final.system.contains("### Summary"))
    }

    // MARK: Ask retrieval

    private func meeting(_ title: String, _ lines: [(Double, String)]) -> AudioLibraryItem {
        var item = AudioLibraryItem(title: title, kind: .recording)
        item.transcripts = [AudioTranscriptRevision(source: .final, segments: lines.map {
            AudioLibrarySegment(start: $0.0, end: $0.0 + 5, text: $0.1)
        })]
        return item
    }

    func testShortMeetingIsSentWhole() {
        var item = meeting("Sync", [(0, "Hello all."), (5, "Budget is approved.")])
        item.userNotes = "budget?"
        let passages = AudioMeetingRetriever.passages(for: [item])
        XCTAssertEqual(passages.first?.kind, .userNotes, "notes lead, so the model reads them first")
        XCTAssertEqual(AudioMeetingRetriever.select(question: "anything", from: passages), passages)
    }

    func testLongMeetingKeepsTheMatchingPassageInTimeOrder() {
        let filler = (0..<400).map { (Double($0) * 5, "general discussion about many unrelated things number \($0)") }
        var lines = filler
        lines.insert((1_000.5, "The vendor contract renewal was rejected by legal."), at: 201)
        let item = meeting("Quarterly", lines)
        let passages = AudioMeetingRetriever.passages(for: [item])
        XCTAssertGreaterThan(passages.reduce(0) { $0 + $1.text.count }, AudioMeetingRetriever.contextCharacterBudget)
        let chosen = AudioMeetingRetriever.select(question: "What happened to the vendor contract?", from: passages, budget: 6_000)
        XCTAssertTrue(chosen.contains { $0.text.contains("vendor contract renewal") })
        XCTAssertLessThanOrEqual(chosen.reduce(0) { $0 + $1.text.count }, 6_000)
        let starts = chosen.compactMap(\.start)
        XCTAssertEqual(starts, starts.sorted(), "passages read in meeting order")
    }

    func testMeaningHitPicksThePassageCoveringItsStart() {
        let item = meeting("M", [(0, "alpha"), (60, "beta")])
        let passages = [
            AudioAskPassage(itemID: item.id, title: "M", kind: .transcript, start: 0, end: 30, text: "alpha"),
            AudioAskPassage(itemID: item.id, title: "M", kind: .transcript, start: 60, end: 90, text: "beta"),
        ]
        let hits = [AudioSemanticHit(itemID: item.id, start: 70, text: nil, score: 0.9),
                    AudioSemanticHit(itemID: UUID(), start: 0, text: nil, score: 0.8)]
        XCTAssertEqual(AudioMeetingRetriever.meaningOrder(hits, passages), [1])
    }

    func testBM25IgnoresStopwordsAndFoldsCase() {
        let id = UUID()
        let passages = [
            AudioAskPassage(itemID: id, title: "", kind: .transcript, start: 0, end: 1, text: "the the the and"),
            AudioAskPassage(itemID: id, title: "", kind: .transcript, start: 1, end: 2, text: "Pricing d\u{e9}cision made"),
        ]
        XCTAssertEqual(AudioMeetingRetriever.bm25Order("What is the PRICING decision?", passages), [1])
    }

    func testAskPromptCitesTimesAndTitlesAcrossMeetings() {
        let id = UUID()
        let one = AudioAskPrompt.build(question: "q", passages: [
            AudioAskPassage(itemID: id, title: "Plan", kind: .transcript, start: 75, end: 80, text: "x"),
        ], meetingCount: 1)
        XCTAssertTrue(one.user.contains("[1:15]\nx"))
        XCTAssertFalse(one.system.contains("meeting title"))
        let many = AudioAskPrompt.build(question: "q", passages: [
            AudioAskPassage(itemID: id, title: "Plan", kind: .transcript, start: 75, end: 80, text: "x"),
            AudioAskPassage(itemID: UUID(), title: "Retro", kind: .userNotes, start: nil, end: nil, text: "y"),
        ], meetingCount: 2)
        XCTAssertTrue(many.user.contains("[Plan, 1:15]"))
        XCTAssertTrue(many.user.contains("[Retro, the user's own notes]"))
        XCTAssertTrue(many.user.hasSuffix("Question: q"))
        XCTAssertTrue(many.system.contains("meeting title"))
    }
}
