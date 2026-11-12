import XCTest
import TurboSpark
@testable import TurboSparkApp

final class AudioMeetingNotesTests: XCTestCase {

    func testAudioMeetingNotesAdvisorRecommendation() {
        let qwen15b = InstalledModel(
            alias: "qwen35-1.5b-4bit",
            repo: "Qwen/Qwen3.5-1.5B",
            path: "/path/to/qwen15b",
            family: "qwen35",
            modality: "text"
        )
        let qwen25_7b = InstalledModel(
            alias: "qwen25-7b-4bit",
            repo: "Qwen/Qwen2.5-7B",
            path: "/path/to/qwen25_7b",
            family: "qwen2",
            modality: "text"
        )
        let gemma = InstalledModel(
            alias: "gemma4-27b",
            repo: "google/gemma-4-27b",
            path: "/path/to/gemma",
            family: "gemma4",
            modality: "text"
        )

        // 1. Recommendation checks
        XCTAssertTrue(AudioMeetingNotesAdvisor.isRecommended(model: qwen15b))
        XCTAssertTrue(AudioMeetingNotesAdvisor.isRecommended(model: qwen25_7b))
        XCTAssertFalse(AudioMeetingNotesAdvisor.isRecommended(model: gemma))

        // 2. Recommendation badge
        let badge15b = AudioMeetingNotesAdvisor.recommendationBadge(for: qwen15b)
        XCTAssertNotNil(badge15b)
        XCTAssertTrue(badge15b?.contains("32k context") == true)

        // 3. Prioritized ordering (recommended first)
        let prioritized = AudioMeetingNotesAdvisor.prioritizedModels([gemma, qwen15b, qwen25_7b])
        XCTAssertEqual(prioritized.first?.id, qwen15b.id)
        XCTAssertEqual(prioritized.last?.id, gemma.id)

        // 4. Default model selection
        let best = AudioMeetingNotesAdvisor.defaultModel(from: [gemma, qwen15b], currentSelected: nil)
        XCTAssertEqual(best?.id, qwen15b.id)
    }

    func testMeetingSummaryDoesNotMutateTranscripts() {
        let seg1 = AudioLibrarySegment(id: UUID(), start: 0, end: 5, text: "Welcome to the meeting.", speaker: "Alice", speakerID: "S1")
        let seg2 = AudioLibrarySegment(id: UUID(), start: 5, end: 10, text: "Let's review the roadmap.", speaker: "Bob", speakerID: "S2")
        let transcript = AudioTranscriptRevision(
            id: UUID(),
            createdAt: Date(),
            source: .final,
            segments: [seg1, seg2],
            language: "en"
        )

        var item = AudioLibraryItem(title: "Sprint Planning", kind: .recording)
        item.transcripts = [transcript]

        let originalTranscripts = item.transcripts

        // Create a summary
        let summary = AudioMeetingSummary(
            text: "### Key Decisions\n- Ship Phase 4 and 5 first.\n\n### Action Items\n- [ ] Alice: review PR",
            sourceTimes: [0, 5],
            transcriptRevisionID: transcript.id,
            modelAlias: "qwen35-1.5b-4bit"
        )
        item.summary = summary
        item.updatedAt = Date()

        // Assert: Transcript is strictly untouched ground truth
        XCTAssertEqual(item.transcripts, originalTranscripts)
        XCTAssertEqual(item.transcripts.count, 1)
        XCTAssertEqual(item.transcripts[0].segments.count, 2)
        XCTAssertEqual(item.transcripts[0].segments[0].text, "Welcome to the meeting.")
        XCTAssertEqual(item.transcripts[0].segments[1].text, "Let's review the roadmap.")
        XCTAssertNotNil(item.summary)
        XCTAssertEqual(item.summary?.modelAlias, "qwen35-1.5b-4bit")
    }

    func testAudioMeetingSummaryChunks() {
        let seg1 = AudioLibrarySegment(id: UUID(), start: 0, end: 10, text: "First discussion topic.")
        let seg2 = AudioLibrarySegment(id: UUID(), start: 10, end: 20, text: "Second discussion topic.")
        let chunks = AudioMeetingSummary.chunks([seg1, seg2])

        XCTAssertFalse(chunks.isEmpty)
        XCTAssertEqual(chunks[0].start, 0)
        XCTAssertTrue(chunks[0].text.contains("First discussion topic."))
    }
}
