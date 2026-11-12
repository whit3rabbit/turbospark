import XCTest

@testable import TurboSparkApp

/// Takes longer than the diarization model accepts are labeled in parts that
/// overlap, and a part's voice takes an earlier label only on agreement in
/// the shared audio. Anything weaker stays a new speaker.
final class AudioSpeakerPartsTests: XCTestCase {
    private typealias Turn = AudioSpeakerLabels.Turn
    private typealias Part = AudioSpeakerParts.Part

    func testPlanCoversTheTakeWithBoundedOverlappingParts() {
        XCTAssertEqual(AudioSpeakerParts.plan(duration: 1800, limit: 1800), [Part(start: 0, end: 1800)])
        XCTAssertEqual(AudioSpeakerParts.plan(duration: 3600, limit: 1800),
                       [Part(start: 0, end: 1400), Part(start: 1100, end: 2500), Part(start: 2200, end: 3600)])
        XCTAssertEqual(AudioSpeakerParts.plan(duration: 1800.5, limit: 1800), [Part(start: 0, end: 1051), Part(start: 751, end: 1800.5)])
        for duration in [1801.0, 2999.9, 3600.5, 5432.1, 86_400] {
            let parts = AudioSpeakerParts.plan(duration: duration, limit: 1800)
            XCTAssertGreaterThan(parts.count, 1, "\(duration)")
            XCTAssertEqual(parts.first?.start, 0)
            XCTAssertEqual(parts.last?.end, duration)
            for part in parts {
                XCTAssertLessThanOrEqual(part.duration, 1800, "\(duration): \(part)")
                XCTAssertEqual(part.start, part.start.rounded(), "parts start on whole seconds")
            }
            for (earlier, later) in zip(parts, parts.dropFirst()) {
                XCTAssertEqual(earlier.end - later.start, AudioSpeakerParts.overlap, accuracy: 1e-9, "\(duration)")
            }
        }
        XCTAssertEqual(AudioSpeakerParts.plan(duration: 0, limit: 1800), [])
        XCTAssertEqual(AudioSpeakerParts.plan(duration: .nan, limit: 1800), [])
        XCTAssertEqual(AudioSpeakerParts.plan(duration: 4000, limit: 300), [], "no room for anything but overlap")
    }

    /// Part B numbers the same two people the other way round and adds a
    /// third. Each keeps one label across the take.
    func testJoinMatchesVoicesThroughTheOverlap() {
        let a = Part(start: 0, end: 1400), b = Part(start: 1100, end: 2500)
        let partA: [Turn] = [
            .init(speaker: "S1", start: 0, end: 100),       // X
            .init(speaker: "S2", start: 100, end: 300),     // Y
            .init(speaker: "S1", start: 1100, end: 1200),   // X, shared
            .init(speaker: "S2", start: 1200, end: 1300),   // Y, shared
            .init(speaker: "S1", start: 1300, end: 1400),   // X, shared
        ]
        let partB: [Turn] = [
            .init(speaker: "S2", start: 1100, end: 1200),   // X
            .init(speaker: "S1", start: 1200, end: 1300),   // Y
            .init(speaker: "S2", start: 1300, end: 1400),   // X
            .init(speaker: "S3", start: 1500, end: 1600),   // Z, new
            .init(speaker: "S1", start: 1600, end: 1700),   // Y
        ]
        let joined = AudioSpeakerParts.join([(a, partA), (b, partB)])
        XCTAssertEqual(joined, [
            .init(speaker: "S1", start: 0, end: 100),
            .init(speaker: "S2", start: 100, end: 300),
            .init(speaker: "S1", start: 1100, end: 1200),
            .init(speaker: "S2", start: 1200, end: 1250),
            .init(speaker: "S2", start: 1250, end: 1300),
            .init(speaker: "S1", start: 1300, end: 1400),
            .init(speaker: "S3", start: 1500, end: 1600),
            .init(speaker: "S2", start: 1600, end: 1700),
        ])
    }

    /// A voice heard for only a few seconds in the shared audio is not
    /// evidence enough. It stays a new speaker for the user to merge.
    func testJoinKeepsAVoiceNewWithoutEnoughAgreement() {
        let a = Part(start: 0, end: 1400), b = Part(start: 1100, end: 2500)
        let partA: [Turn] = [
            .init(speaker: "S1", start: 500, end: 600),
            .init(speaker: "S2", start: 1100, end: 1395),
            .init(speaker: "S1", start: 1395, end: 1400),
        ]
        let partB: [Turn] = [
            .init(speaker: "S1", start: 1100, end: 1395),
            .init(speaker: "S2", start: 1395, end: 1400),
            .init(speaker: "S2", start: 1500, end: 1800),
        ]
        // B's S2 agrees with A's S1 for 5 s, under the 8 s floor.
        let joined = AudioSpeakerParts.join([(a, partA), (b, partB)])
        XCTAssertEqual(joined.map(\.speaker), ["S1", "S2", "S2", "S3", "S3"], "\(joined)")
    }

    /// A voice whose shared time is split between two earlier voices is
    /// ambiguous. Joining it to either could merge two people.
    func testJoinKeepsASplitVoiceNew() {
        let a = Part(start: 0, end: 1400), b = Part(start: 1100, end: 2500)
        let partA: [Turn] = [
            .init(speaker: "S1", start: 600, end: 700),
            .init(speaker: "S2", start: 1100, end: 1250),
            .init(speaker: "S1", start: 1250, end: 1400),
        ]
        // Half of B's shared time is each of A's voices.
        let partB: [Turn] = [
            .init(speaker: "S1", start: 1100, end: 1400),
            .init(speaker: "S1", start: 1500, end: 1800),
        ]
        let joined = AudioSpeakerParts.join([(a, partA), (b, partB)])
        XCTAssertEqual(joined.map(\.speaker), ["S1", "S2", "S3", "S3"], "\(joined)")
    }

    /// Several parts chain: a voice present in every overlap keeps one label.
    func testJoinChainsAcrossThreeParts() {
        let parts = AudioSpeakerParts.plan(duration: 3600, limit: 1800)
        XCTAssertEqual(parts.count, 3)
        // One voice speaks 40 s of every minute. Each part labels the same
        // audio, numbering the voice differently.
        let local = ["S1", "S2", "S3"]
        let spoken = stride(from: 0.0, to: 3600, by: 60).map { $0...($0 + 40) }
        let input: [(part: Part, turns: [Turn])] = parts.enumerated().map { index, part in
            (part, spoken.compactMap { span in
                let start = max(span.lowerBound, part.start), end = min(span.upperBound, part.end)
                return end > start ? Turn(speaker: local[index], start: start, end: end) : nil
            })
        }
        let joined = AudioSpeakerParts.join(input)
        XCTAssertEqual(Set(joined.map(\.speaker)), ["S1"])
        for (earlier, later) in zip(joined, joined.dropFirst()) {
            XCTAssertLessThanOrEqual(earlier.end, later.start, "the cut leaves no overlapping turns")
        }
    }
}
