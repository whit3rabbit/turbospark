import Accelerate
import Foundation
import XCTest
@testable import TurboSparkApp

final class AudioSpeakerMemoryTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Start each test with an empty profile store to prevent interference
        AudioSpeakerProfileStore.save([])
    }

    override func tearDown() {
        AudioSpeakerProfileStore.save([])
        super.tearDown()
    }

    func testNormalizeVector() {
        let v: [Float] = [3.0, 4.0]
        let normalized = AudioSpeakerMemory.normalize(v)
        XCTAssertEqual(normalized.count, 2)
        XCTAssertEqual(normalized[0], 0.6, accuracy: 1e-5)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 1e-5)

        // Empty and zero vectors
        XCTAssertEqual(AudioSpeakerMemory.normalize([]), [])
        XCTAssertEqual(AudioSpeakerMemory.normalize([0, 0, 0]), [0, 0, 0])
    }

    func testMatchCosineSimilarity() {
        let p1 = AudioSpeakerMemory.Profile(name: "Alice", vector: AudioSpeakerMemory.normalize([1.0, 0.0, 0.0]))
        let p2 = AudioSpeakerMemory.Profile(name: "Bob", vector: AudioSpeakerMemory.normalize([0.0, 1.0, 0.0]))
        let profiles = [p1, p2]

        // Query identical to Alice
        let match1 = AudioSpeakerMemory.match(query: [1.0, 0.0, 0.0], against: profiles)
        XCTAssertNotNil(match1)
        XCTAssertEqual(match1?.profile.name, "Alice")
        XCTAssertEqual(match1?.similarity ?? 0, 1.0, accuracy: 1e-4)

        // Query close to Bob (e.g. 80% Bob, 20% Alice)
        let queryBob: [Float] = [0.2, 0.8, 0.0]
        let match2 = AudioSpeakerMemory.match(query: queryBob, against: profiles)
        XCTAssertNotNil(match2)
        XCTAssertEqual(match2?.profile.name, "Bob")
        XCTAssertGreaterThan(match2?.similarity ?? 0, 0.9)

        // Orthogonal query below threshold returns nil
        let orthogonal: [Float] = [0.0, 0.0, 1.0]
        let match3 = AudioSpeakerMemory.match(query: orthogonal, against: profiles, threshold: 0.5)
        XCTAssertNil(match3)
    }

    func testComputeVoiceprint() {
        // Less than 0.5s of audio returns nil
        let shortAudio = [Float](repeating: 0.1, count: 4_000)
        XCTAssertNil(AudioSpeakerMemory.computeVoiceprint(samples: shortAudio))

        // 1 second of speech-like sine wave audio at 16kHz
        let audio = (0..<16_000).map { sin(Float($0) * 0.08) * 0.4 }
        let voiceprint = AudioSpeakerMemory.computeVoiceprint(samples: audio)
        XCTAssertNotNil(voiceprint)
        guard let vp = voiceprint else { return }
        XCTAssertEqual(vp.count, 64)

        // Voiceprint is unit-normalized
        var sumSq: Float = 0
        vDSP_svesq(vp, 1, &sumSq, vDSP_Length(vp.count))
        XCTAssertEqual(sqrt(sumSq), 1.0, accuracy: 1e-4)

        // Identical audio produces identical voiceprint
        let vp2 = AudioSpeakerMemory.computeVoiceprint(samples: audio)
        XCTAssertEqual(vp, vp2)
    }

    func testCrossMeetingSpeakerMemoryWorkflow() {
        // Create Take 1 with an extracted voiceprint for "S1"
        var item1 = AudioLibraryItem(title: "Meeting 1", kind: .recording)
        let v1 = AudioSpeakerMemory.normalize([Float](repeating: 0.1, count: 64))
        item1.speakerVoiceprints = ["S1": AudioSpeakerVoiceprint(vector: v1, talkTime: 12.0)]

        // User names "S1" as "Charlie"
        AudioSpeakerMemory.recordSpeakerNaming(item: item1, renamedSpeakers: ["S1": "Charlie"])

        // Profiles should now persist Charlie
        let savedProfiles = AudioSpeakerProfileStore.load()
        XCTAssertEqual(savedProfiles.count, 1)
        XCTAssertEqual(savedProfiles.first?.name, "Charlie")
        XCTAssertEqual(savedProfiles.first?.meetingCount, 1)

        // In Take 2, diarization produces "S2" which sounds like Charlie (high similarity)
        var item2 = AudioLibraryItem(title: "Meeting 2", kind: .recording)
        let v2 = AudioSpeakerMemory.normalize([Float](repeating: 0.102, count: 64))
        item2.speakerVoiceprints = ["S2": AudioSpeakerVoiceprint(vector: v2, talkTime: 8.0)]
        let segments = [AudioLibrarySegment(start: 0, end: 5, text: "Hello everyone", speaker: "S2", timing: "clip", speakerID: "S2")]

        // Compute suggestions for Take 2
        let suggestions = AudioSpeakerMemory.suggestions(for: item2, segments: segments, profiles: savedProfiles)
        XCTAssertEqual(suggestions["S2"]?.name, "Charlie")
        XCTAssertGreaterThan(suggestions["S2"]?.confidence ?? 0, 0.99)
        if case .voiceprintMatch(let profileName)? = suggestions["S2"]?.reason {
            XCTAssertEqual(profileName, "Charlie")
        } else {
            XCTFail("Expected voiceprintMatch reason")
        }
    }
}
