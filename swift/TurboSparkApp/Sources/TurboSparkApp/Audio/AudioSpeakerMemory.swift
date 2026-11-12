import Accelerate
import Foundation
import TurboSpark

/// Persistent cross-meeting speaker memory and channel-aware identification.
///
/// Phase 4 & Phase 5:
/// 1. Channel Separation: Detects which speaker was captured via the local
///    microphone ("You") vs remote callers on the system audio track.
/// 2. Voiceprint Matching: Compares normalized acoustic embedding vectors
///    against historical confirmed speakers to suggest names across meetings.
/// 3. Late Binding: Preserves ground-truth linkage so past speaker namings
///    automatically inform new meetings without destructive rewrites.
enum AudioSpeakerMemory {

    /// A stored voiceprint profile for a known person.
    struct Profile: Codable, Equatable, Identifiable, Sendable {
        var id: String
        var name: String
        var vector: [Float]
        var meetingCount: Int
        var lastSeen: Date

        init(id: String = UUID().uuidString, name: String, vector: [Float], meetingCount: Int = 1, lastSeen: Date = Date()) {
            self.id = id
            self.name = name
            self.vector = vector
            self.meetingCount = meetingCount
            self.lastSeen = lastSeen
        }
    }

    /// A suggestion for a speaker in the current take.
    struct Suggestion: Equatable, Sendable {
        enum Reason: Equatable, Sendable {
            case microphone
            case voiceprintMatch(profileName: String)
        }

        var name: String
        var confidence: Float
        var reason: Reason
    }

    /// High confidence threshold for auto-suggesting a past speaker name.
    static let highConfidenceThreshold: Float = 0.72
    /// Minimum similarity to present a candidate suggestion chip.
    static let candidateThreshold: Float = 0.55

    /// Identifies which speaker in a take was the local microphone user, if any.
    /// Returns the speaker key (e.g. "S1") that predominates on the microphone track.
    static func detectMicrophoneSpeaker(
        item: AudioLibraryItem,
        segments: [AudioLibrarySegment],
        store: ManagedAssetStore = .shared
    ) -> String? {
        let micClips = item.clips.filter { $0.source == "microphone" }
        let sysClips = item.clips.filter { $0.source == "system" }
        guard !micClips.isEmpty, !sysClips.isEmpty else { return nil }

        let keys = AudioSpeakerRoster.keys(in: segments)
        guard !keys.isEmpty else { return nil }

        var speakerMicScores: [String: Double] = [:]
        for key in keys {
            let speakerSegments = segments.filter { AudioSpeakerRoster.key($0) == key && ($0.end - $0.start) >= 0.5 }
            guard !speakerSegments.isEmpty else { continue }

            var micEnergyTotal = 0.0
            var sysEnergyTotal = 0.0
            var testedSeconds = 0.0

            for seg in speakerSegments.prefix(4) {
                let start = seg.start
                let duration = min(3.0, seg.end - seg.start)
                guard duration >= 0.5 else { continue }

                if let micSamples = try? AudioMediaIO.readMono16k(micClips, store: store, start: start, duration: duration), !micSamples.isEmpty {
                    var rms: Float = 0
                    vDSP_rmsqv(micSamples, 1, &rms, vDSP_Length(micSamples.count))
                    micEnergyTotal += Double(rms) * duration
                }
                if let sysSamples = try? AudioMediaIO.readMono16k(sysClips, store: store, start: start, duration: duration), !sysSamples.isEmpty {
                    var rms: Float = 0
                    vDSP_rmsqv(sysSamples, 1, &rms, vDSP_Length(sysSamples.count))
                    sysEnergyTotal += Double(rms) * duration
                }
                testedSeconds += duration
            }

            guard testedSeconds > 0 else { continue }
            let total = micEnergyTotal + sysEnergyTotal
            if total > 0.005 {
                let ratio = micEnergyTotal / total
                if ratio >= 0.60 {
                    speakerMicScores[key] = ratio
                }
            }
        }

        // Return the speaker key with the highest microphone energy dominance.
        return speakerMicScores.max(by: { $0.value < $1.value })?.key
    }

    /// Computes a normalized 64-band acoustic voiceprint vector from 16kHz mono audio.
    static func computeVoiceprint(samples: [Float], sampleRate: Double = 16_000) -> [Float]? {
        guard samples.count >= 8_000 else { return nil }
        let frameSize = 400
        let hopSize = 160
        let nFft = 512
        guard let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(nFft), .FORWARD) else { return nil }
        defer { vDSP_DFT_DestroySetup(setup) }

        var window = [Float](repeating: 0, count: frameSize)
        vDSP_hann_window(&window, vDSP_Length(frameSize), Int32(vDSP_HANN_NORM))

        let numFilters = 64
        let minFreq: Double = 80.0
        let maxFreq: Double = min(7600.0, sampleRate / 2.0)
        let minMel = 2595.0 * log10(1.0 + minFreq / 700.0)
        let maxMel = 2595.0 * log10(1.0 + maxFreq / 700.0)
        var melPoints = [Double](repeating: 0, count: numFilters + 2)
        for i in 0..<(numFilters + 2) {
            melPoints[i] = minMel + Double(i) * (maxMel - minMel) / Double(numFilters + 1)
        }
        var binIndices = [Int](repeating: 0, count: numFilters + 2)
        for i in 0..<(numFilters + 2) {
            let freq = 700.0 * (pow(10.0, melPoints[i] / 2595.0) - 1.0)
            binIndices[i] = min(nFft / 2, max(0, Int(round((Double(nFft) + 1.0) * freq / sampleRate))))
        }

        var accumulatedMel = [Float](repeating: 0, count: numFilters)
        var activeFrames = 0

        var inReal = [Float](repeating: 0, count: nFft)
        let inImag = [Float](repeating: 0, count: nFft)
        var outReal = [Float](repeating: 0, count: nFft / 2)
        var outImag = [Float](repeating: 0, count: nFft / 2)
        var mags = [Float](repeating: 0, count: nFft / 2)

        var offset = 0
        samples.withUnsafeBufferPointer { sPtr in
            while offset + frameSize <= samples.count {
                var frameEnergy: Float = 0
                vDSP_svesq(sPtr.baseAddress! + offset, 1, &frameEnergy, vDSP_Length(frameSize))
                if frameEnergy > 0.001 {
                    for i in 0..<frameSize {
                        inReal[i] = sPtr[offset + i] * window[i]
                    }
                    for i in frameSize..<nFft {
                        inReal[i] = 0
                    }
                    outReal.withUnsafeMutableBufferPointer { rPtr in
                        outImag.withUnsafeMutableBufferPointer { iPtr in
                            var split = DSPSplitComplex(realp: rPtr.baseAddress!, imagp: iPtr.baseAddress!)
                            vDSP_DFT_Execute(setup, inReal, inImag, &split.realp.pointee, &split.imagp.pointee)
                            vDSP_zvmags(&split, 1, &mags, 1, vDSP_Length(nFft / 2))
                        }
                    }
                    for m in 1...numFilters {
                        let left = binIndices[m - 1]
                        let center = binIndices[m]
                        let right = binIndices[m + 1]
                        var filterEnergy: Float = 0
                        for k in left..<center {
                            let weight = Float(k - left) / Float(max(1, center - left))
                            filterEnergy += mags[k] * weight
                        }
                        for k in center...min(right, nFft / 2 - 1) {
                            let weight = Float(right - k) / Float(max(1, right - center))
                            filterEnergy += mags[k] * weight
                        }
                        accumulatedMel[m - 1] += log(max(1e-6, filterEnergy))
                    }
                    activeFrames += 1
                }
                offset += hopSize
            }
        }

        guard activeFrames > 0 else { return nil }
        let meanMel = accumulatedMel.map { $0 / Float(activeFrames) }
        return normalize(meanMel)
    }

    /// Extracts acoustic voiceprints for all speakers in a take based on their active speech turns.
    static func extractVoiceprints(
        for item: AudioLibraryItem,
        turns: [AudioSpeakerLabels.Turn],
        store: ManagedAssetStore = .shared
    ) -> [String: AudioSpeakerVoiceprint] {
        guard !turns.isEmpty, !item.clips.isEmpty else { return [:] }
        let uniqueSpeakers = Array(Set(turns.map(\.speaker))).sorted()
        var prints: [String: AudioSpeakerVoiceprint] = [:]

        for speaker in uniqueSpeakers {
            let speakerTurns = turns.filter { $0.speaker == speaker && ($0.end - $0.start) >= 0.5 }
            guard !speakerTurns.isEmpty else { continue }

            var accumulatedSamples: [Float] = []
            var totalDuration: Double = 0

            for turn in speakerTurns {
                let duration = min(4.0, turn.end - turn.start)
                guard duration >= 0.5 else { continue }
                if let pcm = try? AudioMediaIO.readMono16kTake(item.clips, store: store, start: turn.start, duration: duration, limit: 30), !pcm.isEmpty {
                    accumulatedSamples.append(contentsOf: pcm)
                    totalDuration += duration
                }
                if accumulatedSamples.count >= 16_000 * 20 { // Cap sample extraction at 20 seconds of speech
                    break
                }
            }

            guard accumulatedSamples.count >= 8_000,
                  let vector = computeVoiceprint(samples: accumulatedSamples) else { continue }
            prints[speaker] = AudioSpeakerVoiceprint(vector: vector, talkTime: totalDuration)
        }

        return prints
    }

    /// Matches a query vector against known speaker profiles using cosine similarity.
    static func match(
        query: [Float],
        against profiles: [Profile],
        threshold: Float = candidateThreshold
    ) -> (profile: Profile, similarity: Float)? {
        guard !query.isEmpty else { return nil }
        let normalizedQuery = normalize(query)
        var best: (profile: Profile, similarity: Float)?

        for profile in profiles where profile.vector.count == query.count {
            let normalizedProfile = normalize(profile.vector)
            var score: Float = 0
            vDSP_dotpr(normalizedQuery, 1, normalizedProfile, 1, &score, vDSP_Length(query.count))
            if score >= threshold {
                if let current = best {
                    if score > current.similarity { best = (profile, score) }
                } else {
                    best = (profile, score)
                }
            }
        }
        return best
    }

    /// Computes suggestions for all speakers in a take based on channel isolation
    /// and cross-meeting voiceprint matching.
    static func suggestions(
        for item: AudioLibraryItem,
        segments: [AudioLibrarySegment],
        profiles: [Profile],
        store: ManagedAssetStore = .shared
    ) -> [String: Suggestion] {
        var results: [String: Suggestion] = [:]
        let micSpeaker = detectMicrophoneSpeaker(item: item, segments: segments, store: store)

        if let micSpeaker {
            results[micSpeaker] = Suggestion(
                name: "You",
                confidence: 0.98,
                reason: .microphone
            )
        }

        // Load any stored voiceprints for this item to match against past profiles.
        if let takeVoiceprints = item.speakerVoiceprints {
            for (key, print) in takeVoiceprints where results[key] == nil {
                if let best = match(query: print.vector, against: profiles) {
                    results[key] = Suggestion(
                        name: best.profile.name,
                        confidence: best.similarity,
                        reason: .voiceprintMatch(profileName: best.profile.name)
                    )
                }
            }
        }

        return results
    }

    /// Updates persistent speaker profiles when the user assigns or renames speakers.
    static func recordSpeakerNaming(
        item: AudioLibraryItem,
        renamedSpeakers: [String: String],
        store: AudioSpeakerProfileStore.Type = AudioSpeakerProfileStore.self
    ) {
        guard let voiceprints = item.speakerVoiceprints, !voiceprints.isEmpty else { return }
        var profiles = store.load()

        for (key, chosenName) in renamedSpeakers {
            let trimmed = chosenName.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, trimmed != AudioSpeakerRoster.defaultName(key) else { continue }
            guard let print = voiceprints[key], !print.vector.isEmpty else { continue }

            if let index = profiles.firstIndex(where: { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame }) {
                // Update existing profile vector with a weighted moving average
                var existing = profiles[index]
                let oldWeight: Float = min(5.0, Float(existing.meetingCount))
                let newWeight: Float = 1.0
                var combined = [Float](repeating: 0, count: print.vector.count)
                for i in 0..<combined.count {
                    combined[i] = (existing.vector[i] * oldWeight + print.vector[i] * newWeight) / (oldWeight + newWeight)
                }
                existing.vector = normalize(combined)
                existing.meetingCount += 1
                existing.lastSeen = Date()
                profiles[index] = existing
            } else {
                // Create a new profile
                let newProfile = Profile(
                    name: trimmed,
                    vector: normalize(print.vector),
                    meetingCount: 1,
                    lastSeen: Date()
                )
                profiles.append(newProfile)
            }
        }

        store.save(profiles)
    }

    /// L2-normalizes a float vector.
    static func normalize(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return vector }
        var sumSquares: Float = 0
        vDSP_svesq(vector, 1, &sumSquares, vDSP_Length(vector.count))
        let norm = sqrt(sumSquares)
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }
}

/// Persistent storage provider for `AudioSpeakerMemory.Profile`s under `~/Library/Application Support/TurboSpark/speaker_profiles.json`.
public enum AudioSpeakerProfileStore {
    private static var fileURL: URL {
        AppStorageRoot.directory.appendingPathComponent("speaker_profiles.json")
    }

    static func load() -> [AudioSpeakerMemory.Profile] {
        AppJSONStore.load([AudioSpeakerMemory.Profile].self, from: fileURL, label: "Speaker Profiles") ?? []
    }

    static func save(_ profiles: [AudioSpeakerMemory.Profile]) {
        AppJSONStore.save(profiles, to: fileURL, label: "Speaker Profiles")
    }
}

/// A stored speaker voiceprint record attached to an item.
struct AudioSpeakerVoiceprint: Codable, Equatable, Sendable {
    var vector: [Float]
    var talkTime: Double
    var sampleRate: Double

    init(vector: [Float], talkTime: Double, sampleRate: Double = 16_000) {
        self.vector = vector
        self.talkTime = talkTime
        self.sampleRate = sampleRate
    }
}
