import Accelerate
import CryptoKit
import Foundation
import TurboSpark

extension AudioLibraryItem {
    /// What search indexes for this item, or nil while it is still
    /// recording: its draft changes every few seconds, and the final pass
    /// indexes it when the take ends.
    var searchProjection: AudioSearchProjection? {
        guard status != .recording, !isEmptyGeneratedAttempt else { return nil }
        let segments = (preferredTranscript?.segments ?? []).map {
            AudioSearchProjection.Segment(start: $0.start, end: $0.end, text: $0.text)
        }
        var hasher = SHA256()
        for part in [title, collection] + tagList { hasher.update(data: Data(part.utf8)); hasher.update(data: Data([0])) }
        hasher.update(data: Data([1]))
        for s in segments { hasher.update(data: Data("\(s.start)|\(s.end)|\(s.text)".utf8)); hasher.update(data: Data([0])) }
        return AudioSearchProjection(itemID: id, hash: Self.hex(hasher.finalize()), title: title, collection: collection,
                                     tags: tagList, segments: segments)
    }

    /// Identifies the audio itself, so sound vectors are rebuilt only when
    /// the clips change, not when the transcript or title does.
    var audioFingerprint: String {
        var hasher = SHA256()
        for clip in clips { hasher.update(data: Data("\(clip.assetReference)|\(clip.offset)|\(clip.frameCount)|\(clip.sampleRate)".utf8)) }
        return Self.hex(hasher.finalize())
    }

    private static func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }
}

enum AudioSearchLane: String, CaseIterable, Identifiable, Sendable {
    case words, meaning, sound
    var id: String { rawValue }
    var titleKey: String {
        switch self {
        case .words: return "Words"
        case .meaning: return "Meaning"
        case .sound: return "Sound"
        }
    }
    var symbol: String {
        switch self {
        case .words: return "textformat"
        case .meaning: return "text.magnifyingglass"
        case .sound: return "waveform"
        }
    }
}

struct AudioSearchHit: Equatable, Sendable, Identifiable {
    let lane: AudioSearchLane
    /// Seconds into the item, or nil for a title or tag match.
    let start: Double?
    let snippet: String
    var id: String { "\(lane.rawValue)-\(start ?? -1)-\(snippet)" }
}

struct AudioSearchResult: Equatable, Sendable, Identifiable {
    let itemID: UUID
    var lanes: [AudioSearchLane]
    var hits: [AudioSearchHit]
    var score: Double
    var id: UUID { itemID }
}

/// A semantic match before fusion: one transcript chunk or audio window.
struct AudioSemanticHit: Equatable, Sendable {
    let itemID: UUID
    let start: Double
    let text: String?
    let score: Float
}

enum AudioSearchFusion {
    static let maxHitsPerLane = 3

    /// Reciprocal-rank fusion at the item level (k = 60, as memory recall
    /// uses): each lane ranks items by their best hit, and an item's score
    /// is the sum of 1 / (k + rank) over the lanes it appears in, so a
    /// match in two lanes outranks one in a single lane. Ties keep the
    /// words lane's order, then meaning's, then sound's.
    static func fuse(words: [AudioLexicalHit], meaning: [AudioSemanticHit], sound: [AudioSemanticHit], k: Double = 60) -> [AudioSearchResult] {
        var results: [UUID: AudioSearchResult] = [:]
        var order: [UUID] = []
        func add(_ lane: AudioSearchLane, _ hits: [(UUID, AudioSearchHit)]) {
            var rank = 0
            var seen = Set<UUID>()
            for (itemID, hit) in hits {
                if results[itemID] == nil {
                    results[itemID] = AudioSearchResult(itemID: itemID, lanes: [], hits: [], score: 0)
                    order.append(itemID)
                }
                if seen.insert(itemID).inserted {
                    rank += 1
                    results[itemID]?.score += 1 / (k + Double(rank))
                    results[itemID]?.lanes.append(lane)
                }
                if (results[itemID]?.hits.filter { $0.lane == lane }.count ?? 0) < maxHitsPerLane {
                    results[itemID]?.hits.append(hit)
                }
            }
        }
        add(.words, words.map { ($0.itemID, AudioSearchHit(lane: .words, start: $0.start, snippet: $0.snippet)) })
        add(.meaning, meaning.map { ($0.itemID, AudioSearchHit(lane: .meaning, start: $0.start, snippet: $0.text ?? "")) })
        add(.sound, sound.map { ($0.itemID, AudioSearchHit(lane: .sound, start: $0.start, snippet: "")) })
        let position = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($1, $0) })
        return results.values.sorted {
            $0.score != $1.score ? $0.score > $1.score : position[$0.itemID]! < position[$1.itemID]!
        }
    }
}

/// Meaning and sound search over stored EmbeddingGemma 2 vectors. Text
/// chunks are 768-d; audio windows are stored at 256-d (Matryoshka), and the
/// query is cut to 256 and renormalized to compare with them, which works
/// because both towers embed into one space. Vectors are unit length, so a
/// dot product is the cosine.
final class AudioSemanticSearch: @unchecked Sendable {
    static let textDimensions = 768
    static let audioDimensions = 256
    /// The sound floor is checked by `crates/runtime/tests/coreml_sound_recall.rs`
    /// (25 real clips, 16 classes): every query-to-clip cosine fell between
    /// 0.59 and 0.77, related or not, so no floor separates them. 0.15 keeps
    /// every relevant clip and drops nothing; only the order carries signal
    /// (right clip in the top three for 75% of queries), so the sound lane
    /// stays labelled experimental. The meaning floor has no gate yet.
    static let minMeaningScore: Float = 0.25
    static let minSoundScore: Float = 0.15

    let modelPath: String
    let modelKey: String
    private let lock = NSLock()
    private var cache: [AudioVectorKind: [AudioVectorRow]] = [:]

    init(modelPath: String, modelKey: String) { self.modelPath = modelPath; self.modelKey = modelKey }

    /// Drops cached vectors after the indexer writes new ones.
    func invalidate() { lock.lock(); cache = [:]; lock.unlock() }

    private func rows(_ kind: AudioVectorKind, index: ProfileDatabase) -> [AudioVectorRow] {
        lock.lock(); defer { lock.unlock() }
        if let cached = cache[kind] { return cached }
        let loaded = (try? index.loadAudioVectors(kind: kind, model: modelKey)) ?? []
        cache[kind] = loaded
        return loaded
    }

    func search(_ query: String, index: ProfileDatabase, limit: Int = 50) async throws -> (meaning: [AudioSemanticHit], sound: [AudioSemanticHit]) {
        let request = EmbeddingRequest(texts: [query], task: "search_query", dimensions: Self.textDimensions)
        guard let full = try await TurboSparkEmbedding.encode(request: request, modelPath: modelPath).texts.first,
              full.count == Self.textDimensions else { return ([], []) }
        let short = Self.normalized(Array(full.prefix(Self.audioDimensions)))
        return (Self.rank(full, rows(.text, index: index), floor: Self.minMeaningScore, limit: limit),
                Self.rank(short, rows(.audio, index: index), floor: Self.minSoundScore, limit: limit))
    }

    static func rank(_ query: [Float], _ rows: [AudioVectorRow], floor: Float, limit: Int) -> [AudioSemanticHit] {
        var hits: [AudioSemanticHit] = []
        for row in rows where row.vector.count == query.count {
            var score: Float = 0
            vDSP_dotpr(query, 1, row.vector, 1, &score, vDSP_Length(query.count))
            if score >= floor { hits.append(AudioSemanticHit(itemID: row.itemID, start: row.start, text: row.text, score: score)) }
        }
        return Array(hits.sorted { $0.score > $1.score }.prefix(limit))
    }

    static func normalized(_ v: [Float]) -> [Float] {
        var norm: Float = 0
        vDSP_svesq(v, 1, &norm, vDSP_Length(v.count))
        let length = norm.squareRoot()
        guard length > 0 else { return v }
        return v.map { $0 / length }
    }
}
