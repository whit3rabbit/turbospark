import Foundation
import TurboSpark

public enum AudioWorkspacePage: String, Codable, CaseIterable, Identifiable, Sendable {
    case library, record, transcribe, voiceover, music, cleanup, advanced
    public var id: String { rawValue }
    static let destinations: [Self] = [.music, .transcribe, .voiceover]
    /// Clean Up only appears when the runtime can open at least one of its
    /// tasks; otherwise the page would offer controls whose every run is
    /// refused at session open (`AudioTask.isRunnable`).
    static var tools: [Self] {
        let cleanupRuns = AudioTask.enhancement.isRunnable || AudioTask.separation.isRunnable
        return [.record, .cleanup, .advanced].filter { $0 != .cleanup || cleanupRuns }
    }
    var title: String {
        switch self {
        case .library: return "Library"
        case .record: return "Record"
        case .transcribe: return "Speech-to-Text"
        case .voiceover: return "Text-to-Speech"
        case .music: return "Music"
        case .cleanup: return "Clean Up"
        case .advanced: return "Advanced"
        }
    }
    var symbol: String {
        switch self {
        case .library: return "books.vertical"
        case .record: return "mic"
        case .transcribe: return "text.bubble"
        case .voiceover: return "waveform"
        case .music: return "music.note"
        case .cleanup: return "slider.horizontal.3"
        case .advanced: return "wrench.and.screwdriver"
        }
    }
}

struct AudioRecipe: Codable, Equatable, Sendable {
    var text = ""
    var caption = ""
    var lyrics = ""
    var language = "auto"
    var voice = "af_heart"
    var speed: Float = 1
    var durationSeconds: Double = 10
    var steps = 30
    var seed: UInt64 = 0
    var modelID: String?
    var task = "speech_to_text"
    var experimentalBackend: String? = nil
}

struct AudioLibraryClip: Codable, Equatable, Identifiable, Sendable {
    var id = UUID()
    var assetReference: String
    var sampleRate: Double
    var channels: Int
    var frameCount: Int64
    var offset: Double = 0
    var source = "output"
    var duration: Double { sampleRate > 0 ? Double(frameCount) / sampleRate : 0 }
}

struct AudioLibrarySegment: Codable, Equatable, Identifiable, Sendable {
    var id = UUID()
    var start: Double
    var end: Double
    var text: String
    var speaker: String? = nil
    var timing = "segment"
}

struct AudioTranscriptRevision: Codable, Equatable, Identifiable, Sendable {
    enum Source: String, Codable, Sendable { case draft, final, user }
    var id = UUID()
    var createdAt = Date()
    var source: Source
    var segments: [AudioLibrarySegment]
    var language: String? = nil
    /// Keep rich native metadata even when the editor does not render it yet.
    var nativeResult: Data? = nil
}

struct AudioRecordingMarker: Codable, Equatable, Identifiable, Sendable {
    var id = UUID()
    var time: Double
    var title: String
}

struct AudioLibraryItem: Codable, Equatable, Identifiable, Sendable {
    enum Kind: String, Codable, Sendable { case recording, imported, transcription, voiceover, music, enhancement, separation, experiment }
    enum Status: String, Codable, Sendable { case draft, recording, processing, completed, interrupted, failed }
    var schemaVersion = 1
    var id = UUID()
    var title: String
    var kind: Kind
    var status: Status = .completed
    var createdAt = Date()
    var updatedAt = Date()
    var collection = ""
    var favorite = false
    var sourceID: UUID? = nil
    var recipe = AudioRecipe()
    var modelIdentity: Data? = nil
    var localModelBookmark: Data? = nil
    var modelFamily: String? = nil
    var resolvedSeed: UInt64? = nil
    var clips: [AudioLibraryClip] = []
    var transcripts: [AudioTranscriptRevision] = []
    var markers: [AudioRecordingMarker] = []
    var failure: String? = nil
    var summary: AudioMeetingSummary? = nil
    var preferredTranscript: AudioTranscriptRevision? {
        transcripts.last(where: { $0.source == .user }) ?? transcripts.last
    }
    var duration: Double { clips.map { $0.offset + $0.duration }.max() ?? 0 }
    var assetReferences: [String] { clips.map(\.assetReference) }
    func validate() throws {
        guard schemaVersion == 1, clips.count <= 100_000,
              clips.allSatisfy({ clip in
                  clip.sampleRate.isFinite && (8_000...192_000).contains(clip.sampleRate)
                      && (1...8).contains(clip.channels) && clip.frameCount > 0
                      && clip.offset.isFinite && clip.offset >= 0
                      && clip.duration + clip.offset <= 86_400
                      && ManagedAssetStore.assetID(from: clip.assetReference) != nil
              }),
              transcripts.allSatisfy({ revision in revision.segments.allSatisfy {
                  $0.start.isFinite && $0.end.isFinite && $0.start >= 0 && $0.end >= $0.start
              } }) else { throw CocoaError(.coderReadCorrupt) }
    }
    mutating func addTranscript(_ segments: [AudioLibrarySegment], source: AudioTranscriptRevision.Source, language: String? = nil, nativeResult: Data? = nil) {
        transcripts.append(AudioTranscriptRevision(source: source, segments: segments, language: language, nativeResult: nativeResult))
        updatedAt = Date()
    }
    @discardableResult mutating func recoverInterrupted() -> Bool {
        guard status == .recording || status == .processing else { return false }
        status = .interrupted
        return true
    }
}

struct AudioPreset: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var name: String
    var recipe: AudioRecipe
}

struct AudioWorkspacePreferences: Codable, Sendable {
    var schemaVersion = 1
    var page: AudioWorkspacePage = .library
    var recipe = AudioRecipe()
    var presets: [AudioPreset] = []
    /// Security-scoped bookmarks retain user-selected local models without absolute paths.
    var localModels: [String: Data] = [:]
    var selectedModels: [String: String] = [:]
}

final class AudioLibraryStore: @unchecked Sendable {
    private let repository: ProfileRepository
    private let lock = NSRecursiveLock()
    init(repository: ProfileRepository = .shared) { self.repository = repository }
    func load() throws -> [AudioLibraryItem] {
        lock.lock(); defer { lock.unlock() }
        return try repository.rawRecords(prefix: "audio:item:").map { _, data in
            let item = try JSONDecoder().decode(AudioLibraryItem.self, from: data)
            try item.validate()
            return item
        }.sorted { $0.createdAt > $1.createdAt }
    }
    func save(_ item: AudioLibraryItem) throws {
        lock.lock(); defer { lock.unlock() }
        try item.validate()
        try repository.saveAudioRecord(key: "audio:item:\(item.id.uuidString)", payload: JSONEncoder().encode(item), references: item.assetReferences)
    }
    func delete(_ id: UUID) throws {
        lock.lock(); defer { lock.unlock() }
        try repository.deleteAudioRecord(key: "audio:item:\(id.uuidString)")
    }
    func item(_ id: UUID) throws -> AudioLibraryItem? {
        lock.lock(); defer { lock.unlock() }
        return try repository.load(AudioLibraryItem.self, key: "audio:item:\(id.uuidString)")
    }
    @discardableResult func update(_ id: UUID, _ mutation: (inout AudioLibraryItem) throws -> Void) throws -> AudioLibraryItem {
        lock.lock(); defer { lock.unlock() }
        guard var item = try repository.load(AudioLibraryItem.self, key: "audio:item:\(id.uuidString)") else {
            throw CocoaError(.fileNoSuchFile)
        }
        try mutation(&item)
        try save(item)
        return item
    }
    func collectUnusedAssets() throws { try repository.collectUnusedAudioAssets() }
    func preferences() throws -> AudioWorkspacePreferences {
        lock.lock(); defer { lock.unlock() }
        let value = try repository.load(AudioWorkspacePreferences.self, key: "audio:preferences") ?? AudioWorkspacePreferences()
        guard value.schemaVersion == 1 else { throw CocoaError(.coderReadCorrupt) }
        return value
    }
    func savePreferences(_ value: AudioWorkspacePreferences) throws {
        lock.lock(); defer { lock.unlock() }
        try repository.save(value, key: "audio:preferences")
    }
}

enum AudioTranscriptExport {
    static func srt(_ segments: [AudioLibrarySegment]) -> String {
        render(segments, separator: ",", indexed: true)
    }
    static func vtt(_ segments: [AudioLibrarySegment]) -> String {
        "WEBVTT\n\n" + render(segments, separator: ".", indexed: false)
    }
    static func text(_ segments: [AudioLibrarySegment]) -> String {
        segments.map { ($0.speaker.map { $0 + ": " } ?? "") + $0.text }.joined(separator: "\n")
    }
    private static func render(_ segments: [AudioLibrarySegment], separator: String, indexed: Bool) -> String {
        segments.enumerated().compactMap { index, segment in
            guard segment.start.isFinite, segment.end.isFinite, segment.end > segment.start, segment.start >= 0 else { return nil }
            let prefix = indexed ? "\(index + 1)\n" : ""
            return prefix + time(segment.start, separator) + " --> " + time(segment.end, separator) + "\n" + text([segment]) + "\n"
        }.joined(separator: "\n")
    }
    private static func time(_ value: Double, _ separator: String) -> String {
        let ms = Int64(min(value, Double(Int32.max)) * 1000)
        return String(format: "%02lld:%02lld:%02lld%@%03lld", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, separator, ms % 1000)
    }
}

struct AudioMeetingSummary: Codable, Equatable, Sendable {
    var text: String
    var sourceTimes: [Double]
    var transcriptRevisionID: UUID
    var modelAlias: String?
    var createdAt = Date()
    struct Chunk { var start: Double; var text: String }
    static func chunks(_ segments: [AudioLibrarySegment]) -> [Chunk] {
        var chunks: [Chunk] = []
        for segment in segments {
            // Bound every local-model prompt, including a single long untimed segment.
            var remaining = AudioTranscriptExport.text([segment])[...]
            while !remaining.isEmpty {
                let part = String(remaining.prefix(6_000))
                remaining = remaining.dropFirst(part.count)
                if let last = chunks.indices.last, chunks[last].text.count + part.count < 8_000 {
                    chunks[last].text += "\n" + part
                } else { chunks.append(Chunk(start: segment.start, text: part)) }
            }
        }
        return chunks
    }
}
