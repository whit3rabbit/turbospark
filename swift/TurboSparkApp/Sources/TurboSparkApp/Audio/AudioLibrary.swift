import Foundation
import TurboSpark

public enum AudioWorkspacePage: String, Codable, CaseIterable, Identifiable, Sendable {
    case library, record, transcribe, voiceover, voiceClone, voiceDesign, music, cleanup, advanced
    public var id: String { rawValue }
    static let destinations: [Self] = [.music, .transcribe, .voiceover, .voiceClone, .voiceDesign]
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
        case .voiceClone: return "Voice Cloning"
        case .voiceDesign: return "Voice Design"
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
        case .voiceClone: return "person.wave.2"
        case .voiceDesign: return "slider.horizontal.3"
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
    /// Music uses the checkpoint recommendation until the user edits its stepper.
    /// Optional so recipes saved before this setting continue to decode.
    var musicStepsOverride: Int? = nil
    var seed: UInt64 = 0
    var modelID: String?
    var task = "speech_to_text"
    var experimentalBackend: String? = nil
    var speech: AudioSpeechSettings? = nil
    var music: AudioMusicSettings? = nil

    var musicSettings: AudioMusicSettings {
        get { music ?? AudioMusicSettings() }
        set { music = newValue }
    }

    var effectiveMusicStepsOverride: Int? {
        musicStepsOverride ?? (steps == 30 ? nil : steps)
    }

    var musicFrames: Int {
        Int((durationSeconds * musicSettings.frameRate).rounded())
    }
}

struct AudioMusicSettings: Codable, Equatable, Sendable {
    var contextSize: Int = 10_240
    var maxFrames: Int = 9_000
    var frameRate: Double = 25.0
}

struct AudioMusicSamplingRecommendation: Equatable, Sendable {
    let profile: String
    let steps: Int

    static let fast = AudioMusicSamplingRecommendation(profile: "Fast", steps: 21)

    static func read(configAt modelURL: URL?) -> Self {
        guard let modelURL,
              let data = try? Data(contentsOf: modelURL.appendingPathComponent("config.json")) else {
            return .fast
        }
        return fromConfig(data)
    }

    static func fromConfig(_ data: Data) -> Self {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let extra = root["extra"] as? [String: Any],
              let variant = extra["turbospark_music3_variant"] as? [String: Any],
              variant["name"] as? String == "turbo_lora_8step_merged",
              variant["flow_steps"] as? Int == 8,
              variant["flow_solver"] as? String == "euler",
              variant["flow_guidance"] as? String == "exact",
              let quantization = root["quantization"] as? [String: Any],
              quantization["mode"] as? String == "affine",
              quantization["bits"] as? Int == 4,
              quantization["group_size"] as? Int == 64 else {
            return .fast
        }
        return AudioMusicSamplingRecommendation(profile: "Turbo", steps: 8)
    }
}

struct AudioSpeechSettings: Codable, Equatable, Sendable {
    enum Mode: String, Codable, Sendable { case customVoice, clone, design }
    var mode: Mode = .customVoice
    var instruction = ""
    var reference: AudioVoiceReference?
    var referenceText = ""
    var embeddingOnly = false
    var temperature: Float = 0.9
    var maxTokens = 2048
    /// Options of a natively cloning engine (MegaTTS3). Optional so settings
    /// saved before it existed still decode.
    var cloneOptions: AudioCloneOptions?
}

/// Named diffusion step counts for the Quality picker. They match the audio
/// crate's `megatts3::Quality`; the MegaTTS3 README's step study measured
/// what fewer steps cost.
enum CloneQuality: CaseIterable, Sendable {
    case fast, balanced, best
    var steps: Int {
        switch self {
        case .fast: 8
        case .balanced: 16
        case .best: 32
        }
    }
    /// The preset a step count belongs to, or nil for a custom count.
    init?(steps: Int) {
        guard let match = Self.allCases.first(where: { $0.steps == steps }) else { return nil }
        self = match
    }
}

struct AudioCloneOptions: Codable, Equatable, Sendable {
    /// Diffusion steps; fewer is faster. Balanced matched Best on
    /// intelligibility and speaker similarity in the step study.
    var steps = CloneQuality.balanced.steps
    /// Intelligibility guidance weight (`p_w`).
    var intelligibility: Float = 1.4
    /// Similarity guidance weight (`t_w`).
    var similarity: Float = 3.0
    /// The exact reference region the user confirmed they may clone. Consent
    /// lapses when the reference or its trim changes.
    var consentedReference: AudioVoiceReference?
}

struct AudioVoiceReference: Codable, Equatable, Sendable {
    var sourceID: UUID
    var title: String
    var clips: [AudioLibraryClip]
    var start: Double = 0
    var duration: Double
    var totalDuration: Double { clips.map { $0.offset + $0.duration }.max() ?? 0 }
    var isValid: Bool {
        start.isFinite && duration.isFinite && start >= 0 && (1...30).contains(duration)
            && start + duration <= totalDuration + 0.001 && !clips.isEmpty
    }
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
    /// The label as first assigned. For a voice with `speakerID` the take's
    /// names and merges decide what shows (`AudioSpeakerRoster`); this copy
    /// is what builds from before speaker ids show.
    var speaker: String? = nil
    var timing = "segment"
    /// The diarization id of the voice (`S1`, `S2`, ...), kept so a rename or
    /// merge on the take can be undone. Optional so older takes decode.
    var speakerID: String? = nil
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
    /// Free-form labels. Optional so records saved before tags still decode
    /// (synthesized Codable requires every non-optional key).
    var tags: [String]? = nil
    /// The `AudioRecordingPreset` a take was recorded with, by raw value.
    var recordingPreset: String? = nil
    /// Names given to voices, by speaker key (`AudioSpeakerRoster.key`).
    /// Kept on the take, never in the segments, so every revision and export
    /// shows the same name and clearing one restores the default label.
    var speakerNames: [String: String]? = nil
    /// Voices shown as another voice, by speaker key. Removing an entry
    /// separates them again.
    var speakerMerges: [String: String]? = nil
    /// How many overlapping parts speaker labeling ran in, when the take was
    /// longer than the diarization model accepts at once.
    var speakerLabelParts: Int? = nil
    /// Extracted voiceprints for speakers in this take, keyed by speaker key.
    /// Used by `AudioSpeakerMemory` for cross-meeting recognition and naming suggestions.
    var speakerVoiceprints: [String: AudioSpeakerVoiceprint]? = nil
    var tagList: [String] { tags ?? [] }
    var preferredTranscript: AudioTranscriptRevision? {
        transcripts.last(where: { $0.source == .user }) ?? transcripts.last
    }
    var duration: Double { clips.map { $0.offset + $0.duration }.max() ?? 0 }
    var hasPlayableAudio: Bool { clips.contains { $0.frameCount > 0 && $0.duration.isFinite && $0.duration > 0 } }
    // Older builds persisted a job before synthesis. Keep explicit drafts,
    // but do not present empty failed/interrupted jobs as saved audio.
    var isEmptyGeneratedAttempt: Bool {
        (kind == .music || kind == .voiceover) && status != .draft && !hasPlayableAudio && transcripts.isEmpty
    }
    var assetReferences: [String] { clips.map(\.assetReference) + (recipe.speech?.reference?.clips.map(\.assetReference) ?? []) }
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
              } }),
              // Only sizes no editor can produce: ordinary tags never fail a load.
              tagList.count <= 256, tagList.allSatisfy({ $0.utf8.count <= 1_024 }),
              Self.isBounded(speakerNames), Self.isBounded(speakerMerges),
              Self.isBoundedVoiceprints(speakerVoiceprints) else { throw CocoaError(.coderReadCorrupt) }
    }
    private static func isBounded(_ map: [String: String]?) -> Bool {
        guard let map else { return true }
        return map.count <= 10_000 && map.allSatisfy { $0.key.utf8.count <= 1_024 && $0.value.utf8.count <= 1_024 }
    }
    private static func isBoundedVoiceprints(_ map: [String: AudioSpeakerVoiceprint]?) -> Bool {
        guard let map else { return true }
        return map.count <= 256 && map.allSatisfy { $0.key.utf8.count <= 1_024 && $0.value.vector.count <= 4_096 }
    }
    @discardableResult mutating func addTranscript(_ segments: [AudioLibrarySegment], source: AudioTranscriptRevision.Source, language: String? = nil, nativeResult: Data? = nil) -> UUID {
        let revision = AudioTranscriptRevision(source: source, segments: segments, language: language, nativeResult: nativeResult)
        transcripts.append(revision)
        updatedAt = Date()
        return revision.id
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
    var speechModes: [String: AudioSpeechSettings]? = nil
    /// Security-scoped bookmarks retain user-selected local models without absolute paths.
    var localModels: [String: Data] = [:]
    var selectedModels: [String: String] = [:]
    /// The Record page's preset (`AudioRecordingPreset` raw value).
    var recordingPreset: String? = nil
    /// Speech-to-Text's mode (`SpeechToTextMode` raw value). Optional and a
    /// raw string so older profiles decode and an unknown value falls back.
    var speechToTextMode: String? = nil
    /// Whether each Speech-to-Text mode labels speakers, by mode raw value.
    /// Optional so older profiles decode. A missing mode uses its default.
    var speakerLabels: [String: Bool]? = nil
    /// Each preset's collection name, localized once at first use and kept,
    /// so changing the app language does not split a collection.
    var presetCollections: [String: String]? = nil
}

final class AudioLibraryStore: @unchecked Sendable {
    private let repository: ProfileRepository
    private let lock = NSRecursiveLock()
    init(repository: ProfileRepository = .shared) { self.repository = repository }
    func load() throws -> [AudioLibraryItem] { try loadReporting().items }
    /// Every readable item, and how many records could not be read. A record
    /// that does not decode or validate is copied aside and skipped, so one
    /// bad record cannot hide the whole library.
    func loadReporting() throws -> (items: [AudioLibraryItem], unreadable: Int) {
        lock.lock(); defer { lock.unlock() }
        var items: [AudioLibraryItem] = []
        var unreadable = 0
        for (key, data) in try repository.rawRecords(prefix: "audio:item:") {
            do {
                let item = try JSONDecoder().decode(AudioLibraryItem.self, from: data)
                try item.validate()
                items.append(item)
            } catch {
                unreadable += 1
                try? repository.quarantineRawRecord(key: key, payload: data)
            }
        }
        return (items.sorted { $0.createdAt > $1.createdAt }, unreadable)
    }
    func save(_ item: AudioLibraryItem) throws {
        lock.lock(); defer { lock.unlock() }
        try item.validate()
        try repository.saveAudioRecord(key: "audio:item:\(item.id.uuidString)", payload: JSONEncoder().encode(item), references: item.assetReferences)
        // The record is the truth. A failed index write is repaired by the
        // next save or the launch backfill, so it never fails the save.
        try? indexForSearch(item)
    }
    /// Brings the search index up to date for one item. Skips the write when
    /// nothing searchable changed (capture saves every 5 s).
    func indexForSearch(_ item: AudioLibraryItem) throws {
        lock.lock(); defer { lock.unlock() }
        guard let index = try repository.audioIndex(), let projection = item.searchProjection,
              try index.audioIndexHash(itemID: item.id) != projection.hash else { return }
        try index.replaceAudioSearchProjection(projection)
    }
    func delete(_ id: UUID) throws {
        lock.lock(); defer { lock.unlock() }
        try repository.deleteAudioRecord(key: "audio:item:\(id.uuidString)")
        try? repository.audioIndex()?.deleteAudioSearchRows(itemID: id)
    }
    /// Exact-word matches across transcripts, titles, collections and tags.
    func lexicalSearch(_ query: String) -> [AudioLexicalHit] {
        lock.lock(); defer { lock.unlock() }
        return (try? repository.audioIndex()?.audioLexicalSearch(query: query)) ?? []
    }
    func searchIndex() -> ProfileDatabase? {
        lock.lock(); defer { lock.unlock() }
        return try? repository.audioIndex()
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
        let recipes: [AudioRecipe] = [value.recipe] + value.presets.map(\.recipe)
        var references: [String] = recipes.flatMap { $0.speech?.reference?.clips.map(\.assetReference) ?? [] }
        for settings in value.speechModes?.values ?? Dictionary<String, AudioSpeechSettings>().values {
            references.append(contentsOf: settings.reference?.clips.map(\.assetReference) ?? [])
        }
        try repository.saveAudioRecord(key: "audio:preferences", payload: JSONEncoder().encode(value), references: references)
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
