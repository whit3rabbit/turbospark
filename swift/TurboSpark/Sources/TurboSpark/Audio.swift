import CTurboSpark
import Foundation

public enum AudioTask: String, Codable, Sendable, CaseIterable {
    case speechToText = "speech_to_text", textToSpeech = "text_to_speech", music
    case enhancement, separation, alignment, diarization, speechDetection = "speech_detection"
    case codec, languageIdentification = "language_identification"
}
public struct AudioProfileIdentity: Codable, Sendable, Hashable {
    public let task: AudioTask
    public let alias: String
    public let repository: String
    public let revision: String
    public let assetFingerprint: String
    public init(task: AudioTask, alias: String, repository: String, revision: String, assetFingerprint: String) {
        self.task = task; self.alias = alias; self.repository = repository
        self.revision = revision; self.assetFingerprint = assetFingerprint
    }
}
public struct AudioCapabilities: Codable, Sendable, Equatable {
    public let languages: [String]
    public let voices: [String]
    public let maxInputSeconds: Int?
    public let supportsLyrics: Bool
    public let operations: [String]
    public let timing: String?
    public let backend: String
    public let cancellation: String
    public let canRun: Bool
    public let unavailableReason: String?
    public init(languages: [String] = [], voices: [String] = [], maxInputSeconds: Int? = nil,
                supportsLyrics: Bool = false, operations: [String], timing: String? = nil,
                backend: String, cancellation: String, canRun: Bool, unavailableReason: String? = nil) {
        self.languages = languages; self.voices = voices; self.maxInputSeconds = maxInputSeconds
        self.supportsLyrics = supportsLyrics; self.operations = operations; self.timing = timing
        self.backend = backend; self.cancellation = cancellation; self.canRun = canRun
        self.unavailableReason = unavailableReason
    }
}
public struct AudioPCMFormat: Codable, Sendable, Equatable {
    public let sampleRate: UInt32
    public let channels: UInt32
    public let interleaved: Bool
    public init(sampleRate: UInt32, channels: UInt32, interleaved: Bool = true) {
        self.sampleRate = sampleRate; self.channels = channels; self.interleaved = interleaved
    }
}
public struct AudioProfile: Codable, Sendable, Identifiable, Equatable {
    public var id: AudioProfileIdentity { identity }
    public let identity: AudioProfileIdentity
    public let family: String
    public let displayName: String
    public let capabilities: AudioCapabilities
    public let pcmFormat: AudioPCMFormat
    public let downloadBytes: UInt64
    public let residentMemoryBytes: UInt64
    public let residentMemoryEvidence: String
    public let readiness: String
    public let evidence: [String]
    /// Local-only descriptions need no invented repository revision or install plan.
    /// Use empty identity source fields and do not pass them to catalog mutation APIs.
    public init(identity: AudioProfileIdentity, family: String, displayName: String,
                capabilities: AudioCapabilities, pcmFormat: AudioPCMFormat,
                downloadBytes: UInt64 = 0, residentMemoryBytes: UInt64 = 0,
                residentMemoryEvidence: String = "unknown", readiness: String, evidence: [String] = []) {
        self.identity = identity; self.family = family; self.displayName = displayName
        self.capabilities = capabilities; self.pcmFormat = pcmFormat; self.downloadBytes = downloadBytes
        self.residentMemoryBytes = residentMemoryBytes; self.residentMemoryEvidence = residentMemoryEvidence
        self.readiness = readiness; self.evidence = evidence
    }
}
public struct AudioInstalledRecord: Codable, Sendable, Equatable, Identifiable {
    public var id: String { alias }
    public let alias: String
    public let path: String
    public let family: String
    public let status: String
    public let installBytes: UInt64
}
public struct AudioLegacyInstall: Codable, Sendable, Equatable {
    public let record: AudioInstalledRecord
    public let identity: AudioProfileIdentity
}
public struct AudioIncompatibleInstall: Codable, Sendable, Equatable {
    public let record: AudioInstalledRecord
    public let identity: AudioProfileIdentity?
    public let error: String
}
public struct AudioInstalledReport: Codable, Sendable, Equatable {
    public let installed: [AudioInstalledRecord]
    public let needsAdoption: [AudioLegacyInstall]
    public let incompatible: [AudioIncompatibleInstall]
    public let otherAudio: [AudioInstalledRecord]
}
public struct AudioFamilyCapability: Codable, Sendable, Identifiable, Equatable {
    public var id: String { family }
    public let family: String
    public let task: AudioTask
    public let operations: [String]
    public let backend: String
    public let canRun: Bool
    public let component: Bool
    public let unavailableReason: String?
    public let source: String
    public let evidence: String
}
public enum AudioCatalog {
    public static func capabilities() throws -> [AudioFamilyCapability] {
        try audioDecode([AudioFamilyCapability].self, try takeString { ts_audio_capabilities_json($0) })
    }
    public static func profiles() throws -> [AudioProfile] {
        try audioDecode([AudioProfile].self, try takeString { ts_audio_catalog_json($0) })
    }
    public static func installed() throws -> AudioInstalledReport {
        try audioDecode(AudioInstalledReport.self, try takeString { ts_audio_installed_json($0) })
    }
    public static func resolve(_ identity: AudioProfileIdentity) throws -> String {
        let json = try audioEncode(identity)
        return try takeString { out in json.withCString { ts_audio_resolve($0, out) } }
    }
    public static func delete(_ identity: AudioProfileIdentity) throws {
        let json = try audioEncode(identity)
        let status = json.withCString { ts_audio_delete($0) }
        if status != 0 { throw TurboSparkError.fromLastError(status) }
    }
    public static func adopt(_ identity: AudioProfileIdentity) throws -> AudioInstalledRecord {
        let json = try audioEncode(identity)
        return try audioDecode(AudioInstalledRecord.self, takeString { out in json.withCString { ts_audio_adopt($0, out) } })
    }
    /// Blocking transfer; call off the main actor. Existing Catalog install controls apply.
    public static func install(_ identity: AudioProfileIdentity, progress: @escaping @Sendable (AudioDownloadProgress) -> Void = { _ in }) throws -> AudioInstalledRecord {
        let json = try audioEncode(identity)
        let box = AudioDownloadBox(progress)
        let context = Unmanaged.passUnretained(box).toOpaque()
        return try audioDecode(AudioInstalledRecord.self, takeString { out in json.withCString { identity in
            ts_audio_install(identity, { userdata, _, path, length, completed, total in
                guard let userdata else { return }
                let box = Unmanaged<AudioDownloadBox>.fromOpaque(userdata).takeUnretainedValue()
                let name = path.map { String(decoding: UnsafeBufferPointer(start: UnsafeRawPointer($0).assumingMemoryBound(to: UInt8.self), count: length), as: UTF8.self) }
                box.callback(AudioDownloadProgress(path: name, completedBytes: completed, totalBytes: total))
            }, context, out)
        } })
    }
}
public struct AudioDownloadProgress: Sendable {
    public let path: String?
    public let completedBytes: UInt64
    public let totalBytes: UInt64
}
private final class AudioDownloadBox {
    let callback: @Sendable (AudioDownloadProgress) -> Void
    init(_ callback: @escaping @Sendable (AudioDownloadProgress) -> Void) { self.callback = callback }
}
public struct AudioRequest: Codable, Sendable, Equatable {
    public var task: AudioTask
    public var language: String?
    public var text = ""
    public var voice = "af_heart"
    public var speed: Float = 1
    public var caption = ""
    public var lyrics = ""
    public var durationSeconds: Double? = 10
    public var steps: Int? = 30
    public var seed: UInt64? = 0
    public init(task: AudioTask, language: String? = nil) { self.task = task; self.language = language }
}
public struct AudioWord: Codable, Sendable, Equatable {
    public var text: String
    public let startSeconds: Double
    public let endSeconds: Double
    public init(text: String, startSeconds: Double, endSeconds: Double) { self.text = text; self.startSeconds = startSeconds; self.endSeconds = endSeconds }
}
public struct AudioSegment: Codable, Sendable, Equatable {
    public let startSeconds: Double?
    public let endSeconds: Double?
    public var text: String
    /// "window" and "clip" are extents, not measured word alignment.
    public let timing: String
    public var speaker: String?
    public let words: [AudioWord]?
    public init(startSeconds: Double?, endSeconds: Double?, text: String, timing: String, speaker: String? = nil, words: [AudioWord]? = nil) {
        self.startSeconds = startSeconds; self.endSeconds = endSeconds; self.text = text; self.timing = timing; self.speaker = speaker; self.words = words
    }
}
public struct AudioTranscript: Codable, Sendable, Equatable {
    public var text: String
    public let language: String?
    public var segments: [AudioSegment]
    public let tags: [String]?
    public let tokenLogprobs: [Float]?
    public init(text: String, language: String? = nil, segments: [AudioSegment], tags: [String]? = nil, tokenLogprobs: [Float]? = nil) {
        self.text = text; self.language = language; self.segments = segments; self.tags = tags; self.tokenLogprobs = tokenLogprobs
    }
}
public struct AudioPCM: Codable, Sendable, Equatable {
    public let format: AudioPCMFormat
    public let samples: [Float]
    public init(format: AudioPCMFormat, samples: [Float]) { self.format = format; self.samples = samples }
    public var durationSeconds: Double { Double(samples.count) / Double(max(1, format.channels)) / Double(max(1, format.sampleRate)) }
}
public struct AudioResult: Codable, Sendable, Equatable {
    public let transcript: AudioTranscript?
    public let pcm: AudioPCM?
    public let seed: UInt64?
    public init(transcript: AudioTranscript? = nil, pcm: AudioPCM? = nil, seed: UInt64? = nil) { self.transcript = transcript; self.pcm = pcm; self.seed = seed }
}
public struct AudioProgress: Codable, Sendable, Equatable {
    public let stage: String
    public let completed: Int
    public let total: Int?
}
private func audioEncode<T: Encodable>(_ value: T) throws -> String {
    let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase
    return String(decoding: try encoder.encode(value), as: UTF8.self)
}
private func audioDecode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: Data(json.utf8))
}
private struct AudioWireResult: Decodable {
    let transcript: AudioTranscript?
    let pcmFormat: AudioPCMFormat?
    let sampleCount: Int
    let seed: UInt64?
}
private final class AudioCallbackBox {
    var samples: [Float] = []
    let progress: @Sendable (AudioProgress) -> Void
    init(progress: @escaping @Sendable (AudioProgress) -> Void) { self.progress = progress }
}
public struct AudioJobStatus: Codable, Sendable, Equatable {
    public let state: String
    public let cancellationRequested: Bool
}
/// One job owns its native cancellation handle until its execution has drained.
public final class AudioJob: @unchecked Sendable {
    private let lock = NSLock()
    private var handle: OpaquePointer?
    init(_ handle: OpaquePointer) { self.handle = handle }
    public func cancel() { lock.lock(); defer { lock.unlock() }; if let handle { _ = ts_audio_job_cancel(handle) } }
    public func status() throws -> AudioJobStatus {
        lock.lock(); defer { lock.unlock() }
        guard let handle else { throw TurboSparkError(code: .invalidArgument, message: "Audio job is closed") }
        return try audioDecode(AudioJobStatus.self, takeString { ts_audio_job_status_json(handle, $0) })
    }
    func close() { lock.lock(); defer { lock.unlock() }; if let handle { _ = ts_audio_job_close(handle); self.handle = nil } }
}
/// A serial facade over a dedicated native worker. Cancellation never waits for the operation lock.
public final class AudioSession: @unchecked Sendable {
    private let operation = NSLock()
    private let control = NSLock()
    private var handle: OpaquePointer?
    private var job: AudioJob?
    private var stopping = false
    public init(modelPath: String, task: AudioTask, allowPortable: Bool = false, allowExperimentalMetal: Bool = false, expectedFamily: String? = nil) throws {
        struct Options: Encodable { let task: AudioTask; let allowPortable: Bool; let allowExperimentalMetal: Bool; let expectedFamily: String? }
        let json = try audioEncode(Options(task: task, allowPortable: allowPortable, allowExperimentalMetal: allowExperimentalMetal, expectedFamily: expectedFamily))
        let path = NSString(string: modelPath).expandingTildeInPath
        var native: OpaquePointer?
        let status = path.withCString { path in json.withCString { ts_audio_session_open(path, $0, &native) } }
        guard status == 0, let native else { throw TurboSparkError.fromLastError(status) }
        handle = native
    }
    deinit { close() }
    public func cancel() { control.lock(); let active = job; control.unlock(); active?.cancel() }
    /// Cancels and drains synchronously. Call off MainActor; progress must not wait on MainActor.
    public func close() {
        control.lock(); stopping = true; let active = job; control.unlock(); active?.cancel()
        operation.lock(); defer { operation.unlock() }
        if let handle { _ = ts_audio_session_close(handle); self.handle = nil }
    }
    public func execute(_ request: AudioRequest, pcm: [Float] = [], progress: @escaping @Sendable (AudioProgress) -> Void = { _ in }) async throws -> AudioResult {
        let cancellation = AudioCancellationSlot()
        return try await withTaskCancellationHandler {
            try await Task.detached { [self] in try executeBlocking(request, pcm: pcm, progress: progress, cancellation: cancellation) }.value
        } onCancel: { cancellation.cancel() }
    }
    private func executeBlocking(_ request: AudioRequest, pcm: [Float], progress: @escaping @Sendable (AudioProgress) -> Void, cancellation: AudioCancellationSlot) throws -> AudioResult {
        operation.lock(); defer { operation.unlock() }
        control.lock(); let closed = stopping; control.unlock()
        guard !closed, let handle else { throw TurboSparkError(code: .invalidArgument, message: "Audio session is closed") }
        guard pcm.count <= 16_000 * 1800, pcm.allSatisfy(\.isFinite) else { throw TurboSparkError(code: .invalidArgument, message: "Audio input must be finite and no longer than 30 minutes") }
        let json = try audioEncode(request)
        var raw: OpaquePointer?
        let status = json.withCString { ts_audio_job_open(handle, $0, &raw) }
        guard status == 0, let raw else { throw TurboSparkError.fromLastError(status) }
        let active = AudioJob(raw)
        control.lock(); job = active; let stopNow = stopping; control.unlock()
        cancellation.install(active)
        if stopNow { active.cancel() }
        defer { cancellation.clear(); control.lock(); job = nil; control.unlock(); active.close() }
        for start in stride(from: 0, to: pcm.count, by: 32_000) {
            let count = min(32_000, pcm.count - start)
            let status = pcm.withUnsafeBufferPointer { ts_audio_job_append(raw, $0.baseAddress!.advanced(by: start), count) }
            if status != 0 { throw TurboSparkError.fromLastError(status) }
        }
        let box = AudioCallbackBox(progress: progress)
        let context = Unmanaged.passUnretained(box).toOpaque()
        let result = try takeString { out in
            ts_audio_job_run(raw, { userdata, kind, json, length, samples, count in
                guard let userdata else { return }
                let box = Unmanaged<AudioCallbackBox>.fromOpaque(userdata).takeUnretainedValue()
                if kind == 1, let json, let update = try? audioDecode(AudioProgress.self, String(decoding: UnsafeBufferPointer(start: UnsafeRawPointer(json).assumingMemoryBound(to: UInt8.self), count: length), as: UTF8.self)) { box.progress(update) }
                if kind == 2, let samples { box.samples.append(contentsOf: UnsafeBufferPointer(start: samples, count: count)) }
            }, context, out)
        }
        let wire = try audioDecode(AudioWireResult.self, result)
        guard wire.sampleCount == box.samples.count else { throw TurboSparkError(code: .invalidArgument, message: "Native audio output count mismatch") }
        return AudioResult(transcript: wire.transcript, pcm: wire.pcmFormat.map { AudioPCM(format: $0, samples: box.samples) }, seed: wire.seed)
    }
}
private final class AudioCancellationSlot: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var job: AudioJob?
    func cancel() { lock.lock(); cancelled = true; let active = job; lock.unlock(); active?.cancel() }
    func install(_ job: AudioJob) { lock.lock(); self.job = job; let cancelNow = cancelled; lock.unlock(); if cancelNow { job.cancel() } }
    func clear() { lock.lock(); job = nil; lock.unlock() }
}

/// Reserve native heavyweight execution for external MLX work. Do not wrap Rust generation.
public final class NativeHeavyWorkPermit: @unchecked Sendable {
    private let lock = NSLock()
    private var handle: OpaquePointer?
    private init(_ handle: OpaquePointer) { self.handle = handle }
    public static func tryAcquire() throws -> NativeHeavyWorkPermit? {
        var handle: OpaquePointer?
        let status = ts_heavy_work_try_acquire(&handle)
        guard status == 0 else { throw TurboSparkError.fromLastError(status) }
        return handle.map(NativeHeavyWorkPermit.init)
    }
    public func release() {
        lock.lock(); defer { lock.unlock() }
        if let handle { _ = ts_heavy_work_release(handle); self.handle = nil }
    }
    deinit { release() }
}
