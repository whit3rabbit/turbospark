import Foundation
import Metal
import MLX
import Qwen3TTS
import TurboSpark

/// Profile teardown must drain the producer before its reference asset keys expire.
final class Qwen3TTSSpeechJob: @unchecked Sendable {
    private let lock = NSLock()
    private let drain = DispatchGroup()
    private var worker: Task<AudioResult, Error>?
    private var cancelled = false
    let openToken: AudioOpenToken

    init() throws { openToken = try AudioOpenToken() }

    func start(_ operation: @escaping @Sendable () async throws -> AudioResult) throws -> Task<AudioResult, Error> {
        lock.lock(); defer { lock.unlock() }
        guard worker == nil, !cancelled else { throw CancellationError() }
        drain.enter()
        let group = drain
        let task = Task.detached(priority: .userInitiated) {
            defer { group.leave() }
            try Task.checkCancellation()
            return try await operation()
        }
        worker = task
        return task
    }
    func cancel() {
        lock.lock(); cancelled = true; let task = worker; lock.unlock()
        openToken.cancel(); task?.cancel()
    }
    func stopAndDrain() { cancel(); drain.wait() }

    static func generate(recipe: AudioRecipe, profile: AudioProfile, bookmark: Data?, installedPath: String?,
                         assets: ManagedAssetStore, lifetime: AudioOperationLifetime, openToken: AudioOpenToken,
                         report: @escaping @Sendable (AudioProgress) -> Void) async throws -> AudioResult {
        try Task.checkCancellation(); try lifetime.check()
        var permit: NativeHeavyWorkPermit?
        while permit == nil {
            try Task.checkCancellation()
            permit = try NativeHeavyWorkPermit.tryAcquire()
            if permit == nil { try await Task.sleep(for: .milliseconds(50)) }
        }
        defer { permit?.release() }
        let directory: URL
        var scoped = false
        if let bookmark {
            var stale = false
            directory = try URL(resolvingBookmarkData: bookmark, options: [.withSecurityScope, .withoutUI, .withoutMounting], relativeTo: nil, bookmarkDataIsStale: &stale)
            guard !stale else { throw AudioWorkspaceError.message(String(localized: "The model folder moved. Choose it again.", bundle: .module)) }
            scoped = directory.startAccessingSecurityScopedResource()
        } else {
            let verified = try AudioCatalog.verify(profile.identity, token: openToken, progress: report)
            guard verified == installedPath else { throw CancellationError() }
            directory = URL(fileURLWithPath: verified, isDirectory: true)
        }
        defer { if scoped { directory.stopAccessingSecurityScopedResource() } }
        let settings = recipe.speech ?? AudioSpeechSettings()
        let checkpoint = try Qwen3TTSCheckpoint(data: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let expected: Qwen3TTSCheckpoint.Variant = settings.mode == .clone ? .base : settings.mode == .design ? .voiceDesign : .customVoice
        let expectedSize = profile.identity.repository.contains("-0.6B-") ? "0b6" : "1b7"
        guard checkpoint.variant == expected, checkpoint.size == expectedSize else { throw Qwen3TTSError.incompatibleProfile }
        guard settings.temperature.isFinite, (0...2).contains(settings.temperature), (64...4096).contains(settings.maxTokens) else {
            throw Qwen3TTSError.invalidRequest
        }
        // This is a conservative admission estimate, not measured peak memory.
        let weightBytes = try [directory, directory.appendingPathComponent("speech_tokenizer")].reduce(UInt64(0)) { sum, folder in
            try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.fileSizeKey])
                .filter { $0.pathExtension == "safetensors" }.reduce(sum) { try $0 + UInt64($1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
        }
        let budget = min(ProcessInfo.processInfo.physicalMemory * 3 / 4, MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
        guard weightBytes > 0, weightBytes <= budget / 2 else {
            throw AudioWorkspaceError.message(String(localized: "This checkpoint exceeds the available audio model memory budget.", bundle: .module))
        }
        let reference: [Float]?
        if settings.mode == .clone {
            guard let source = settings.reference, source.isValid,
                  settings.embeddingOnly || !settings.referenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw Qwen3TTSError.invalidReference
            }
            report(AudioProgress(stage: "Preparing reference audio", completed: 0))
            reference = try AudioMediaIO.readMono24k(source.clips, store: assets, start: source.start, duration: source.duration) {
                try Task.checkCancellation(); try lifetime.check()
            }
        } else { reference = nil }
        let voice: String?
        switch settings.mode {
        case .clone: voice = nil
        case .design: voice = settings.instruction
        case .customVoice:
            voice = recipe.voice + (settings.instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : "," + settings.instruction)
        }
        try checkpoint.validate(text: recipe.text, voice: voice, reference: reference,
                                referenceText: settings.referenceText, language: recipe.language)
        report(AudioProgress(stage: "Loading speech model", completed: 0))
        var model: Qwen3TTSModel?
        defer {
            // MLX keeps freed buffers for reuse. A completed clip should release
            // those buffers before another heavyweight task takes the permit.
            model = nil
            MLX.Memory.clearCache()
        }
        model = try await Qwen3TTSModel.fromModelDirectory(directory)
        try Task.checkCancellation(); try lifetime.check()
        let seed = recipe.seed == 0 ? UInt64.random(in: 1...UInt64.max) : recipe.seed
        MLXRandom.seed(seed)
        var count = 0
        report(AudioProgress(stage: "Generating speech", completed: 0))
        let samples = try model!.synthesize(text: recipe.text, voice: voice, reference: reference,
            referenceText: settings.embeddingOnly ? nil : settings.referenceText, language: recipe.language,
            parameters: GenerateParameters(maxTokens: settings.maxTokens, temperature: settings.temperature), onToken: { _ in
                count += 1
                if count % 12 == 0 { report(AudioProgress(stage: "Generating speech", completed: count)) }
            })
        try Task.checkCancellation(); try lifetime.check()
        guard !samples.isEmpty, samples.count <= 24_000 * 360, samples.allSatisfy(\.isFinite),
              samples.contains(where: { abs($0) > 1e-5 }) else { throw Qwen3TTSError.invalidRequest }
        return AudioResult(pcm: AudioPCM(format: AudioPCMFormat(sampleRate: 24_000, channels: 1), samples: samples), seed: seed)
    }
}

extension AudioWorkspaceController {
    func synthesizeQwenSpeech(_ recipe: AudioRecipe, profile: AudioProfile, source: AudioLibraryItem,
                              report: @escaping @Sendable (AudioProgress) -> Void) async throws -> AudioResult {
        idleTask?.cancel()
        let old = session; let access = modelAccessURL
        session = nil; sessionKey = nil; modelAccessURL = nil
        if let old { await Task.detached { old.close() }.value }
        access?.stopAccessingSecurityScopedResource()
        try Task.checkCancellation()
        let job = try Qwen3TTSSpeechJob()
        qwenSpeechJob = job
        defer { if qwenSpeechJob === job { qwenSpeechJob = nil } }
        let assets = self.assets; let lifetime = self.lifetime; let path = installedPaths[profile.identity.alias]
        let worker = try job.start {
            try await Qwen3TTSSpeechJob.generate(recipe: recipe, profile: profile, bookmark: source.localModelBookmark,
                installedPath: path, assets: assets, lifetime: lifetime, openToken: job.openToken, report: report)
        }
        return try await withTaskCancellationHandler { try await worker.value } onCancel: { job.cancel() }
    }
}
