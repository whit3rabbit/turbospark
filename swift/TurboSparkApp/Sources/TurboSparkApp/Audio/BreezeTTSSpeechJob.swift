import Foundation
import Metal
import MLX
import Qwen3TTS
import TurboSpark

/// Breeze TTS 2 runs in the vendored Swift MLX package. The run's
/// `AudioRequest` is built by the shared native-clone path
/// (`runRequest`), so consent, the reference transcript and the seed policy
/// are the same ones the Rust cloners get; this maps that request onto the
/// model.
enum BreezeTTSSpeech {
    /// Upstream recommends guidance 4 whenever an instruction is given
    /// (voice design and direction). Without one, guidance does not run.
    static let instructionGuidance: Float = 4

    /// `[S0]`..`[S9]` from the recipe's voice; anything else is `S0`, the
    /// model's documented default.
    static func speaker(_ voice: String?) -> Int {
        guard let voice, voice.count == 2, voice.first == "S", let n = Int(voice.dropFirst()), (0 ... 9).contains(n) else { return 0 }
        return n
    }

    /// The model request for one run. A reference is used only with consent
    /// for it: the job checks again here rather than trusting that every
    /// caller went through the page's gate.
    static func modelRequest(_ request: AudioRequest, reference: [Float]) throws -> BreezeTTSRequest {
        let instruction = request.instruction?.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasInstruction = !(instruction ?? "").isEmpty
        var model = BreezeTTSRequest(
            text: request.text ?? "",
            speaker: speaker(request.voice),
            instruction: hasInstruction ? instruction : nil,
            cfgScale: hasInstruction ? instructionGuidance : nil,
            seed: request.seed.flatMap { $0 == 0 ? nil : $0 })
        if !reference.isEmpty {
            guard request.consent else {
                throw AudioWorkspaceError.message(AppStrings.localized("Confirm you have the right to use this voice."))
            }
            guard let transcript = request.referenceText?.trimmingCharacters(in: .whitespacesAndNewlines), !transcript.isEmpty else {
                throw AudioWorkspaceError.message(AppStrings.localized("Enter the exact transcript of the selected reference region."))
            }
            model.referenceAudio = reference
            model.referenceText = transcript
        }
        return model
    }

    static func generate(_ request: AudioRequest, reference: [Float], model: BreezeTTSModel,
                         lifetime: AudioOperationLifetime,
                         report: @escaping @Sendable (AudioProgress) -> Void) async throws -> AudioResult {
        try Task.checkCancellation(); try lifetime.check()
        let modelRequest = try modelRequest(request, reference: reference)
        var permit: NativeHeavyWorkPermit?
        while permit == nil {
            try Task.checkCancellation()
            permit = try NativeHeavyWorkPermit.tryAcquire()
            if permit == nil { try await Task.sleep(for: .milliseconds(50)) }
        }
        defer { permit?.release() }
        report(AudioProgress(stage: "Generating speech", completed: 0))
        let result = try model.generate(modelRequest, onFrame: { frames, _, _ in
            if frames % 12 == 0 { report(AudioProgress(stage: "Generating speech", completed: frames)) }
        })
        try Task.checkCancellation(); try lifetime.check()
        // A segment that never reached the end class likely trails off or
        // loops; a fresh seed is the remedy, as for Drama 3 (`retakeRequest`).
        return AudioResult(pcm: AudioPCM(format: AudioPCMFormat(sampleRate: UInt32(result.sampleRate), channels: 1), samples: result.samples),
                           seed: modelRequest.seed, stoppedEarly: result.hitFrameLimit)
    }
}

extension AudioWorkspaceController {
    func synthesizeBreezeSpeech(_ request: AudioRequest, reference: [Float], profile: AudioProfile, source: AudioLibraryItem,
                                report: @escaping @Sendable (AudioProgress) -> Void) async throws -> AudioResult {
        // Like `synthesizeQwenSpeech`: the model opens from its folder, not a
        // Rust session, so the move check and session handoff happen here.
        if let installed = installedPaths[profile.identity.alias], modelMoving?(installed) == true {
            throw AudioWorkspaceError.message(ModelMoveText.modelBeingMoved())
        }
        idleTask?.cancel()
        let old = session; let access = modelAccessURL
        session = nil; sessionKey = nil; modelAccessURL = nil
        if let old { await Task.detached { old.close() }.value }
        access?.stopAccessingSecurityScopedResource()
        try Task.checkCancellation()
        let model = try await residentBreezeSession(for: profile, source: source)
        // The Qwen job type is model-agnostic: it owns the producer task so
        // Stop cancels it and shutdown drains it (`qwenSpeechJob`).
        let job = try Qwen3TTSSpeechJob()
        qwenSpeechJob = job
        defer { if qwenSpeechJob === job { qwenSpeechJob = nil } }
        let lifetime = self.lifetime
        let worker = try job.start {
            try await BreezeTTSSpeech.generate(request, reference: reference, model: model, lifetime: lifetime, report: report)
        }
        return try await withTaskCancellationHandler { try await worker.value } onCancel: { job.cancel() }
    }
}

extension AudioWorkspaceController {
    /// Loads Breeze as the one resident audio model, as `residentQwenSession`
    /// does for Qwen3-TTS: a managed install loads only after its bytes
    /// verify, inside `AudioModelOpening.perform` so Stop cancels it and a
    /// shutdown drain never waits on an open that never comes.
    func residentBreezeSession(for profile: AudioProfile, source: AudioLibraryItem? = nil) async throws -> BreezeTTSModel {
        idleTask?.cancel()
        var url: URL?
        let directory: URL
        let path: String
        if let bookmark = try modelBookmark(for: profile, source: source) {
            var stale = false
            let resolved = try URL(resolvingBookmarkData: bookmark, options: .withSecurityScope, relativeTo: nil, bookmarkDataIsStale: &stale)
            guard !stale else { throw AudioWorkspaceError.message(AppStrings.localized("The model folder moved. Choose it again.")) }
            _ = resolved.startAccessingSecurityScopedResource(); url = resolved; directory = resolved; path = resolved.path
        } else if let installed = installedPaths[profile.identity.alias] {
            path = installed
            directory = URL(fileURLWithPath: installed, isDirectory: true)
        } else {
            throw AudioWorkspaceError.message(AppStrings.localized("Download this model or choose its checkpoint folder."))
        }
        let key = "breeze_tts:\(path)"
        if sessionKey == key, let breezeModel {
            url?.stopAccessingSecurityScopedResource()
            return breezeModel
        }
        if let previous = session { await Task.detached { previous.close() }.value }
        session = nil
        modelAccessURL?.stopAccessingSecurityScopedResource(); modelAccessURL = nil
        qwenModel = nil; breezeModel = nil; voiceIsolatorModel = nil; liveVoiceIsolator = nil; fluidTranscriber = nil; sessionKey = nil; loadedModelAlias = nil
        MLX.Memory.clearCache()

        let token = epoch
        do {
            // A conservative admission estimate from weight bytes, not a
            // measured peak.
            let weightBytes = try [directory, directory.appendingPathComponent("audio_tokenizer")].reduce(UInt64(0)) { sum, folder in
                guard FileManager.default.fileExists(atPath: folder.path) else { return sum }
                return try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.fileSizeKey])
                    .filter { $0.pathExtension == "safetensors" }.reduce(sum) { try $0 + UInt64($1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
            }
            let budget = min(ProcessInfo.processInfo.physicalMemory * 3 / 4, MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
            guard weightBytes > 0, weightBytes <= budget / 2 else {
                throw AudioWorkspaceError.message(AppStrings.localized("This checkpoint exceeds the available audio model memory budget."))
            }

            let opening = AudioModelOpening(token: qwenLoadDependencies.openToken())
            opening.onProgress = { [weak self, weak opening] fraction in
                Task { @MainActor in
                    guard let self, let opening, self.valid(token), self.opening === opening else { return }
                    self.status = AppStrings.localized("Verifying")
                    self.progress = fraction
                }
            }
            self.opening = opening
            let verifies = url == nil
            let verify = qwenLoadDependencies.verify
            let verified = try await Task.detached {
                try opening.perform { () throws -> String in
                    guard verifies else { return path }
                    guard let openToken = opening.token else { throw CancellationError() }
                    return try verify(profile.identity, openToken) { progress in
                        guard let total = progress.total, total > 0 else { return }
                        opening.onProgress?(min(1, Double(progress.completed) / Double(total)))
                    }
                }
            }.value
            guard verified == path else { throw CancellationError() }

            self.status = AppStrings.localized("Loading model")
            let load = breezeLoad
            let loaded = try await Task.detached {
                try Task.checkCancellation()
                return try await load(directory)
            }.value

            self.opening = nil
            self.progress = nil
            guard valid(token), !Task.isCancelled else { throw CancellationError() }
            breezeModel = loaded
            sessionKey = key
            loadedModelAlias = profile.identity.alias
            modelAccessURL = url
            return loaded
        } catch {
            opening = nil
            progress = nil
            url?.stopAccessingSecurityScopedResource()
            throw error
        }
    }
}
